from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.experiment import JobState, load_ledger, load_manifest
from extensions.benchmark_tools.runner import _jobs_to_run, _workflow, run_benchmark
from extensions.benchmark_tools.schema import Condition
from extensions.benchmark_tools.site import build_site
from multi_agent_research.llm import LLMCallError
from multi_agent_research.models import (
    CallError,
    ModelCallRecord,
    UsageStats,
    utc_now,
)
from tests.fakes import DelayedFakeLLMClient, FakeLLMClient


def _write_tasks(path, count: int = 1) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "id": f"task-{index + 1}",
            "prompt": f"Question {index + 1}?",
            "answer": "A",
            "answer_type": "multiple_choice",
            "choices": [
                {"label": "A", "text": "Alpha"},
                {"label": "B", "text": "Beta"},
            ],
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_run_persists_manifest_ledger_and_resumes(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)

    first = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="pilot",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient(["<final_answer>A</final_answer>"]),
        max_attempts=1,
    )
    second = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="pilot",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient([]),
        concurrency=2,
        max_in_flight_requests=2,
        tokens_per_minute=150_000,
        max_attempts=4,
        request_max_attempts=2,
    )

    experiment_root = results / "pilot"
    manifest = load_manifest(experiment_root / "experiment-manifest.json")
    ledger = load_ledger(
        experiment_root / "experiment-ledger.json",
        manifest=manifest,
    )
    record = next(iter(ledger.jobs.values()))

    assert first["scheduled_jobs"] == 1
    assert second["scheduled_jobs"] == 0
    assert record.state.value == "success"
    assert record.attempt_count == 1
    assert record.latest_run_id
    assert manifest.policy.concurrency == 2
    assert manifest.policy.tokens_per_minute == 150_000
    assert (experiment_root / record.latest_run_id / "artifact-manifest.json").exists()


@pytest.mark.asyncio
async def test_runtime_workflow_exclusion_preserves_jobs_for_later_resume(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)
    conditions = [
        Condition(id="solo", workflow="solo"),
        Condition(
            id="sample",
            workflow="sample",
            agents=1,
            aggregation="plurality_vote",
        ),
    ]
    events = []

    first = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="filtered",
        output_dir=results,
        conditions=conditions,
        excluded_workflows={"sample"},
        llm=FakeLLMClient(["<final_answer>A</final_answer>"]),
        max_attempts=1,
        event_handler=events.append,
        emit_json_events=False,
    )

    root = results / "filtered"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    records = {record.spec.condition_id: record for record in ledger.jobs.values()}

    assert len(manifest.conditions) == 2
    assert records["solo"].state.value == "success"
    assert records["sample"].state.value == "pending"
    assert first["scheduled_jobs"] == 1
    assert first["scope_jobs"] == 1
    assert first["deferred_jobs"] == 1
    assert events[0]["total_jobs"] == 1
    assert events[0]["completed_jobs"] == 0
    assert events[0]["deferred_jobs"] == 1

    second = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="filtered",
        output_dir=results,
        conditions=conditions,
        llm=FakeLLMClient(["<final_answer>A</final_answer>"]),
        max_attempts=1,
    )
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    records = {record.spec.condition_id: record for record in ledger.jobs.values()}

    assert second["scheduled_jobs"] == 1
    assert records["solo"].attempt_count == 1
    assert records["sample"].state.value == "success"
    assert records["sample"].attempt_count == 1


@pytest.mark.asyncio
async def test_unknown_workflow_exclusion_fails_before_writing(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)

    with pytest.raises(ValueError, match="not present"):
        await run_benchmark(
            tasks_path=tasks,
            model="fake/model",
            experiment_id="bad-filter",
            output_dir=results,
            conditions=[Condition(id="solo", workflow="solo")],
            excluded_workflows={"debate"},
        )

    assert not results.exists()


@pytest.mark.asyncio
async def test_reasoning_effort_exclusion_preserves_jobs_for_later_resume(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)
    conditions = [
        Condition(id="solo-none", workflow="solo", reasoning_effort="none"),
        Condition(id="solo-high", workflow="solo", reasoning_effort="high"),
    ]

    first = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="effort-filtered",
        output_dir=results,
        conditions=conditions,
        excluded_reasoning_efforts={"high"},
        llm=FakeLLMClient(["<final_answer>A</final_answer>"]),
        max_attempts=1,
    )

    root = results / "effort-filtered"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    records = {record.spec.condition_id: record for record in ledger.jobs.values()}

    assert first["scheduled_jobs"] == 1
    assert first["excluded_reasoning_efforts"] == ["high"]
    assert records["solo-none"].state.value == "success"
    assert records["solo-high"].state.value == "pending"


@pytest.mark.asyncio
async def test_drain_leaves_queued_jobs_pending_without_starting_attempts(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks, count=2)
    drain_event = asyncio.Event()
    drain_event.set()

    summary = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="drained",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient([]),
        drain_event=drain_event,
    )

    root = results / "drained"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)

    assert summary["scheduled_jobs"] == 2
    assert summary["started_jobs"] == 0
    assert summary["drained_jobs"] == 2
    assert all(record.state.value == "pending" for record in ledger.jobs.values())
    assert all(record.attempt_count == 0 for record in ledger.jobs.values())


@pytest.mark.asyncio
async def test_drain_finishes_active_call_and_leaves_next_job_pending(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks, count=2)
    drain_event = asyncio.Event()
    call_started = asyncio.Event()
    release_call = asyncio.Event()
    wrapped = FakeLLMClient(["<final_answer>A</final_answer>"])

    class BlockingClient:
        async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
            call_started.set()
            await release_call.wait()
            return await wrapped.complete(**kwargs)

    running = asyncio.create_task(
        run_benchmark(
            tasks_path=tasks,
            model="fake/model",
            experiment_id="drain-active",
            output_dir=results,
            conditions=[Condition(id="solo", workflow="solo")],
            llm=BlockingClient(),
            concurrency=1,
            drain_event=drain_event,
        )
    )
    await call_started.wait()
    drain_event.set()
    release_call.set()
    summary = await running

    root = results / "drain-active"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    states = sorted(record.state.value for record in ledger.jobs.values())

    assert summary["started_jobs"] == 1
    assert summary["drained_jobs"] == 1
    assert states == ["pending", "success"]


def test_cancelled_attempt_does_not_consume_job_attempt_ceiling() -> None:
    spec = SimpleNamespace(condition_id="solo")
    ledger = SimpleNamespace(
        jobs={
            "job": SimpleNamespace(
                spec=spec,
                state=JobState.FAILED,
                latest_error="retryable:cancelled",
                attempts=[
                    SimpleNamespace(error="retryable:cancelled"),
                    SimpleNamespace(error="retryable:cancelled"),
                ],
            )
        }
    )

    assert _jobs_to_run(ledger, max_attempts=1) == [spec]


@pytest.mark.asyncio
async def test_run_emits_structured_progress_events(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks)
    events = []

    await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="events",
        output_dir=tmp_path / "results",
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient(["<final_answer>A</final_answer>"]),
        max_attempts=1,
        event_handler=events.append,
        emit_json_events=False,
    )

    assert [event["event"] for event in events] == [
        "benchmark_started",
        "attempt_started",
        "model_call_started",
        "model_call_completed",
        "attempt_completed",
        "benchmark_finished",
    ]
    assert events[0]["total_jobs"] == 1
    assert events[0]["experiment_id"] == "events"
    assert events[0]["model"] == "fake/model"
    assert events[0]["manifest_schema_version"] == 1
    assert events[1]["condition"] == "solo"
    assert events[1]["workflow"] == "solo"
    assert events[1]["workflow_version"] == "2.0.0"
    assert events[1]["task_id"] == "task-1"
    assert events[1]["task_prompt"] == "Question 1?"
    assert events[1]["estimated_model_calls"] == 1
    assert events[2]["job_id"] == events[1]["job_id"]
    assert events[2]["step"] == "answer"
    assert events[2]["agent_id"] == "agent-1"
    assert events[3]["output_tokens"] == 5
    assert events[3]["total_tokens"] == 15
    assert events[3]["job_id"] == events[1]["job_id"]
    assert events[4]["status"] == "success"
    assert events[4]["cost_usd"] == 0.01
    assert events[4]["output_tokens"] == 5


class Flaky429Client:
    def __init__(self) -> None:
        self.calls = 0
        self.success = FakeLLMClient(["<final_answer>A</final_answer>"])

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls > 1:
            return await self.success.complete(**kwargs)
        timestamp = utc_now()
        raise LLMCallError(
            ModelCallRecord(
                sequence=kwargs["sequence"],
                run_id=kwargs["run_id"],
                task_id=kwargs["task_id"],
                workflow=kwargs["workflow"],
                step=kwargs["step"],
                agent_id=kwargs["agent"].id,
                requested_model=kwargs["agent"].model,
                request_parameters={},
                messages=kwargs["messages"],
                usage=UsageStats(),
                started_at=timestamp,
                ended_at=timestamp,
                latency_ms=0,
                status="failed",
                error=CallError(
                    type="RateLimitError",
                    message="slow down",
                    details={"status_code": 429, "retry_after": 0},
                ),
            )
        )


@pytest.mark.asyncio
async def test_429_retries_only_the_failed_request_inside_one_job_attempt(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)
    client = Flaky429Client()

    summary = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="retry-pilot",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=client,
        max_attempts=2,
        request_max_attempts=2,
        request_retry_base_delay_seconds=0,
        request_retry_max_delay_seconds=0,
        request_retry_jitter_ratio=0,
        sleep=_no_sleep,
    )

    root = results / "retry-pilot"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    record = next(iter(ledger.jobs.values()))

    assert client.calls == 2
    assert record.attempt_count == 1
    assert [attempt.state.value for attempt in record.attempts] == ["success"]
    assert len(list(root.glob("*/result.json"))) == 1
    assert summary["results"][0]["status"] == "success"
    run_result = json.loads(
        next(root.glob("*/result.json")).read_text(encoding="utf-8")
    )
    assert len(run_result["calls"]) == 1
    assert run_result["calls"][0]["metadata"]["request_attempt"] == 2
    assert (
        run_result["calls"][0]["metadata"]["request_retries"][0]["reason"] == "http_429"
    )

    report = analyze_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="retry-pilot",
        output_dir=tmp_path / "analysis",
    )
    condition = report["conditions"][0]
    assert condition["expected_jobs"] == 1
    assert condition["attempts"] == 1
    assert condition["valid_completed_accuracy"] == 1.0
    site_path = build_site(
        analysis_dir=tmp_path / "analysis",
        output_dir=tmp_path / "site",
    )
    site = site_path.read_text(encoding="utf-8")
    assert "Coverage And Execution Health" in site
    assert "fake/model" in site


async def _no_sleep(delay: float) -> None:
    del delay


class TimeoutClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        timestamp = utc_now()
        raise LLMCallError(
            ModelCallRecord(
                sequence=kwargs["sequence"],
                run_id=kwargs["run_id"],
                task_id=kwargs["task_id"],
                workflow=kwargs["workflow"],
                step=kwargs["step"],
                agent_id=kwargs["agent"].id,
                requested_model=kwargs["agent"].model,
                request_parameters={},
                messages=kwargs["messages"],
                usage=UsageStats(),
                started_at=timestamp,
                ended_at=timestamp,
                latency_ms=120_000,
                status="failed",
                error=CallError(
                    type="Timeout",
                    message="request timed out",
                    details={"status_code": 408},
                ),
            )
        )


class PartialSampleTimeoutClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failed_agent_6 = False

    async def complete(self, **kwargs):  # type: ignore[no-untyped-def]
        agent_id = kwargs["agent"].id
        self.calls.append(agent_id)
        if agent_id == "agent-6" and not self.failed_agent_6:
            self.failed_agent_6 = True
            timestamp = utc_now()
            raise LLMCallError(
                ModelCallRecord(
                    sequence=kwargs["sequence"],
                    run_id=kwargs["run_id"],
                    task_id=kwargs["task_id"],
                    workflow=kwargs["workflow"],
                    step=kwargs["step"],
                    agent_id=agent_id,
                    requested_model=kwargs["agent"].model,
                    request_parameters=kwargs["agent"].completion_parameters(),
                    messages=kwargs["messages"],
                    prompt_references=kwargs["prompt_references"],
                    usage=UsageStats(),
                    started_at=timestamp,
                    ended_at=timestamp,
                    latency_ms=120_000,
                    status="failed",
                    error=CallError(
                        type="Timeout",
                        message="request timed out",
                        details={"status_code": 408},
                    ),
                    metadata=kwargs["metadata"],
                )
            )
        return await FakeLLMClient(
            ["<final_answer>A</final_answer>"]
        ).complete(**kwargs)


@pytest.mark.asyncio
async def test_timeout_fails_once_without_request_or_job_retry(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)
    client = TimeoutClient()
    events = []

    summary = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="timeout-pilot",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=client,
        max_attempts=3,
        request_max_attempts=3,
        event_handler=events.append,
        emit_json_events=False,
    )

    root = results / "timeout-pilot"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    record = next(iter(ledger.jobs.values()))

    assert client.calls == 1
    assert record.attempt_count == 1
    assert record.state.value == "failed"
    assert summary["results"][0]["status"] == "failed"
    assert [event["event"] for event in events] == [
        "benchmark_started",
        "attempt_started",
        "model_call_started",
        "model_call_failed",
        "attempt_completed",
        "benchmark_finished",
    ]
    assert events[3]["job_id"] == record.spec.job_id
    assert events[3]["step"] == "answer"
    assert events[3]["will_retry"] is False


@pytest.mark.asyncio
async def test_sample_retry_recovers_successful_agents_from_prior_attempt(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks)
    client = PartialSampleTimeoutClient()
    condition = Condition(
        id="sample-6",
        workflow="sample",
        agents=6,
        aggregation="plurality_vote",
    )

    first = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="checkpoint-pilot",
        output_dir=results,
        conditions=[condition],
        llm=client,
        max_attempts=2,
        emit_json_events=False,
    )

    root = results / "checkpoint-pilot"
    assert first["results"][0]["status"] == "failed"
    assert sorted(client.calls) == [f"agent-{index}" for index in range(1, 7)]
    assert len(list((root / "_checkpoints").glob("*/*.json"))) == 5

    # Exercise artifact backfill, which also recovers runs made before checkpoints
    # existed.
    shutil.rmtree(root / "_checkpoints")
    events: list[dict] = []
    second = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="checkpoint-pilot",
        output_dir=results,
        conditions=[condition],
        llm=client,
        max_attempts=2,
        event_handler=events.append,
        emit_json_events=False,
    )

    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    record = next(iter(ledger.jobs.values()))
    result = json.loads(
        (root / str(record.latest_run_id) / "result.json").read_text(
            encoding="utf-8"
        )
    )
    reused = [
        call for call in result["calls"] if call["metadata"].get("checkpoint_reused")
    ]
    completed_events = [
        event for event in events if event["event"] == "model_call_completed"
    ]

    assert second["results"][0]["status"] == "success"
    assert client.calls.count("agent-6") == 2
    assert len(client.calls) == 7
    assert record.attempt_count == 2
    assert record.state.value == "success"
    assert len(result["calls"]) == 6
    assert len(reused) == 5
    assert {
        call["metadata"]["checkpoint_source"] for call in reused
    } == {"prior_attempt"}
    assert len(list((root / "_checkpoints").glob("*/*.json"))) == 6
    assert sum(event["checkpoint_reused"] for event in completed_events) == 5
    assert sum(event["total_tokens"] for event in completed_events) == 15


@pytest.mark.asyncio
async def test_global_request_cap_bounds_parallel_sample_jobs(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, count=2)
    client = DelayedFakeLLMClient(
        ["<final_answer>A</final_answer>"] * 12,
        {f"agent-{index}": 0.01 for index in range(1, 7)},
    )

    await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="parallel-pilot",
        output_dir=tmp_path / "results",
        conditions=[
            Condition(
                id="sample-6",
                workflow="sample",
                agents=6,
                aggregation="plurality_vote",
            )
        ],
        llm=client,
        concurrency=2,
        max_in_flight_requests=3,
        max_attempts=1,
    )

    assert client.max_active_calls == 3


@pytest.mark.asyncio
async def test_dry_run_writes_nothing_and_counts_nested_calls(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_tasks(tasks, count=2)

    plan = await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="dry",
        output_dir=results,
        conditions=[
            Condition(
                id="debate",
                workflow="debate",
                agents=3,
                rounds=1,
                aggregation="plurality_vote",
            )
        ],
        repetitions=2,
        dry_run=True,
    )

    assert plan["jobs"] == 4
    assert plan["estimated_minimum_model_calls"] == 24
    assert not results.exists()


@pytest.mark.asyncio
async def test_required_answer_type_fails_before_writing(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    tasks.write_text(
        json.dumps(
            {
                "id": "short",
                "prompt": "Name the answer.",
                "answer": "alpha",
                "answer_type": "short_answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain only"):
        await run_benchmark(
            tasks_path=tasks,
            model="fake/model",
            experiment_id="bad",
            output_dir=results,
            conditions=[Condition(id="solo", workflow="solo")],
            required_answer_type="multiple_choice",
            dry_run=True,
        )

    assert not results.exists()


def test_aggregation_judge_model_does_not_replace_supervisor_model() -> None:
    sample = _workflow(
        condition=Condition(
            id="sample",
            workflow="sample",
            agents=3,
            aggregation="plurality_vote",
            vote_tie_break="judge",
            judge_reasoning_effort="low",
        ),
        model="primary/model",
        judge_model="aggregation/judge",
        system_prompt="",
        requires_semantic_judge=True,
    )
    supervisor = _workflow(
        condition=Condition(
            id="supervisor",
            workflow="supervisor",
            rounds=2,
            reasoning_effort="medium",
            supervisor_reasoning_effort="high",
            judge_reasoning_effort="low",
        ),
        model="primary/model",
        judge_model="aggregation/judge",
        system_prompt="",
        requires_semantic_judge=True,
    )

    assert sample.wrapped.judge.model == "aggregation/judge"
    assert sample.wrapped.judge.reasoning_effort == "low"
    assert supervisor.wrapped.supervisor.model == "primary/model"
    assert supervisor.wrapped.supervisor.reasoning_effort == "high"
