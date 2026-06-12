from __future__ import annotations

import json

import pytest

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.experiment import load_ledger, load_manifest
from extensions.benchmark_tools.runner import run_benchmark
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
        max_attempts=1,
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
    assert (experiment_root / record.latest_run_id / "artifact-manifest.json").exists()


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
async def test_transient_failure_creates_a_separate_retry_attempt(tmp_path) -> None:
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
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        retry_jitter_ratio=0,
    )

    root = results / "retry-pilot"
    manifest = load_manifest(root / "experiment-manifest.json")
    ledger = load_ledger(root / "experiment-ledger.json", manifest=manifest)
    record = next(iter(ledger.jobs.values()))

    assert client.calls == 2
    assert record.attempt_count == 2
    assert [attempt.state.value for attempt in record.attempts] == [
        "failed",
        "success",
    ]
    assert record.attempts[0].metadata["retry_reason"] == "http_429"
    assert record.attempts[0].metadata["retry_delay_seconds"] == 0
    assert len(list(root.glob("*/result.json"))) == 2
    assert summary["results"][0]["status"] == "success"

    report = analyze_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="retry-pilot",
        output_dir=tmp_path / "analysis",
    )
    condition = report["conditions"][0]
    assert condition["expected_jobs"] == 1
    assert condition["attempts"] == 2
    assert condition["valid_completed_accuracy"] == 1.0
    site_path = build_site(
        analysis_dir=tmp_path / "analysis",
        output_dir=tmp_path / "site",
    )
    site = site_path.read_text(encoding="utf-8")
    assert "Coverage And Execution Health" in site
    assert "fake/model" in site


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
