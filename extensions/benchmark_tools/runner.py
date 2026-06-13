from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.llm import LLMCallError, LLMClient
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import AgentSpec, ModelCallRecord, utc_now
from multi_agent_research.prompts import (
    CROSS_EXAMINATION_CHALLENGE_PROMPT,
    CROSS_EXAMINATION_CLAIM_PROMPT,
    CROSS_EXAMINATION_FINAL_REVISION_PROMPT,
    CROSS_EXAMINATION_RESPONSE_PROMPT,
    CROSS_EXAMINATION_VERDICT_PROMPT,
    DEBATE_ADVERSARIAL_CHALLENGE_PROMPT,
    DEBATE_ADVERSARIAL_RESOLUTION_PROMPT,
    DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT,
    DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
    DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
    DEBATE_DERIVATION_ROLE_PROMPT,
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SELF_CRITIC_REVISION_PROMPT,
    SUPERVISOR_REVIEW_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TIE_BREAK_JUDGE_PROMPT,
    WORKER_REVISION_PROMPT,
)
from multi_agent_research.runner import ExperimentRunner
from multi_agent_research.storage import FileRunStore
from multi_agent_research.workflows import (
    AdversarialDebateWorkflow,
    CrossExaminationDebateWorkflow,
    DebateWorkflow,
    IndependentSampleWorkflow,
    SelfCriticWorkflow,
    SoloWorkflow,
    SupervisorWorkflow,
    Workflow,
)

from extensions.benchmark_tools.connector import load_jsonl, task_from_example
from extensions.benchmark_tools.experiment import (
    ExecutionPolicy,
    ExperimentLedger,
    ExperimentManifest,
    JobSpec,
    JobState,
    load_ledger,
    load_manifest,
    save_ledger,
    save_manifest,
)
from extensions.benchmark_tools.rate_limit import (
    AsyncRateLimiter,
    RetryDecision,
    RetryPolicy,
    classify_retry,
    provider_tpm_limit,
)
from extensions.benchmark_tools.schema import Condition


DEFAULT_CONDITIONS = [
    Condition(id="solo", workflow="solo"),
    Condition(id="sample-3", workflow="sample", agents=3, aggregation="plurality_vote"),
    Condition(id="sample-6", workflow="sample", agents=6, aggregation="plurality_vote"),
    Condition(
        id="debate-3-full",
        workflow="debate",
        agents=3,
        rounds=1,
        aggregation="plurality_vote",
        debate_peer_view="full_response",
    ),
    Condition(
        id="debate-3-answer",
        workflow="debate",
        agents=3,
        rounds=1,
        aggregation="plurality_vote",
        debate_peer_view="answer_only",
    ),
    Condition(
        id="adversarial-debate-3",
        workflow="adversarial-debate",
        agents=3,
        rounds=2,
        aggregation="plurality_vote",
        debate_peer_view="full_response",
    ),
]


def load_conditions(path: Path | str | None) -> list[Condition]:
    if path is None:
        return DEFAULT_CONDITIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["conditions"] if isinstance(data, dict) else data
    return [Condition(**row) for row in rows]


async def run_benchmark(
    *,
    tasks_path: Path | str,
    model: str,
    experiment_id: str,
    output_dir: Path | str = "results",
    conditions: list[Condition] | None = None,
    judge_model: str | None = None,
    system_prompt: str = "",
    concurrency: int = 1,
    max_in_flight_requests: int = 1,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    estimated_tokens_per_request: int = 4096,
    repetitions: int = 1,
    max_attempts: int = 3,
    retry_base_delay_seconds: float = 1.0,
    retry_max_delay_seconds: float = 30.0,
    retry_jitter_ratio: float = 0.2,
    request_max_attempts: int = 3,
    request_retry_base_delay_seconds: float = 1.0,
    request_retry_max_delay_seconds: float = 60.0,
    request_retry_jitter_ratio: float = 0.2,
    resume: bool = True,
    dry_run: bool = False,
    required_answer_type: str | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    excluded_workflows: set[str] | None = None,
    excluded_reasoning_efforts: set[str] | None = None,
    llm: LLMClient | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
    emit_json_events: bool = True,
    drain_event: asyncio.Event | None = None,
    sleep=asyncio.sleep,  # type: ignore[no-untyped-def]
    random_source=random.random,
) -> dict[str, Any]:
    load_dotenv()
    tasks_path = Path(tasks_path)
    examples = load_jsonl(tasks_path)
    selected_conditions = conditions or DEFAULT_CONDITIONS
    if not examples:
        raise ValueError("benchmark task set is empty")
    if required_answer_type and any(
        example.answer_type != required_answer_type for example in examples
    ):
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.answer_type] = counts.get(example.answer_type, 0) + 1
        raise ValueError(
            f"task set must contain only {required_answer_type!r}; found {counts}"
        )
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("benchmark task ids must be unique")
    if not selected_conditions:
        raise ValueError("at least one condition is required")

    condition_by_id = {condition.id: condition for condition in selected_conditions}
    if len(condition_by_id) != len(selected_conditions):
        raise ValueError("condition ids must be unique")
    excluded_workflow_set = set(excluded_workflows or ())
    excluded_effort_set = set(excluded_reasoning_efforts or ())
    available_workflows = {condition.workflow for condition in selected_conditions}
    unknown_exclusions = excluded_workflow_set - available_workflows
    if unknown_exclusions:
        raise ValueError(
            "excluded workflows are not present in the experiment: "
            + ", ".join(sorted(unknown_exclusions))
        )
    available_efforts = {
        condition.reasoning_effort
        for condition in selected_conditions
        if condition.reasoning_effort is not None
    }
    unknown_effort_exclusions = excluded_effort_set - available_efforts
    if unknown_effort_exclusions:
        raise ValueError(
            "excluded reasoning efforts are not present in the experiment: "
            + ", ".join(sorted(unknown_effort_exclusions))
        )
    scoped_condition_ids = {
        condition.id
        for condition in selected_conditions
        if condition.workflow not in excluded_workflow_set
        and condition.reasoning_effort not in excluded_effort_set
    }
    requires_semantic_judge = any(
        example.answer_type == "short_answer" for example in examples
    )
    for condition in selected_conditions:
        _workflow(
            condition=condition,
            model=model,
            judge_model=judge_model,
            system_prompt=system_prompt,
            requires_semantic_judge=requires_semantic_judge,
        )

    policy = ExecutionPolicy(
        concurrency=concurrency,
        max_in_flight_requests=max_in_flight_requests,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        estimated_tokens_per_request=estimated_tokens_per_request,
        max_attempts=max_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
        retry_max_delay_seconds=retry_max_delay_seconds,
        retry_jitter_ratio=retry_jitter_ratio,
        request_max_attempts=request_max_attempts,
        request_retry_base_delay_seconds=request_retry_base_delay_seconds,
        request_retry_max_delay_seconds=request_retry_max_delay_seconds,
        request_retry_jitter_ratio=request_retry_jitter_ratio,
    )
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        task_set_path=str(tasks_path),
        task_set_sha256=sha256(tasks_path.read_bytes()).hexdigest(),
        task_count=len(examples),
        conditions=tuple(asdict(condition) for condition in selected_conditions),
        model=model,
        judge_model=judge_model,
        generation_settings={"required_answer_type": required_answer_type},
        system_settings={"system_prompt": system_prompt},
        repetitions=repetitions,
        policy=policy,
        metadata=experiment_metadata or {},
    )
    planning_ledger = ExperimentLedger.create(
        manifest, [example.id for example in examples]
    )
    planning_examples = {example.id: example for example in examples}
    planned_calls = sum(
        _estimated_calls(
            condition_by_id[record.spec.condition_id],
            planning_examples[record.spec.task_id],
        )
        for record in planning_ledger.jobs.values()
    )
    scoped_planning_jobs = [
        record
        for record in planning_ledger.jobs.values()
        if record.spec.condition_id in scoped_condition_ids
    ]
    scoped_planned_calls = sum(
        _estimated_calls(
            condition_by_id[record.spec.condition_id],
            planning_examples[record.spec.task_id],
        )
        for record in scoped_planning_jobs
    )
    plan = {
        "experiment_id": experiment_id,
        "tasks": len(examples),
        "conditions": len(selected_conditions),
        "repetitions": repetitions,
        "jobs": len(examples) * len(selected_conditions) * repetitions,
        "estimated_minimum_model_calls": planned_calls,
        "scope_conditions": len(scoped_condition_ids),
        "scope_jobs": len(scoped_planning_jobs),
        "scope_estimated_minimum_model_calls": scoped_planned_calls,
        "excluded_workflows": sorted(excluded_workflow_set),
        "excluded_reasoning_efforts": sorted(excluded_effort_set),
        "estimate_note": (
            "Excludes conditional tie-break judge calls, early supervisor "
            "approval, and the separate semantic HLE grading pass."
        ),
    }
    if dry_run:
        return {"dry_run": True, **plan}

    results_root = Path(output_dir)
    experiment_root = results_root / experiment_id
    manifest_path = experiment_root / "experiment-manifest.json"
    ledger_path = experiment_root / "experiment-ledger.json"
    persisted_manifest = None
    if manifest_path.exists():
        if not resume:
            raise ValueError(
                f"experiment already exists: {experiment_id}; enable resume or use a new id"
            )
        persisted_manifest = load_manifest(manifest_path)
        persisted_manifest.assert_compatible(manifest)
    else:
        if experiment_root.exists() and any(experiment_root.iterdir()):
            raise ValueError(
                f"experiment directory has legacy artifacts but no manifest: {experiment_root}"
            )
        save_manifest(manifest_path, manifest)

    if ledger_path.exists():
        ledger = load_ledger(
            ledger_path,
            manifest=persisted_manifest or manifest,
        )
        ledger.requeue_cancelled_jobs()
        ledger.manifest_fingerprint = manifest.compatibility_fingerprint
        if persisted_manifest is not None and persisted_manifest != manifest:
            save_manifest(
                manifest_path,
                replace(
                    persisted_manifest,
                    task_set_path=manifest.task_set_path,
                    policy=manifest.policy,
                    metadata=manifest.metadata,
                ),
            )
        save_ledger(ledger_path, ledger)
    else:
        ledger = ExperimentLedger.create(manifest, [example.id for example in examples])
        save_ledger(ledger_path, ledger)

    limiter = AsyncRateLimiter(
        max_in_flight=max_in_flight_requests,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        sleep=sleep,
    )
    checkpointed_llm = SampleCheckpointLLMClient(
        llm or LiteLLMClient(),
        experiment_root=experiment_root,
        ledger=ledger,
    )
    limited_llm = RateLimitedLLMClient(
        checkpointed_llm,
        limiter=limiter,
        estimated_tokens=estimated_tokens_per_request,
        retry_policy=RetryPolicy(
            max_attempts=request_max_attempts,
            base_delay_seconds=request_retry_base_delay_seconds,
            max_delay_seconds=request_retry_max_delay_seconds,
            jitter_ratio=request_retry_jitter_ratio,
        ),
        event_handler=event_handler,
        random_source=random_source,
    )
    runner = ExperimentRunner(
        llm=limited_llm,
        store=FileRunStore(results_root),
    )
    example_by_id = {example.id: example for example in examples}
    ledger_lock = asyncio.Lock()
    job_semaphore = asyncio.Semaphore(concurrency)
    summaries: list[dict[str, Any]] = []

    scoped_ledger_jobs = [
        record
        for record in ledger.jobs.values()
        if record.spec.condition_id in scoped_condition_ids
    ]
    jobs = _jobs_to_run(
        ledger,
        max_attempts=max_attempts,
        allowed_condition_ids=scoped_condition_ids,
    )
    if event_handler:
        event_handler(
            {
                "event": "benchmark_started",
                "experiment_id": experiment_id,
                "purpose": manifest.metadata.get("purpose"),
                "model": model,
                "judge_model": judge_model,
                "manifest_schema_version": manifest.schema_version,
                "task_count": len(examples),
                "condition_count": len(selected_conditions),
                "scope_condition_count": len(scoped_condition_ids),
                "repetitions": repetitions,
                "concurrency": concurrency,
                "max_in_flight_requests": max_in_flight_requests,
                "requests_per_minute": requests_per_minute,
                "tokens_per_minute": tokens_per_minute,
                "estimated_minimum_model_calls": scoped_planned_calls,
                "total_jobs": len(scoped_ledger_jobs),
                "scheduled_jobs": len(jobs),
                "completed_jobs": len(scoped_ledger_jobs) - len(jobs),
                "deferred_jobs": len(ledger.jobs) - len(scoped_ledger_jobs),
                "excluded_workflows": sorted(excluded_workflow_set),
                "excluded_reasoning_efforts": sorted(excluded_effort_set),
            }
        )

    async def run_one(job: JobSpec) -> dict[str, Any] | None:
        condition = condition_by_id[job.condition_id]
        example = example_by_id[job.task_id]
        async with job_semaphore:
            if drain_event is not None and drain_event.is_set():
                return None
            workflow = _workflow(
                condition=condition,
                model=model,
                judge_model=judge_model,
                system_prompt=system_prompt,
                requires_semantic_judge=requires_semantic_judge,
            )
            async with ledger_lock:
                attempt = ledger.start_attempt(job.job_id)
                save_ledger(ledger_path, ledger)
            if event_handler:
                event_handler(
                    {
                        "event": "attempt_started",
                        "job_id": job.job_id,
                        "attempt": attempt.number,
                        "condition": condition.id,
                        "workflow": condition.workflow,
                        "workflow_version": workflow.version,
                        "agents": condition.agents,
                        "rounds": condition.rounds,
                        "reasoning_effort": condition.reasoning_effort,
                        "task_id": example.id,
                        "task_category": example.category,
                        "task_prompt": example.prompt,
                        "repetition": job.repetition,
                        "estimated_model_calls": _estimated_calls(condition, example),
                    }
                )
            try:
                result = await runner.run(
                    task=task_from_example(
                        example,
                        include_confidence=condition.include_confidence,
                    ),
                    workflow=workflow,
                    experiment_id=experiment_id,
                    call_metadata={
                        "benchmark_job_id": job.job_id,
                        "condition_id": condition.id,
                    },
                )
            except asyncio.CancelledError:
                async with ledger_lock:
                    ledger.finish_attempt(
                        job.job_id,
                        JobState.FAILED,
                        error="retryable:cancelled",
                        metadata={
                            "retryable": True,
                            "retry_reason": "cancelled",
                        },
                    )
                    save_ledger(ledger_path, ledger)
                raise

            decision = _retry_decision(result)
            state = JobState(result.status)
            error = _result_error(result, decision)
            async with ledger_lock:
                ledger.finish_attempt(
                    job.job_id,
                    state,
                    run_id=result.run_id,
                    error=error,
                    metadata={
                        "retryable": decision.retryable,
                        "retry_reason": decision.reason,
                        "retry_after_seconds": decision.retry_after_seconds,
                    },
                )
                save_ledger(ledger_path, ledger)

            summary = {
                "job_id": job.job_id,
                "attempt_id": attempt.attempt_id,
                "attempt": attempt.number,
                "repetition": job.repetition,
                "condition": condition.id,
                "task_id": example.id,
                "run_id": result.run_id,
                "status": result.status,
                "final_answer": result.final_answer,
                "cost_usd": result.metrics.cost_usd,
                "total_tokens": result.metrics.total_tokens,
                "output_tokens": result.metrics.output_tokens,
                "retryable": decision.retryable,
                "retry_reason": decision.reason,
            }
            if event_handler:
                event_handler(
                    {
                        "event": "attempt_completed",
                        **summary,
                        "will_retry": False,
                    }
                )
            if emit_json_events:
                print(json.dumps(summary, sort_keys=True), flush=True)
            return summary

    if jobs:
        completed = await asyncio.gather(*(run_one(job) for job in jobs))
        summaries.extend(summary for summary in completed if summary is not None)
    if event_handler:
        event_handler({"event": "benchmark_finished"})
    return {
        "dry_run": False,
        **plan,
        "scheduled_jobs": len(jobs),
        "started_jobs": len(summaries),
        "drained_jobs": len(jobs) - len(summaries),
        "skipped_jobs": len(ledger.jobs) - len(jobs),
        "deferred_jobs": len(ledger.jobs) - len(scoped_ledger_jobs),
        "results": summaries,
    }


class SampleCheckpointLLMClient:
    """Persist successful independent samples so retries only run missing agents."""

    SCHEMA_VERSION = 1
    TRANSPORT_PARAMETERS = {"max_retries", "num_retries", "timeout"}

    def __init__(
        self,
        wrapped: LLMClient,
        *,
        experiment_root: Path,
        ledger: ExperimentLedger,
    ) -> None:
        self.wrapped = wrapped
        self.experiment_root = experiment_root
        self.checkpoint_root = experiment_root / "_checkpoints"
        self.ledger = ledger
        self._historical_results: dict[str, dict[str, Any] | None] = {}

    async def complete(self, **kwargs: Any) -> ModelCallRecord:
        if not self._eligible(kwargs):
            return await self.wrapped.complete(**kwargs)

        fingerprint = self._fingerprint(kwargs)
        checkpoint_path = self._checkpoint_path(kwargs)
        call = self._load_checkpoint(checkpoint_path, fingerprint)
        source = "checkpoint"
        if call is None:
            call = self._find_historical_call(kwargs)
            source = "prior_attempt"
            if call is not None:
                self._save_checkpoint(checkpoint_path, fingerprint, call)
        if call is not None:
            return self._reuse_call(
                call,
                kwargs=kwargs,
                checkpoint_path=checkpoint_path,
                fingerprint=fingerprint,
                source=source,
            )

        call = await self.wrapped.complete(**kwargs)
        if call.status == "success":
            self._save_checkpoint(checkpoint_path, fingerprint, call)
        return call

    @staticmethod
    def _eligible(kwargs: dict[str, Any]) -> bool:
        metadata = kwargs.get("metadata") or {}
        return (
            kwargs.get("workflow") == "independent_sample"
            and str(kwargs.get("step") or "").startswith("sample_")
            and bool(metadata.get("benchmark_job_id"))
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    def _fingerprint(self, kwargs: dict[str, Any]) -> str:
        agent = kwargs["agent"]
        metadata = kwargs.get("metadata") or {}
        material = {
            "job_id": metadata["benchmark_job_id"],
            "condition_id": metadata.get("condition_id"),
            "task_id": kwargs["task_id"],
            "workflow": kwargs["workflow"],
            "step": kwargs["step"],
            "agent": agent.model_dump(mode="json"),
            "messages": [
                self._json_value(message) for message in kwargs["messages"]
            ],
            "prompt_references": [
                self._json_value(reference)
                for reference in kwargs.get("prompt_references") or []
            ],
        }
        encoded = json.dumps(
            material,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _checkpoint_path(self, kwargs: dict[str, Any]) -> Path:
        metadata = kwargs.get("metadata") or {}
        job_id = str(metadata["benchmark_job_id"])
        key = sha256(
            f"{kwargs['step']}:{kwargs['agent'].id}".encode("utf-8")
        ).hexdigest()[:16]
        return self.checkpoint_root / job_id / f"{key}.json"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def _load_checkpoint(
        self,
        path: Path,
        fingerprint: str,
    ) -> ModelCallRecord | None:
        data = self._load_json(path)
        if (
            data is None
            or data.get("schema_version") != self.SCHEMA_VERSION
            or data.get("request_fingerprint") != fingerprint
            or not isinstance(data.get("call"), dict)
        ):
            return None
        try:
            call = ModelCallRecord.model_validate(data["call"])
        except ValueError:
            return None
        return call if call.status == "success" else None

    def _save_checkpoint(
        self,
        path: Path,
        fingerprint: str,
        call: ModelCallRecord,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "request_fingerprint": fingerprint,
            "step": call.step,
            "agent_id": call.agent_id,
            "call": call.model_dump(mode="json"),
        }
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _find_historical_call(
        self,
        kwargs: dict[str, Any],
    ) -> ModelCallRecord | None:
        metadata = kwargs.get("metadata") or {}
        record = self.ledger.jobs.get(str(metadata["benchmark_job_id"]))
        if record is None:
            return None
        for attempt in reversed(record.attempts):
            if not attempt.run_id:
                continue
            result = self._historical_result(attempt.run_id)
            if result is None or not self._result_matches(result, kwargs):
                continue
            for call_data in reversed(result.get("calls") or []):
                if not isinstance(call_data, dict):
                    continue
                try:
                    call = ModelCallRecord.model_validate(call_data)
                except ValueError:
                    continue
                if self._call_matches(call, kwargs):
                    return call
        return None

    def _historical_result(self, run_id: str) -> dict[str, Any] | None:
        if run_id not in self._historical_results:
            self._historical_results[run_id] = self._load_json(
                self.experiment_root / run_id / "result.json"
            )
        return self._historical_results[run_id]

    @staticmethod
    def _result_matches(result: dict[str, Any], kwargs: dict[str, Any]) -> bool:
        workflow = result.get("workflow") or {}
        config = workflow.get("config") or {}
        metadata = kwargs.get("metadata") or {}
        return (
            result.get("task_id") == kwargs["task_id"]
            and workflow.get("name") == kwargs["workflow"]
            and config.get("condition_id") == metadata.get("condition_id")
        )

    def _call_matches(
        self,
        call: ModelCallRecord,
        kwargs: dict[str, Any],
    ) -> bool:
        agent = kwargs["agent"]
        expected_parameters = agent.completion_parameters()
        stored_parameters = {
            key: value
            for key, value in call.request_parameters.items()
            if key not in self.TRANSPORT_PARAMETERS
        }
        return (
            call.status == "success"
            and call.step == kwargs["step"]
            and call.agent_id == agent.id
            and call.requested_model == agent.model
            and stored_parameters == expected_parameters
            and [message.model_dump(mode="json") for message in call.messages]
            == [
                self._json_value(message)
                for message in kwargs["messages"]
            ]
            and [
                reference.model_dump(mode="json")
                for reference in call.prompt_references
            ]
            == [
                self._json_value(reference)
                for reference in kwargs.get("prompt_references") or []
            ]
        )

    @staticmethod
    def _reuse_call(
        call: ModelCallRecord,
        *,
        kwargs: dict[str, Any],
        checkpoint_path: Path,
        fingerprint: str,
        source: str,
    ) -> ModelCallRecord:
        timestamp = utc_now()
        metadata = {
            **call.metadata,
            **(kwargs.get("metadata") or {}),
            "checkpoint_reused": True,
            "checkpoint_source": source,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_request_fingerprint": fingerprint,
            "checkpoint_source_call_id": call.id,
            "checkpoint_source_run_id": call.run_id,
        }
        return call.model_copy(
            deep=True,
            update={
                "id": str(uuid4()),
                "sequence": kwargs["sequence"],
                "run_id": kwargs["run_id"],
                "started_at": timestamp,
                "ended_at": timestamp,
                "latency_ms": 0.0,
                "metadata": metadata,
            },
        )


class RateLimitedLLMClient:
    ESTIMATED_TOKENS_BY_EFFORT = {
        "none": 5_000,
        "low": 8_000,
        "medium": 40_000,
        "high": 55_000,
        "xhigh": 65_000,
    }

    def __init__(
        self,
        wrapped: LLMClient,
        *,
        limiter: AsyncRateLimiter,
        estimated_tokens: int,
        retry_policy: RetryPolicy | None = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
        random_source=random.random,
    ) -> None:
        self.wrapped = wrapped
        self.limiter = limiter
        self.estimated_tokens = estimated_tokens
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=1)
        self.event_handler = event_handler
        self.random_source = random_source

    async def complete(self, **kwargs: Any) -> ModelCallRecord:
        agent = kwargs["agent"]
        metadata = kwargs.get("metadata") or {}
        progress_fields = {
            "job_id": metadata.get("benchmark_job_id"),
            "condition": metadata.get("condition_id"),
            "run_id": kwargs["run_id"],
            "sequence": kwargs["sequence"],
            "step": kwargs["step"],
            "agent_id": agent.id,
            "model": agent.model,
        }
        estimated_tokens = max(
            self.estimated_tokens,
            self.ESTIMATED_TOKENS_BY_EFFORT.get(
                agent.reasoning_effort or "",
                self.estimated_tokens,
            ),
        )
        retries: list[dict[str, Any]] = []
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            lease = await self.limiter.acquire(estimated_tokens=estimated_tokens)
            if self.event_handler:
                self.event_handler(
                    {
                        "event": "model_call_started",
                        **progress_fields,
                        "request_attempt": attempt_number,
                    }
                )
            try:
                call = await self.wrapped.complete(**kwargs)
            except LLMCallError as exc:
                decision = classify_retry(exc)
                reported_tpm_limit = (
                    provider_tpm_limit(exc) if decision.status_code == 429 else None
                )
                effective_tpm_limit = None
                if reported_tpm_limit is not None:
                    effective_tpm_limit = await self.limiter.cap_tokens_per_minute(
                        reported_tpm_limit
                    )
                retry_summary = {
                    "attempt": attempt_number,
                    "reason": decision.reason,
                    "status_code": decision.status_code,
                    "retry_after_seconds": decision.retry_after_seconds,
                    "error_type": (
                        exc.record.error.type
                        if exc.record.error
                        else type(exc).__name__
                    ),
                    "error_message": (
                        exc.record.error.message if exc.record.error else str(exc)
                    ),
                    "latency_ms": exc.record.latency_ms,
                    "rate_limit": asdict(lease.metadata),
                    "provider_tpm_limit": reported_tpm_limit,
                    "effective_tpm_limit": effective_tpm_limit,
                }
                retries.append(retry_summary)
                exc.record.metadata = {
                    **exc.record.metadata,
                    "rate_limit": asdict(lease.metadata),
                    "request_attempt": attempt_number,
                    "request_retries": retries,
                }
                should_retry = (
                    decision.status_code == 429
                    and attempt_number < self.retry_policy.max_attempts
                )
                if self.event_handler:
                    self.event_handler(
                        {
                            "event": "model_call_failed",
                            **progress_fields,
                            "request_attempt": attempt_number,
                            "latency_ms": exc.record.latency_ms,
                            "reason": decision.reason,
                            "status_code": decision.status_code,
                            "will_retry": should_retry,
                        }
                    )
                if not should_retry:
                    raise
                delay = self.retry_policy.delay(
                    attempt_number,
                    retry_after_seconds=decision.retry_after_seconds,
                    random_value=self.random_source(),
                )
                await self.limiter.impose_cooldown(max(1.0, delay))
                if self.event_handler:
                    self.event_handler(
                        {
                            "event": "model_call_retry_scheduled",
                            "model": agent.model,
                            "attempt": attempt_number,
                            "delay_seconds": max(1.0, delay),
                            "reason": decision.reason,
                            "status_code": decision.status_code,
                            "provider_tpm_limit": reported_tpm_limit,
                            "effective_tpm_limit": effective_tpm_limit,
                        }
                    )
            else:
                checkpoint_reused = bool(call.metadata.get("checkpoint_reused"))
                call.metadata = {
                    **call.metadata,
                    "rate_limit": asdict(lease.metadata),
                    "request_attempt": attempt_number,
                    "request_retries": retries,
                }
                provider_total_tokens = (
                    0 if checkpoint_reused else call.usage.total_tokens or 0
                )
                provider_output_tokens = (
                    0 if checkpoint_reused else call.usage.output_tokens or 0
                )
                await lease.reconcile_tokens(provider_total_tokens)
                if self.event_handler:
                    self.event_handler(
                        {
                            "event": "model_call_completed",
                            **progress_fields,
                            "model": call.response_model or call.requested_model,
                            "total_tokens": provider_total_tokens,
                            "output_tokens": provider_output_tokens,
                            "recorded_total_tokens": call.usage.total_tokens or 0,
                            "checkpoint_reused": checkpoint_reused,
                            "latency_ms": call.latency_ms,
                            "request_attempt": attempt_number,
                        }
                    )
                return call
            finally:
                lease.release()
        raise AssertionError("unreachable")


def _jobs_to_run(
    ledger: ExperimentLedger,
    *,
    max_attempts: int,
    allowed_condition_ids: set[str] | None = None,
) -> list[JobSpec]:
    jobs = []
    for record in ledger.jobs.values():
        if (
            allowed_condition_ids is not None
            and record.spec.condition_id not in allowed_condition_ids
        ):
            continue
        attempts_toward_limit = sum(
            not (attempt.error or "").startswith("retryable:cancelled")
            for attempt in record.attempts
        )
        if attempts_toward_limit >= max_attempts:
            continue
        if record.state in {JobState.PENDING, JobState.RUNNING}:
            jobs.append(record.spec)
        elif record.state == JobState.FAILED and (record.latest_error or "").startswith(
            "retryable:"
        ):
            jobs.append(record.spec)
    return jobs


def _retry_decision(result: Any) -> RetryDecision:
    if result.status != "failed":
        return RetryDecision(False, result.status)
    failed_calls = [call for call in result.calls if call.status == "failed"]
    if failed_calls:
        return classify_retry(LLMCallError(failed_calls[-1]))
    return RetryDecision(False, result.error.type if result.error else "run_failed")


def _result_error(result: Any, decision: RetryDecision) -> str | None:
    if result.status != "failed":
        return None
    prefix = "retryable" if decision.retryable else "terminal"
    message = result.error.message if result.error else "run failed"
    return f"{prefix}:{decision.reason}:{message}"


def _estimated_calls(condition: Condition, example: Any) -> int:
    aggregation_call = int(condition.aggregation == "judge")
    semantic_vote_call = int(
        condition.aggregation != "judge"
        and example.answer_type == "short_answer"
        and not (condition.workflow == "sample" and condition.agents == 1)
        and condition.workflow
        in {"sample", "debate", "adversarial-debate", "cross-examination-debate"}
    )
    if condition.workflow == "solo":
        return 1
    if condition.workflow == "self-critic":
        return 1 + condition.rounds
    if condition.workflow == "sample":
        return condition.agents + aggregation_call + semantic_vote_call
    if condition.workflow in {"debate", "adversarial-debate"}:
        return (
            condition.agents * (1 + condition.rounds)
            + aggregation_call
            + semantic_vote_call
        )
    if condition.workflow == "cross-examination-debate":
        return (
            condition.agents * (3 + 3 * condition.rounds)
            + aggregation_call
            + semantic_vote_call
        )
    if condition.workflow == "supervisor":
        return 1 + (2 * condition.rounds)
    raise ValueError(f"unsupported workflow: {condition.workflow}")


def _workflow(
    *,
    condition: Condition,
    model: str,
    judge_model: str | None,
    system_prompt: str,
    requires_semantic_judge: bool = False,
) -> Workflow:
    parameters: dict[str, Any] = {}
    if condition.temperature is not None:
        parameters["temperature"] = condition.temperature
    if condition.max_tokens is not None:
        parameters["max_tokens"] = condition.max_tokens
    base_agent = AgentSpec(
        id="agent-1",
        model=model,
        system_prompt=system_prompt,
        system_prompt_name="agent.primary.system",
        reasoning_effort=condition.reasoning_effort,
        service_tier=condition.service_tier,  # type: ignore[arg-type]
        parameters=parameters,
    )
    if condition.workflow == "solo":
        return LabeledWorkflow(SoloWorkflow(base_agent), condition.id)
    if condition.workflow == "self-critic":
        return LabeledWorkflow(
            SelfCriticWorkflow(
                base_agent,
                rounds=condition.rounds,
                revision_prompt=SELF_CRITIC_REVISION_PROMPT,
            ),
            condition.id,
        )

    agents = [
        AgentSpec(
            id=f"agent-{index + 1}",
            model=model,
            system_prompt=system_prompt,
            system_prompt_name="agent.primary.system",
            reasoning_effort=condition.reasoning_effort,
            service_tier=condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
        for index in range(condition.agents)
    ]
    judge = None
    if (
        condition.aggregation == "judge"
        or condition.vote_tie_break == "judge"
        or (
            requires_semantic_judge
            and condition.aggregation != "judge"
            and condition.workflow
            in {
                "sample",
                "debate",
                "adversarial-debate",
                "cross-examination-debate",
            }
        )
    ):
        judge = AgentSpec(
            id="judge",
            model=judge_model or model,
            system_prompt=JUDGE_SYSTEM_PROMPT.template,
            system_prompt_name=JUDGE_SYSTEM_PROMPT.name,
            system_prompt_version=JUDGE_SYSTEM_PROMPT.version,
            reasoning_effort=condition.judge_reasoning_effort
            or condition.reasoning_effort,
            service_tier=condition.judge_service_tier or condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
    voting = VotingConfig(tie_break=condition.vote_tie_break)  # type: ignore[arg-type]

    if condition.workflow == "sample":
        return LabeledWorkflow(
            IndependentSampleWorkflow(
                agents,
                judge,
                judge_prompt=JUDGE_SELECTION_PROMPT,
                tie_break_judge_prompt=TIE_BREAK_JUDGE_PROMPT,
                parallel=not condition.sequential,
                aggregation=condition.aggregation,  # type: ignore[arg-type]
                voting=voting,
            ),
            condition.id,
        )
    if condition.workflow == "cross-examination-debate":
        return LabeledWorkflow(
            CrossExaminationDebateWorkflow(
                agents,
                judge,
                rounds=condition.rounds,
                claim_prompt=CROSS_EXAMINATION_CLAIM_PROMPT,
                challenge_prompt=CROSS_EXAMINATION_CHALLENGE_PROMPT,
                response_prompt=CROSS_EXAMINATION_RESPONSE_PROMPT,
                verdict_prompt=CROSS_EXAMINATION_VERDICT_PROMPT,
                final_revision_prompt=CROSS_EXAMINATION_FINAL_REVISION_PROMPT,
                judge_prompt=JUDGE_SELECTION_PROMPT,
                tie_break_judge_prompt=TIE_BREAK_JUDGE_PROMPT,
                parallel=not condition.sequential,
                aggregation=condition.aggregation,  # type: ignore[arg-type]
                voting=voting,
                claim_max_tokens=condition.cross_exam_claim_max_tokens,
                challenge_max_tokens=condition.cross_exam_challenge_max_tokens,
                response_max_tokens=condition.cross_exam_response_max_tokens,
                verdict_max_tokens=condition.cross_exam_verdict_max_tokens,
            ),
            condition.id,
        )
    if condition.workflow in {"debate", "adversarial-debate"}:
        workflow_type = (
            AdversarialDebateWorkflow
            if condition.workflow == "adversarial-debate"
            else DebateWorkflow
        )
        return LabeledWorkflow(
            workflow_type(
                agents,
                judge,
                rounds=condition.rounds,
                debate_prompt=DEBATE_REVIEW_PROMPT,
                judge_prompt=JUDGE_SELECTION_PROMPT,
                tie_break_judge_prompt=TIE_BREAK_JUDGE_PROMPT,
                parallel=not condition.sequential,
                aggregation=condition.aggregation,  # type: ignore[arg-type]
                voting=voting,
                peer_view=condition.debate_peer_view,  # type: ignore[arg-type]
                adversarial_role_prompts=(
                    DEBATE_DERIVATION_ROLE_PROMPT,
                    DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
                    DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
                ),
                adversarial_challenge_prompt=DEBATE_ADVERSARIAL_CHALLENGE_PROMPT,
                adversarial_unanimous_prompt=DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT,
                adversarial_resolution_prompt=DEBATE_ADVERSARIAL_RESOLUTION_PROMPT,
            ),
            condition.id,
        )
    if condition.workflow == "supervisor":
        supervisor = AgentSpec(
            id="supervisor",
            model=model,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT.template,
            system_prompt_name=SUPERVISOR_SYSTEM_PROMPT.name,
            system_prompt_version=SUPERVISOR_SYSTEM_PROMPT.version,
            reasoning_effort=(
                condition.supervisor_reasoning_effort
                or condition.judge_reasoning_effort
                or condition.reasoning_effort
            ),
            service_tier=condition.judge_service_tier or condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
        return LabeledWorkflow(
            SupervisorWorkflow(
                worker=base_agent,
                supervisor=supervisor,
                max_revisions=condition.rounds,
                review_prompt=SUPERVISOR_REVIEW_PROMPT,
                revision_prompt=WORKER_REVISION_PROMPT,
            ),
            condition.id,
        )
    raise ValueError(f"unsupported workflow: {condition.workflow}")


class LabeledWorkflow(Workflow):
    def __init__(self, wrapped: Workflow, condition_id: str) -> None:
        self.wrapped = wrapped
        self.condition_id = condition_id
        self.name = wrapped.name
        self.version = wrapped.version

    async def run(self, task, context):  # type: ignore[no-untyped-def]
        return await self.wrapped.run(task, context)

    def config(self) -> dict[str, Any]:
        return {
            **self.wrapped.config(),
            "condition_id": self.condition_id,
        }

    def prompt_templates(self):  # type: ignore[no-untyped-def]
        return self.wrapped.prompt_templates()
