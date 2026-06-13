from __future__ import annotations

import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Callable

from dotenv import load_dotenv

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.llm import LLMCallError, LLMClient
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import AgentSpec, ModelCallRecord
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
    resume: bool = True,
    dry_run: bool = False,
    required_answer_type: str | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
    emit_json_events: bool = True,
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
    plan = {
        "experiment_id": experiment_id,
        "tasks": len(examples),
        "conditions": len(selected_conditions),
        "repetitions": repetitions,
        "jobs": len(examples) * len(selected_conditions) * repetitions,
        "estimated_minimum_model_calls": planned_calls,
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
        ledger = load_ledger(ledger_path, manifest=manifest)
    else:
        ledger = ExperimentLedger.create(
            manifest, [example.id for example in examples]
        )
        save_ledger(ledger_path, ledger)

    limiter = AsyncRateLimiter(
        max_in_flight=max_in_flight_requests,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
    )
    limited_llm = RateLimitedLLMClient(
        llm or LiteLLMClient(),
        limiter=limiter,
        estimated_tokens=estimated_tokens_per_request,
        event_handler=event_handler,
    )
    runner = ExperimentRunner(
        llm=limited_llm,
        store=FileRunStore(results_root),
    )
    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=retry_base_delay_seconds,
        max_delay_seconds=retry_max_delay_seconds,
        jitter_ratio=retry_jitter_ratio,
    )
    example_by_id = {example.id: example for example in examples}
    ledger_lock = asyncio.Lock()
    job_semaphore = asyncio.Semaphore(concurrency)
    summaries: list[dict[str, Any]] = []

    jobs = _jobs_to_run(ledger, max_attempts=max_attempts)
    if event_handler:
        event_handler(
            {
                "event": "benchmark_started",
                "total_jobs": len(ledger.jobs),
                "scheduled_jobs": len(jobs),
                "completed_jobs": len(ledger.jobs) - len(jobs),
            }
        )

    async def run_one(job: JobSpec) -> dict[str, Any]:
        condition = condition_by_id[job.condition_id]
        example = example_by_id[job.task_id]
        async with job_semaphore:
            while True:
                async with ledger_lock:
                    attempt = ledger.start_attempt(job.job_id)
                    save_ledger(ledger_path, ledger)
                if event_handler:
                    event_handler(
                        {
                            "event": "attempt_started",
                            "job_id": job.job_id,
                            "attempt": attempt.number,
                        }
                    )
                try:
                    result = await runner.run(
                        task=task_from_example(
                            example,
                            include_confidence=condition.include_confidence,
                        ),
                        workflow=_workflow(
                            condition=condition,
                            model=model,
                            judge_model=judge_model,
                            system_prompt=system_prompt,
                            requires_semantic_judge=requires_semantic_judge,
                        ),
                        experiment_id=experiment_id,
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
                should_retry = (
                    result.status == "failed"
                    and decision.retryable
                    and attempt.number < max_attempts
                )
                delay = (
                    retry_policy.delay(
                        attempt.number,
                        retry_after_seconds=decision.retry_after_seconds,
                        random_value=random_source(),
                    )
                    if should_retry
                    else None
                )
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
                            "retry_delay_seconds": delay,
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
                            "will_retry": should_retry,
                        }
                    )
                if emit_json_events:
                    print(json.dumps(summary, sort_keys=True), flush=True)

                if not should_retry:
                    return summary

                assert delay is not None
                retry_event = {
                    "event": "retry_scheduled",
                    "job_id": job.job_id,
                    "attempt": attempt.number,
                    "delay_seconds": delay,
                    "reason": decision.reason,
                }
                if event_handler:
                    event_handler(retry_event)
                if emit_json_events:
                    print(json.dumps(retry_event, sort_keys=True), flush=True)
                await sleep(delay)

    if jobs:
        summaries.extend(await asyncio.gather(*(run_one(job) for job in jobs)))
    if event_handler:
        event_handler({"event": "benchmark_finished"})
    return {
        "dry_run": False,
        **plan,
        "scheduled_jobs": len(jobs),
        "skipped_jobs": len(ledger.jobs) - len(jobs),
        "results": summaries,
    }


class RateLimitedLLMClient:
    def __init__(
        self,
        wrapped: LLMClient,
        *,
        limiter: AsyncRateLimiter,
        estimated_tokens: int,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.limiter = limiter
        self.estimated_tokens = estimated_tokens
        self.event_handler = event_handler

    async def complete(self, **kwargs: Any) -> ModelCallRecord:
        lease = await self.limiter.acquire(estimated_tokens=self.estimated_tokens)
        try:
            call = await self.wrapped.complete(**kwargs)
        except LLMCallError as exc:
            exc.record.metadata = {
                **exc.record.metadata,
                "rate_limit": asdict(lease.metadata),
            }
            raise
        else:
            call.metadata = {
                **call.metadata,
                "rate_limit": asdict(lease.metadata),
            }
            await lease.reconcile_tokens(call.usage.total_tokens or 0)
            if self.event_handler:
                self.event_handler(
                    {
                        "event": "model_call_completed",
                        "model": call.response_model or call.requested_model,
                        "output_tokens": call.usage.output_tokens or 0,
                        "latency_ms": call.latency_ms,
                    }
                )
            return call
        finally:
            lease.release()


def _jobs_to_run(
    ledger: ExperimentLedger,
    *,
    max_attempts: int,
) -> list[JobSpec]:
    jobs = []
    for record in ledger.jobs.values():
        if record.attempt_count >= max_attempts:
            continue
        if record.state in {JobState.PENDING, JobState.RUNNING}:
            jobs.append(record.spec)
        elif (
            record.state == JobState.FAILED
            and (record.latest_error or "").startswith("retryable:")
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
            SelfCriticWorkflow(base_agent, rounds=condition.rounds, revision_prompt=SELF_CRITIC_REVISION_PROMPT),
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
            reasoning_effort=condition.judge_reasoning_effort or condition.reasoning_effort,
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
        workflow_type = AdversarialDebateWorkflow if condition.workflow == "adversarial-debate" else DebateWorkflow
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
