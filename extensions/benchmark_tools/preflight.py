from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

from dotenv import load_dotenv

from extensions.benchmark_tools.config import ExperimentConfig, load_experiment_config
from extensions.benchmark_tools.connector import load_jsonl
from extensions.benchmark_tools.grading import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_GRADER_REASONING_EFFORT,
    HLEGrade,
    HLE_GRADER_PROMPT,
)
from extensions.benchmark_tools.rate_limit import (
    AsyncRateLimiter,
    RetryPolicy,
    RetryableContractError,
    retry_call,
)
from extensions.benchmark_tools.runner import RateLimitedLLMClient
from multi_agent_research.llm import LLMCallError, LLMClient
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import (
    AgentSpec,
    AnswerSpec,
    Message,
    ModelCallRecord,
    WorkflowOutput,
)
from multi_agent_research.prompts import (
    JUDGE_SELECTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SHORT_ANSWER_SEMANTIC_VOTE_PROMPT,
)


_EFFORT_ORDER = {
    name: index for index, name in enumerate(("none", "low", "medium", "high", "xhigh"))
}


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    kind: str
    model: str
    effort: str | None
    schema: str | None
    prompt: str
    expected: str
    response_format: dict[str, Any] | None = None
    max_completion_tokens: int | None = None

    def plan_summary(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "kind": self.kind,
            "model": self.model,
            "effort": self.effort,
            "schema": self.schema,
            "prompt": self.prompt,
            "expected": self.expected,
            "status": "planned",
            "call_count": 0,
            "tokens": 0,
            "cost": 0.0,
            "error": None,
        }


def plan_preflight_checks(
    *,
    config: ExperimentConfig,
    config_path: Path | str,
    primary_model: str,
    grader_model: str | None = None,
    grader_reasoning_effort: str | None = None,
) -> list[PreflightCheck]:
    task_path = _resolve_task_path(config_path, config.tasks_path)
    examples = load_jsonl(task_path)
    has_short_answers = any(
        example.answer_type == "short_answer" for example in examples
    )
    metadata = config.metadata
    selected_grader_model = (
        grader_model
        or _metadata_string(metadata, "grader_model")
        or DEFAULT_GRADER_MODEL
    )
    selected_grader_effort = (
        grader_reasoning_effort
        if grader_reasoning_effort is not None
        else _metadata_string(metadata, "grader_reasoning_effort")
        or DEFAULT_GRADER_REASONING_EFFORT
    )

    checks: list[PreflightCheck] = []
    primary_efforts = {condition.reasoning_effort for condition in config.conditions}
    primary_efforts.update(
        condition.supervisor_reasoning_effort
        or condition.judge_reasoning_effort
        or condition.reasoning_effort
        for condition in config.conditions
        if condition.workflow == "supervisor"
    )
    for effort in sorted(primary_efforts, key=_effort_sort_key):
        checks.append(_primary_check(primary_model, effort))

    judge_requirements = {
        (
            config.aggregation_judge_model or primary_model,
            condition.judge_reasoning_effort or condition.reasoning_effort,
        )
        for condition in config.conditions
        if _requires_aggregation_judge(condition, has_short_answers)
    }
    for model, effort in sorted(
        judge_requirements,
        key=lambda item: (item[0], _effort_sort_key(item[1])),
    ):
        checks.append(_aggregation_judge_check(model, effort))
        if has_short_answers and any(
            _requires_semantic_vote(condition)
            and (condition.judge_reasoning_effort or condition.reasoning_effort)
            == effort
            for condition in config.conditions
        ):
            checks.append(_semantic_vote_check(model, effort))

    checks.append(_grader_check(selected_grader_model, selected_grader_effort))
    return _deduplicate(checks)


async def preflight_experiment(
    *,
    config_path: Path | str,
    primary_model: str,
    grader_model: str | None = None,
    grader_reasoning_effort: str | None = None,
    dry_run: bool = False,
    max_attempts: int = 2,
    retry_base_delay_seconds: float = 1.0,
    retry_max_delay_seconds: float = 10.0,
    retry_jitter_ratio: float = 0.0,
    max_in_flight_requests: int = 1,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    estimated_tokens_per_request: int = 512,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    load_dotenv()
    config = load_experiment_config(config_path)
    checks = plan_preflight_checks(
        config=config,
        config_path=config_path,
        primary_model=primary_model,
        grader_model=grader_model,
        grader_reasoning_effort=grader_reasoning_effort,
    )
    if dry_run:
        return _summary(checks=[check.plan_summary() for check in checks], dry_run=True)

    limited_llm = RateLimitedLLMClient(
        llm or LiteLLMClient(),
        limiter=AsyncRateLimiter(
            max_in_flight=max_in_flight_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        ),
        estimated_tokens=estimated_tokens_per_request,
    )
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=retry_base_delay_seconds,
        max_delay_seconds=retry_max_delay_seconds,
        jitter_ratio=retry_jitter_ratio,
    )
    results = []
    for sequence, check in enumerate(checks):
        results.append(
            await _run_check(
                check=check,
                sequence=sequence,
                llm=limited_llm,
                policy=policy,
            )
        )
    return _summary(checks=results, dry_run=False)


async def _run_check(
    *,
    check: PreflightCheck,
    sequence: int,
    llm: LLMClient,
    policy: RetryPolicy,
) -> dict[str, Any]:
    attempts: list[ModelCallRecord] = []

    async def operation(attempt_number: int) -> ModelCallRecord:
        parameters = (
            {"max_completion_tokens": check.max_completion_tokens}
            if check.max_completion_tokens is not None
            else {}
        )
        agent = AgentSpec(
            id=f"preflight-{check.kind}",
            model=check.model,
            system_prompt=(
                JUDGE_SYSTEM_PROMPT.template
                if check.kind in {"aggregation_judge", "semantic_vote"}
                else ""
            ),
            reasoning_effort=check.effort,
            parameters={
                **parameters,
                **(
                    {"response_format": check.response_format}
                    if check.response_format is not None
                    else {}
                ),
            },
        )
        try:
            call = await llm.complete(
                sequence=sequence,
                run_id="provider-preflight",
                task_id=check.check_id,
                workflow="provider_preflight",
                step=f"{check.kind}_attempt_{attempt_number}",
                agent=agent,
                messages=_check_messages(check),
                metadata={
                    "preflight_check_id": check.check_id,
                    "attempt": attempt_number,
                },
            )
        except LLMCallError as exc:
            attempts.append(exc.record)
            raise
        attempts.append(call)
        try:
            _validator(check.kind)(call)
        except Exception as exc:
            raise RetryableContractError(str(exc)) from exc
        return call

    error: str | None = None
    try:
        await retry_call(operation, policy=policy)
        status = "passed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    return {
        "check_id": check.check_id,
        "kind": check.kind,
        "model": check.model,
        "effort": check.effort,
        "schema": check.schema,
        "status": status,
        "call_count": len(attempts),
        "tokens": sum(call.usage.total_tokens or 0 for call in attempts),
        "cost": sum(call.cost_usd or 0.0 for call in attempts),
        "error": error,
        "response_preview": (
            _output(attempts[-1])[:500] if status == "failed" and attempts else None
        ),
    }


def _primary_check(model: str, effort: str | None) -> PreflightCheck:
    return PreflightCheck(
        check_id=f"primary.reasoning_effort.{effort or 'default'}",
        kind="primary_reasoning",
        model=model,
        effort=effort,
        schema=None,
        prompt=("Provider compatibility preflight. Reply with exactly the word READY."),
        expected="non-empty text response",
    )


def _aggregation_judge_check(model: str, effort: str | None) -> PreflightCheck:
    candidates = (
        "Candidate 1 (agent-1):\n<final_answer>alpha</final_answer>\n\n"
        "Candidate 2 (agent-2):\n<final_answer>beta</final_answer>"
    )
    return PreflightCheck(
        check_id=f"aggregation_judge.final_answer.{effort or 'default'}",
        kind="aggregation_judge",
        model=model,
        effort=effort,
        schema="final_answer",
        prompt=(
            "Synthetic task: choose the candidate whose final answer is beta.\n\n"
            + JUDGE_SELECTION_PROMPT.render(candidates=candidates)
        ),
        expected="valid <final_answer>beta</final_answer> response",
    )


def _semantic_vote_check(model: str, effort: str | None) -> PreflightCheck:
    candidates = (
        "Candidate 1 (agent-1):\n<final_answer>0.5</final_answer>\n\n"
        "Candidate 2 (agent-2):\n<final_answer>one half</final_answer>\n\n"
        "Candidate 3 (agent-3):\n<final_answer>2</final_answer>"
    )
    prompt = SHORT_ANSWER_SEMANTIC_VOTE_PROMPT.render(
        mode="plurality_vote",
        candidate_count=3,
        tie_policy="inconclusive",
        candidates=candidates,
    )
    return PreflightCheck(
        check_id=f"aggregation_judge.semantic_vote.{effort or 'default'}",
        kind="semantic_vote",
        model=model,
        effort=effort,
        schema="vote_status+final_answer",
        prompt=(
            "Synthetic task: identify the value represented by the candidates. "
            "This is a provider compatibility check.\n\n" + prompt
        ),
        expected=(
            "<vote_status>winner</vote_status> and a valid final_answer "
            "equivalent to one half"
        ),
    )


def _grader_check(model: str, effort: str | None) -> PreflightCheck:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ExtractedAnswer",
            "strict": True,
            "schema": HLEGrade.model_json_schema(),
        },
    }
    return PreflightCheck(
        check_id=f"hle_grader.strict_json_schema.{effort or 'default'}",
        kind="hle_grader",
        model=model,
        effort=effort,
        schema="HLEGrade.strict",
        prompt=HLE_GRADER_PROMPT.render(
            question="Synthetic check: what is two plus two?",
            response="<final_answer>4</final_answer>",
            correct_answer="4",
        ),
        expected="strict HLEGrade JSON matching the provider response schema",
        response_format=response_format,
    )


def _validate_primary(call: ModelCallRecord) -> None:
    if not _output(call).strip():
        raise ValueError("provider returned an empty response")


def _validate_aggregation_judge(call: ModelCallRecord) -> None:
    output = WorkflowOutput.from_response(_output(call), AnswerSpec())
    if not output.contract_valid:
        raise ValueError(
            "aggregation judge violated final-answer contract: "
            + ", ".join(output.validation_errors)
        )
    if output.answer.casefold() != "beta":
        raise ValueError(
            f"aggregation judge selected {output.answer!r}, expected 'beta'"
        )


def _validate_semantic_vote(call: ModelCallRecord) -> None:
    response = _output(call)
    statuses = re.findall(
        r"<vote_status>\s*(winner|inconclusive)\s*</vote_status>",
        response,
        flags=re.IGNORECASE,
    )
    if not statuses or statuses[-1].casefold() != "winner":
        raise ValueError("semantic vote response omitted winner vote_status")
    output = WorkflowOutput.from_response(
        response,
        AnswerSpec(type="short_answer"),
    )
    if not output.contract_valid:
        raise ValueError(
            "semantic vote response violated final-answer contract: "
            + ", ".join(output.validation_errors)
        )
    normalized = re.sub(r"\s+", " ", output.answer.strip().casefold())
    if normalized not in {"0.5", "one half", "1/2"}:
        raise ValueError(f"semantic vote selected {output.answer!r}, expected one half")


def _validate_hle_grader(call: ModelCallRecord) -> None:
    HLEGrade.model_validate_json(_output(call))


def _validator(kind: str) -> Callable[[ModelCallRecord], None]:
    return {
        "primary_reasoning": _validate_primary,
        "aggregation_judge": _validate_aggregation_judge,
        "semantic_vote": _validate_semantic_vote,
        "hle_grader": _validate_hle_grader,
    }[kind]


def _output(call: ModelCallRecord) -> str:
    return str(call.output.content) if call.output is not None else ""


def _check_messages(check: PreflightCheck) -> list[Message]:
    messages = [Message(role="user", content=check.prompt)]
    if check.kind in {"aggregation_judge", "semantic_vote"}:
        messages.append(
            Message(
                role="user",
                content=AnswerSpec(type="short_answer").instruction(),
            )
        )
    return messages


def _requires_aggregation_judge(condition: Any, has_short_answers: bool) -> bool:
    return (
        condition.aggregation == "judge"
        or condition.vote_tie_break == "judge"
        or (has_short_answers and _requires_semantic_vote(condition))
    )


def _requires_semantic_vote(condition: Any) -> bool:
    return (
        condition.aggregation != "judge"
        and condition.workflow
        in {
            "sample",
            "debate",
            "adversarial-debate",
            "cross-examination-debate",
        }
        and not (condition.workflow == "sample" and condition.agents == 1)
    )


def _deduplicate(checks: list[PreflightCheck]) -> list[PreflightCheck]:
    unique: dict[tuple[str, str | None, str | None], PreflightCheck] = {}
    for check in checks:
        response_format = (
            json.dumps(check.response_format, sort_keys=True)
            if check.response_format is not None
            else check.schema
        )
        key = (check.model, check.effort, response_format)
        unique.setdefault(key, check)
    return list(unique.values())


def _summary(*, checks: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    passed = sum(check["status"] == "passed" for check in checks)
    failed = sum(check["status"] == "failed" for check in checks)
    return {
        "dry_run": dry_run,
        "status": "planned" if dry_run else ("failed" if failed else "passed"),
        "required_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "call_count": sum(check["call_count"] for check in checks),
        "tokens": sum(check["tokens"] for check in checks),
        "cost": sum(check["cost"] for check in checks),
        "checks": checks,
    }


def _resolve_task_path(config_path: Path | str, tasks_path: str) -> Path:
    path = Path(tasks_path)
    if path.is_absolute() or path.exists():
        return path
    relative = Path(config_path).resolve().parent / path
    return relative if relative.exists() else path


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _effort_sort_key(effort: str | None) -> tuple[int, str]:
    if effort is None:
        return (-1, "")
    return (_EFFORT_ORDER.get(effort, len(_EFFORT_ORDER)), effort)
