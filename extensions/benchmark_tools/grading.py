from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from extensions.benchmark_tools.connector import load_jsonl
from extensions.benchmark_tools.experiment import load_ledger, load_manifest
from extensions.benchmark_tools.rate_limit import RetryPolicy, classify_retry
from extensions.benchmark_tools.runner import RateLimitedLLMClient
from multi_agent_research.llm import LLMCallError, LLMClient
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import AgentSpec, Message, PromptTemplate


GRADING_SCHEMA_VERSION = 1
DEFAULT_GRADER_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_GRADER_REASONING_EFFORT = "low"
HLE_GRADER_PROMPT = PromptTemplate(
    name="benchmark.hle.semantic_grader",
    version="2.0.0",
    template="""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: $question

[response]: $response

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: $correct_answer

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.""",
)


class HLEGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int = Field(ge=0, le=100)
    strict: Literal[True]


@dataclass(frozen=True)
class GradeSetManifest:
    experiment_id: str
    task_set_sha256: str
    grader_model: str
    scope: str
    prompt_name: str
    prompt_version: str
    prompt_sha256: str
    max_tokens: int | None
    reasoning_effort: str | None
    grade_set_id: str
    schema_version: int = GRADING_SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                key: value
                for key, value in asdict(self).items()
                if key != "grade_set_id"
            }
        )


@dataclass(frozen=True)
class GradeTarget:
    task_id: str
    question: str
    correct_answer: str
    response: str
    answer_type: str
    contexts: tuple[dict[str, Any], ...] = ()

    @property
    def response_sha256(self) -> str:
        return sha256(self.response.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return _digest(
            {
                "task_id": self.task_id,
                "correct_answer": self.correct_answer,
                "response": self.response,
            }
        )


@dataclass
class GradeRecord:
    id: str
    task_id: str
    answer_type: str
    response_sha256: str
    contexts: list[dict[str, Any]]
    status: str = "pending"
    attempts: list[dict[str, Any]] = field(default_factory=list)
    grade: dict[str, Any] | None = None
    error: str | None = None
    schema_version: int = GRADING_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradeRecord:
        return cls(**data)


@dataclass(frozen=True)
class SemanticGradeSet:
    manifest: GradeSetManifest
    records: dict[tuple[str, str], GradeRecord]

    def lookup(self, task_id: str, response: str) -> GradeRecord | None:
        return self.records.get(
            (task_id, sha256(response.encode("utf-8")).hexdigest())
        )


async def grade_experiment(
    *,
    tasks_path: Path | str,
    results_dir: Path | str,
    experiment_id: str,
    grader_model: str = DEFAULT_GRADER_MODEL,
    scope: Literal["final", "all"] = "final",
    grade_set_id: str | None = None,
    concurrency: int = 8,
    max_in_flight_requests: int = 8,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    estimated_tokens_per_request: int = 2048,
    max_attempts: int = 3,
    retry_base_delay_seconds: float = 1.0,
    retry_max_delay_seconds: float = 30.0,
    retry_jitter_ratio: float = 0.2,
    max_tokens: int | None = None,
    reasoning_effort: str | None = DEFAULT_GRADER_REASONING_EFFORT,
    dry_run: bool = False,
    llm: LLMClient | None = None,
    sleep=asyncio.sleep,  # type: ignore[no-untyped-def]
    random_source=random.random,
) -> dict[str, Any]:
    load_dotenv()
    if scope not in {"final", "all"}:
        raise ValueError("grading scope must be 'final' or 'all'")
    if concurrency < 1:
        raise ValueError("grading concurrency must be positive")

    tasks_path = Path(tasks_path)
    examples = load_jsonl(tasks_path)
    experiment_root = Path(results_dir) / experiment_id
    targets = _grade_targets(
        examples=examples,
        experiment_root=experiment_root,
        scope=scope,
    )
    requested_manifest = _grade_manifest(
        experiment_id=experiment_id,
        task_set_sha256=sha256(tasks_path.read_bytes()).hexdigest(),
        grader_model=grader_model,
        scope=scope,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        grade_set_id=grade_set_id,
    )
    output_root = (
        experiment_root / "grades" / requested_manifest.grade_set_id
    )
    records = {
        target.id: _load_or_create_record(output_root, target)
        for target in targets
    }
    pending = [
        target
        for target in targets
        if records[target.id].status != "success"
        and len(records[target.id].attempts) < max_attempts
    ]
    plan = {
        "experiment_id": experiment_id,
        "grade_set_id": requested_manifest.grade_set_id,
        "scope": scope,
        "canonical_responses": sum(len(target.contexts) for target in targets),
        "unique_responses": len(targets),
        "scheduled_grades": len(pending),
        "cached_grades": sum(
            record.status == "success" for record in records.values()
        ),
        "exhausted_failed_grades": sum(
            record.status != "success"
            and len(record.attempts) >= max_attempts
            for record in records.values()
        ),
        "grader_model": grader_model,
        "reasoning_effort": reasoning_effort,
    }
    if dry_run:
        return {"dry_run": True, **plan}

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        persisted = _manifest_from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if persisted.fingerprint != requested_manifest.fingerprint:
            raise ValueError(
                "grade set is incompatible with the existing grading manifest"
            )
    else:
        _atomic_write_json(manifest_path, asdict(requested_manifest))

    from extensions.benchmark_tools.rate_limit import AsyncRateLimiter

    limited_llm = RateLimitedLLMClient(
        llm or LiteLLMClient(),
        limiter=AsyncRateLimiter(
            max_in_flight=max_in_flight_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        ),
        estimated_tokens=estimated_tokens_per_request,
    )
    parameters: dict[str, Any] = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedAnswer",
                "strict": True,
                "schema": HLEGrade.model_json_schema(),
            },
        },
    }
    if max_tokens is not None:
        parameters["max_completion_tokens"] = max_tokens
    agent = AgentSpec(
        id="hle-grader",
        model=grader_model,
        reasoning_effort=reasoning_effort,
        parameters=parameters,
    )
    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=retry_base_delay_seconds,
        max_delay_seconds=retry_max_delay_seconds,
        jitter_ratio=retry_jitter_ratio,
    )
    semaphore = asyncio.Semaphore(concurrency)
    sequence_lock = asyncio.Lock()
    sequence = 0

    async def next_sequence() -> int:
        nonlocal sequence
        async with sequence_lock:
            value = sequence
            sequence += 1
            return value

    async def grade_one(target: GradeTarget) -> dict[str, Any]:
        record = records[target.id]
        async with semaphore:
            while len(record.attempts) < max_attempts:
                attempt_number = len(record.attempts) + 1
                prompt = HLE_GRADER_PROMPT.render(
                    question=target.question,
                    response=target.response,
                    correct_answer=target.correct_answer,
                )
                try:
                    call = await limited_llm.complete(
                        sequence=await next_sequence(),
                        run_id=f"grade-{requested_manifest.grade_set_id}",
                        task_id=target.task_id,
                        workflow="hle_semantic_grader",
                        step=f"grade_attempt_{attempt_number}",
                        agent=agent,
                        messages=[Message(role="user", content=prompt)],
                        prompt_references=[HLE_GRADER_PROMPT.reference()],
                        metadata={
                            "grade_target_id": target.id,
                            "answer_type": target.answer_type,
                        },
                    )
                    content = (
                        str(call.output.content)
                        if call.output is not None
                        else ""
                    )
                    try:
                        grade = HLEGrade.model_validate(_parse_json(content))
                    except Exception as exc:
                        record.attempts.append(
                            {
                                "number": attempt_number,
                                "status": "failed",
                                "error": f"invalid_grader_output:{exc}",
                                "call": call.model_dump(mode="json"),
                            }
                        )
                        record.status = "failed"
                        record.error = f"invalid_grader_output:{exc}"
                        _save_record(output_root, record)
                        should_retry = attempt_number < max_attempts
                        reason = "invalid_grader_output"
                        retry_after = None
                    else:
                        record.attempts.append(
                            {
                                "number": attempt_number,
                                "status": "success",
                                "call": call.model_dump(mode="json"),
                            }
                        )
                        record.status = "success"
                        record.grade = grade.model_dump()
                        record.error = None
                        _save_record(output_root, record)
                        return _record_summary(record)
                except LLMCallError as exc:
                    decision = classify_retry(exc)
                    record.attempts.append(
                        {
                            "number": attempt_number,
                            "status": "failed",
                            "error": (
                                f"{decision.reason}:"
                                f"{exc.record.error.message if exc.record.error else exc}"
                            ),
                            "call": exc.record.model_dump(mode="json"),
                        }
                    )
                    record.status = "failed"
                    record.error = record.attempts[-1]["error"]
                    _save_record(output_root, record)
                    should_retry = (
                        decision.retryable and attempt_number < max_attempts
                    )
                    reason = decision.reason
                    retry_after = decision.retry_after_seconds

                if not should_retry:
                    return _record_summary(record)
                delay = retry_policy.delay(
                    attempt_number,
                    retry_after_seconds=retry_after,
                    random_value=random_source(),
                )
                print(
                    json.dumps(
                        {
                            "event": "grade_retry_scheduled",
                            "grade_target_id": target.id,
                            "attempt": attempt_number,
                            "delay_seconds": delay,
                            "reason": reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                await sleep(delay)
        return _record_summary(record)

    completed = await asyncio.gather(*(grade_one(target) for target in pending))
    return {
        "dry_run": False,
        **plan,
        "successful_grades": sum(
            record.status == "success" for record in records.values()
        ),
        "failed_grades": sum(
            record.status != "success" for record in records.values()
        ),
        "results": completed,
        "output_dir": str(output_root),
    }


def load_grade_set(
    *,
    results_dir: Path | str,
    experiment_id: str,
    grade_set_id: str | None = None,
) -> SemanticGradeSet | None:
    root = Path(results_dir) / experiment_id / "grades"
    if not root.exists():
        return None
    if grade_set_id is None:
        candidates = sorted(
            path for path in root.iterdir() if (path / "manifest.json").exists()
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError(
                "multiple semantic grade sets exist; specify grade_set_id"
            )
        selected = candidates[0]
    else:
        selected = root / grade_set_id
        if not (selected / "manifest.json").exists():
            raise ValueError(f"semantic grade set not found: {grade_set_id}")

    manifest = _manifest_from_dict(
        json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    )
    records: dict[tuple[str, str], GradeRecord] = {}
    for path in sorted((selected / "items").glob("*.json")):
        record = GradeRecord.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        records[(record.task_id, record.response_sha256)] = record
    return SemanticGradeSet(manifest=manifest, records=records)


def _grade_targets(
    *,
    examples: list[Any],
    experiment_root: Path,
    scope: str,
) -> list[GradeTarget]:
    example_by_id = {example.id: example for example in examples}
    results = _canonical_results(experiment_root)
    targets: dict[str, GradeTarget] = {}
    contexts: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "success":
            continue
        example = example_by_id.get(str(result.get("task_id")))
        if example is None:
            continue
        output = result.get("output") or {}
        response = str(
            output.get("raw_response")
            or result.get("final_answer")
            or output.get("answer")
            or ""
        ).strip()
        if response:
            target = GradeTarget(
                task_id=example.id,
                question=example.prompt,
                correct_answer=example.answer,
                response=response,
                answer_type=example.answer_type,
            )
            targets[target.id] = target
            contexts.setdefault(target.id, []).append(
                {
                    "run_id": result.get("run_id"),
                    "kind": "final",
                    "sequence": None,
                }
            )
        if scope != "all":
            continue
        for stage in result.get("stage_answers") or []:
            stage_output = stage.get("output") or {}
            stage_response = str(
                stage_output.get("raw_response")
                or stage_output.get("answer")
                or ""
            ).strip()
            if not stage_response:
                continue
            target = GradeTarget(
                task_id=example.id,
                question=example.prompt,
                correct_answer=example.answer,
                response=stage_response,
                answer_type=example.answer_type,
            )
            targets[target.id] = target
            contexts.setdefault(target.id, []).append(
                {
                    "run_id": result.get("run_id"),
                    "kind": "stage",
                    "sequence": stage.get("sequence"),
                    "step": stage.get("step"),
                    "agent_id": stage.get("agent_id"),
                }
            )
    return [
        GradeTarget(
            task_id=target.task_id,
            question=target.question,
            correct_answer=target.correct_answer,
            response=target.response,
            answer_type=target.answer_type,
            contexts=tuple(contexts[target_id]),
        )
        for target_id, target in sorted(targets.items())
    ]


def _canonical_results(experiment_root: Path) -> list[dict[str, Any]]:
    results = {
        str(result.get("run_id")): result
        for path in sorted(experiment_root.glob("*/result.json"))
        for result in [json.loads(path.read_text(encoding="utf-8"))]
    }
    manifest_path = experiment_root / "experiment-manifest.json"
    ledger_path = experiment_root / "experiment-ledger.json"
    if not manifest_path.exists() or not ledger_path.exists():
        return list(results.values())
    manifest = load_manifest(manifest_path)
    ledger = load_ledger(ledger_path, manifest=manifest)
    selected = []
    for record in ledger.jobs.values():
        attempt = next(
            (
                item
                for item in record.attempts
                if item.attempt_id == record.selected_attempt_id
            ),
            None,
        )
        if attempt and attempt.run_id in results:
            selected.append(results[attempt.run_id])
    return selected


def _grade_manifest(
    *,
    experiment_id: str,
    task_set_sha256: str,
    grader_model: str,
    scope: str,
    max_tokens: int | None,
    reasoning_effort: str | None,
    grade_set_id: str | None,
) -> GradeSetManifest:
    material = {
        "experiment_id": experiment_id,
        "task_set_sha256": task_set_sha256,
        "grader_model": grader_model,
        "scope": scope,
        "prompt_name": HLE_GRADER_PROMPT.name,
        "prompt_version": HLE_GRADER_PROMPT.version,
        "prompt_sha256": HLE_GRADER_PROMPT.content_sha256,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "schema_version": GRADING_SCHEMA_VERSION,
    }
    generated_id = f"hle-{_digest(material)[:12]}"
    return GradeSetManifest(
        **material,
        grade_set_id=grade_set_id or generated_id,
    )


def _manifest_from_dict(data: dict[str, Any]) -> GradeSetManifest:
    return GradeSetManifest(**data)


def _load_or_create_record(
    output_root: Path,
    target: GradeTarget,
) -> GradeRecord:
    path = output_root / "items" / f"{target.id}.json"
    if path.exists():
        record = GradeRecord.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        record.contexts = [dict(context) for context in target.contexts]
        return record
    return GradeRecord(
        id=target.id,
        task_id=target.task_id,
        answer_type=target.answer_type,
        response_sha256=target.response_sha256,
        contexts=[dict(context) for context in target.contexts],
    )


def _save_record(output_root: Path, record: GradeRecord) -> None:
    _atomic_write_json(
        output_root / "items" / f"{record.id}.json",
        asdict(record),
    )


def _record_summary(record: GradeRecord) -> dict[str, Any]:
    return {
        "grade_target_id": record.id,
        "task_id": record.task_id,
        "status": record.status,
        "attempts": len(record.attempts),
        "correct": (record.grade or {}).get("correct"),
        "error": record.error,
    }


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("grader response must be a JSON object")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
