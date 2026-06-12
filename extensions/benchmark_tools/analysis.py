from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from multi_agent_research.aggregation import normalize_answer
from multi_agent_research.models import AnswerChoice, AnswerSpec

from extensions.benchmark_tools.connector import load_jsonl
from extensions.benchmark_tools.experiment import load_ledger, load_manifest
from extensions.benchmark_tools.grading import SemanticGradeSet, load_grade_set


JobKey = tuple[str, str, str]


def analyze_experiment(
    *,
    tasks_path: Path | str,
    results_dir: Path | str,
    experiment_id: str,
    output_dir: Path | str,
    grade_set_id: str | None = None,
    require_semantic_grades: bool | None = None,
) -> dict[str, Any]:
    examples = load_jsonl(tasks_path)
    semantic_grades = load_grade_set(
        results_dir=results_dir,
        experiment_id=experiment_id,
        grade_set_id=grade_set_id,
    )
    semantic_required = (
        any(example.answer_type == "short_answer" for example in examples)
        if require_semantic_grades is None
        else require_semantic_grades
    )
    if semantic_required and semantic_grades is None:
        raise ValueError(
            "semantic HLE grades are required for this task set; run the "
            "benchmark-tools grade command first"
        )
    if (
        semantic_required
        and semantic_grades is not None
        and semantic_grades.manifest.scope != "all"
    ):
        raise ValueError(
            "full semantic analysis requires a grade set with scope='all'"
        )
    experiment_root = Path(results_dir) / experiment_id
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(experiment_root.glob("*/result.json"))
    ]
    expected_jobs = None
    experiment_metadata: dict[str, Any] = {
        "experiment_id": experiment_id,
        "scoring": (
            "official_contract_semantic_hle"
            if semantic_grades is not None
            else "canonical_exact"
        ),
        **_task_set_metadata(examples),
    }
    if semantic_grades is not None:
        experiment_metadata.update(
            {
                "grade_set_id": semantic_grades.manifest.grade_set_id,
                "grader_model": semantic_grades.manifest.grader_model,
                "grader_prompt": semantic_grades.manifest.prompt_name,
                "grader_prompt_version": semantic_grades.manifest.prompt_version,
                "grader_prompt_sha256": semantic_grades.manifest.prompt_sha256,
                "grading_scope": semantic_grades.manifest.scope,
                **_grade_set_metrics(semantic_grades),
            }
        )
    manifest_path = experiment_root / "experiment-manifest.json"
    ledger_path = experiment_root / "experiment-ledger.json"
    if manifest_path.exists() and ledger_path.exists():
        manifest = load_manifest(manifest_path)
        ledger = load_ledger(ledger_path, manifest=manifest)
        run_to_job = {
            attempt.run_id: record
            for record in ledger.jobs.values()
            for attempt in record.attempts
            if attempt.run_id
        }
        for result in results:
            record = run_to_job.get(result.get("run_id"))
            if record is not None:
                result["job_id"] = record.spec.job_id
                result["repetition"] = record.spec.repetition
        expected_jobs = [
            {
                "job_id": record.spec.job_id,
                "condition": record.spec.condition_id,
                "task_id": record.spec.task_id,
                "repetition": record.spec.repetition,
                "attempt_count": record.attempt_count,
            }
            for record in ledger.jobs.values()
        ]
        experiment_metadata.update(
            {
                "model": manifest.model,
                "judge_model": manifest.judge_model,
                "repetitions": manifest.repetitions,
                "task_set_sha256": manifest.task_set_sha256,
                "manifest_schema_version": manifest.schema_version,
            }
        )
    summary, run_rows, stage_rows = analyze_attempts(
        results,
        expected_answers={example.id: example.answer for example in examples},
        answer_specs={
            example.id: AnswerSpec(
                type=example.answer_type,  # type: ignore[arg-type]
                choices=[
                    AnswerChoice(label=choice.label, text=choice.text)
                    for choice in example.choices
                ],
                include_confidence=True,
            )
            for example in examples
        },
        categories={example.id: example.category or "unknown" for example in examples},
        answer_types={example.id: example.answer_type for example in examples},
        experiment_id=experiment_id,
        expected_jobs=expected_jobs,
        metadata=experiment_metadata,
        semantic_grades=semantic_grades,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "stage_answers.csv", stage_rows)
    _write_csv(output / "summary.csv", summary["conditions"])
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "runs.json").write_text(
        json.dumps(run_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "stage_answers.json").write_text(
        json.dumps(stage_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def analyze_attempts(
    results: Iterable[dict[str, Any]],
    *,
    expected_answers: dict[str, str],
    answer_specs: dict[str, AnswerSpec],
    categories: dict[str, str] | None = None,
    answer_types: dict[str, str] | None = None,
    experiment_id: str = "",
    expected_jobs: Iterable[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    semantic_grades: SemanticGradeSet | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze terminal attempts as logical jobs.

    New ledgers should supply stable ``job_id`` and ``repetition`` fields. Legacy
    artifacts fall back to condition + task_id + repetition (default 0), which
    intentionally collapses reruns of the same condition/task into one job.
    """
    attempts = [
        result for result in results if result.get("task_id") in expected_answers
    ]
    grouped: dict[JobKey, list[dict[str, Any]]] = defaultdict(list)
    for result in attempts:
        grouped[job_key(result)].append(result)

    planned = (
        {_expected_job_key(job): dict(job) for job in expected_jobs}
        if expected_jobs is not None
        else _infer_expected_jobs(attempts, expected_answers)
    )
    for key, rows in grouped.items():
        planned.setdefault(
            key,
            {
                "job_id": _job_id(rows[0]),
                "condition": _condition_id(rows[0]),
                "task_id": str(rows[0]["task_id"]),
                "repetition": _repetition(rows[0]),
            },
        )

    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for key, job in sorted(planned.items()):
        task_id = str(job["task_id"])
        job_attempts = grouped.get(key, [])
        selected = select_canonical_attempt(job_attempts)
        row = _job_row(
            job=job,
            selected=selected,
            attempts=job_attempts,
            expected=expected_answers[task_id],
            answer_spec=answer_specs[task_id],
            category=(categories or {}).get(task_id, "unknown"),
            answer_type=(answer_types or {}).get(
                task_id,
                answer_specs[task_id].type,
            ),
            experiment_id=experiment_id,
            semantic_grades=semantic_grades,
        )
        run_rows.append(row)
        if selected is not None:
            stage_rows.extend(
                _stage_rows(
                    selected,
                    row,
                    expected_answers[task_id],
                    answer_specs[task_id],
                    semantic_grades,
                )
            )

    summary = summarize(run_rows, stage_rows, metadata=metadata)
    return summary, run_rows, stage_rows


def job_key(result: dict[str, Any]) -> JobKey:
    condition = _condition_id(result)
    task_id = str(result["task_id"])
    stable_job_id = _job_id(result)
    if stable_job_id:
        return condition, task_id, f"job:{stable_job_id}"
    return condition, task_id, f"repetition:{_repetition(result)}"


def select_canonical_attempt(
    attempts: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    rows = list(attempts)
    if not rows:
        return None
    successful = [row for row in rows if row.get("status") == "success"]
    return max(successful or rows, key=_attempt_order)


def summarize(
    run_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_condition[row["condition"]].append(row)

    condition_summary = [
        _condition_summary(condition, rows)
        for condition, rows in sorted(by_condition.items())
    ]
    error_counts = Counter(
        row["error_reason"]
        for row in run_rows
        if row["outcome"] != "completed_valid"
        or row["correct"] is not True
    )
    return {
        "schema_version": "3",
        "metadata": metadata or {},
        "conditions": condition_summary,
        "answer_type_breakdown": answer_type_breakdown(run_rows),
        "paired_comparisons": paired_comparisons(run_rows),
        "error_reasons": dict(sorted(error_counts.items())),
        "revision_transitions": revision_transitions(stage_rows),
        "question_heatmap": question_heatmap(run_rows),
    }


def paired_comparisons(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(
        dict
    )
    for row in run_rows:
        if row["correct"] is not None:
            pair_key = (str(row["task_id"]), str(row["repetition"]))
            by_condition[row["condition"]][pair_key] = row

    comparisons = []
    for condition_a, condition_b in combinations(sorted(by_condition), 2):
        matched = sorted(
            set(by_condition[condition_a]) & set(by_condition[condition_b])
        )
        correct_a = sum(
            bool(by_condition[condition_a][key]["correct"]) for key in matched
        )
        correct_b = sum(
            bool(by_condition[condition_b][key]["correct"]) for key in matched
        )
        count = len(matched)
        accuracy_a = correct_a / count if count else None
        accuracy_b = correct_b / count if count else None
        comparisons.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "matched_completed_pairs": count,
                "accuracy_a": accuracy_a,
                "accuracy_b": accuracy_b,
                "accuracy_delta_b_minus_a": (
                    accuracy_b - accuracy_a if count else None
                ),
            }
        )
    return comparisons


def answer_type_breakdown(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["condition"], row["answer_type"])].append(row)
    return [
        {
            "condition": condition,
            "answer_type": answer_type,
            "expected_jobs": len(rows),
            "completed_answer_jobs": sum(
                row["outcome"] == "completed_valid" for row in rows
            ),
            "graded_jobs": sum(row["correct"] is not None for row in rows),
            "grading_failures": sum(
                row["grading_status"] in {"missing", "failed"} for row in rows
            ),
            "correct": sum(row["correct"] is True for row in rows),
            "planned_job_accuracy": (
                sum(row["correct"] is True for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "graded_accuracy": (
                sum(row["correct"] is True for row in rows)
                / sum(row["correct"] is not None for row in rows)
                if any(row["correct"] is not None for row in rows)
                else None
            ),
        }
        for (condition, answer_type), rows in sorted(grouped.items())
    ]


def revision_transitions(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        if row["kind"] != "candidate":
            continue
        key = (
            row["condition"],
            row["job_key"],
            row.get("agent_id") or row["step"],
        )
        grouped[key].append(row)
    counts: Counter[tuple[str, str]] = Counter()
    for (condition, _, _), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["sequence"]))
        if len(ordered) < 2:
            continue
        if ordered[0]["correct"] is None or ordered[-1]["correct"] is None:
            continue
        first = bool(ordered[0]["correct"])
        last = bool(ordered[-1]["correct"])
        label = (
            ("right" if first else "wrong")
            + "_to_"
            + ("right" if last else "wrong")
        )
        counts[(condition, label)] += 1
    return [
        {"condition": condition, "transition": transition, "count": count}
        for (condition, transition), count in sorted(counts.items())
    ]


def question_heatmap(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "repetition": row["repetition"],
            "condition": row["condition"],
            "correct": row["correct"],
            "outcome": row["outcome"],
            "category": row["category"],
        }
        for row in sorted(
            run_rows,
            key=lambda row: (
                row["task_id"],
                str(row["repetition"]),
                row["condition"],
            ),
        )
    ]


def error_reason(
    result: dict[str, Any],
    task_id: str,
    expected: str,
    answer_spec: AnswerSpec,
    semantic_grades: SemanticGradeSet | None,
) -> str:
    if result.get("status") == "inconclusive":
        return "inconclusive"
    if result.get("status") != "success":
        return "provider_execution_failure"
    output = result.get("output") or {}
    final_answer = output.get("answer") or result.get("final_answer") or ""
    if not str(final_answer).strip():
        return "inconclusive"
    final_response = str(output.get("raw_response") or final_answer)
    final_score = _score(
        task_id,
        final_response,
        str(final_answer),
        expected,
        answer_spec,
        semantic_grades,
    )
    if final_score["correct"] is None:
        return f"semantic_grade_{final_score['status']}"
    if final_score["correct"]:
        if not output.get("contract_valid", False):
            return "correct_contract_invalid"
        return "correct"
    stages = result.get("stage_answers") or []
    candidate_stages = [
        stage for stage in stages if stage.get("kind") == "candidate"
    ]
    aggregate_stages = [
        stage for stage in stages if stage.get("kind") == "aggregate"
    ]
    candidate_correct = [
        stage
        for stage in candidate_stages
        if _stage_score(
            stage,
            task_id,
            expected,
            answer_spec,
            semantic_grades,
        )["correct"]
    ]
    if candidate_correct and aggregate_stages:
        return "aggregation_error"
    if _has_right_to_wrong(
        candidate_stages,
        task_id,
        expected,
        answer_spec,
        semantic_grades,
    ):
        return "bad_revision"
    if candidate_correct:
        return "peer_persuasion_failure"
    if result["workflow"]["name"] in {"debate", "adversarial_debate"}:
        return "shared_reasoning_error"
    return "model_error"


def _condition_summary(
    condition: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = len(rows)
    valid = [row for row in rows if row["outcome"] == "completed_valid"]
    graded = [row for row in rows if row["correct"] is not None]
    correct = sum(row["correct"] is True for row in rows)
    cost = sum(float(row["cost_usd"]) for row in rows)
    tokens = sum(int(row["total_tokens"]) for row in rows)
    return {
        "condition": condition,
        "workflow": next(
            (row["workflow"] for row in rows if row["workflow"]),
            "unknown",
        ),
        "expected_jobs": expected,
        "completed_answer_jobs": len(valid),
        "graded_jobs": len(graded),
        "grading_failures": sum(
            row["grading_status"] in {"missing", "failed"} for row in rows
        ),
        "inconclusive_jobs": _count_outcome(rows, "inconclusive"),
        "provider_execution_failures": _count_outcome(
            rows, "provider_execution_failure"
        ),
        "contract_invalid_outputs": _count_outcome(rows, "contract_invalid"),
        "missing_jobs": _count_outcome(rows, "missing"),
        "attempts": sum(int(row["attempt_count"]) for row in rows),
        "retried_jobs": sum(int(row["attempt_count"] > 1) for row in rows),
        "coverage_rate": len(valid) / expected if expected else 0.0,
        "planned_job_accuracy": correct / expected if expected else 0.0,
        "valid_completed_accuracy": (
            sum(row["correct"] is True for row in valid)
            / sum(row["correct"] is not None for row in valid)
            if any(row["correct"] is not None for row in valid)
            else None
        ),
        "graded_accuracy": correct / len(graded) if graded else None,
        "correct_valid_completed": sum(
            row["correct"] is True for row in valid
        ),
        "total_tokens": tokens,
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
        "cost_usd": cost,
        "correct_per_dollar": correct / cost if cost else None,
        "correct_per_million_tokens": (
            correct / (tokens / 1_000_000) if tokens else None
        ),
        "avg_wall_time_ms": (
            sum(float(row["wall_time_ms"]) for row in rows) / expected
            if expected
            else 0.0
        ),
    }


def _job_row(
    *,
    job: dict[str, Any],
    selected: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
    expected: str,
    answer_spec: AnswerSpec,
    category: str,
    answer_type: str,
    experiment_id: str,
    semantic_grades: SemanticGradeSet | None,
) -> dict[str, Any]:
    condition = str(job["condition"])
    task_id = str(job["task_id"])
    repetition = job.get("repetition", 0)
    if selected is None:
        return {
            "experiment_id": experiment_id,
            "job_id": job.get("job_id"),
            "job_key": "|".join((condition, task_id, str(repetition))),
            "condition": condition,
            "workflow": job.get("workflow", ""),
            "run_id": "",
            "task_id": task_id,
            "repetition": repetition,
            "category": category,
            "answer_type": answer_type,
            "status": "missing",
            "outcome": "missing",
            "attempt_count": int(job.get("attempt_count", 0)),
            "successful_attempt_count": 0,
            "selected_attempt_ended_at": "",
            "final_answer": "",
            "expected_answer": expected,
            "correct": None,
            "grading_status": "not_available",
            "grader_extracted_answer": "",
            "grader_reasoning": "",
            "grader_confidence": None,
            "contract_valid": 0,
            "error_reason": "missing",
            **_empty_metrics(),
        }

    output = selected.get("output") or {}
    final_answer = output.get("answer") or selected.get("final_answer") or ""
    final_response = str(output.get("raw_response") or final_answer)
    outcome = _outcome(selected)
    score = (
        _score(
            task_id,
            final_response,
            str(final_answer),
            expected,
            answer_spec,
            semantic_grades,
        )
        if selected.get("status") == "success" and final_response.strip()
        else {
            "correct": None,
            "status": "not_available",
            "extracted_final_answer": "",
            "reasoning": "",
            "confidence": None,
        }
    )
    metrics = selected.get("metrics") or {}
    return {
        "experiment_id": experiment_id,
        "job_id": _job_id(selected) or job.get("job_id"),
        "job_key": "|".join(job_key(selected)),
        "condition": condition,
        "workflow": selected["workflow"]["name"],
        "run_id": selected.get("run_id", ""),
        "task_id": task_id,
        "repetition": job.get("repetition", _repetition(selected)),
        "category": category,
        "answer_type": answer_type,
        "status": selected.get("status", ""),
        "outcome": outcome,
        "attempt_count": max(len(attempts), int(job.get("attempt_count", 0))),
        "successful_attempt_count": sum(
            attempt.get("status") == "success" for attempt in attempts
        ),
        "selected_attempt_ended_at": selected.get("ended_at") or "",
        "final_answer": final_answer,
        "expected_answer": expected,
        "correct": score["correct"],
        "grading_status": score["status"],
        "grader_extracted_answer": score["extracted_final_answer"],
        "grader_reasoning": score["reasoning"],
        "grader_confidence": score["confidence"],
        "contract_valid": int(bool(output.get("contract_valid"))),
        "error_reason": error_reason(
            selected,
            task_id,
            expected,
            answer_spec,
            semantic_grades,
        ),
        "model_calls": metrics.get("model_calls", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "reasoning_tokens": metrics.get("reasoning_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "cost_usd": metrics.get("cost_usd", 0.0),
        "wall_time_ms": metrics.get("wall_time_ms", 0.0),
    }


def _stage_rows(
    result: dict[str, Any],
    job_row: dict[str, Any],
    expected: str,
    answer_spec: AnswerSpec,
    semantic_grades: SemanticGradeSet | None,
) -> list[dict[str, Any]]:
    rows = []
    for stage in result.get("stage_answers") or []:
        output = stage.get("output") or {}
        stage_answer = output.get("answer") or ""
        score = _stage_score(
            stage,
            str(job_row["task_id"]),
            expected,
            answer_spec,
            semantic_grades,
        )
        rows.append(
            {
                "condition": job_row["condition"],
                "job_key": job_row["job_key"],
                "repetition": job_row["repetition"],
                "workflow": job_row["workflow"],
                "run_id": job_row["run_id"],
                "task_id": job_row["task_id"],
                "sequence": stage.get("sequence"),
                "step": stage.get("step"),
                "kind": stage.get("kind"),
                "agent_id": stage.get("agent_id"),
                "answer": stage_answer,
                "correct": score["correct"],
                "grading_status": score["status"],
                "grader_extracted_answer": score["extracted_final_answer"],
                "grader_reasoning": score["reasoning"],
                "grader_confidence": score["confidence"],
                "contract_valid": int(bool(output.get("contract_valid"))),
                "confidence": output.get("confidence"),
            }
        )
    return rows


def _infer_expected_jobs(
    attempts: list[dict[str, Any]],
    expected_answers: dict[str, str],
) -> dict[JobKey, dict[str, Any]]:
    if any(_job_id(result) for result in attempts):
        return {
            job_key(result): {
                "job_id": _job_id(result),
                "condition": _condition_id(result),
                "task_id": str(result["task_id"]),
                "repetition": _repetition(result),
            }
            for result in attempts
        }
    conditions = sorted({_condition_id(result) for result in attempts})
    repetitions = sorted({_repetition(result) for result in attempts}, key=str) or [0]
    return {
        (condition, task_id, f"repetition:{repetition}"): {
            "job_id": None,
            "condition": condition,
            "task_id": task_id,
            "repetition": repetition,
        }
        for condition in conditions
        for task_id in expected_answers
        for repetition in repetitions
    }


def _expected_job_key(job: dict[str, Any]) -> JobKey:
    condition = str(job["condition"])
    task_id = str(job["task_id"])
    if job.get("job_id"):
        return condition, task_id, f"job:{job['job_id']}"
    return condition, task_id, f"repetition:{job.get('repetition', 0)}"


def _outcome(result: dict[str, Any]) -> str:
    if result.get("status") == "inconclusive":
        return "inconclusive"
    if result.get("status") != "success":
        return "provider_execution_failure"
    output = result.get("output") or {}
    if not output.get("contract_valid", False):
        return "contract_invalid"
    if not str(output.get("answer") or result.get("final_answer") or "").strip():
        return "inconclusive"
    return "completed_valid"


def _attempt_order(result: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(result.get("ended_at") or result.get("started_at") or ""),
        str(result.get("started_at") or ""),
        str(result.get("run_id") or ""),
    )


def _job_id(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") or {}
    value = result.get("job_id") or metadata.get("job_id")
    return str(value) if value is not None else None


def _repetition(result: dict[str, Any]) -> Any:
    metadata = result.get("metadata") or {}
    return result.get("repetition", metadata.get("repetition", 0))


def _has_right_to_wrong(
    stages: list[dict[str, Any]],
    task_id: str,
    expected: str,
    answer_spec: AnswerSpec,
    semantic_grades: SemanticGradeSet | None,
) -> bool:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stage in stages:
        by_agent[stage.get("agent_id") or stage.get("step")].append(stage)
    for rows in by_agent.values():
        ordered = sorted(rows, key=lambda row: int(row.get("sequence") or 0))
        values = [
            _stage_score(
                row,
                task_id,
                expected,
                answer_spec,
                semantic_grades,
            )["correct"]
            for row in ordered
        ]
        if any(
            values[index] is True and values[index + 1] is False
            for index in range(len(values) - 1)
        ):
            return True
    return False


def _condition_id(result: dict[str, Any]) -> str:
    config = result["workflow"].get("config") or {}
    if config.get("condition_id"):
        return str(config["condition_id"])
    workflow = result["workflow"]["name"]
    if workflow == "solo":
        return "solo"
    if workflow == "independent_sample":
        return f"sample-{len(config.get('agents') or [])}"
    if workflow in {"debate", "adversarial_debate"}:
        prefix = (
            "adversarial-debate"
            if workflow == "adversarial_debate"
            else "debate"
        )
        return (
            f"{prefix}-{len(config.get('agents') or [])}-"
            f"{config.get('peer_view', 'full_response')}"
        )
    return workflow


def _task_set_metadata(examples: list[Any]) -> dict[str, Any]:
    sources = {
        (
            str(example.source.get("benchmark") or ""),
            str(example.source.get("version") or ""),
            str(example.source.get("split") or ""),
        )
        for example in examples
        if example.source
    }
    metadata: dict[str, Any] = {"task_count": len(examples)}
    if len(sources) == 1:
        benchmark, version, split = next(iter(sources))
        metadata.update(
            {
                "task_set": benchmark or None,
                "task_set_version": version or None,
                "task_set_split": split or None,
            }
        )
    return {key: value for key, value in metadata.items() if value is not None}


def _grade_set_metrics(grade_set: SemanticGradeSet) -> dict[str, Any]:
    calls = [
        attempt.get("call") or {}
        for record in grade_set.records.values()
        for attempt in record.attempts
    ]
    return {
        "grading_unique_responses": len(grade_set.records),
        "grading_successes": sum(
            record.status == "success" for record in grade_set.records.values()
        ),
        "grading_failures": sum(
            record.status != "success" for record in grade_set.records.values()
        ),
        "grading_model_calls": len(calls),
        "grading_total_tokens": sum(
            int(((call.get("usage") or {}).get("total_tokens") or 0))
            for call in calls
        ),
        "grading_cost_usd": sum(
            float(call.get("cost_usd") or 0.0) for call in calls
        ),
    }


def _count_outcome(rows: list[dict[str, Any]], outcome: str) -> int:
    return sum(row["outcome"] == outcome for row in rows)


def _empty_metrics() -> dict[str, int | float]:
    return {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "wall_time_ms": 0.0,
    }


def _stage_score(
    stage: dict[str, Any],
    task_id: str,
    expected: str,
    answer_spec: AnswerSpec,
    semantic_grades: SemanticGradeSet | None,
) -> dict[str, Any]:
    output = stage.get("output") or {}
    answer = str(output.get("answer") or "")
    response = str(output.get("raw_response") or answer)
    return _score(
        task_id,
        response,
        answer,
        expected,
        answer_spec,
        semantic_grades,
    )


def _score(
    task_id: str,
    response: str,
    answer: str,
    expected: str,
    answer_spec: AnswerSpec,
    semantic_grades: SemanticGradeSet | None,
) -> dict[str, Any]:
    if semantic_grades is None:
        return {
            "correct": (
                normalize_answer(answer, answer_spec)
                == normalize_answer(expected, answer_spec)
            ),
            "status": "exact",
            "extracted_final_answer": answer,
            "reasoning": "",
            "confidence": None,
        }
    record = semantic_grades.lookup(task_id, response)
    if record is None:
        return {
            "correct": None,
            "status": "missing",
            "extracted_final_answer": "",
            "reasoning": "",
            "confidence": None,
        }
    if record.status != "success" or record.grade is None:
        return {
            "correct": None,
            "status": "failed",
            "extracted_final_answer": "",
            "reasoning": record.error or "",
            "confidence": None,
        }
    return {
        "correct": record.grade["correct"] == "yes",
        "status": "graded",
        "extracted_final_answer": record.grade["extracted_final_answer"],
        "reasoning": record.grade["reasoning"],
        "confidence": record.grade["confidence"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
