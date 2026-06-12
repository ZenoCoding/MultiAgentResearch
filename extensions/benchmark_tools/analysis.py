from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from multi_agent_research.aggregation import normalize_answer
from multi_agent_research.models import AnswerChoice, AnswerSpec

from extensions.benchmark_tools.connector import load_jsonl


def analyze_experiment(
    *,
    tasks_path: Path | str,
    results_dir: Path | str,
    experiment_id: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    examples = load_jsonl(tasks_path)
    expected = {example.id: example.answer for example in examples}
    answer_specs = {
        example.id: AnswerSpec(
            type=example.answer_type,  # type: ignore[arg-type]
            choices=[AnswerChoice(label=choice.label, text=choice.text) for choice in example.choices],
            include_confidence=True,
        )
        for example in examples
    }
    categories = {example.id: example.category or "unknown" for example in examples}
    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    experiment_root = Path(results_dir) / experiment_id
    for result_path in sorted(experiment_root.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = result["task_id"]
        if task_id not in expected:
            continue
        answer_spec = answer_specs[task_id]
        final_answer = result.get("final_answer") or ""
        correct = _correct(final_answer, expected[task_id], answer_spec)
        condition = _condition_id(result)
        metrics = result.get("metrics") or {}
        row = {
            "experiment_id": experiment_id,
            "condition": condition,
            "workflow": result["workflow"]["name"],
            "run_id": result["run_id"],
            "task_id": task_id,
            "category": categories[task_id],
            "status": result["status"],
            "final_answer": final_answer,
            "expected_answer": expected[task_id],
            "correct": int(correct),
            "contract_valid": int(bool((result.get("output") or {}).get("contract_valid"))),
            "error_reason": error_reason(result, expected[task_id], answer_spec),
            "model_calls": metrics.get("model_calls", 0),
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "reasoning_tokens": metrics.get("reasoning_tokens", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "cost_usd": metrics.get("cost_usd", 0.0),
            "wall_time_ms": metrics.get("wall_time_ms", 0.0),
        }
        run_rows.append(row)
        for stage in result.get("stage_answers") or []:
            output = stage.get("output") or {}
            stage_answer = output.get("answer") or ""
            stage_rows.append(
                {
                    "condition": condition,
                    "workflow": result["workflow"]["name"],
                    "run_id": result["run_id"],
                    "task_id": task_id,
                    "sequence": stage.get("sequence"),
                    "step": stage.get("step"),
                    "kind": stage.get("kind"),
                    "agent_id": stage.get("agent_id"),
                    "answer": stage_answer,
                    "correct": int(_correct(stage_answer, expected[task_id], answer_spec)),
                    "contract_valid": int(bool(output.get("contract_valid"))),
                    "confidence": output.get("confidence"),
                }
            )

    summary = summarize(run_rows, stage_rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "stage_answers.csv", stage_rows)
    _write_csv(output / "summary.csv", summary["conditions"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "runs.json").write_text(json.dumps(run_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "stage_answers.json").write_text(json.dumps(stage_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def summarize(run_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_condition[row["condition"]].append(row)

    condition_summary: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        total = len(rows)
        correct = sum(row["correct"] for row in rows)
        cost = sum(float(row["cost_usd"]) for row in rows)
        tokens = sum(int(row["total_tokens"]) for row in rows)
        condition_summary.append(
            {
                "condition": condition,
                "workflow": rows[0]["workflow"],
                "runs": total,
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "total_tokens": tokens,
                "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
                "cost_usd": cost,
                "accuracy_per_dollar": correct / cost if cost else None,
                "accuracy_per_million_tokens": correct / (tokens / 1_000_000) if tokens else None,
                "contract_valid_rate": sum(row["contract_valid"] for row in rows) / total if total else 0.0,
                "avg_wall_time_ms": sum(float(row["wall_time_ms"]) for row in rows) / total if total else 0.0,
            }
        )

    error_counts = Counter(row["error_reason"] for row in run_rows if not row["correct"])
    return {
        "schema_version": "1",
        "conditions": condition_summary,
        "error_reasons": dict(sorted(error_counts.items())),
        "revision_transitions": revision_transitions(stage_rows),
        "question_heatmap": question_heatmap(run_rows),
    }


def revision_transitions(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        if row["kind"] != "candidate":
            continue
        key = (row["condition"], row["run_id"], row.get("agent_id") or row["step"])
        grouped[key].append(row)
    counts: Counter[tuple[str, str]] = Counter()
    for (condition, _, _), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["sequence"]))
        if len(ordered) < 2:
            continue
        first = bool(ordered[0]["correct"])
        last = bool(ordered[-1]["correct"])
        label = ("right" if first else "wrong") + "_to_" + ("right" if last else "wrong")
        counts[(condition, label)] += 1
    return [
        {"condition": condition, "transition": transition, "count": count}
        for (condition, transition), count in sorted(counts.items())
    ]


def question_heatmap(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "condition": row["condition"],
            "correct": row["correct"],
            "category": row["category"],
        }
        for row in sorted(run_rows, key=lambda row: (row["task_id"], row["condition"]))
    ]


def error_reason(result: dict[str, Any], expected: str, answer_spec: AnswerSpec) -> str:
    output = result.get("output") or {}
    if not output.get("contract_valid", False):
        return "format_error"
    final_answer = output.get("answer") or ""
    if _correct(final_answer, expected, answer_spec):
        return "correct"
    stages = result.get("stage_answers") or []
    candidate_stages = [stage for stage in stages if stage.get("kind") == "candidate"]
    aggregate_stages = [stage for stage in stages if stage.get("kind") == "aggregate"]
    candidate_correct = [
        stage for stage in candidate_stages
        if _correct(((stage.get("output") or {}).get("answer") or ""), expected, answer_spec)
    ]
    if candidate_correct and aggregate_stages:
        return "aggregation_error"
    if _has_right_to_wrong(candidate_stages, expected, answer_spec):
        return "bad_revision"
    if candidate_correct:
        return "peer_persuasion_failure"
    if result["workflow"]["name"] in {"debate", "adversarial_debate"}:
        return "shared_reasoning_error"
    return "model_error"


def _has_right_to_wrong(stages: list[dict[str, Any]], expected: str, answer_spec: AnswerSpec) -> bool:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stage in stages:
        by_agent[stage.get("agent_id") or stage.get("step")].append(stage)
    for rows in by_agent.values():
        ordered = sorted(rows, key=lambda row: int(row.get("sequence") or 0))
        values = [_correct(((row.get("output") or {}).get("answer") or ""), expected, answer_spec) for row in ordered]
        if any(values[index] and not values[index + 1] for index in range(len(values) - 1)):
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
        prefix = "adversarial-debate" if workflow == "adversarial_debate" else "debate"
        return f"{prefix}-{len(config.get('agents') or [])}-{config.get('peer_view', 'full_response')}"
    return workflow


def _correct(answer: str, expected: str, answer_spec: AnswerSpec) -> bool:
    return normalize_answer(answer, answer_spec) == normalize_answer(expected, answer_spec)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
