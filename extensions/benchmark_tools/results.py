from __future__ import annotations

from typing import Any


def render_results(summary: dict[str, Any]) -> str:
    conditions = summary["conditions"]
    metadata = summary.get("metadata") or {}
    expected = sum(int(row["expected_jobs"]) for row in conditions)
    completed = sum(int(row["completed_answer_jobs"]) for row in conditions)
    graded = sum(int(row["graded_jobs"]) for row in conditions)
    correct = sum(int(row["correct_valid_completed"]) for row in conditions)
    workflow_cost = sum(float(row["cost_usd"]) for row in conditions)
    grading_cost = float(metadata.get("grading_cost_usd") or 0.0)
    total_tokens = sum(int(row["total_tokens"]) for row in conditions)

    lines = [
        f"Experiment: {metadata.get('experiment_id', 'unknown')}",
        (
            f"Completed: {completed}/{expected} ({_percent(completed, expected)})"
            f"  Graded: {graded}/{expected}"
            f"  Accuracy: {_fraction(correct, graded)}"
        ),
        (
            f"Workflow spend: ${workflow_cost:.4f}"
            f"  Grading spend: ${grading_cost:.4f}"
            f"  Total: ${workflow_cost + grading_cost:.4f}"
            f"  Tokens: {total_tokens:,}"
        ),
        "",
    ]

    headers = ("Condition", "Done", "Accuracy", "Cost", "Tokens", "Avg time")
    rows = [
        (
            str(row["condition"]),
            f"{row['completed_answer_jobs']}/{row['expected_jobs']}",
            _fraction(
                int(row["correct_valid_completed"]),
                int(row["graded_jobs"]),
            ),
            f"${float(row['cost_usd']):.4f}",
            f"{int(row['total_tokens']):,}",
            f"{float(row['avg_wall_time_ms']) / 1000:.1f}s",
        )
        for row in conditions
    ]
    lines.extend(_table(headers, rows))

    if metadata.get("grade_set_id"):
        lines.extend(
            [
                "",
                (
                    f"Grades: {metadata['grade_set_id']} using "
                    f"{metadata.get('grader_model', 'unknown')}"
                ),
            ]
        )
    return "\n".join(lines)


def _fraction(numerator: int, denominator: int) -> str:
    if not denominator:
        return "ungraded"
    return f"{numerator}/{denominator} ({_percent(numerator, denominator)})"


def _percent(numerator: int, denominator: int) -> str:
    if not denominator:
        return "--"
    return f"{100 * numerator / denominator:.1f}%"


def _table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> list[str]:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()

    return [
        format_row(headers),
        format_row(tuple("-" * width for width in widths)),
        *(format_row(row) for row in rows),
    ]
