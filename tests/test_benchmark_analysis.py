from __future__ import annotations

from typing import Any

import pytest

from multi_agent_research.models import AnswerChoice, AnswerSpec

from extensions.benchmark_tools.analysis import analyze_attempts


SPEC = AnswerSpec(
    type="multiple_choice",
    choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
)


def _result(
    *,
    run_id: str,
    condition: str,
    task_id: str = "task-1",
    answer: str = "A",
    status: str = "success",
    contract_valid: bool = True,
    ended_at: str = "2026-06-12T01:00:00Z",
    repetition: int | None = None,
    stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_id,
        "workflow": {
            "name": "solo",
            "config": {"condition_id": condition},
        },
        "status": status,
        "final_answer": answer,
        "output": {
            "answer": answer,
            "contract_valid": contract_valid,
        },
        "ended_at": ended_at,
        "metrics": {
            "total_tokens": 10,
            "input_tokens": 7,
            "output_tokens": 3,
            "reasoning_tokens": 0,
            "cost_usd": 0.01,
            "wall_time_ms": 100,
        },
        "stage_answers": stages or [],
    }
    if repetition is not None:
        result["repetition"] = repetition
    if status != "success":
        result["output"] = None
        result["final_answer"] = ""
        result["error"] = {"type": "ProviderError", "message": "failed"}
    return result


def _analyze(
    results: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...] = ("task-1",),
):
    return analyze_attempts(
        results,
        expected_answers={task: "A" for task in tasks},
        answer_specs={task: SPEC for task in tasks},
    )


def test_duplicate_reruns_count_as_one_logical_job() -> None:
    summary, rows, _ = _analyze(
        [
            _result(run_id="old", condition="solo"),
            _result(
                run_id="new",
                condition="solo",
                ended_at="2026-06-12T02:00:00Z",
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["run_id"] == "new"
    assert rows[0]["attempt_count"] == 2
    assert summary["conditions"][0]["expected_jobs"] == 1
    assert summary["conditions"][0]["attempts"] == 2


def test_failed_then_successful_selects_success_and_its_stages() -> None:
    failed_stages = [
        {
            "sequence": 0,
            "step": "answer",
            "kind": "candidate",
            "agent_id": "agent",
            "output": {"answer": "B", "contract_valid": True},
        }
    ]
    success_stages = [
        {
            "sequence": 0,
            "step": "answer",
            "kind": "candidate",
            "agent_id": "agent",
            "output": {"answer": "A", "contract_valid": True},
        }
    ]
    summary, rows, stage_rows = _analyze(
        [
            _result(
                run_id="failed",
                condition="solo",
                status="failed",
                ended_at="2026-06-12T03:00:00Z",
                stages=failed_stages,
            ),
            _result(
                run_id="success",
                condition="solo",
                ended_at="2026-06-12T02:00:00Z",
                stages=success_stages,
            ),
        ]
    )

    assert rows[0]["run_id"] == "success"
    assert rows[0]["outcome"] == "completed_valid"
    assert rows[0]["attempt_count"] == 2
    assert [row["run_id"] for row in stage_rows] == ["success"]
    assert summary["conditions"][0]["planned_job_accuracy"] == 1.0


def test_all_failed_condition_has_health_metrics_and_no_valid_accuracy() -> None:
    summary, rows, _ = _analyze(
        [_result(run_id="failed", condition="solo", status="failed")]
    )
    condition = summary["conditions"][0]

    assert rows[0]["outcome"] == "provider_execution_failure"
    assert condition["provider_execution_failures"] == 1
    assert condition["completed_answer_jobs"] == 0
    assert condition["planned_job_accuracy"] == 0.0
    assert condition["valid_completed_accuracy"] is None


def test_missing_jobs_are_materialized_from_planned_task_grid() -> None:
    summary, rows, _ = _analyze(
        [_result(run_id="one", condition="solo", task_id="task-1")],
        tasks=("task-1", "task-2"),
    )
    condition = summary["conditions"][0]

    assert len(rows) == 2
    assert {row["outcome"] for row in rows} == {"completed_valid", "missing"}
    assert condition["expected_jobs"] == 2
    assert condition["missing_jobs"] == 1
    assert condition["planned_job_accuracy"] == 0.5
    assert condition["valid_completed_accuracy"] == 1.0


def test_paired_comparison_uses_only_matched_completed_repetitions() -> None:
    summary, _, _ = _analyze(
        [
            _result(
                run_id="a-r0",
                condition="a",
                repetition=0,
                answer="A",
            ),
            _result(
                run_id="a-r1",
                condition="a",
                repetition=1,
                answer="A",
            ),
            _result(
                run_id="b-r0",
                condition="b",
                repetition=0,
                answer="B",
            ),
            _result(
                run_id="b-r1",
                condition="b",
                repetition=1,
                status="failed",
            ),
        ]
    )
    paired = summary["paired_comparisons"][0]

    assert paired["matched_completed_pairs"] == 1
    assert paired["accuracy_a"] == 1.0
    assert paired["accuracy_b"] == 0.0
    assert paired["accuracy_delta_b_minus_a"] == -1.0


def test_summary_splits_metrics_by_answer_type() -> None:
    summary, _, _ = analyze_attempts(
        [
            _result(run_id="mcq", condition="solo", task_id="mcq"),
            _result(
                run_id="short",
                condition="solo",
                task_id="short",
                answer="wrong",
            ),
        ],
        expected_answers={"mcq": "A", "short": "expected"},
        answer_specs={
            "mcq": SPEC,
            "short": AnswerSpec(type="short_answer"),
        },
        answer_types={"mcq": "multiple_choice", "short": "short_answer"},
    )

    rows = {
        row["answer_type"]: row for row in summary["answer_type_breakdown"]
    }
    assert rows["multiple_choice"]["planned_job_accuracy"] == 1.0
    assert rows["short_answer"]["planned_job_accuracy"] == 0.0


def test_contract_invalid_is_not_answer_bearing_completion() -> None:
    summary, rows, _ = _analyze(
        [
            _result(
                run_id="invalid",
                condition="solo",
                contract_valid=False,
            )
        ]
    )

    assert rows[0]["outcome"] == "contract_invalid"
    assert summary["conditions"][0]["contract_invalid_outputs"] == 1
    assert summary["conditions"][0]["completed_answer_jobs"] == 0


def test_inconclusive_is_not_reported_as_provider_failure() -> None:
    summary, rows, _ = _analyze(
        [_result(run_id="tie", condition="solo", status="inconclusive")]
    )

    assert rows[0]["outcome"] == "inconclusive"
    assert rows[0]["error_reason"] == "inconclusive"
    assert summary["conditions"][0]["inconclusive_jobs"] == 1
    assert summary["conditions"][0]["provider_execution_failures"] == 0


def test_explicit_expected_jobs_support_stable_job_ids() -> None:
    result = _result(run_id="attempt", condition="solo")
    result["job_id"] = "job-7"
    summary, rows, _ = analyze_attempts(
        [result],
        expected_answers={"task-1": "A"},
        answer_specs={"task-1": SPEC},
        expected_jobs=[
            {
                "job_id": "job-7",
                "condition": "solo",
                "task_id": "task-1",
                "repetition": 3,
            }
        ],
    )

    assert rows[0]["job_id"] == "job-7"
    assert rows[0]["repetition"] == 3
    assert summary["conditions"][0]["expected_jobs"] == 1


def test_stable_job_ids_without_ledger_do_not_create_legacy_duplicates() -> None:
    result = _result(run_id="attempt", condition="solo", repetition=2)
    result["job_id"] = "job-7"

    summary, rows, _ = _analyze([result])

    assert len(rows) == 1
    assert rows[0]["job_id"] == "job-7"
    assert rows[0]["repetition"] == 2
    assert summary["conditions"][0]["expected_jobs"] == 1


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("", "inconclusive"), ("A", "completed_valid")],
)
def test_success_outcome_requires_an_answer(answer: str, expected: str) -> None:
    _, rows, _ = _analyze(
        [_result(run_id="run", condition="solo", answer=answer)]
    )
    assert rows[0]["outcome"] == expected
