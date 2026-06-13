from __future__ import annotations

from extensions.benchmark_tools.results import render_results


def test_render_results_prints_terminal_summary_and_condition_table() -> None:
    output = render_results(
        {
            "metadata": {
                "experiment_id": "smoke",
                "grade_set_id": "grades-1",
                "grader_model": "grader/model",
                "grading_cost_usd": 0.02,
            },
            "conditions": [
                {
                    "condition": "solo",
                    "expected_jobs": 5,
                    "completed_answer_jobs": 5,
                    "graded_jobs": 5,
                    "correct_valid_completed": 2,
                    "cost_usd": 0.01,
                    "total_tokens": 1000,
                    "avg_wall_time_ms": 2500,
                },
                {
                    "condition": "sample-3",
                    "expected_jobs": 5,
                    "completed_answer_jobs": 4,
                    "graded_jobs": 4,
                    "correct_valid_completed": 1,
                    "cost_usd": 0.03,
                    "total_tokens": 2000,
                    "avg_wall_time_ms": 5000,
                },
            ],
        }
    )

    assert "Experiment: smoke" in output
    assert "Completed: 9/10 (90.0%)" in output
    assert "Accuracy: 3/9 (33.3%)" in output
    assert "Total: $0.0600" in output
    assert "solo" in output
    assert "sample-3" in output
    assert "Grades: grades-1 using grader/model" in output
