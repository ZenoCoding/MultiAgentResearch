from __future__ import annotations

import json

import pytest

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.grading import grade_experiment, load_grade_set
from extensions.benchmark_tools.grading import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_GRADER_REASONING_EFFORT,
)
from extensions.benchmark_tools.runner import run_benchmark
from extensions.benchmark_tools.schema import Condition
from tests.fakes import FakeLLMClient


def _write_short_answer_task(path) -> None:  # type: ignore[no-untyped-def]
    path.write_text(
        json.dumps(
            {
                "id": "short-1",
                "prompt": "Give one half as a decimal.",
                "answer": "1/2",
                "answer_type": "short_answer",
                "category": "Math",
                "source": {
                    "benchmark": "hle",
                    "version": "cais/hle",
                    "split": "test",
                    "original_id": "short-1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _judge_json(*, correct: str = "yes") -> str:
    return json.dumps(
        {
            "extracted_final_answer": "0.5",
            "reasoning": "0.5 and 1/2 are mathematically equivalent.",
            "correct": correct,
            "confidence": 100,
            "strict": True,
        }
    )


@pytest.mark.asyncio
async def test_semantic_hle_grading_is_resumable_and_drives_analysis(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_short_answer_task(tasks)
    await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="semantic",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient(
            ["Reasoning\n<final_answer>0.5</final_answer>\n"]
        ),
        max_attempts=1,
    )

    with pytest.raises(ValueError, match="semantic HLE grades are required"):
        analyze_experiment(
            tasks_path=tasks,
            results_dir=results,
            experiment_id="semantic",
            output_dir=tmp_path / "ungraded-analysis",
        )

    first = await grade_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="semantic",
        grader_model="fake/grader",
        scope="all",
        llm=FakeLLMClient([_judge_json()]),
        max_attempts=1,
    )
    second = await grade_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="semantic",
        grader_model="fake/grader",
        scope="all",
        llm=FakeLLMClient([]),
        max_attempts=1,
    )

    assert first["canonical_responses"] == 2
    assert first["unique_responses"] == 1
    assert first["successful_grades"] == 1
    assert second["scheduled_grades"] == 0
    grade_set = load_grade_set(
        results_dir=results,
        experiment_id="semantic",
    )
    assert grade_set is not None
    record = next(iter(grade_set.records.values()))
    assert record.grade is not None
    assert record.grade["correct"] == "yes"

    summary = analyze_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="semantic",
        output_dir=tmp_path / "analysis",
    )
    condition = summary["conditions"][0]
    assert condition["graded_jobs"] == 1
    assert condition["planned_job_accuracy"] == 1.0
    assert condition["graded_accuracy"] == 1.0
    runs = json.loads(
        (tmp_path / "analysis" / "runs.json").read_text(encoding="utf-8")
    )
    stages = json.loads(
        (tmp_path / "analysis" / "stage_answers.json").read_text(
            encoding="utf-8"
        )
    )
    assert runs[0]["grading_status"] == "graded"
    assert runs[0]["grader_extracted_answer"] == "0.5"
    assert stages[0]["grading_status"] == "graded"
    assert summary["metadata"]["grading_model_calls"] == 1


@pytest.mark.asyncio
async def test_invalid_grader_json_retries_as_a_new_attempt(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_short_answer_task(tasks)
    await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="retry-grader",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient(["<final_answer>0.5</final_answer>"]),
        max_attempts=1,
    )

    summary = await grade_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="retry-grader",
        grader_model="fake/grader",
        llm=FakeLLMClient(["not json", _judge_json()]),
        max_attempts=2,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        retry_jitter_ratio=0,
    )

    assert summary["successful_grades"] == 1
    grade_set = load_grade_set(
        results_dir=results,
        experiment_id="retry-grader",
    )
    assert grade_set is not None
    record = next(iter(grade_set.records.values()))
    assert [attempt["status"] for attempt in record.attempts] == [
        "failed",
        "success",
    ]


@pytest.mark.asyncio
async def test_default_grader_is_pinned_low_effort_with_strict_schema(
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    results = tmp_path / "results"
    _write_short_answer_task(tasks)
    await run_benchmark(
        tasks_path=tasks,
        model="fake/model",
        experiment_id="default-grader",
        output_dir=results,
        conditions=[Condition(id="solo", workflow="solo")],
        llm=FakeLLMClient(["<final_answer>0.5</final_answer>"]),
        max_attempts=1,
    )
    grader = FakeLLMClient([_judge_json()])

    summary = await grade_experiment(
        tasks_path=tasks,
        results_dir=results,
        experiment_id="default-grader",
        llm=grader,
        max_attempts=1,
    )

    grade_set = load_grade_set(
        results_dir=results,
        experiment_id="default-grader",
        grade_set_id=summary["grade_set_id"],
    )
    assert grade_set is not None
    assert grade_set.manifest.grader_model == DEFAULT_GRADER_MODEL
    assert (
        grade_set.manifest.reasoning_effort
        == DEFAULT_GRADER_REASONING_EFFORT
    )
    record = next(iter(grade_set.records.values()))
    call = record.attempts[0]["call"]
    assert call["requested_model"] == DEFAULT_GRADER_MODEL
    assert call["request_parameters"]["reasoning_effort"] == "low"
    assert "max_completion_tokens" not in call["request_parameters"]
    response_format = call["request_parameters"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    prompt = grader.requests[0][0].content
    assert isinstance(prompt, str)
    assert "Your judgement must be in the format and criteria specified below" in prompt
