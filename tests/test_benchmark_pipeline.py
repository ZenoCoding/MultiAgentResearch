from __future__ import annotations

import asyncio
import json

import pytest

from extensions.benchmark_tools import pipeline


def _write_config(tmp_path) -> object:  # type: ignore[no-untyped-def]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "task-1",
                "prompt": "Question?",
                "answer": "A",
                "answer_type": "multiple_choice",
                "choices": [
                    {"label": "A", "text": "Alpha"},
                    {"label": "B", "text": "Beta"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "pipeline-test",
                "tasks": str(tasks),
                "families": {"solo": {"efforts": ["low"]}},
            }
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.asyncio
async def test_pipeline_runs_all_stages_and_returns_summary(
    tmp_path,
    monkeypatch,
) -> None:
    config = _write_config(tmp_path)
    stages = []
    run_kwargs = {}

    async def fake_preflight(**kwargs):  # type: ignore[no-untyped-def]
        return {"status": "passed", "call_count": 2}

    async def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        run_kwargs.update(kwargs)
        return {"scheduled_jobs": 1}

    async def fake_grade(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "grade_set_id": "grades-1",
            "cached_grades": 0,
            "scheduled_grades": 1,
        }

    def fake_analyze(**kwargs):  # type: ignore[no-untyped-def]
        return {"metadata": {"experiment_id": "pipeline-test"}, "conditions": []}

    monkeypatch.setattr(pipeline, "preflight_experiment", fake_preflight)
    monkeypatch.setattr(pipeline, "run_benchmark", fake_run)
    monkeypatch.setattr(pipeline, "grade_experiment", fake_grade)
    monkeypatch.setattr(pipeline, "analyze_experiment", fake_analyze)

    result = await pipeline.run_experiment_pipeline(
        config_path=config,
        model="fake/model",
        results_dir=tmp_path / "results",
        excluded_workflows={"debate"},
        stage_handler=lambda name, data: stages.append(name),
    )

    assert result["run"]["scheduled_jobs"] == 1
    assert run_kwargs["excluded_workflows"] == {"debate"}
    assert stages == [
        "preflight_started",
        "preflight_finished",
        "run_started",
        "run_finished",
        "grading_started",
        "grading_finished",
    ]
    assert result["grading"]["grade_set_id"] == "grades-1"
    assert result["summary"]["metadata"]["experiment_id"] == "pipeline-test"


@pytest.mark.asyncio
async def test_pipeline_skips_preflight_for_existing_experiment(
    tmp_path,
    monkeypatch,
) -> None:
    config = _write_config(tmp_path)
    stages = []
    experiment_root = tmp_path / "results" / "pipeline-test"
    experiment_root.mkdir(parents=True)
    (experiment_root / "experiment-manifest.json").write_text("{}", encoding="utf-8")

    async def fail_preflight(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("preflight should be skipped")

    monkeypatch.setattr(pipeline, "preflight_experiment", fail_preflight)
    monkeypatch.setattr(
        pipeline,
        "run_benchmark",
        lambda **kwargs: _async_value({"scheduled_jobs": 0}),
    )
    monkeypatch.setattr(
        pipeline,
        "grade_experiment",
        lambda **kwargs: _async_value(
            {
                "grade_set_id": "grades-1",
                "cached_grades": 1,
                "scheduled_grades": 0,
            }
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_experiment",
        lambda **kwargs: {"metadata": {}, "conditions": []},
    )

    result = await pipeline.run_experiment_pipeline(
        config_path=config,
        model="fake/model",
        results_dir=tmp_path / "results",
        stage_handler=lambda name, data: stages.append(name),
    )

    assert result["preflight"]["status"] == "skipped"
    assert stages[0] == "preflight_skipped"


@pytest.mark.asyncio
async def test_pipeline_raises_structured_preflight_failure(
    tmp_path,
    monkeypatch,
) -> None:
    config = _write_config(tmp_path)
    failed = {
        "status": "failed",
        "checks": [
            {
                "check_id": "semantic-vote",
                "status": "failed",
                "error": "missing final answer",
            }
        ],
    }

    monkeypatch.setattr(
        pipeline,
        "preflight_experiment",
        lambda **kwargs: _async_value(failed),
    )

    with pytest.raises(pipeline.PreflightFailedError) as exc_info:
        await pipeline.run_experiment_pipeline(
            config_path=config,
            model="fake/model",
            results_dir=tmp_path / "results",
        )

    assert exc_info.value.summary is failed


@pytest.mark.asyncio
async def test_pipeline_stops_after_a_drained_run(tmp_path, monkeypatch) -> None:
    config = _write_config(tmp_path)
    drain_event = asyncio.Event()
    drain_event.set()

    monkeypatch.setattr(
        pipeline,
        "preflight_experiment",
        lambda **kwargs: _async_value({"status": "passed"}),
    )
    monkeypatch.setattr(
        pipeline,
        "run_benchmark",
        lambda **kwargs: _async_value({"started_jobs": 1, "drained_jobs": 2}),
    )
    monkeypatch.setattr(
        pipeline,
        "grade_experiment",
        lambda **kwargs: pytest.fail("grading should not run after drain"),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_experiment",
        lambda **kwargs: pytest.fail("analysis should not run after drain"),
    )

    result = await pipeline.run_experiment_pipeline(
        config_path=config,
        model="fake/model",
        results_dir=tmp_path / "results",
        drain_event=drain_event,
    )

    assert result["drained"] is True
    assert result["summary"] is None


async def _async_value(value):  # type: ignore[no-untyped-def]
    return value
