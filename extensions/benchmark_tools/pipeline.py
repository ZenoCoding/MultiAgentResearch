from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.config import load_experiment_config
from extensions.benchmark_tools.grading import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_GRADER_REASONING_EFFORT,
    grade_experiment,
)
from extensions.benchmark_tools.preflight import preflight_experiment
from extensions.benchmark_tools.runner import run_benchmark
from extensions.benchmark_tools.site import build_site


class PreflightFailedError(RuntimeError):
    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__("provider preflight failed")
        self.summary = summary


async def run_experiment_pipeline(
    *,
    config_path: Path | str,
    model: str,
    results_dir: Path | str = "results",
    reports_dir: Path | str = "reports",
    grader_model: str = DEFAULT_GRADER_MODEL,
    grader_reasoning_effort: str | None = DEFAULT_GRADER_REASONING_EFFORT,
    concurrency: int = 1,
    max_in_flight_requests: int = 1,
    grading_concurrency: int = 8,
    grading_max_in_flight_requests: int = 8,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    max_attempts: int = 3,
    request_max_attempts: int = 3,
    request_retry_base_delay_seconds: float = 1.0,
    request_retry_max_delay_seconds: float = 60.0,
    request_retry_jitter_ratio: float = 0.2,
    excluded_workflows: set[str] | None = None,
    excluded_reasoning_efforts: set[str] | None = None,
    skip_preflight: bool = False,
    skip_grading: bool = False,
    html: bool = False,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
    stage_handler: Callable[[str, dict[str, Any]], None] | None = None,
    drain_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    experiment_root = Path(results_dir) / config.experiment_id
    manifest_path = experiment_root / "experiment-manifest.json"

    preflight_summary: dict[str, Any]
    if skip_preflight or manifest_path.exists():
        preflight_summary = {
            "status": "skipped",
            "reason": (
                "disabled" if skip_preflight else "existing experiment manifest"
            ),
        }
        _stage(stage_handler, "preflight_skipped", preflight_summary)
    else:
        _stage(stage_handler, "preflight_started", {})
        preflight_summary = await preflight_experiment(
            config_path=config_path,
            primary_model=model,
            grader_model=grader_model,
            grader_reasoning_effort=grader_reasoning_effort,
            max_attempts=min(max_attempts, 2),
            max_in_flight_requests=1,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
        )
        _stage(stage_handler, "preflight_finished", preflight_summary)
        if preflight_summary["status"] != "passed":
            raise PreflightFailedError(preflight_summary)

    _stage(stage_handler, "run_started", {})
    run_summary = await run_benchmark(
        tasks_path=config.tasks_path,
        model=model,
        judge_model=config.aggregation_judge_model,
        experiment_id=config.experiment_id,
        output_dir=results_dir,
        conditions=list(config.conditions),
        concurrency=concurrency,
        max_in_flight_requests=max_in_flight_requests,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        repetitions=config.repetitions,
        max_attempts=max_attempts,
        request_max_attempts=request_max_attempts,
        request_retry_base_delay_seconds=request_retry_base_delay_seconds,
        request_retry_max_delay_seconds=request_retry_max_delay_seconds,
        request_retry_jitter_ratio=request_retry_jitter_ratio,
        excluded_workflows=excluded_workflows,
        excluded_reasoning_efforts=excluded_reasoning_efforts,
        experiment_metadata=config.metadata,
        event_handler=event_handler,
        emit_json_events=False,
        drain_event=drain_event,
    )
    _stage(stage_handler, "run_finished", run_summary)
    if drain_event is not None and drain_event.is_set():
        return {
            "preflight": preflight_summary,
            "run": run_summary,
            "grading": None,
            "summary": None,
            "site_path": None,
            "drained": True,
        }

    grade_summary: dict[str, Any] | None = None
    grade_set_id: str | None = None
    if not skip_grading:
        _stage(stage_handler, "grading_started", {})
        grade_summary = await grade_experiment(
            tasks_path=config.tasks_path,
            results_dir=results_dir,
            experiment_id=config.experiment_id,
            grader_model=grader_model,
            scope="final",
            concurrency=grading_concurrency,
            max_in_flight_requests=grading_max_in_flight_requests,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            max_attempts=max_attempts,
            reasoning_effort=grader_reasoning_effort,
        )
        grade_set_id = str(grade_summary["grade_set_id"])
        _stage(stage_handler, "grading_finished", grade_summary)

    analysis_dir = Path(reports_dir) / "analysis" / config.experiment_id
    summary = analyze_experiment(
        tasks_path=config.tasks_path,
        results_dir=results_dir,
        experiment_id=config.experiment_id,
        output_dir=analysis_dir if html else None,
        grade_set_id=grade_set_id,
        require_semantic_grades=not skip_grading,
    )

    site_path = None
    if html:
        site_path = build_site(
            analysis_dir=analysis_dir,
            output_dir=Path(reports_dir) / "site" / config.experiment_id,
        )

    return {
        "preflight": preflight_summary,
        "run": run_summary,
        "grading": grade_summary,
        "summary": summary,
        "site_path": str(site_path) if site_path else None,
        "drained": False,
    }


def _stage(
    handler: Callable[[str, dict[str, Any]], None] | None,
    name: str,
    data: dict[str, Any],
) -> None:
    if handler:
        handler(name, data)
