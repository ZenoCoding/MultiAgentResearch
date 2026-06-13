from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.charts import write_charts
from extensions.benchmark_tools.config import load_experiment_config
from extensions.benchmark_tools.connector import load_jsonl, write_fixed_sample
from extensions.benchmark_tools.grading import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_GRADER_REASONING_EFFORT,
    grade_experiment,
)
from extensions.benchmark_tools.preflight import preflight_experiment
from extensions.benchmark_tools.pipeline import (
    PreflightFailedError,
    run_experiment_pipeline,
)
from extensions.benchmark_tools.progress import TerminalProgress
from extensions.benchmark_tools.results import render_results
from extensions.benchmark_tools.runner import load_conditions, run_benchmark
from extensions.benchmark_tools.site import build_site


def main() -> None:
    args = _parser().parse_args()
    if args.command == "sample":
        path = write_fixed_sample(
            input_path=args.input,
            output_dir=args.output_dir,
            sample_size=args.size,
            seed=args.seed,
            stratify_key=args.stratify_key,
            answer_type=args.answer_type,
            sampling_strategy=args.sampling_strategy,
        )
        print(path)
    elif args.command == "validate-tasks":
        examples = load_jsonl(args.tasks)
        if args.require_answer_type and any(
            example.answer_type != args.require_answer_type
            for example in examples
        ):
            raise ValueError(
                f"task set contains types other than {args.require_answer_type!r}"
            )
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.answer_type] = counts.get(example.answer_type, 0) + 1
        print(
            json.dumps(
                {
                    "tasks": len(examples),
                    "answer_types": dict(sorted(counts.items())),
                    "valid": True,
                },
                indent=2,
            )
        )
    elif args.command == "run":
        config = load_experiment_config(args.config) if args.config else None
        if config and args.conditions:
            raise ValueError("--config and --conditions cannot be used together")
        tasks = args.tasks or (config.tasks_path if config else None)
        experiment_id = args.experiment_id or (
            config.experiment_id if config else None
        )
        if not tasks or not experiment_id:
            raise ValueError(
                "run requires --tasks and --experiment-id, or a --config"
            )
        conditions = (
            list(config.conditions)
            if config
            else load_conditions(args.conditions)
        )
        judge_model = args.judge_model or (
            config.aggregation_judge_model if config else None
        )
        repetitions = args.repetitions or (
            config.repetitions if config else 1
        )
        progress = TerminalProgress() if args.progress and not args.dry_run else None
        try:
            summary = asyncio.run(
                run_benchmark(
                    tasks_path=tasks,
                    model=args.model,
                    judge_model=judge_model,
                    experiment_id=experiment_id,
                    output_dir=args.results_dir,
                    conditions=conditions,
                    system_prompt=args.system_prompt,
                    concurrency=args.concurrency,
                    max_in_flight_requests=args.max_in_flight_requests,
                    requests_per_minute=args.requests_per_minute,
                    tokens_per_minute=args.tokens_per_minute,
                    estimated_tokens_per_request=args.estimated_tokens_per_request,
                    repetitions=repetitions,
                    max_attempts=args.max_attempts,
                    retry_base_delay_seconds=args.retry_base_delay,
                    retry_max_delay_seconds=args.retry_max_delay,
                    retry_jitter_ratio=args.retry_jitter,
                    resume=args.resume,
                    dry_run=args.dry_run,
                    required_answer_type=args.require_answer_type,
                    experiment_metadata=config.metadata if config else None,
                    event_handler=progress.handle if progress else None,
                    emit_json_events=progress is None,
                )
            )
        finally:
            if progress:
                progress.close()
        if progress is None:
            print(json.dumps(summary, indent=2))
    elif args.command == "experiment":
        progress = TerminalProgress()

        def stage(name: str, data: dict[str, object]) -> None:
            if name == "preflight_started":
                print("Preflight: running...", flush=True)
            elif name == "preflight_skipped":
                print(f"Preflight: skipped ({data['reason']})", flush=True)
            elif name == "preflight_finished":
                print(
                    f"Preflight: {data['status']} "
                    f"({data.get('call_count', 0)} calls)",
                    flush=True,
                )
            elif name == "grading_started":
                print("Grading: running...", flush=True)
            elif name == "grading_finished":
                print(
                    f"Grading: {data.get('cached_grades', 0)} cached, "
                    f"{data.get('scheduled_grades', 0)} newly scheduled",
                    flush=True,
                )

        try:
            try:
                pipeline = asyncio.run(
                    run_experiment_pipeline(
                        config_path=args.config,
                        model=args.model,
                        results_dir=args.results_dir,
                        reports_dir=args.reports_dir,
                        grader_model=args.grader_model,
                        grader_reasoning_effort=args.grader_reasoning_effort,
                        concurrency=args.concurrency,
                        max_in_flight_requests=args.max_in_flight_requests,
                        grading_concurrency=args.grading_concurrency,
                        grading_max_in_flight_requests=(
                            args.grading_max_in_flight_requests
                        ),
                        requests_per_minute=args.requests_per_minute,
                        tokens_per_minute=args.tokens_per_minute,
                        max_attempts=args.max_attempts,
                        skip_preflight=args.skip_preflight,
                        skip_grading=args.skip_grading,
                        html=args.html,
                        event_handler=progress.handle,
                        stage_handler=stage,
                    )
                )
            except PreflightFailedError as exc:
                print("\nFailed preflight checks:", flush=True)
                for check in exc.summary["checks"]:
                    if check["status"] == "failed":
                        print(
                            f"- {check['check_id']}: {check['error']}",
                            flush=True,
                        )
                raise SystemExit(1) from None
        finally:
            progress.close()
        print(render_results(pipeline["summary"]))
        if pipeline["site_path"]:
            print(f"\nHTML: {pipeline['site_path']}")
    elif args.command == "preflight":
        summary = asyncio.run(
            preflight_experiment(
                config_path=args.config,
                primary_model=args.model,
                grader_model=args.grader_model,
                grader_reasoning_effort=args.grader_reasoning_effort,
                dry_run=args.dry_run,
                max_attempts=args.max_attempts,
                retry_base_delay_seconds=args.retry_base_delay,
                retry_max_delay_seconds=args.retry_max_delay,
                retry_jitter_ratio=args.retry_jitter,
                max_in_flight_requests=args.max_in_flight_requests,
                requests_per_minute=args.requests_per_minute,
                tokens_per_minute=args.tokens_per_minute,
                estimated_tokens_per_request=args.estimated_tokens_per_request,
            )
        )
        print(json.dumps(summary, indent=2))
        if summary["status"] == "failed":
            raise SystemExit(1)
    elif args.command == "analyze":
        summary = analyze_experiment(
            tasks_path=args.tasks,
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            output_dir=args.output_dir,
            grade_set_id=args.grade_set_id,
            require_semantic_grades=args.require_semantic_grades,
        )
        print(json.dumps({"conditions": len(summary["conditions"])}, indent=2))
    elif args.command == "results":
        config = load_experiment_config(args.config) if args.config else None
        tasks = args.tasks or (config.tasks_path if config else None)
        experiment_id = args.experiment_id or (
            config.experiment_id if config else None
        )
        if not tasks or not experiment_id:
            raise ValueError(
                "results requires --tasks and --experiment-id, or a --config"
            )
        summary = analyze_experiment(
            tasks_path=tasks,
            results_dir=args.results_dir,
            experiment_id=experiment_id,
            output_dir=None,
            grade_set_id=args.grade_set_id,
        )
        print(render_results(summary))
    elif args.command == "grade":
        summary = asyncio.run(
            grade_experiment(
                tasks_path=args.tasks,
                results_dir=args.results_dir,
                experiment_id=args.experiment_id,
                grader_model=args.grader_model,
                scope=args.scope,
                grade_set_id=args.grade_set_id,
                concurrency=args.concurrency,
                max_in_flight_requests=args.max_in_flight_requests,
                requests_per_minute=args.requests_per_minute,
                tokens_per_minute=args.tokens_per_minute,
                estimated_tokens_per_request=args.estimated_tokens_per_request,
                max_attempts=args.max_attempts,
                retry_base_delay_seconds=args.retry_base_delay,
                retry_max_delay_seconds=args.retry_max_delay,
                retry_jitter_ratio=args.retry_jitter,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                dry_run=args.dry_run,
            )
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "charts":
        paths = write_charts(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
        print(json.dumps([str(path) for path in paths], indent=2))
    elif args.command == "site":
        index = build_site(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
        print(index)
    elif args.command == "dashboard":
        from extensions.benchmark_tools.dashboard import serve
        serve(
            results_dir=args.results_dir,
            port=args.port,
            open_browser=args.open,
        )
    elif args.command == "update-report":
        analysis_dir = Path(args.output_dir) / "analysis" / args.experiment_id
        site_dir = Path(args.output_dir) / "site" / args.experiment_id
        grade_set_id = args.grade_set_id
        if args.grader_model:
            grade_summary = asyncio.run(
                grade_experiment(
                    tasks_path=args.tasks,
                    results_dir=args.results_dir,
                    experiment_id=args.experiment_id,
                    grader_model=args.grader_model,
                    scope=args.grading_scope,
                    grade_set_id=grade_set_id,
                    concurrency=args.grading_concurrency,
                    max_in_flight_requests=args.max_in_flight_requests,
                    requests_per_minute=args.requests_per_minute,
                    tokens_per_minute=args.tokens_per_minute,
                    max_attempts=args.max_attempts,
                    max_tokens=args.grader_max_tokens,
                    reasoning_effort=args.grader_reasoning_effort,
                )
            )
            grade_set_id = str(grade_summary["grade_set_id"])
        analyze_experiment(
            tasks_path=args.tasks,
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            output_dir=analysis_dir,
            grade_set_id=grade_set_id,
            require_semantic_grades=args.require_semantic_grades,
        )
        index = build_site(analysis_dir=analysis_dir, output_dir=site_dir)
        print(index)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Create a deterministic fixed benchmark sample.")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--size", type=int, default=30)
    sample.add_argument("--seed", type=int, default=20260611)
    sample.add_argument("--stratify-key", default="category")
    sample.add_argument(
        "--answer-type",
        default=None,
        help="Filter the sampled task set by canonical answer type.",
    )
    sample.add_argument(
        "--sampling-strategy",
        choices=["balanced", "proportional"],
        default="balanced",
    )

    validate = sub.add_parser(
        "validate-tasks",
        help="Validate and summarize a benchmark JSONL file without model calls.",
    )
    validate.add_argument("--tasks", required=True)
    validate.add_argument("--require-answer-type")

    run = sub.add_parser("run", help="Run benchmark tasks across workflow conditions.")
    run.add_argument("--config")
    run.add_argument("--tasks")
    run.add_argument("--model", required=True)
    run.add_argument(
        "--judge-model",
        help=(
            "Aggregation and vote tie-break judge model. "
            "Does not replace the supervisor model."
        ),
    )
    run.add_argument("--experiment-id")
    run.add_argument("--results-dir", default="results")
    run.add_argument("--conditions")
    run.add_argument("--system-prompt", default="")
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrently active logical jobs.",
    )
    run.add_argument(
        "--max-in-flight-requests",
        type=int,
        default=1,
        help="Global cap across model calls made inside all active workflows.",
    )
    run.add_argument("--requests-per-minute", type=int)
    run.add_argument("--tokens-per-minute", type=int)
    run.add_argument("--estimated-tokens-per-request", type=int, default=4096)
    run.add_argument("--repetitions", type=int)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--retry-base-delay", type=float, default=1.0)
    run.add_argument("--retry-max-delay", type=float, default=30.0)
    run.add_argument("--retry-jitter", type=float, default=0.2)
    run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the job/call plan without writing or calling a model.",
    )
    run.add_argument(
        "--progress",
        action="store_true",
        help="Show a live terminal summary instead of per-job JSON output.",
    )
    run.add_argument(
        "--require-answer-type",
        help="Reject the run unless every task has this canonical answer type.",
    )

    experiment = sub.add_parser(
        "experiment",
        help="Preflight, run, grade, and summarize an experiment end to end.",
    )
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--model", required=True)
    experiment.add_argument("--results-dir", default="results")
    experiment.add_argument("--reports-dir", default="reports")
    experiment.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    experiment.add_argument(
        "--grader-reasoning-effort",
        default=DEFAULT_GRADER_REASONING_EFFORT,
    )
    experiment.add_argument("--concurrency", type=int, default=1)
    experiment.add_argument("--max-in-flight-requests", type=int, default=1)
    experiment.add_argument("--grading-concurrency", type=int, default=8)
    experiment.add_argument(
        "--grading-max-in-flight-requests",
        type=int,
        default=8,
    )
    experiment.add_argument("--requests-per-minute", type=int)
    experiment.add_argument("--tokens-per-minute", type=int)
    experiment.add_argument("--max-attempts", type=int, default=3)
    experiment.add_argument("--skip-preflight", action="store_true")
    experiment.add_argument("--skip-grading", action="store_true")
    experiment.add_argument(
        "--html",
        action="store_true",
        help="Also generate the optional static HTML report.",
    )

    preflight = sub.add_parser(
        "preflight",
        help="Test live provider compatibility before a paid benchmark run.",
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--grader-model")
    preflight.add_argument("--grader-reasoning-effort")
    preflight.add_argument("--max-in-flight-requests", type=int, default=1)
    preflight.add_argument("--requests-per-minute", type=int)
    preflight.add_argument("--tokens-per-minute", type=int)
    preflight.add_argument("--estimated-tokens-per-request", type=int, default=512)
    preflight.add_argument("--max-attempts", type=int, default=2)
    preflight.add_argument("--retry-base-delay", type=float, default=1.0)
    preflight.add_argument("--retry-max-delay", type=float, default=10.0)
    preflight.add_argument("--retry-jitter", type=float, default=0.0)
    preflight.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact planned checks with zero provider calls or writes.",
    )

    analyze = sub.add_parser("analyze", help="Score saved run artifacts.")
    analyze.add_argument("--tasks", required=True)
    analyze.add_argument("--results-dir", default="results")
    analyze.add_argument("--experiment-id", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--grade-set-id")
    analyze.add_argument(
        "--require-semantic-grades",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    results = sub.add_parser(
        "results",
        help="Show a scored experiment summary directly in the terminal.",
    )
    results.add_argument("--config")
    results.add_argument("--tasks")
    results.add_argument("--experiment-id")
    results.add_argument("--results-dir", default="results")
    results.add_argument("--grade-set-id")

    grade = sub.add_parser(
        "grade",
        help="Semantically grade canonical HLE final and stage responses.",
    )
    grade.add_argument("--tasks", required=True)
    grade.add_argument("--results-dir", default="results")
    grade.add_argument("--experiment-id", required=True)
    grade.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help=f"Semantic grader model (default: {DEFAULT_GRADER_MODEL}).",
    )
    grade.add_argument("--scope", choices=["final", "all"], default="final")
    grade.add_argument("--grade-set-id")
    grade.add_argument("--concurrency", type=int, default=8)
    grade.add_argument("--max-in-flight-requests", type=int, default=8)
    grade.add_argument("--requests-per-minute", type=int)
    grade.add_argument("--tokens-per-minute", type=int)
    grade.add_argument("--estimated-tokens-per-request", type=int, default=2048)
    grade.add_argument("--max-attempts", type=int, default=3)
    grade.add_argument("--retry-base-delay", type=float, default=1.0)
    grade.add_argument("--retry-max-delay", type=float, default=30.0)
    grade.add_argument("--retry-jitter", type=float, default=0.2)
    grade.add_argument(
        "--max-tokens",
        type=int,
        help="Optional explicit grader completion cap; omitted by default.",
    )
    grade.add_argument(
        "--reasoning-effort",
        default=DEFAULT_GRADER_REASONING_EFFORT,
        help=(
            "Explicit grader reasoning effort "
            f"(default: {DEFAULT_GRADER_REASONING_EFFORT})."
        ),
    )
    grade.add_argument("--dry-run", action="store_true")

    charts = sub.add_parser("charts", help="Generate SVG charts from analysis JSON.")
    charts.add_argument("--analysis-dir", required=True)
    charts.add_argument("--output-dir", required=True)

    site = sub.add_parser("site", help="Generate a static benchmark website.")
    site.add_argument("--analysis-dir", required=True)
    site.add_argument("--output-dir", required=True)

    update = sub.add_parser("update-report", help="Analyze runs and rebuild the static website.")
    update.add_argument("--tasks", required=True)
    update.add_argument("--results-dir", default="results")
    update.add_argument("--experiment-id", required=True)
    update.add_argument("--output-dir", default="reports")
    update.add_argument(
        "--grader-model",
        help=(
            "Run semantic grading before rebuilding the report. "
            f"Recommended: {DEFAULT_GRADER_MODEL}."
        ),
    )
    update.add_argument("--grade-set-id")
    update.add_argument("--grading-concurrency", type=int, default=8)
    update.add_argument(
        "--grading-scope",
        choices=["final", "all"],
        default="final",
    )
    update.add_argument("--max-in-flight-requests", type=int, default=8)
    update.add_argument("--requests-per-minute", type=int)
    update.add_argument("--tokens-per-minute", type=int)
    update.add_argument("--max-attempts", type=int, default=3)
    update.add_argument(
        "--grader-max-tokens",
        type=int,
        help="Optional explicit grader completion cap; omitted by default.",
    )
    update.add_argument(
        "--grader-reasoning-effort",
        default=DEFAULT_GRADER_REASONING_EFFORT,
    )
    update.add_argument(
        "--require-semantic-grades",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    dashboard = sub.add_parser("dashboard", help="Start the local results dashboard.")
    dashboard.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically open the dashboard in a web browser.",
    )
    dashboard.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on.")
    dashboard.add_argument("--results-dir", default="results", help="Directory where experiment results are saved.")

    return parser


if __name__ == "__main__":
    main()
