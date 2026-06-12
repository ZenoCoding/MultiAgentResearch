from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.charts import write_charts
from extensions.benchmark_tools.config import load_experiment_config
from extensions.benchmark_tools.connector import load_jsonl, write_fixed_sample
from extensions.benchmark_tools.grading import grade_experiment
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
        repetitions = args.repetitions or (
            config.repetitions if config else 1
        )
        summary = asyncio.run(
            run_benchmark(
                tasks_path=tasks,
                model=args.model,
                judge_model=args.judge_model,
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
            )
        )
        print(json.dumps(summary, indent=2))
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
                    scope="all",
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
    run.add_argument("--judge-model")
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
        "--require-answer-type",
        help="Reject the run unless every task has this canonical answer type.",
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

    grade = sub.add_parser(
        "grade",
        help="Semantically grade canonical HLE final and stage responses.",
    )
    grade.add_argument("--tasks", required=True)
    grade.add_argument("--results-dir", default="results")
    grade.add_argument("--experiment-id", required=True)
    grade.add_argument("--grader-model", required=True)
    grade.add_argument("--scope", choices=["final", "all"], default="all")
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
    grade.add_argument("--max-tokens", type=int, default=4096)
    grade.add_argument("--reasoning-effort")
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
    update.add_argument("--grader-model")
    update.add_argument("--grade-set-id")
    update.add_argument("--grading-concurrency", type=int, default=8)
    update.add_argument("--max-in-flight-requests", type=int, default=8)
    update.add_argument("--requests-per-minute", type=int)
    update.add_argument("--tokens-per-minute", type=int)
    update.add_argument("--max-attempts", type=int, default=3)
    update.add_argument("--grader-max-tokens", type=int, default=4096)
    update.add_argument("--grader-reasoning-effort")
    update.add_argument(
        "--require-semantic-grades",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


if __name__ == "__main__":
    main()
