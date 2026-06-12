from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from extensions.benchmark_tools.analysis import analyze_experiment
from extensions.benchmark_tools.charts import write_charts
from extensions.benchmark_tools.connector import write_fixed_sample
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
        )
        print(path)
    elif args.command == "run":
        summaries = asyncio.run(
            run_benchmark(
                tasks_path=args.tasks,
                model=args.model,
                judge_model=args.judge_model,
                experiment_id=args.experiment_id,
                output_dir=args.results_dir,
                conditions=load_conditions(args.conditions),
                system_prompt=args.system_prompt,
                concurrency=args.concurrency,
            )
        )
        print(json.dumps({"runs": len(summaries)}, indent=2))
    elif args.command == "analyze":
        summary = analyze_experiment(
            tasks_path=args.tasks,
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            output_dir=args.output_dir,
        )
        print(json.dumps({"conditions": len(summary["conditions"])}, indent=2))
    elif args.command == "charts":
        paths = write_charts(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
        print(json.dumps([str(path) for path in paths], indent=2))
    elif args.command == "site":
        index = build_site(analysis_dir=args.analysis_dir, output_dir=args.output_dir)
        print(index)
    elif args.command == "update-report":
        analysis_dir = Path(args.output_dir) / "analysis" / args.experiment_id
        site_dir = Path(args.output_dir) / "site" / args.experiment_id
        analyze_experiment(
            tasks_path=args.tasks,
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            output_dir=analysis_dir,
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

    run = sub.add_parser("run", help="Run benchmark tasks across workflow conditions.")
    run.add_argument("--tasks", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--judge-model")
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--results-dir", default="results")
    run.add_argument("--conditions")
    run.add_argument("--system-prompt", default="")
    run.add_argument("--concurrency", type=int, default=1)

    analyze = sub.add_parser("analyze", help="Score saved run artifacts.")
    analyze.add_argument("--tasks", required=True)
    analyze.add_argument("--results-dir", default="results")
    analyze.add_argument("--experiment-id", required=True)
    analyze.add_argument("--output-dir", required=True)

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
    return parser


if __name__ == "__main__":
    main()

