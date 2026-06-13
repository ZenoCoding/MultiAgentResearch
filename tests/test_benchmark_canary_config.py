from __future__ import annotations

from collections import Counter
from pathlib import Path

from extensions.benchmark_tools.config import load_experiment_config
from extensions.benchmark_tools.connector import load_jsonl
from extensions.benchmark_tools.runner import run_benchmark


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "benchmarks/hle-matrix-canary-2/tasks.jsonl"
CONFIG_PATH = ROOT / "data/hle-matrix-canary-2-experiment.json"


async def test_canary_uses_two_answer_types_across_full_matrix() -> None:
    tasks = load_jsonl(TASKS_PATH)
    config = load_experiment_config(CONFIG_PATH)

    assert Counter(task.answer_type for task in tasks) == {
        "multiple_choice": 1,
        "short_answer": 1,
    }
    assert len(config.conditions) == 35
    assert config.experiment_id == "hle-matrix-canary-2-v1"

    plan = await run_benchmark(
        tasks_path=TASKS_PATH,
        model="fake/non-paid",
        judge_model=config.aggregation_judge_model,
        experiment_id=config.experiment_id,
        conditions=list(config.conditions),
        repetitions=config.repetitions,
        dry_run=True,
    )

    assert plan["jobs"] == 70
    assert plan["estimated_minimum_model_calls"] > 500
