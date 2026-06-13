from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from extensions.benchmark_tools.config import load_experiment_config
from extensions.benchmark_tools.connector import load_jsonl
from extensions.benchmark_tools.runner import _workflow, run_benchmark


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "benchmarks/hle-smoke-5/tasks.jsonl"
SOURCE_TASKS_PATH = ROOT / "benchmarks/hle-representative-40/tasks.jsonl"
MANIFEST_PATH = ROOT / "benchmarks/hle-smoke-5/manifest.json"
CONFIG_PATH = ROOT / "data/hle-smoke-5-experiment.json"
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_smoke_tasks_are_canonical_and_have_requested_mix() -> None:
    rows = _jsonl_rows(TASKS_PATH)
    source_by_id = {
        row["id"]: row for row in _jsonl_rows(SOURCE_TASKS_PATH)
    }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ids = [row["id"] for row in rows]
    assert len(rows) == 5
    assert len(ids) == len(set(ids))
    assert Counter(row["answer_type"] for row in rows) == {
        "multiple_choice": 2,
        "short_answer": 3,
    }
    assert Counter(row["category"] for row in rows) == {
        "Biology/Medicine": 1,
        "Chemistry": 1,
        "Math": 1,
        "Other": 1,
        "Physics": 1,
    }
    assert rows == [source_by_id[task_id] for task_id in ids]
    assert manifest["source_ids"] == ids
    assert manifest["answer_type_counts"] == {
        "multiple_choice": 2,
        "short_answer": 3,
    }
    assert manifest["subject_counts"] == dict(
        sorted(Counter(row["category"] for row in rows).items())
    )
    assert manifest["seed"] == 20260612
    assert manifest["selection_rationale"]
    assert len(load_jsonl(TASKS_PATH)) == 5


async def test_smoke_config_expands_to_five_workflows_and_25_jobs() -> None:
    config = load_experiment_config(CONFIG_PATH)
    conditions = {condition.workflow: condition for condition in config.conditions}

    assert config.tasks_path == "benchmarks/hle-smoke-5/tasks.jsonl"
    assert config.aggregation_judge_model == JUDGE_MODEL
    assert config.repetitions == 1
    assert len(config.conditions) == 5
    assert set(conditions) == {
        "solo",
        "self-critic",
        "sample",
        "debate",
        "supervisor",
    }
    assert {condition.judge_reasoning_effort for condition in config.conditions} == {
        "low"
    }

    assert conditions["self-critic"].rounds == 1
    assert conditions["sample"].agents == 3
    assert conditions["sample"].aggregation == "plurality_vote"
    assert conditions["sample"].vote_tie_break == "judge"
    assert conditions["debate"].agents == 2
    assert conditions["debate"].rounds == 1
    assert conditions["debate"].debate_peer_view == "full_response"
    assert conditions["supervisor"].rounds == 1
    assert conditions["supervisor"].supervisor_reasoning_effort == "low"

    plan = await run_benchmark(
        tasks_path=ROOT / config.tasks_path,
        model="fake/non-paid",
        judge_model=config.aggregation_judge_model,
        experiment_id=config.experiment_id,
        conditions=list(config.conditions),
        repetitions=config.repetitions,
        dry_run=True,
    )
    assert plan["jobs"] == 25
    assert plan["estimated_minimum_model_calls"] == 71


def test_smoke_judge_pin_does_not_replace_supervisor_primary_model() -> None:
    config = load_experiment_config(CONFIG_PATH)
    conditions = {condition.workflow: condition for condition in config.conditions}

    sample = _workflow(
        condition=conditions["sample"],
        model="fake/primary",
        judge_model=config.aggregation_judge_model,
        system_prompt="",
        requires_semantic_judge=True,
    )
    supervisor = _workflow(
        condition=conditions["supervisor"],
        model="fake/primary",
        judge_model=config.aggregation_judge_model,
        system_prompt="",
        requires_semantic_judge=True,
    )

    assert sample.wrapped.judge.model == JUDGE_MODEL
    assert sample.wrapped.judge.reasoning_effort == "low"
    assert supervisor.wrapped.supervisor.model == "fake/primary"
    assert supervisor.wrapped.supervisor.reasoning_effort == "low"
