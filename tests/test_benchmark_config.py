from __future__ import annotations

import json

import pytest

from extensions.benchmark_tools.config import load_experiment_config


CONFIG_PATH = "data/hle-representative-40-experiment.json"


def test_representative_experiment_config_expands_requested_matrix() -> None:
    config = load_experiment_config(CONFIG_PATH)
    conditions = {condition.id: condition for condition in config.conditions}

    assert config.experiment_id == "hle-representative-40-scaling-v1"
    assert config.tasks_path == "benchmarks/hle-representative-40/tasks.jsonl"
    assert (
        config.aggregation_judge_model
        == "gpt-5.4-mini-2026-03-17"
    )
    assert len(conditions) == 35
    assert {
        condition.judge_reasoning_effort for condition in conditions.values()
    } == {"low"}

    assert {
        condition.reasoning_effort
        for condition in conditions.values()
        if condition.workflow == "solo"
    } == {"none", "low", "medium", "high", "xhigh"}
    assert conditions["self-critic-e-none-r9"].rounds == 9
    assert conditions["sample-e-medium-a9"].agents == 9

    debate = conditions["debate-e-medium-a9-r3"]
    assert debate.agents == 9
    assert debate.rounds == 3
    assert debate.aggregation == "plurality_vote"
    assert debate.vote_tie_break == "judge"

    mixed = conditions["supervisor-w-low-s-high-r5"]
    assert mixed.reasoning_effort == "low"
    assert mixed.supervisor_reasoning_effort == "high"
    assert mixed.rounds == 5


def test_config_rejects_unknown_effort(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "bad",
                "tasks": "tasks.jsonl",
                "families": {"solo": {"efforts": ["extreme"]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        load_experiment_config(path)
