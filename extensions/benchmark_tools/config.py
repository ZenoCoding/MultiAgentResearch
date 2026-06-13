from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from extensions.benchmark_tools.schema import Condition


CONFIG_SCHEMA_VERSION = 1
EFFORTS = {"none", "low", "medium", "high", "xhigh"}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    tasks_path: str
    aggregation_judge_model: str | None
    repetitions: int
    conditions: tuple[Condition, ...]
    metadata: dict[str, Any]
    schema_version: int = CONFIG_SCHEMA_VERSION


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("experiment config must be an object")
    allowed = {
        "schema_version",
        "experiment_id",
        "tasks",
        "aggregation_judge_model",
        "repetitions",
        "defaults",
        "families",
        "metadata",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown experiment config fields: {sorted(unknown)}")
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported experiment config schema_version")

    experiment_id = _nonempty_string(data.get("experiment_id"), "experiment_id")
    tasks_path = _nonempty_string(data.get("tasks"), "tasks")
    aggregation_judge_model = (
        _nonempty_string(
            data.get("aggregation_judge_model"),
            "aggregation_judge_model",
        )
        if data.get("aggregation_judge_model") is not None
        else None
    )
    repetitions = int(data.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    defaults = _object(data.get("defaults", {}), "defaults")
    families = _object(data.get("families"), "families")
    metadata = _object(data.get("metadata", {}), "metadata")

    conditions = [
        *_solo_conditions(families.get("solo"), defaults),
        *_round_conditions(
            "self-critic",
            families.get("self_critic"),
            defaults,
        ),
        *_sample_conditions(families.get("sampling"), defaults),
        *_debate_conditions(families.get("debate"), defaults),
        *_supervisor_conditions(families.get("supervisor_worker"), defaults),
    ]
    if not conditions:
        raise ValueError("experiment config expands to no conditions")
    ids = [condition.id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment config expands to duplicate condition ids")
    return ExperimentConfig(
        experiment_id=experiment_id,
        tasks_path=tasks_path,
        aggregation_judge_model=aggregation_judge_model,
        repetitions=repetitions,
        conditions=tuple(conditions),
        metadata=metadata,
    )


def _solo_conditions(value: Any, defaults: dict[str, Any]) -> list[Condition]:
    if value is None:
        return []
    spec = _object(value, "families.solo")
    return [
        _condition(
            defaults,
            id=f"solo-e-{effort}",
            workflow="solo",
            reasoning_effort=_effort(effort),
        )
        for effort in _list(spec.get("efforts"), "solo.efforts")
    ]


def _round_conditions(
    workflow: str,
    value: Any,
    defaults: dict[str, Any],
) -> list[Condition]:
    conditions = []
    for variant in _variants(value, workflow):
        effort = _effort(variant.get("effort"))
        for rounds in _positive_ints(variant.get("rounds"), f"{workflow}.rounds"):
            conditions.append(
                _condition(
                    defaults,
                    id=f"{workflow}-e-{effort}-r{rounds}",
                    workflow=workflow,
                    rounds=rounds,
                    reasoning_effort=effort,
                )
            )
    return conditions


def _sample_conditions(value: Any, defaults: dict[str, Any]) -> list[Condition]:
    conditions = []
    for variant in _variants(value, "sampling"):
        effort = _effort(variant.get("effort"))
        for agents in _positive_ints(variant.get("agents"), "sampling.agents"):
            conditions.append(
                _condition(
                    defaults,
                    id=f"sample-e-{effort}-a{agents}",
                    workflow="sample",
                    agents=agents,
                    reasoning_effort=effort,
                )
            )
    return conditions


def _debate_conditions(value: Any, defaults: dict[str, Any]) -> list[Condition]:
    conditions = []
    for variant in _variants(value, "debate"):
        effort = _effort(variant.get("effort"))
        pairs = _list(variant.get("agent_round_pairs"), "debate.agent_round_pairs")
        for pair in pairs:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(item, int) or item < 1 for item in pair)
            ):
                raise ValueError(
                    "debate.agent_round_pairs must contain [agents, rounds]"
                )
            agents, rounds = pair
            conditions.append(
                _condition(
                    defaults,
                    id=f"debate-e-{effort}-a{agents}-r{rounds}",
                    workflow="debate",
                    agents=agents,
                    rounds=rounds,
                    reasoning_effort=effort,
                )
            )
    return conditions


def _supervisor_conditions(value: Any, defaults: dict[str, Any]) -> list[Condition]:
    conditions = []
    for variant in _variants(value, "supervisor_worker"):
        worker_effort = _effort(variant.get("worker_effort"))
        supervisor_effort = _effort(
            variant.get("supervisor_effort", worker_effort)
        )
        for rounds in _positive_ints(
            variant.get("max_rounds"),
            "supervisor_worker.max_rounds",
        ):
            conditions.append(
                _condition(
                    defaults,
                    id=(
                        f"supervisor-w-{worker_effort}-"
                        f"s-{supervisor_effort}-r{rounds}"
                    ),
                    workflow="supervisor",
                    rounds=rounds,
                    reasoning_effort=worker_effort,
                    supervisor_reasoning_effort=supervisor_effort,
                )
            )
    return conditions


def _condition(
    defaults: dict[str, Any],
    **overrides: Any,
) -> Condition:
    values = {
        "aggregation": defaults.get("aggregation", "plurality_vote"),
        "vote_tie_break": defaults.get("vote_tie_break", "judge"),
        "debate_peer_view": defaults.get("debate_peer_view", "full_response"),
        "include_confidence": bool(defaults.get("include_confidence", False)),
        "judge_reasoning_effort": (
            _effort(defaults["judge_reasoning_effort"])
            if defaults.get("judge_reasoning_effort") is not None
            else None
        ),
        **overrides,
    }
    return Condition(**values)


def _variants(value: Any, label: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [
        _object(item, f"{label} variant")
        for item in _list(value, f"families.{label}")
    ]


def _effort(value: Any) -> str:
    effort = _nonempty_string(value, "reasoning effort")
    aliases = {"med": "medium"}
    effort = aliases.get(effort, effort)
    if effort not in EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {effort}")
    return effort


def _positive_ints(value: Any, label: str) -> list[int]:
    values = _list(value, label)
    if any(not isinstance(item, int) or item < 1 for item in values):
        raise ValueError(f"{label} must contain positive integers")
    return values


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    return list(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
