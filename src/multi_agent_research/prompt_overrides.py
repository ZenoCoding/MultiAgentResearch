from __future__ import annotations

import json
from pathlib import Path
from string import Template

from multi_agent_research.models import PromptTemplate


def load_prompt_overrides(
    path: Path | str | None,
) -> dict[str, PromptTemplate]:
    if path is None:
        return {}

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Prompt override file must contain a JSON object")

    overrides: dict[str, PromptTemplate] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Prompt override {name!r} must be an object")
        overrides[name] = PromptTemplate(
            name=name,
            version=value["version"],
            template=value["template"],
        )
    return overrides


def overridden(
    default: PromptTemplate,
    overrides: dict[str, PromptTemplate],
) -> PromptTemplate:
    override = overrides.get(default.name)
    if override is None:
        return default

    expected_variables = set(Template(default.template).get_identifiers())
    actual_variables = set(Template(override.template).get_identifiers())
    if actual_variables != expected_variables:
        raise ValueError(
            f"Prompt override {default.name!r} must use variables "
            f"{sorted(expected_variables)}, got {sorted(actual_variables)}"
        )
    return override
