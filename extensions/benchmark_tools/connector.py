from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from multi_agent_research.models import (
    AnswerChoice,
    AnswerSpec,
    TaskInput,
    TaskSource,
)

from extensions.benchmark_tools.schema import BenchmarkChoice, BenchmarkExample


GOLD_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "correct_answer",
    "reference",
    "reference_answer",
    "solution",
    "rubric",
}


def load_jsonl(path: Path | str) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(handle)
    for line_number, line in enumerate(rows, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        examples.append(example_from_row(row, line_number=line_number))
    return examples


def example_from_row(row: dict[str, Any], *, line_number: int | None = None) -> BenchmarkExample:
    prefix = f"row {line_number}: " if line_number is not None else ""
    task_id = str(row.get("id") or row.get("original_id") or "").strip()
    prompt = str(row.get("prompt") or row.get("question") or "").strip()
    answer = str(
        row.get("answer")
        or row.get("gold")
        or row.get("gold_answer")
        or row.get("correct_answer")
        or ""
    ).strip()
    if not task_id:
        raise ValueError(prefix + "missing id")
    if not prompt:
        raise ValueError(prefix + "missing prompt/question")
    if not answer:
        raise ValueError(prefix + "missing answer/gold_answer/correct_answer")

    choices = tuple(_choice(choice) for choice in row.get("choices", []) or [])
    answer_type = str(row.get("answer_type") or ("multiple_choice" if choices else "short_answer"))
    source = dict(row.get("source") or {})
    public_metadata = {
        key: value
        for key, value in dict(row.get("metadata") or {}).items()
        if key not in GOLD_KEYS
    }
    for key in ("difficulty", "subject", "category", "subcategory"):
        if key in row and key not in public_metadata:
            public_metadata[key] = row[key]

    return BenchmarkExample(
        id=task_id,
        prompt=prompt,
        answer=answer,
        answer_type=answer_type,
        choices=choices,
        category=str(row.get("category") or row.get("subject") or "") or None,
        source=source,
        public_metadata=public_metadata,
    )


def task_from_example(
    example: BenchmarkExample,
    *,
    include_confidence: bool = True,
    include_explanation: bool = True,
) -> TaskInput:
    source = None
    if example.source:
        source = TaskSource(
            benchmark=str(example.source.get("benchmark") or "benchmark"),
            version=example.source.get("version"),
            split=example.source.get("split"),
            original_id=str(example.source.get("original_id") or example.id),
        )
    return TaskInput.from_prompt(
        id=example.id,
        prompt=_prompt_with_choices(example),
        answer_spec=AnswerSpec(
            type=example.answer_type,  # type: ignore[arg-type]
            choices=[
                AnswerChoice(label=choice.label, text=choice.text)
                for choice in example.choices
            ],
            include_confidence=include_confidence,
            include_explanation=include_explanation,
        ),
        source=source,
        metadata=example.public_metadata,
    )


def reference_map(examples: Iterable[BenchmarkExample]) -> dict[str, str]:
    return {example.id: example.answer for example in examples}


def write_fixed_sample(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    sample_size: int = 30,
    seed: int = 20260611,
    stratify_key: str = "category",
) -> Path:
    examples = load_jsonl(input_path)
    selected = fixed_sample(examples, sample_size=sample_size, seed=seed, stratify_key=stratify_key)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks_path = output / "tasks.jsonl"
    manifest_path = output / "manifest.json"
    rows = [_row_from_example(example) for example in selected]
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "input_path": str(input_path),
                "sample_size": len(selected),
                "requested_sample_size": sample_size,
                "seed": seed,
                "stratify_key": stratify_key,
                "task_ids": [example.id for example in selected],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tasks_path


def fixed_sample(
    examples: list[BenchmarkExample],
    *,
    sample_size: int,
    seed: int,
    stratify_key: str = "category",
) -> list[BenchmarkExample]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(examples):
        return sorted(examples, key=lambda example: example.id)

    rng = random.Random(seed)
    groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        value = getattr(example, stratify_key, None) or example.public_metadata.get(stratify_key) or "unknown"
        groups[str(value)].append(example)

    for group in groups.values():
        group.sort(key=lambda example: example.id)
        rng.shuffle(group)

    selected: list[BenchmarkExample] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < sample_size and any(groups.values()):
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        cursor += 1
    return sorted(selected, key=lambda example: example.id)


def _choice(value: Any) -> BenchmarkChoice:
    if isinstance(value, str):
        label, separator, text = value.partition("=")
        return BenchmarkChoice(label=label.strip(), text=text.strip() if separator else None)
    if isinstance(value, dict):
        return BenchmarkChoice(label=str(value["label"]).strip(), text=value.get("text"))
    raise ValueError(f"unsupported choice: {value!r}")


def _prompt_with_choices(example: BenchmarkExample) -> str:
    if not example.choices:
        return example.prompt
    rendered = []
    for choice in example.choices:
        rendered.append(f"{choice.label}. {choice.text}" if choice.text else choice.label)
    return example.prompt.rstrip() + "\n\nChoose one:\n" + "\n".join(rendered)


def _row_from_example(example: BenchmarkExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "prompt": example.prompt,
        "answer": example.answer,
        "answer_type": example.answer_type,
        "choices": [
            {"label": choice.label, "text": choice.text}
            for choice in example.choices
        ],
        "category": example.category,
        "source": example.source,
        "metadata": example.public_metadata,
    }
