from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from multi_agent_research.models import (
    AnswerChoice,
    AnswerSpec,
    TaskInput,
    TaskSource,
)

from extensions.benchmark_tools.schema import BenchmarkChoice, BenchmarkExample
from extensions.benchmark_tools.task_validation import (
    normalize_mcq_answer,
    validate_benchmark_example,
    validate_choices,
)


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

ANSWER_CHOICES_HEADING = "Answer Choices:"
CHOICE_LINE = re.compile(r"^([A-Z])\.\s*(.*)$")


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

    explicit_choices = tuple(_choice(choice) for choice in row.get("choices", []) or [])
    prompt, embedded_choices = _extract_embedded_choices(prompt, prefix=prefix)
    if explicit_choices and embedded_choices and explicit_choices != embedded_choices:
        raise ValueError(prefix + "embedded choices do not match choices field")
    choices = embedded_choices or explicit_choices

    declared_answer_type = str(row.get("answer_type") or "").strip()
    if embedded_choices:
        answer_type = "multiple_choice"
    else:
        answer_type = declared_answer_type or (
            "multiple_choice" if choices else "short_answer"
        )
    if answer_type == "multiple_choice":
        answer = normalize_mcq_answer(answer, choices, prefix=prefix)

    source = dict(row.get("source") or {})
    public_metadata = {
        key: value
        for key, value in dict(row.get("metadata") or {}).items()
        if key.lower() not in GOLD_KEYS
    }
    for key in ("difficulty", "subject", "category", "subcategory"):
        if key in row and key not in public_metadata:
            public_metadata[key] = row[key]

    example = BenchmarkExample(
        id=task_id,
        prompt=prompt,
        answer=answer,
        answer_type=answer_type,
        choices=choices,
        category=str(row.get("category") or row.get("subject") or "") or None,
        source=source,
        public_metadata=public_metadata,
    )
    validate_benchmark_example(example, prefix=prefix)
    return example


def task_from_example(
    example: BenchmarkExample,
    *,
    include_confidence: bool = True,
    include_explanation: bool = True,
) -> TaskInput:
    validate_benchmark_example(example)
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
    answer_type: str | None = None,
    sampling_strategy: str = "balanced",
) -> Path:
    examples = load_jsonl(input_path)
    if answer_type is not None:
        examples = [
            example for example in examples if example.answer_type == answer_type
        ]
        if not examples:
            raise ValueError(f"no tasks match answer_type={answer_type!r}")
    if sampling_strategy == "balanced":
        selected = fixed_sample(
            examples,
            sample_size=sample_size,
            seed=seed,
            stratify_key=stratify_key,
        )
    elif sampling_strategy == "proportional":
        selected = proportional_sample(
            examples,
            sample_size=sample_size,
            seed=seed,
            primary_key=stratify_key,
            secondary_key="answer_type",
        )
    else:
        raise ValueError(f"unsupported sampling_strategy: {sampling_strategy}")
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
                "sampling_strategy": sampling_strategy,
                "answer_type": answer_type,
                "source_distribution": _distribution(examples),
                "sample_distribution": _distribution(selected),
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


def proportional_sample(
    examples: list[BenchmarkExample],
    *,
    sample_size: int,
    seed: int,
    primary_key: str = "category",
    secondary_key: str = "answer_type",
) -> list[BenchmarkExample]:
    """Sample proportionally by subject, then by answer type within subject."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(examples):
        return sorted(examples, key=lambda example: example.id)

    primary_groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        primary_groups[_stratum_value(example, primary_key)].append(example)
    primary_quotas = _proportional_quotas(
        {key: len(group) for key, group in primary_groups.items()},
        sample_size,
    )

    rng = random.Random(seed)
    selected: list[BenchmarkExample] = []
    for primary in sorted(primary_groups):
        group = primary_groups[primary]
        secondary_groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
        for example in group:
            secondary_groups[_stratum_value(example, secondary_key)].append(example)
        secondary_quotas = _proportional_quotas(
            {key: len(rows) for key, rows in secondary_groups.items()},
            primary_quotas[primary],
        )
        for secondary in sorted(secondary_groups):
            candidates = sorted(
                secondary_groups[secondary],
                key=lambda example: example.id,
            )
            rng.shuffle(candidates)
            selected.extend(candidates[: secondary_quotas[secondary]])
    return sorted(selected, key=lambda example: example.id)


def _proportional_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if sample_size < 0 or sample_size > total:
        raise ValueError("sample_size must fit within the available stratum counts")
    if total == 0:
        return {key: 0 for key in counts}

    exact = {key: sample_size * count / total for key, count in counts.items()}
    quotas = {key: min(counts[key], int(value)) for key, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - int(exact[key])), key),
    )
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("could not allocate proportional sample quotas")
    return quotas


def _stratum_value(example: BenchmarkExample, key: str) -> str:
    value = getattr(example, key, None) or example.public_metadata.get(key)
    return str(value or "unknown")


def _distribution(examples: list[BenchmarkExample]) -> dict[str, dict[str, int]]:
    by_category = Counter(example.category or "unknown" for example in examples)
    by_answer_type = Counter(example.answer_type for example in examples)
    return {
        "category": dict(sorted(by_category.items())),
        "answer_type": dict(sorted(by_answer_type.items())),
    }


def _choice(value: Any) -> BenchmarkChoice:
    if isinstance(value, str):
        label, separator, text = value.partition("=")
        return BenchmarkChoice(label=label.strip(), text=text.strip() if separator else None)
    if isinstance(value, dict):
        return BenchmarkChoice(label=str(value["label"]).strip(), text=value.get("text"))
    raise ValueError(f"unsupported choice: {value!r}")


def _extract_embedded_choices(
    prompt: str,
    *,
    prefix: str = "",
) -> tuple[str, tuple[BenchmarkChoice, ...]]:
    if ANSWER_CHOICES_HEADING not in prompt:
        return prompt, ()

    lines = prompt.splitlines()
    headings = [
        index
        for index, line in enumerate(lines)
        if line.strip() == ANSWER_CHOICES_HEADING
    ]
    if len(headings) != 1:
        raise ValueError(prefix + "malformed embedded Answer Choices block")

    heading_index = headings[0]
    base_prompt = "\n".join(lines[:heading_index]).strip()
    if not base_prompt:
        raise ValueError(prefix + "missing prompt before Answer Choices block")

    parsed: list[BenchmarkChoice] = []
    current_label: str | None = None
    current_text: list[str] = []

    def finish_choice() -> None:
        if current_label is not None:
            parsed.append(
                BenchmarkChoice(
                    label=current_label,
                    text="\n".join(current_text).strip(),
                )
            )

    for line in lines[heading_index + 1 :]:
        match = CHOICE_LINE.match(line.strip())
        if match:
            finish_choice()
            current_label = match.group(1)
            current_text = [match.group(2)]
        elif current_label is None:
            if line.strip():
                raise ValueError(prefix + "malformed embedded Answer Choices block")
        else:
            current_text.append(line.rstrip())
    finish_choice()

    choices = tuple(parsed)
    if not choices:
        raise ValueError(prefix + "malformed embedded Answer Choices block")
    validate_choices(choices, prefix=prefix)
    return base_prompt, choices


def _prompt_with_choices(example: BenchmarkExample) -> str:
    if not example.choices:
        return example.prompt
    rendered = [f"{choice.label}. {choice.text}" for choice in example.choices]
    return (
        example.prompt.rstrip()
        + f"\n\n{ANSWER_CHOICES_HEADING}\n"
        + "\n".join(rendered)
    )


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
