from __future__ import annotations

import re
from typing import Iterable

from extensions.benchmark_tools.schema import BenchmarkChoice, BenchmarkExample


MCQ_ANSWER = re.compile(r"^([A-Z])(?:[.)])?(?:\s+(.*))?$", re.DOTALL)


def validate_choices(
    choices: Iterable[BenchmarkChoice],
    *,
    prefix: str = "",
) -> None:
    choice_list = list(choices)
    labels = [choice.label for choice in choice_list]
    if any(not label for label in labels):
        raise ValueError(prefix + "answer choice labels cannot be empty")
    if len(labels) != len(set(labels)):
        raise ValueError(prefix + "answer choice labels must be unique")
    if any(not choice.text or not choice.text.strip() for choice in choice_list):
        raise ValueError(prefix + "answer choice text cannot be empty")

    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected:
        raise ValueError(
            prefix
            + "answer choice labels must be contiguous and ordered from A; "
            + f"got {labels}"
        )


def normalize_mcq_answer(
    answer: str,
    choices: Iterable[BenchmarkChoice],
    *,
    prefix: str = "",
) -> str:
    choice_list = list(choices)
    validate_choices(choice_list, prefix=prefix)
    match = MCQ_ANSWER.fullmatch(answer.strip())
    if not match:
        raise ValueError(prefix + f"invalid multiple-choice gold answer: {answer!r}")

    label = match.group(1)
    by_label = {choice.label: choice for choice in choice_list}
    if label not in by_label:
        raise ValueError(
            prefix + f"multiple-choice gold label {label!r} is not in parsed choices"
        )

    supplied_text = (match.group(2) or "").strip()
    expected_text = (by_label[label].text or "").strip()
    if supplied_text and supplied_text != expected_text:
        raise ValueError(
            prefix + f"gold answer text does not match choice {label!r}"
        )
    return label


def validate_benchmark_example(
    example: BenchmarkExample,
    *,
    prefix: str = "",
) -> None:
    if example.answer_type == "multiple_choice":
        if not example.choices:
            raise ValueError(prefix + "multiple_choice tasks require choices")
        validate_choices(example.choices, prefix=prefix)
        normalized = normalize_mcq_answer(
            example.answer,
            example.choices,
            prefix=prefix,
        )
        if normalized != example.answer:
            raise ValueError(prefix + "multiple-choice gold answer must be a label")
    elif example.choices:
        raise ValueError(
            prefix
            + f"{example.answer_type} tasks cannot define multiple-choice choices"
        )
