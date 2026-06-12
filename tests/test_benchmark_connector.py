from __future__ import annotations

import json
from pathlib import Path

import pytest

from extensions.benchmark_tools.connector import (
    example_from_row,
    load_jsonl,
    proportional_sample,
    task_from_example,
    write_fixed_sample,
)
from extensions.benchmark_tools.schema import BenchmarkChoice, BenchmarkExample
from extensions.benchmark_tools.task_validation import validate_benchmark_example


HLE_SAMPLE_PATH = Path("benchmarks/hle-small-30/tasks.jsonl")


def _current_hle_row(task_id: str) -> dict[str, object]:
    with HLE_SAMPLE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] == task_id:
                return row
    raise AssertionError(f"missing HLE fixture row {task_id}")


@pytest.mark.parametrize(
    ("task_id", "gold", "choice_text"),
    [
        (
            "671d91bcad7fb0793a0e93bd",
            "C",
            "Chaps influences the structure of Kag1.",
        ),
        ("670fb58f2ca6bea76e9354a7", "H", "min 2, max 3"),
    ],
)
def test_current_hle_mcq_rows_preserve_structured_choices(
    task_id: str,
    gold: str,
    choice_text: str,
) -> None:
    example = example_from_row(_current_hle_row(task_id))

    assert example.answer_type == "multiple_choice"
    assert example.answer == gold
    assert "Answer Choices:" not in example.prompt
    assert example.choices[ord(gold) - ord("A")] == BenchmarkChoice(
        label=gold,
        text=choice_text,
    )

    task = task_from_example(example)
    prompt = task.messages[0].content
    assert isinstance(prompt, str)
    assert task.answer_spec.type == "multiple_choice"
    assert task.answer_spec.choices[ord(gold) - ord("A")].text == choice_text
    assert prompt.count("Answer Choices:") == 1
    assert prompt.count(f"{gold}. {choice_text}") == 1
    assert "answer" not in task.metadata


def test_genuine_short_answer_stays_short_answer() -> None:
    row = {
        "id": "668828540a642802bdfeadfc",
        "prompt": "Output the requested four-letter string.",
        "answer": "yeyo",
        "answer_type": "short_answer",
        "metadata": {"gold_answer": "leak", "subject": "Other"},
    }

    example = example_from_row(row)
    task = task_from_example(example)

    assert example.answer_type == "short_answer"
    assert example.choices == ()
    assert task.answer_spec.type == "short_answer"
    assert task.answer_spec.choices == []
    assert task.metadata == {"subject": "Other"}


def test_mcq_gold_with_matching_choice_text_normalizes_to_label() -> None:
    row = {
        "id": "mcq",
        "prompt": "Question?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
        "answer": "B. Beta",
        "answer_type": "short_answer",
    }

    assert example_from_row(row).answer == "B"


@pytest.mark.parametrize(
    ("prompt", "error"),
    [
        (
            "Question?\n\nAnswer Choices:\nA. Alpha\nA. Again",
            "labels must be unique",
        ),
        (
            "Question?\n\nAnswer Choices:\nA. Alpha\nC. Charlie",
            "contiguous and ordered",
        ),
        (
            "Question?\n\nAnswer Choices:\nA.\nB. Beta",
            "choice text cannot be empty",
        ),
        (
            "Question?\n\nAnswer Choices:\nnot a labeled choice",
            "malformed embedded",
        ),
        (
            "Question? Answer Choices:\nA. Alpha\nB. Beta",
            "malformed embedded",
        ),
    ],
)
def test_malformed_embedded_choice_blocks_are_rejected(
    prompt: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        example_from_row({"id": "bad", "prompt": prompt, "answer": "A"})


def test_mcq_gold_outside_parsed_labels_is_rejected() -> None:
    row = {
        "id": "bad-gold",
        "prompt": "Question?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
        "answer": "C",
    }

    with pytest.raises(ValueError, match="is not in parsed choices"):
        example_from_row(row)


@pytest.mark.parametrize(
    "example",
    [
        BenchmarkExample(
            id="mcq-without-choices",
            prompt="Question?",
            answer="A",
            answer_type="multiple_choice",
        ),
        BenchmarkExample(
            id="short-with-choices",
            prompt="Question?",
            answer="Alpha",
            answer_type="short_answer",
            choices=(BenchmarkChoice(label="A", text="Alpha"),),
        ),
    ],
)
def test_contradictory_answer_type_and_choices_are_rejected(
    example: BenchmarkExample,
) -> None:
    with pytest.raises(ValueError):
        validate_benchmark_example(example)


def test_load_jsonl_reports_row_for_validation_failure(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "prompt": "Question?\n\nAnswer Choices:\nA. Alpha\nC. Charlie",
                "answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"row 1: .*contiguous"):
        load_jsonl(path)


def test_fixed_sample_can_filter_to_multiple_choice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "id": "mcq-a",
            "prompt": "Question?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
            "answer": "A",
            "category": "one",
        },
        {
            "id": "short",
            "prompt": "Name the answer.",
            "answer": "alpha",
            "answer_type": "short_answer",
            "category": "two",
        },
        {
            "id": "mcq-b",
            "prompt": "Question?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
            "answer": "B",
            "category": "two",
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    tasks_path = write_fixed_sample(
        input_path=source,
        output_dir=tmp_path / "sample",
        sample_size=2,
        answer_type="multiple_choice",
    )

    examples = load_jsonl(tasks_path)
    manifest = json.loads((tasks_path.parent / "manifest.json").read_text())
    assert [example.id for example in examples] == ["mcq-a", "mcq-b"]
    assert {example.answer_type for example in examples} == {"multiple_choice"}
    assert manifest["answer_type"] == "multiple_choice"


def test_proportional_sample_matches_primary_distribution_and_type_mix() -> None:
    examples = load_jsonl("data/hle-all.jsonl")
    selected = proportional_sample(
        examples,
        sample_size=40,
        seed=20260612,
    )
    categories: dict[str, int] = {}
    answer_types: dict[str, int] = {}
    for example in selected:
        categories[example.category or "unknown"] = (
            categories.get(example.category or "unknown", 0) + 1
        )
        answer_types[example.answer_type] = (
            answer_types.get(example.answer_type, 0) + 1
        )

    assert categories == {
        "Biology/Medicine": 4,
        "Chemistry": 2,
        "Computer Science/AI": 4,
        "Engineering": 1,
        "Humanities/Social Science": 4,
        "Math": 18,
        "Other": 3,
        "Physics": 4,
    }
    assert answer_types == {"multiple_choice": 11, "short_answer": 29}
