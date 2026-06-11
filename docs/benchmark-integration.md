# Benchmark Integration Contract

The benchmark layer owns dataset loading, gold answers, and scoring. The agent
harness owns model-visible task input, workflow execution, and normalized
workflow output.

## Safety boundary

`TaskInput` is deliberately gold-free. Do not place expected answers, scoring
rubrics, or hidden evaluator material in `TaskInput.metadata`; everything in a
task may be persisted and is available to a workflow.

A benchmark adapter should keep its reference data separately:

```python
reference_by_task_id = {
    "sample-1": "B",
}

task = TaskInput(...)
result = await runner.run(
    task=task,
    workflow=workflow,
    experiment_id="hle-baseline",
)
score = result.final_answer == reference_by_task_id[task.id]
```

## Text task

```python
from multi_agent_research import AnswerSpec, TaskInput

task = TaskInput.from_prompt(
    id="math-1",
    prompt="What is 12 * 12?",
    answer_spec=AnswerSpec(
        type="number",
        include_explanation=True,
    ),
)
```

## Multiple-choice task

The benchmark question and choices should remain in the model-visible
messages. `AnswerSpec.choices` defines the allowed scoreable labels.

```python
from multi_agent_research import AnswerChoice, AnswerSpec, TaskInput

task = TaskInput.from_prompt(
    id="mc-1",
    prompt="Which value is prime?\nA. 9\nB. 11\nC. 15",
    answer_spec=AnswerSpec(
        type="multiple_choice",
        choices=[
            AnswerChoice(label="A", text="9"),
            AnswerChoice(label="B", text="11"),
            AnswerChoice(label="C", text="15"),
        ],
        include_confidence=True,
    ),
)
```

## Multimodal task

Image URLs can be normal HTTP URLs or data URLs. The benchmark adapter is
responsible for converting local dataset image bytes into a provider-readable
URL or data URL.

```python
from multi_agent_research.models import (
    AnswerSpec,
    ImageContent,
    ImageURL,
    Message,
    TaskInput,
    TextContent,
)

task = TaskInput(
    id="image-1",
    messages=[
        Message(
            role="user",
            content=[
                TextContent(text="What object is shown?"),
                ImageContent(
                    image_url=ImageURL(
                        url="data:image/png;base64,...",
                    )
                ),
            ],
        )
    ],
    answer_spec=AnswerSpec(type="short_answer"),
)
```

When sampling or debate uses `majority_vote` or `plurality_vote` for a
`short_answer` task, a semantic vote judge groups equivalent answers before
applying the vote rule. For example, `NYC` and `New York City` can count
together. The judge must return a representative answer from the winning
group; it cannot substitute a minority answer it independently prefers.

`aggregation="judge"` remains a separate behavior for every answer type: the
judge chooses or synthesizes the answer it considers best.

## Output contract

Every answer-producing model call receives an instruction to end with:

```text
<final_answer>...</final_answer>
```

When requested, it must also include:

```text
<confidence>0-100</confidence>
```

`RunResult` provides:

* `output.raw_response`: the complete final model response
* `output.answer`: the extracted answer
* `output.confidence`: extracted confidence when requested
* `output.contract_valid`: whether formatting and answer type were valid
* `output.validation_errors`: protocol problems
* `final_answer`: convenience alias for `output.answer`

Benchmark correctness remains the benchmark adapter's responsibility.

## Supported answer types

* `short_answer`
* `multiple_choice`
* `number`
* `json`
* `code`
* `free_text`

These types constrain output formatting. The `short_answer` type also uses
semantic grouping when a voting aggregation is selected; benchmark
correctness remains the benchmark adapter's responsibility.

## Source identity

Populate `TaskInput.source` with the benchmark name, dataset version, split,
and original sample ID. This makes result files traceable without exposing
gold answers.
