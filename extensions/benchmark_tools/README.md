# Benchmark Tools Extension

This directory is intentionally standalone. It adds benchmark sampling, batch
runs, scoring, chart generation, and a static report site without changing the
existing harness files.

## JSONL input format

Each row should contain only one task:

```json
{"id":"hle-1","prompt":"Question text","answer":"B","answer_type":"multiple_choice","choices":[{"label":"A","text":"..."},{"label":"B","text":"..."}],"category":"math","source":{"benchmark":"hle","version":"local","split":"sample"}}
```

Gold answers stay in this JSONL file and are used only by analysis. They are
not copied into `TaskInput.metadata`.

## Suggested personal sample

Start with 30 canonical multiple-choice HLE tasks:

```bash
uv run python -m extensions.benchmark_tools.cli sample \
  --input data/hle-all.jsonl \
  --output-dir benchmarks/hle-mcq-30 \
  --size 30 \
  --seed 20260611 \
  --answer-type multiple_choice
```

Thirty questions is small enough for repeated personal runs but large enough to
surface obvious differences between workflow conditions.

Validate the generated task set before any paid run:

```bash
uv run python -m extensions.benchmark_tools.cli validate-tasks \
  --tasks benchmarks/hle-mcq-30/tasks.jsonl \
  --require-answer-type multiple_choice
```

## Run and update report

First inspect the exact job grid and minimum call count. Dry runs do not create
experiment files or call a model:

```bash
uv run python -m extensions.benchmark_tools.cli run \
  --tasks benchmarks/hle-mcq-30/tasks.jsonl \
  --model openai/gpt-5.4-nano \
  --experiment-id hle-mcq-30-pilot \
  --conditions data/cheap-conditions.json \
  --require-answer-type multiple_choice \
  --concurrency 2 \
  --max-in-flight-requests 3 \
  --requests-per-minute 60 \
  --tokens-per-minute 120000 \
  --dry-run
```

Remove `--dry-run` to execute. Each logical
`(condition, task, repetition)` job has a stable ID. Provider failures create
separate attempts, and successful or inconclusive jobs are skipped when the
same command resumes. Experiment state is saved atomically under:

```text
results/<experiment-id>/experiment-manifest.json
results/<experiment-id>/experiment-ledger.json
results/<experiment-id>/<run-id>/...
```

Then rebuild the report:

```bash
uv run python -m extensions.benchmark_tools.cli update-report \
  --tasks benchmarks/hle-mcq-30/tasks.jsonl \
  --experiment-id hle-mcq-30-pilot \
  --output-dir reports
```

Open `reports/site/hle-mcq-30-pilot/index.html` after each update. Reports use
the experiment ledger to count each logical job once and show coverage,
execution failures, invalid output contracts, missing jobs, planned-job
accuracy, and accuracy among valid completed outputs separately.

## Representative mixed HLE matrix

`benchmarks/hle-representative-40` is a deterministic subject-proportional
sample of 40 HLE tasks. Its subject counts follow the full dataset's
distribution and it preserves the canonical answer-type mix within subjects:
11 multiple-choice and 29 short-answer tasks.

The complete 35-condition scaling matrix lives in
`data/hle-representative-40-experiment.json`. It expands solo, self-critic,
sampling, debate, and supervisor-worker variants across the requested effort,
agent, and round settings. Load and inspect it without spending:

```bash
uv run python -m extensions.benchmark_tools.cli run \
  --config data/hle-representative-40-experiment.json \
  --model openai/gpt-5.4-nano \
  --concurrency 2 \
  --max-in-flight-requests 3 \
  --dry-run
```

The current matrix is 1,400 logical jobs and at least 12,126 workflow model
calls. The estimate excludes conditional tie-break calls and the separate HLE
grading pass.

## Semantic HLE grading

Reports for mixed HLE task sets require a semantic grade set. The grader mirrors
the official HLE contract: it sees the question, full model response, and
reference answer, then records an extracted final answer, binary correctness,
reasoning, and confidence. It grades both canonical final responses and saved
stage responses so revision metrics use the same semantic standard.

Preview the number of unique responses after a run:

```bash
uv run python -m extensions.benchmark_tools.cli grade \
  --tasks benchmarks/hle-representative-40/tasks.jsonl \
  --results-dir results \
  --experiment-id hle-representative-40-scaling-v1 \
  --grader-model "$GRADER_MODEL" \
  --scope all \
  --dry-run
```

Run or resume grading:

```bash
uv run python -m extensions.benchmark_tools.cli grade \
  --tasks benchmarks/hle-representative-40/tasks.jsonl \
  --results-dir results \
  --experiment-id hle-representative-40-scaling-v1 \
  --grader-model "$GRADER_MODEL" \
  --scope all \
  --concurrency 8 \
  --max-in-flight-requests 8
```

Each unique task/response pair is cached atomically under the experiment's
`grades/` directory. Identical final and stage responses are judged once.
Malformed judge output and transient provider errors are retried explicitly.
The report shows semantic grading coverage and failures separately from model
execution and answer-contract validity.
