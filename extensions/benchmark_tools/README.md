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

Start with 30 text-only multiple-choice HLE tasks:

```bash
uv run python -m extensions.benchmark_tools.cli sample \
  --input data/hle-all.jsonl \
  --output-dir benchmarks/hle-small-30 \
  --size 30 \
  --seed 20260611
```

Thirty questions is small enough for repeated personal runs but large enough to
surface obvious differences between workflow conditions.

## Run and update report

```bash
uv run python -m extensions.benchmark_tools.cli run \
  --tasks benchmarks/hle-small-30/tasks.jsonl \
  --model openai/gpt-5.4-nano \
  --experiment-id hle-small-30-v1 \
  --conditions extensions/benchmark_tools/example_conditions.json \
  --concurrency 1

uv run python -m extensions.benchmark_tools.cli update-report \
  --tasks benchmarks/hle-small-30/tasks.jsonl \
  --experiment-id hle-small-30-v1 \
  --output-dir reports
```

Open `reports/site/hle-small-30-v1/index.html` after each update.

