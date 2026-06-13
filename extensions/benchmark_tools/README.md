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
`(condition, task, repetition)` job has a stable ID. A 429 retries only the
failed model request inside the active workflow; completed workflow calls are
not repeated. Timeouts fail the job without an automatic retry. Each scheduled
job runs at most once per invocation, and successful jobs are skipped when the
same command resumes. `--max-attempts` is the persisted attempt ceiling across
explicit resumes, while `--request-max-attempts` controls 429 recovery inside a
single workflow call. Experiment state is saved atomically under:

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

The matrix pins semantic aggregation and vote tie-break judging to
`gpt-5.4-mini-2026-03-17` at `low` reasoning effort. Supervisor-worker
conditions continue to use the primary experiment model with their configured
supervisor effort.

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

To resume the non-debate portion on the 2M TPM provider tier, leave headroom
for token-estimation variance and defer grading until this phase completes:

```bash
uv run python -m extensions.benchmark_tools.cli experiment \
  --config data/hle-representative-40-experiment.json \
  --model openai/gpt-5.4-nano \
  --concurrency 32 \
  --max-in-flight-requests 48 \
  --requests-per-minute 1000 \
  --tokens-per-minute 1500000 \
  --max-attempts 4 \
  --request-max-attempts 2 \
  --exclude-workflow debate \
  --exclude-reasoning-effort high \
  --exclude-reasoning-effort xhigh \
  --skip-preflight \
  --skip-grading
```

Changing these operational limits is resume-compatible and does not rerun
successful jobs. Workflow exclusions apply only to the current invocation: the
full condition matrix remains in the manifest and deferred jobs remain pending
for a later resume.

On macOS or Linux, request a graceful drain without cancelling active model
calls:

```bash
kill -USR1 <benchmark-pid>
```

The process finishes calls already admitted by the concurrency semaphore,
leaves queued jobs pending, saves the ledger, and exits normally. Resume with
the same experiment ID and new operational limits.

## First paid smoke run

`benchmarks/hle-smoke-5` contains five fixed tasks from the representative set:
two multiple-choice and three short-answer questions across five subjects.
`data/hle-smoke-5-experiment.json` expands to five workflow conditions and 25
logical jobs, with at least 71 workflow model calls.

The normal end-to-end path is one resumable command:

```bash
uv run python -m extensions.benchmark_tools.cli experiment \
  --config data/hle-smoke-5-experiment.json \
  --model "$PRIMARY_MODEL" \
  --concurrency 2 \
  --max-in-flight-requests 3
```

It runs provider preflight for a new experiment, executes or resumes benchmark
jobs, grades final answers, and prints the scored terminal table. Existing
experiment jobs and grades are reused automatically. Add `--html` only when a
shareable static report is needed.

The lower-level commands remain available for inspecting a stage without
running the full pipeline. For example, inspect the job plan without spending:

```bash
uv run python -m extensions.benchmark_tools.cli run \
  --config data/hle-smoke-5-experiment.json \
  --model "$PRIMARY_MODEL" \
  --concurrency 2 \
  --max-in-flight-requests 3 \
  --dry-run
```

Remove `--dry-run` only after preflight passes. The same command resumes the
experiment from its atomic ledger if interrupted.

For a live terminal summary with completion percentage, job status counts,
retries, session spend, rolling tokens per minute (TPM), rolling successful
requests per minute (RPM), aggregate output tokens per second (TPS), elapsed
time, ETA, and an animated comet moving through unfinished work, add
`--progress`:

```bash
uv run python -m extensions.benchmark_tools.cli run \
  --config data/hle-smoke-5-experiment.json \
  --model "$PRIMARY_MODEL" \
  --concurrency 2 \
  --max-in-flight-requests 3 \
  --progress
```

After grading, inspect scored results directly in the terminal without
generating a site:

```bash
uv run python -m extensions.benchmark_tools.cli results \
  --config data/hle-smoke-5-experiment.json
```

Use `update-report` only when an HTML artifact is useful for sharing or deeper
inspection.

## Semantic HLE grading

Reports for mixed HLE task sets require a semantic grade set. The grader mirrors
the official HLE prompt and structured-output contract: it sees the question,
full model response, and reference answer, then records an extracted final
answer, binary correctness, reasoning, confidence, and the required strict
flag. The primary pass grades canonical final responses. An explicit
`--scope all` diagnostic pass also grades saved stage responses for semantic
revision metrics.

The default grader is pinned to `gpt-5.4-mini-2026-03-17` with
`reasoning_effort=low`. This replaces HLE's older
`o3-mini-2025-01-31` default. The model and effort are written into the grade
manifest and changing either produces a separate grade set.

Preview the number of unique responses after a run:

```bash
uv run python -m extensions.benchmark_tools.cli grade \
  --tasks benchmarks/hle-representative-40/tasks.jsonl \
  --results-dir results \
  --experiment-id hle-representative-40-scaling-v1 \
  --dry-run
```

Run or resume grading:

```bash
uv run python -m extensions.benchmark_tools.cli grade \
  --tasks benchmarks/hle-representative-40/tasks.jsonl \
  --results-dir results \
  --experiment-id hle-representative-40-scaling-v1 \
  --concurrency 8 \
  --max-in-flight-requests 8
```

Each unique task/response pair is cached atomically under the experiment's
`grades/` directory. The final-only pass requires at most one successful judge
call per successful logical job: 1,400 calls for the complete matrix before
deduplication or retries. Use `--scope all` only when stage-transition analysis
is needed; its current worst-case upper bound is 11,062 unique judge calls.
Identical final and stage responses are judged once. Malformed judge output and
transient provider errors are retried explicitly. The report shows semantic
grading coverage and failures separately from model execution and
answer-contract validity.
