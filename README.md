# Multi-Agent Research

**TL;DR:** People say many LLMs working together is good. Is it? Why or why not?

This project aims to dissect key weaknesses of multi-agent reasoning, investigate whether meaningful improvements can be made to models through techniques like adversarial review and multi-agent debate, examine how these approaches affect token efficiency, and analyze their impact on qualitative output.

---

## Abstract

It is a common belief that harnessing multi-agent workflows with thorough review and at the expense of efficiency can lead to meaningful improvements in agent capabilities across fields like mathematics, coding, and general problem solving. In fact, these multi-agent teams were credited with advancing model capabilities to push the frontier of AI-Mathematics (e.g., Gemini's agentic workflow with verifiers and checkers to solve IMO Gold problems, or harnesses like Poetiq and consumer products like Grok). However, these improvements have focused on particular benchmarks and have not led to sweeping improvements across the frontier.

Furthermore, as coding agents (like Claude Code and Codex) continuously improve their capability, and the amount of real work that agents are able to perform independently increases, agents will need to be able to collaborate with humans or their LLM peers. 

This paper/project investigates these dynamics to identify where the limits lie, how interaction changes token efficiency, and how multi-agent interaction may reveal insights about human-agent interaction for subjective results (like writing emails).

---

## Method

The goal is to assess simple agent workflows and identify patterns in how agent responses change when they work with other agents. The experiment is constructed of three primary components:

### 1. LLM Harness
An abstraction layer that systemizes LLM outputs from various vendors, allowing them to be plug-and-play.
* **Implementation:** Built using **LiteLLM** (and optionally configured with OpenRouter for cost-effective model routing).
* Support for batch requests to optimize testing throughput.

### 2. Agent Harness
Allows multiple kinds of agent interactions to be abstracted and structured for testing:
* **Solo Agent:** Single agent, single answer.
* **Self-Critic:** An agent is continuously asked to revise its own output.
* **N-Debate:** $N$ agents debate a topic/problem, and a final "judge" model chooses the winner.
* **Supervisor-Subordinate:** A supervisor agent reviews a subordinate's output iteratively until satisfied.
* **System Prompt Experimentation:** Differentiating agent behaviors using specialized prompt configurations.

### 3. Tester
Utilizes the Agent Harness to run evaluations across a suite of reasoning, coding, and math benchmarks.
* Provides quantitative reports at scale.
* Saves all model outputs and intermediate debate logs for qualitative review.

---

## Project Tasks

- [x] Set up the GitHub repository
- [ ] Choose benchmark providers
- [x] Build the LLM Harness (`litellm`)
- [x] Build the initial Agent Harness
- [ ] Choose models for evaluation

---

## Harness

The Python package standardizes benchmark tasks, agent configuration, model
calls, workflow results, and usage data. It currently includes:

* Solo agent
* Independent sampling with a judge
* Self-critique and revision
* Multi-agent debate with a judge
* Supervisor-worker revision

Install the project:

```bash
uv sync --extra dev
```

Run a workflow:

```bash
uv run mar \
  --workflow debate \
  --model openai/gpt-5.4-nano \
  --agents 2 \
  --rounds 1 \
  --experiment-id first-debate \
  --prompt "Solve the problem and explain your answer."
```

Run the same synthetic hard problem through all five workflows:

```bash
./examples/run-baseline-workflows.sh openai/gpt-5.4-nano
```

The task, gold answer, derivation, and distractor rationale are documented in
[evals/baseline-grid-parity.md](evals/baseline-grid-parity.md). This is an
end-to-end smoke baseline; it is not a substitute for the versioned HLE task
set described in the experiment plan.

Reasoning effort is a first-class agent setting:

```bash
uv run mar \
  --workflow debate \
  --model openai/gpt-5.4-nano \
  --reasoning-effort low \
  --judge-reasoning-effort high \
  --prompt "Solve this problem."
```

`--reasoning-effort` applies to the primary agents.
`--judge-reasoning-effort` and `--supervisor-reasoning-effort` override it for
those roles. Values are passed through LiteLLM because supported effort names
vary by provider and model.

Processing priority is also configurable per role:

```bash
uv run mar \
  --workflow sample \
  --model openai/gpt-5.4-nano \
  --service-tier flex \
  --judge-service-tier priority \
  --prompt "Solve this problem."
```

Supported tier values are `auto`, `default`, `flex`, and `priority`.
`--service-tier` applies to primary agents; judge and supervisor flags override
it for those roles. The requested tier is included in the workflow
fingerprint, and the provider's returned tier is saved on each model call when
available.

Independent samples, debate initial answers, and all agents within each debate
round run concurrently. Debate rounds themselves remain sequential and use a
shared previous-round snapshot. Pass `--sequential` to disable parallel phases
for timing comparisons or provider constraints.

Debate can control what agents learn from their peers:

```bash
uv run mar \
  --workflow debate \
  --debate-peer-view answer_only \
  --agents 3 \
  --model openai/gpt-5.4-nano \
  --prompt "Solve this problem."
```

Available peer views are `full_response`, `answer_only`, and
`answer_and_confidence`. The last mode requires `--include-confidence` to
expose a reported confidence value.

Independent sampling and debate share configurable aggregation:

```bash
uv run mar \
  --workflow debate \
  --aggregation majority_vote \
  --vote-tie-break inconclusive \
  --agents 3 \
  --model openai/gpt-5.4-nano \
  --prompt "Solve this problem."
```

Available modes are `judge`, `majority_vote`, and `plurality_vote`. Voting
modes normally make no judge call. Tie handling is explicit: record an
inconclusive run, choose the first candidate-order answer, use a seeded random
choice, or call a judge only when the top vote is tied. Invalid formatted
ballots can be excluded or fail the run. Inconclusive runs retain the complete
tally and ballot records without being classified as execution failures.

The CLI default is `judge` for both independent sampling and debate. The
aggregation judge receives the original task and every candidate's full
response. It is prompted to choose or synthesize the best response; it does
not receive the gold answer and is not instructed to follow the majority.

Voting extracts each candidate's `<final_answer>`. `majority_vote` requires one
answer to receive more than half of valid ballots. `plurality_vote` selects the
most frequent answer even without an absolute majority. Voting currently uses
text normalization, not semantic equivalence, so it is best suited to
multiple-choice, numeric, JSON, or otherwise canonical answers.

Each run is stored under:

```text
results/<experiment-id>/<run-id>/
  request.json
  result.json
  provenance.json
  source-reference.json
  calls.jsonl
  events.jsonl
  artifact-manifest.json

results/_artifacts/sources/<source-sha256>.tar.gz
```

`result.json` contains the final answer, full call records, aggregate token
usage, estimated cost, latency, workflow events, and any failure information.

It also contains `stage_answers`, an ordered record of every answer-producing
stage. Candidate records include independent samples, debate initial answers
and revisions, self-critic revisions, and worker revisions. Judge or voting
outputs are marked as aggregate answers. Each record preserves the agent,
step, round metadata, raw response, parsed answer, confidence, and output
contract validity.

`provenance.json` records the command line, working directory, Git commit,
branch and dirty status, Python/platform versions, installed dependency
versions, relevant non-secret provider environment settings, credential and
`.env` fingerprints, `uv.lock` hash, and a SHA-256 manifest of the source tree.
The content-addressed source archive contains the exact non-secret repository
files present when the runner was created. Identical runs share one archive
under `results/_artifacts/sources`; each run stores only `source-reference.json`.
Environment files such as `.env` and generated result directories are
intentionally excluded; credentials are hashed rather than stored.

Each model call records the exact rendered messages, requested model, completion
parameters sent to LiteLLM, normalized output, provider response model, usage,
cost, timing, raw provider response, and failure traceback/details.

LiteLLM and provider-SDK retries are set to zero. Hidden fallback settings are
rejected so orchestration retries remain visible as distinct recorded
attempts. `artifact-manifest.json` stores the size and SHA-256 hash of every
other run artifact for later integrity checks.

The harness does not label a transition as correct or incorrect because
`TaskInput` contains no gold answer. A benchmark scorer can compare consecutive
stage answers against its private gold data to classify:

* Wrong to right: good correction
* Right to wrong: bad correction
* Wrong to wrong without adopting a correct peer answer: failed persuasion
* Right to right: retained correctness
* Wrong to wrong with an answer change: unsuccessful revision

### Benchmark boundary

Benchmark adapters provide gold-free `TaskInput` objects containing text or
multimodal messages plus an `AnswerSpec`. Gold answers and scoring remain
outside the workflow harness. Results contain both the complete final response
and an extracted, validated answer for the benchmark scorer.

See [docs/benchmark-integration.md](docs/benchmark-integration.md) for the
integration contract and examples.

### Prompt and workflow versioning

Each saved run includes:

* A semantic workflow version, such as `debate@1.0.0`
* A deterministic workflow fingerprint
* Every workflow prompt's name, semantic version, full template, and SHA-256
  content hash
* Prompt references on each individual model call
* The exact rendered messages sent to the model

The fingerprint changes when workflow configuration, agent configuration,
prompt versions, or prompt contents change. This prevents two materially
different runs from being grouped together merely because they share a
workflow name.

Built-in prompts live in
`src/multi_agent_research/prompts.py`. Override them without changing code by
passing a JSON file:

```bash
uv run mar \
  --workflow debate \
  --model openai/gpt-5.4-nano \
  --prompt-overrides examples/prompt-overrides.json \
  --prompt "Solve this problem."
```

An override is keyed by the stable prompt name:

```json
{
  "workflow.debate.peer_review": {
    "version": "1.1.0",
    "template": "Review these answers:\n$peer_answers"
  }
}
```

Overrides must preserve the variables required by the built-in prompt. When
prompt wording changes, increment its version. When workflow control flow or
message routing changes, increment the workflow class version.

Run the deterministic workflow tests without making API calls:

```bash
uv run pytest
```

---

## Prior Art & References

### 1. *Can LLM Agents Really Debate?*
* Focuses on the mechanics of Multi-Agent Debate (MAD) using logic puzzles in a controlled environment with clear steps.
* **Framework:** Iterative player discussion phase followed by self-adjustment, aggregated via majority vote or GPT-4 tiebreaker.
* **Key Findings:**
  * Performance is influenced by team size, composition, confidence visibility, debate order, depth, and task difficulty.
  * Weaker models often defer to consensus, while stronger models correct errors.
  * Overall ceiling is bounded by the strongest model in the mix.

### 2. *More Agents Is All You Need*
* Demonstrates that logarithmic scaling performance can be achieved via sampling and voting.
* **Key Findings:**
  * Stronger models benefit less from scaling.
  * Performance gains increase and then decrease, driven primarily by temperature.
  * Scales orthogonally and stacks using step-wise and hierarchical configurations.
