# Experiment Plan

## Research question

When a fixed model is given additional inference budget, does communication
between agents improve closed-ended reasoning more than spending the same
budget on independent attempts?

The primary study is intended to run on Humanity's Last Exam (HLE), but
benchmark integration is outside the core harness. The harness receives
gold-free `TaskInput` objects and does not load, filter, sample, or score HLE.

## Primary hypotheses

1. One round of peer review improves accuracy over the same agents' initial
   answers.
2. Debate outperforms independent sampling at a similar model-call and token
   budget.
3. Full reasoning exchange performs differently from sharing final answers
   alone.
4. Reported peer confidence changes revision behavior and calibration.

## Pilot requirements

Sarah's experiment layer supplies a fixed, versioned task set and experiment
manifest. The infrastructure pilot uses a deterministic 40-question HLE subset
sampled proportionally by subject and then by canonical answer type within each
subject. This preserves both multiple-choice and short-answer representation.
Report the two answer types separately: multiple choice provides objective
voting, while short answer requires semantic grouping during aggregation.
Final correctness for both answer types uses the versioned HLE semantic grading
judge. Stage responses are graded only in explicit diagnostic passes for
revision-transition analysis.

Use one primary model, fixed generation parameters, and the same questions for
every condition. Run three repetitions when provider nondeterminism or a nonzero
temperature is present.

| ID | Condition | Calls before aggregation | Purpose |
| --- | --- | ---: | --- |
| S1 | Solo | 1 | Baseline |
| S3 | Three independent samples | 3 | Benefit from multiple attempts |
| S6 | Six independent samples | 6 | Call-budget control for one-round debate |
| D3-A | Three agents, one round, answer only | 6 | Interaction without reasoning exchange |
| D3-R | Three agents, one round, full response | 6 | Interaction with reasoning exchange |
| D3-C | Three agents, one round, answer and confidence | 6 | Confidence-signaling effect |
| CX3-1 | Three-agent cross-examination, one round | 18 | Directed challenge-response interaction |

Use plurality voting for the primary comparison so judge behavior is not
confounded with agent interaction. Run judge aggregation as a separate
secondary experiment.

Short-answer and multimodal study phases depend on task sets and scoring
provided by the experiment layer. Do not mix those results into the primary
multiple-choice analysis.

For every debate run, score both:

- The plurality answer from the three initial responses.
- The plurality answer after the debate round.
- Every individual agent answer at the initial and post-debate stages.

Treat standard debate and adversarial debate as separate workflow conditions.
Adversarial debate assigns heterogeneous initial reasoning roles, challenges
unanimous initial answers as possible correlated errors, and uses a later
evidence-resolution round. Compare it directly with standard debate at the
same agent count, round count, model settings, peer visibility, and
aggregation policy.

Treat cross-examination debate as a third, higher-compute condition rather than
another prompt variant. It adds claim extraction, named challenge, direct
response, challenger verdict, and private final-revision phases. For `N` agents
and `R` rounds it makes `N * (3 + 3R)` calls before aggregation. Its primary
controls should use matched measured-token budgets rather than assuming that a
cross-examination round is comparable to a snapshot debate round.

This paired within-run comparison measures whether communication changes
correct answers to incorrect ones or incorrect answers to correct ones.

Join `stage_answers` to the benchmark layer's private correctness labels by
task ID. For each agent and adjacent answer stage, classify:

- Good correction: incorrect to correct
- Bad correction: correct to incorrect
- Retained correctness: correct to correct
- Failed correction: incorrect to incorrect

For debate, additionally record whether a correct peer answer was visible
before an agent remained incorrect. This is an operational measure of failed
persuasion; it does not prove that the peer's reasoning caused or should have
caused a change.

## Aggregation terminology

The workflow aggregation judge and the HLE grading judge are different:

- The aggregation judge sees the original task and candidate responses, but no
  gold answer. It chooses or synthesizes the best final response regardless of
  answer type.
- The HLE grading judge belongs to the benchmark layer. It sees the model
  response and reference answer and decides whether they match.

The HLE grader follows the official structured contract and stores the
extracted answer, binary correctness, grading reasoning, confidence, model,
prompt fingerprint, usage, and full call provenance. Grading is resumable and
cached by task and full response. A mixed-HLE report must not fall back to
normalized exact matching when final semantic grades are missing. Full
stage-level grading is optional because it substantially increases judge cost.
The default is pinned to `gpt-5.4-mini-2026-03-17` at `low` reasoning effort;
the official HLE script's `o3-mini-2025-01-31` call does not specify reasoning
effort.

Voting extracts each candidate's final answer and selects by strict majority
or plurality. Canonical answers such as multiple-choice labels are normalized
and counted directly without a model call. For short answers, a semantic vote
judge groups equivalent answers and then applies the requested voting rule.
This judge cannot choose a semantically smaller group merely because it thinks
that answer is more accurate.

The aggregation and vote tie-break judge is pinned to
`gpt-5.4-mini-2026-03-17` at `low` reasoning effort. This setting does not
change the model used by supervisor-worker conditions.

## Metrics

Primary:

- Accuracy
- Paired accuracy difference on identical tasks
- Wrong-to-right and right-to-wrong revision counts

Efficiency:

- Input, output, reasoning, and total tokens
- Estimated cost
- Wall-clock latency
- Accuracy per dollar and per million tokens

Calibration:

- Brier score
- Expected calibration error
- Accuracy by confidence bucket

Report confidence intervals for paired accuracy differences. Do not treat
model calls from the same task as independent samples.

## Required controls

The harness must preserve these variables in the workflow fingerprint and saved
configuration:

- Per-agent model, system prompt, reasoning effort, service tier, and generation
  parameters
- Agent count and debate rounds
- Peer view: full response, final answer only, or answer plus confidence
- Cross-examination routing and per-phase output-token caps
- Aggregation method and tie policy
- Prompt versions
- Parallel or sequential execution

The current Python workflow API already supports heterogeneous `AgentSpec`
objects. The experiment layer should construct those agents from its config
rather than adding a second agent implementation.

## Deferred controls

Do not add these to the first pilot:

- Sparse or directed communication topologies
- Agent identities or assigned personas
- Early stopping on consensus
- Dynamic agent replacement
- Tool access
- Mixed-model teams

They are useful follow-up variables, but including them now would make the
initial experiment too broad to interpret.

## Ownership boundary

The core harness owns workflow behavior, prompts, model calls, fingerprints,
and per-run artifacts.

Sarah's experiment layer owns all benchmark concerns: HLE access, importing,
conversion, filtering, sampling, task-set manifests, gold data, scoring, and
benchmark-version tracking. It also owns condition matrices, repetitions,
concurrency, retries, resume behavior, and aggregate reports.

The core harness only consumes the resulting gold-free `TaskInput` objects and
executes the requested workflow configuration.
