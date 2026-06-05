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
- [ ] Build the LLM Harness (`litellm`)
- [ ] Build the Agent Harness
- [ ] Choose models for evaluation

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
