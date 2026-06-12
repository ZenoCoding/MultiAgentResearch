from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import AgentSpec
from multi_agent_research.prompts import (
    DEBATE_ADVERSARIAL_CHALLENGE_PROMPT,
    DEBATE_ADVERSARIAL_RESOLUTION_PROMPT,
    DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT,
    DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
    DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
    DEBATE_DERIVATION_ROLE_PROMPT,
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SELF_CRITIC_REVISION_PROMPT,
    SUPERVISOR_REVIEW_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TIE_BREAK_JUDGE_PROMPT,
    WORKER_REVISION_PROMPT,
)
from multi_agent_research.runner import ExperimentRunner
from multi_agent_research.storage import FileRunStore
from multi_agent_research.workflows import (
    AdversarialDebateWorkflow,
    DebateWorkflow,
    IndependentSampleWorkflow,
    SelfCriticWorkflow,
    SoloWorkflow,
    SupervisorWorkflow,
    Workflow,
)

from extensions.benchmark_tools.connector import load_jsonl, task_from_example
from extensions.benchmark_tools.schema import Condition


DEFAULT_CONDITIONS = [
    Condition(id="solo", workflow="solo"),
    Condition(id="sample-3", workflow="sample", agents=3, aggregation="plurality_vote"),
    Condition(id="sample-6", workflow="sample", agents=6, aggregation="plurality_vote"),
    Condition(
        id="debate-3-full",
        workflow="debate",
        agents=3,
        rounds=1,
        aggregation="plurality_vote",
        debate_peer_view="full_response",
    ),
    Condition(
        id="debate-3-answer",
        workflow="debate",
        agents=3,
        rounds=1,
        aggregation="plurality_vote",
        debate_peer_view="answer_only",
    ),
    Condition(
        id="adversarial-debate-3",
        workflow="adversarial-debate",
        agents=3,
        rounds=2,
        aggregation="plurality_vote",
        debate_peer_view="full_response",
    ),
]


def load_conditions(path: Path | str | None) -> list[Condition]:
    if path is None:
        return DEFAULT_CONDITIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["conditions"] if isinstance(data, dict) else data
    return [Condition(**row) for row in rows]


async def run_benchmark(
    *,
    tasks_path: Path | str,
    model: str,
    experiment_id: str,
    output_dir: Path | str = "results",
    conditions: list[Condition] | None = None,
    judge_model: str | None = None,
    system_prompt: str = "",
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    load_dotenv()
    examples = load_jsonl(tasks_path)
    selected_conditions = conditions or DEFAULT_CONDITIONS
    semaphore = asyncio.Semaphore(concurrency)
    llm = LiteLLMClient()
    store = FileRunStore(Path(output_dir))
    summaries: list[dict[str, Any]] = []

    async def run_one(condition: Condition, example_index: int) -> dict[str, Any]:
        example = examples[example_index]
        async with semaphore:
            runner = ExperimentRunner(llm=llm, store=store)
            task = task_from_example(
                example,
                include_confidence=condition.include_confidence,
            )
            result = await runner.run(
                task=task,
                workflow=_workflow(
                    condition=condition,
                    model=model,
                    judge_model=judge_model,
                    system_prompt=system_prompt,
                ),
                experiment_id=experiment_id,
            )
            summary = {
                "condition": condition.id,
                "task_id": example.id,
                "run_id": result.run_id,
                "status": result.status,
                "final_answer": result.final_answer,
                "cost_usd": result.metrics.cost_usd,
                "total_tokens": result.metrics.total_tokens,
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
            return summary

    jobs = [
        run_one(condition, index)
        for condition in selected_conditions
        for index in range(len(examples))
    ]
    for result in await asyncio.gather(*jobs):
        summaries.append(result)
    return summaries


def _workflow(
    *,
    condition: Condition,
    model: str,
    judge_model: str | None,
    system_prompt: str,
) -> Workflow:
    parameters: dict[str, Any] = {}
    if condition.temperature is not None:
        parameters["temperature"] = condition.temperature
    if condition.max_tokens is not None:
        parameters["max_tokens"] = condition.max_tokens
    base_agent = AgentSpec(
        id="agent-1",
        model=model,
        system_prompt=system_prompt,
        system_prompt_name="agent.primary.system",
        reasoning_effort=condition.reasoning_effort,
        service_tier=condition.service_tier,  # type: ignore[arg-type]
        parameters=parameters,
    )
    if condition.workflow == "solo":
        return LabeledWorkflow(SoloWorkflow(base_agent), condition.id)
    if condition.workflow == "self-critic":
        return LabeledWorkflow(
            SelfCriticWorkflow(base_agent, rounds=condition.rounds, revision_prompt=SELF_CRITIC_REVISION_PROMPT),
            condition.id,
        )

    agents = [
        AgentSpec(
            id=f"agent-{index + 1}",
            model=model,
            system_prompt=system_prompt,
            system_prompt_name="agent.primary.system",
            reasoning_effort=condition.reasoning_effort,
            service_tier=condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
        for index in range(condition.agents)
    ]
    judge = None
    if condition.aggregation == "judge" or condition.vote_tie_break == "judge":
        judge = AgentSpec(
            id="judge",
            model=judge_model or model,
            system_prompt=JUDGE_SYSTEM_PROMPT.template,
            system_prompt_name=JUDGE_SYSTEM_PROMPT.name,
            system_prompt_version=JUDGE_SYSTEM_PROMPT.version,
            reasoning_effort=condition.judge_reasoning_effort or condition.reasoning_effort,
            service_tier=condition.judge_service_tier or condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
    voting = VotingConfig(tie_break=condition.vote_tie_break)  # type: ignore[arg-type]

    if condition.workflow == "sample":
        return LabeledWorkflow(
            IndependentSampleWorkflow(
                agents,
                judge,
                judge_prompt=JUDGE_SELECTION_PROMPT,
                tie_break_judge_prompt=TIE_BREAK_JUDGE_PROMPT,
                parallel=not condition.sequential,
                aggregation=condition.aggregation,  # type: ignore[arg-type]
                voting=voting,
            ),
            condition.id,
        )
    if condition.workflow in {"debate", "adversarial-debate"}:
        workflow_type = AdversarialDebateWorkflow if condition.workflow == "adversarial-debate" else DebateWorkflow
        return LabeledWorkflow(
            workflow_type(
                agents,
                judge,
                rounds=condition.rounds,
                debate_prompt=DEBATE_REVIEW_PROMPT,
                judge_prompt=JUDGE_SELECTION_PROMPT,
                tie_break_judge_prompt=TIE_BREAK_JUDGE_PROMPT,
                parallel=not condition.sequential,
                aggregation=condition.aggregation,  # type: ignore[arg-type]
                voting=voting,
                peer_view=condition.debate_peer_view,  # type: ignore[arg-type]
                adversarial_role_prompts=(
                    DEBATE_DERIVATION_ROLE_PROMPT,
                    DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
                    DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
                ),
                adversarial_challenge_prompt=DEBATE_ADVERSARIAL_CHALLENGE_PROMPT,
                adversarial_unanimous_prompt=DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT,
                adversarial_resolution_prompt=DEBATE_ADVERSARIAL_RESOLUTION_PROMPT,
            ),
            condition.id,
        )
    if condition.workflow == "supervisor":
        supervisor = AgentSpec(
            id="supervisor",
            model=judge_model or model,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT.template,
            system_prompt_name=SUPERVISOR_SYSTEM_PROMPT.name,
            system_prompt_version=SUPERVISOR_SYSTEM_PROMPT.version,
            reasoning_effort=condition.judge_reasoning_effort or condition.reasoning_effort,
            service_tier=condition.judge_service_tier or condition.service_tier,  # type: ignore[arg-type]
            parameters=parameters,
        )
        return LabeledWorkflow(
            SupervisorWorkflow(
                worker=base_agent,
                supervisor=supervisor,
                max_revisions=condition.rounds,
                review_prompt=SUPERVISOR_REVIEW_PROMPT,
                revision_prompt=WORKER_REVISION_PROMPT,
            ),
            condition.id,
        )
    raise ValueError(f"unsupported workflow: {condition.workflow}")


class LabeledWorkflow(Workflow):
    def __init__(self, wrapped: Workflow, condition_id: str) -> None:
        self.wrapped = wrapped
        self.condition_id = condition_id
        self.name = wrapped.name
        self.version = wrapped.version

    async def run(self, task, context):  # type: ignore[no-untyped-def]
        return await self.wrapped.run(task, context)

    def config(self) -> dict[str, Any]:
        return {
            **self.wrapped.config(),
            "condition_id": self.condition_id,
        }

    def prompt_templates(self):  # type: ignore[no-untyped-def]
        return self.wrapped.prompt_templates()
