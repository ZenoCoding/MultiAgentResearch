from __future__ import annotations

from typing import Any

from multi_agent_research.aggregation import (
    aggregate_votes,
    AggregationMode,
    VALID_AGGREGATION_MODES,
    VotingConfig,
)
from multi_agent_research.context import CompletionSpec, RunContext, task_messages
from multi_agent_research.models import AgentSpec, Message, PromptTemplate, TaskInput
from multi_agent_research.prompts import (
    JUDGE_SELECTION_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow


class IndependentSampleWorkflow(Workflow):
    name = "independent_sample"
    version = "2.2.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec | None = None,
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
        parallel: bool = True,
        aggregation: AggregationMode = "judge",
        voting: VotingConfig | None = None,
    ) -> None:
        if not agents:
            raise ValueError("Independent sampling requires at least one agent")
        if aggregation not in VALID_AGGREGATION_MODES:
            raise ValueError(f"Unsupported aggregation mode: {aggregation}")
        if aggregation == "judge" and judge is None:
            raise ValueError("Judge aggregation requires a judge")
        self.agents = agents
        self.judge = judge
        self.judge_prompt = judge_prompt
        self.parallel = parallel
        self.aggregation = aggregation
        self.voting = voting or VotingConfig()

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"sample_{index}",
                    agent=agent,
                    messages=task_messages(task, agent),
                    prompt_references=[task.answer_spec.prompt_reference()],
                    metadata={"sample_index": index},
                    track_answer=True,
                )
                for index, agent in enumerate(self.agents)
            ],
            parallel=self.parallel,
        )
        answers = [
            (agent.id, response)
            for agent, response in zip(self.agents, responses, strict=True)
        ]

        if self.aggregation != "judge":
            vote = aggregate_votes(
                task=task,
                candidates=answers,
                mode=self.aggregation,
                config=self.voting,
            )
            context.emit("votes_aggregated", **vote.model_dump())
            context.record_stage_answer(
                step="aggregation",
                response=vote.response(task.answer_spec),
                kind="aggregate",
                metadata={"aggregation": self.aggregation},
            )
            context.emit("workflow_completed", workflow=self.name)
            return vote.response(task.answer_spec)

        assert self.judge is not None
        judge_prompt = _judge_prompt(self.judge_prompt, answers)
        final_answer = await context.complete(
            step="judge",
            agent=self.judge,
            messages=task_messages(
                task,
                self.judge,
                [Message(role="user", content=judge_prompt)],
            ),
            prompt_references=[
                self.judge_prompt.reference(),
                task.answer_spec.prompt_reference(),
            ],
            metadata={"aggregation": "judge"},
            track_answer=True,
            answer_kind="aggregate",
        )
        context.emit("workflow_completed", workflow=self.name)
        return final_answer

    def config(self) -> dict[str, Any]:
        return {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump() if self.judge else None,
            "parallel": self.parallel,
            "aggregation": self.aggregation,
            "voting": self.voting.model_dump(),
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge) if self.judge else None,
                self.judge_prompt if self.aggregation == "judge" else None,
            ]
        )


def _judge_prompt(
    prompt: PromptTemplate,
    answers: list[tuple[str, str]],
) -> str:
    candidates = "\n\n".join(
        f"Candidate {index + 1} ({agent_id}):\n{answer}"
        for index, (agent_id, answer) in enumerate(answers)
    )
    return prompt.render(candidates=candidates)
