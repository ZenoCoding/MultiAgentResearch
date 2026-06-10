from __future__ import annotations

from typing import Any

from multi_agent_research.context import RunContext, task_messages
from multi_agent_research.models import AgentSpec, Message, PromptTemplate, TaskInput
from multi_agent_research.prompts import (
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow
from multi_agent_research.workflows.sample import _judge_prompt


class DebateWorkflow(Workflow):
    name = "debate"
    version = "2.0.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec,
        rounds: int = 1,
        debate_prompt: PromptTemplate = DEBATE_REVIEW_PROMPT,
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Debate requires at least two agents")
        if rounds < 1:
            raise ValueError("Debate rounds must be at least 1")
        self.agents = agents
        self.judge = judge
        self.rounds = rounds
        self.debate_prompt = debate_prompt
        self.judge_prompt = judge_prompt

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        answers: dict[str, str] = {}

        for index, agent in enumerate(self.agents):
            answers[agent.id] = await context.complete(
                step=f"initial_{index}",
                agent=agent,
                messages=task_messages(task, agent),
                prompt_references=[task.answer_spec.prompt_reference()],
                metadata={"phase": "initial", "agent_index": index},
            )

        for round_index in range(self.rounds):
            previous_answers = dict(answers)
            for index, agent in enumerate(self.agents):
                peer_text = "\n\n".join(
                    f"{peer_id}:\n{answer}"
                    for peer_id, answer in previous_answers.items()
                    if peer_id != agent.id
                )
                messages = task_messages(
                    task,
                    agent,
                    [
                        Message(
                            role="assistant",
                            content=previous_answers[agent.id],
                        ),
                        Message(
                            role="user",
                            content=self.debate_prompt.render(peer_answers=peer_text),
                        ),
                    ],
                )
                answers[agent.id] = await context.complete(
                    step=f"debate_{round_index + 1}_{index}",
                    agent=agent,
                    messages=messages,
                    prompt_references=[
                        self.debate_prompt.reference(),
                        task.answer_spec.prompt_reference(),
                    ],
                    metadata={
                        "phase": "debate",
                        "round": round_index + 1,
                        "agent_index": index,
                    },
                )

        final_answer = await context.complete(
            step="judge",
            agent=self.judge,
            messages=task_messages(
                task,
                self.judge,
                [
                    Message(
                        role="user",
                        content=_judge_prompt(
                            self.judge_prompt,
                            list(answers.items()),
                        ),
                    ),
                ],
            ),
            prompt_references=[
                self.judge_prompt.reference(),
                task.answer_spec.prompt_reference(),
            ],
            metadata={"phase": "judge"},
        )
        context.emit("workflow_completed", workflow=self.name)
        return final_answer

    def config(self) -> dict[str, Any]:
        return {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump(),
            "rounds": self.rounds,
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge),
                self.debate_prompt,
                self.judge_prompt,
            ]
        )
