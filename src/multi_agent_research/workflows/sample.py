from __future__ import annotations

from typing import Any

from multi_agent_research.context import RunContext, task_messages
from multi_agent_research.models import AgentSpec, Message, PromptTemplate, TaskInput
from multi_agent_research.prompts import (
    JUDGE_SELECTION_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow


class IndependentSampleWorkflow(Workflow):
    name = "independent_sample"
    version = "2.0.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec,
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
    ) -> None:
        if not agents:
            raise ValueError("Independent sampling requires at least one agent")
        self.agents = agents
        self.judge = judge
        self.judge_prompt = judge_prompt

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        answers: list[tuple[str, str]] = []
        for index, agent in enumerate(self.agents):
            answer = await context.complete(
                step=f"sample_{index}",
                agent=agent,
                messages=task_messages(task, agent),
                prompt_references=[task.answer_spec.prompt_reference()],
                metadata={"sample_index": index},
            )
            answers.append((agent.id, answer))

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
        )
        context.emit("workflow_completed", workflow=self.name)
        return final_answer

    def config(self) -> dict[str, Any]:
        return {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump(),
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge),
                self.judge_prompt,
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
