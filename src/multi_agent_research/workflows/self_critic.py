from __future__ import annotations

from typing import Any

from multi_agent_research.context import RunContext, task_messages
from multi_agent_research.models import AgentSpec, Message, PromptTemplate, TaskInput
from multi_agent_research.prompts import (
    SELF_CRITIC_REVISION_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow


class SelfCriticWorkflow(Workflow):
    name = "self_critic"
    version = "2.0.0"

    def __init__(
        self,
        agent: AgentSpec,
        rounds: int = 1,
        revision_prompt: PromptTemplate = SELF_CRITIC_REVISION_PROMPT,
    ) -> None:
        if rounds < 1:
            raise ValueError("Self-critic rounds must be at least 1")
        self.agent = agent
        self.rounds = rounds
        self.revision_prompt = revision_prompt

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        answer = await context.complete(
            step="initial_answer",
            agent=self.agent,
            messages=task_messages(task, self.agent),
            prompt_references=[task.answer_spec.prompt_reference()],
        )

        for round_index in range(self.rounds):
            messages = task_messages(
                task,
                self.agent,
                [
                    Message(role="assistant", content=answer),
                    Message(
                        role="user",
                        content=self.revision_prompt.render(),
                    ),
                ],
            )
            answer = await context.complete(
                step=f"revision_{round_index + 1}",
                agent=self.agent,
                messages=messages,
                prompt_references=[
                    self.revision_prompt.reference(),
                    task.answer_spec.prompt_reference(),
                ],
                metadata={"round": round_index + 1},
            )

        context.emit("workflow_completed", workflow=self.name)
        return answer

    def config(self) -> dict[str, Any]:
        return {"agent": self.agent.model_dump(), "rounds": self.rounds}

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [system_prompt_template(self.agent), self.revision_prompt]
        )
