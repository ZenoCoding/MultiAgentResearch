from __future__ import annotations

from typing import Any

from multi_agent_research.context import RunContext, task_messages
from multi_agent_research.models import AgentSpec, PromptTemplate, TaskInput
from multi_agent_research.prompts import system_prompt_template, unique_prompts
from multi_agent_research.workflows.base import Workflow


class SoloWorkflow(Workflow):
    name = "solo"
    version = "2.0.0"

    def __init__(self, agent: AgentSpec) -> None:
        self.agent = agent

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        answer = await context.complete(
            step="answer",
            agent=self.agent,
            messages=task_messages(task, self.agent),
            prompt_references=[task.answer_spec.prompt_reference()],
            track_answer=True,
        )
        context.emit("workflow_completed", workflow=self.name)
        return answer

    def config(self) -> dict[str, Any]:
        return {"agent": self.agent.model_dump()}

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts([system_prompt_template(self.agent)])
