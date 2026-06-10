from __future__ import annotations

from typing import Any

from multi_agent_research.context import benchmark_messages, RunContext, task_messages
from multi_agent_research.models import AgentSpec, Message, PromptTemplate, TaskInput
from multi_agent_research.prompts import (
    SUPERVISOR_REVIEW_PROMPT,
    WORKER_REVISION_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow


class SupervisorWorkflow(Workflow):
    name = "supervisor"
    version = "2.0.0"

    def __init__(
        self,
        worker: AgentSpec,
        supervisor: AgentSpec,
        max_revisions: int = 2,
        review_prompt: PromptTemplate = SUPERVISOR_REVIEW_PROMPT,
        revision_prompt: PromptTemplate = WORKER_REVISION_PROMPT,
    ) -> None:
        if max_revisions < 1:
            raise ValueError("Supervisor max_revisions must be at least 1")
        self.worker = worker
        self.supervisor = supervisor
        self.max_revisions = max_revisions
        self.review_prompt = review_prompt
        self.revision_prompt = revision_prompt

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        answer = await context.complete(
            step="worker_initial",
            agent=self.worker,
            messages=task_messages(task, self.worker),
            prompt_references=[task.answer_spec.prompt_reference()],
        )

        for revision in range(self.max_revisions):
            review = await context.complete(
                step=f"supervisor_review_{revision + 1}",
                agent=self.supervisor,
                messages=benchmark_messages(
                    task,
                    self.supervisor,
                    [
                        Message(
                            role="user",
                            content=self.review_prompt.render(
                                answer=answer,
                            ),
                        ),
                    ],
                ),
                prompt_references=[self.review_prompt.reference()],
                metadata={"revision": revision + 1},
            )
            if review.lstrip().upper().startswith("APPROVE"):
                context.emit(
                    "supervisor_approved",
                    revision=revision,
                )
                break

            messages = task_messages(
                task,
                self.worker,
                [
                    Message(role="assistant", content=answer),
                    Message(
                        role="user",
                        content=self.revision_prompt.render(feedback=review),
                    ),
                ],
            )
            answer = await context.complete(
                step=f"worker_revision_{revision + 1}",
                agent=self.worker,
                messages=messages,
                prompt_references=[
                    self.revision_prompt.reference(),
                    task.answer_spec.prompt_reference(),
                ],
                metadata={"revision": revision + 1},
            )

        context.emit("workflow_completed", workflow=self.name)
        return answer

    def config(self) -> dict[str, Any]:
        return {
            "worker": self.worker.model_dump(),
            "supervisor": self.supervisor.model_dump(),
            "max_revisions": self.max_revisions,
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                system_prompt_template(self.worker),
                system_prompt_template(self.supervisor),
                self.review_prompt,
                self.revision_prompt,
            ]
        )
