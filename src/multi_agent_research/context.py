from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from multi_agent_research.llm import LLMCallError, LLMClient
from multi_agent_research.models import (
    AgentSpec,
    Message,
    ModelCallRecord,
    PromptReference,
    TaskInput,
    WorkflowEvent,
)
from multi_agent_research.prompts import system_prompt_reference


@dataclass(frozen=True)
class CompletionSpec:
    step: str
    agent: AgentSpec
    messages: list[Message]
    prompt_references: list[PromptReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RunContext:
    def __init__(
        self,
        *,
        run_id: str,
        task: TaskInput,
        workflow_name: str,
        llm: LLMClient,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.workflow_name = workflow_name
        self.llm = llm
        self.calls: list[ModelCallRecord] = []
        self.events: list[WorkflowEvent] = []
        self._next_sequence = 0

    def emit(self, event_type: str, **data: Any) -> None:
        self.events.append(WorkflowEvent(type=event_type, data=data))

    async def complete(
        self,
        *,
        step: str,
        agent: AgentSpec,
        messages: list[Message],
        prompt_references: list[PromptReference] | None = None,
        metadata: dict[str, Any] | None = None,
        sequence: int | None = None,
    ) -> str:
        if sequence is None:
            sequence = self.reserve_sequence()
        references = list(prompt_references or [])
        system_reference = system_prompt_reference(agent)
        if system_reference:
            references.insert(0, system_reference)
        self.emit("model_call_started", step=step, agent_id=agent.id)
        try:
            call = await self.llm.complete(
                sequence=sequence,
                run_id=self.run_id,
                task_id=self.task.id,
                workflow=self.workflow_name,
                step=step,
                agent=agent,
                messages=messages,
                prompt_references=references,
                metadata=metadata,
            )
        except LLMCallError as exc:
            self.calls.append(exc.record)
            self.emit(
                "model_call_failed",
                step=step,
                agent_id=agent.id,
                error=exc.record.error.model_dump() if exc.record.error else None,
            )
            raise

        self.calls.append(call)
        self.emit(
            "model_call_completed",
            step=step,
            agent_id=agent.id,
            call_id=call.id,
        )
        return call.output.content if call.output else ""

    def reserve_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    async def complete_many(
        self,
        specs: list[CompletionSpec],
        *,
        parallel: bool = True,
    ) -> list[str]:
        sequences = [self.reserve_sequence() for _ in specs]
        if not parallel:
            results: list[str] = []
            for spec, sequence in zip(specs, sequences, strict=True):
                results.append(
                    await self.complete(
                        step=spec.step,
                        agent=spec.agent,
                        messages=spec.messages,
                        prompt_references=spec.prompt_references,
                        metadata=spec.metadata,
                        sequence=sequence,
                    )
                )
            return results

        coroutines = [
            self.complete(
                step=spec.step,
                agent=spec.agent,
                messages=spec.messages,
                prompt_references=spec.prompt_references,
                metadata=spec.metadata,
                sequence=sequence,
            )
            for spec, sequence in zip(specs, sequences, strict=True)
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return [result for result in results if isinstance(result, str)]


def benchmark_messages(
    task: TaskInput,
    agent: AgentSpec,
    followups: list[Message] | None = None,
) -> list[Message]:
    messages: list[Message] = []
    if agent.system_prompt:
        messages.append(Message(role="system", content=agent.system_prompt))
    messages.extend(message.model_copy(deep=True) for message in task.messages)
    messages.extend(followups or [])
    return messages


def task_messages(
    task: TaskInput,
    agent: AgentSpec,
    followups: list[Message] | None = None,
) -> list[Message]:
    messages = benchmark_messages(task, agent, followups)
    messages.append(
        Message(
            role="user",
            content=task.answer_spec.instruction(),
        )
    )
    return messages
