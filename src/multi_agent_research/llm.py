from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from multi_agent_research.models import (
    AgentSpec,
    Message,
    ModelCallRecord,
    PromptReference,
)


class LLMCallError(RuntimeError):
    def __init__(self, record: ModelCallRecord):
        self.record = record
        message = record.error.message if record.error else "Model call failed"
        super().__init__(message)


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        run_id: str,
        task_id: str,
        workflow: str,
        step: str,
        agent: AgentSpec,
        messages: list[Message],
        prompt_references: list[PromptReference] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallRecord: ...


def elapsed_ms(started_at: datetime, ended_at: datetime) -> float:
    return (ended_at - started_at).total_seconds() * 1000
