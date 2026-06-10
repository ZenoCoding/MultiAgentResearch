from __future__ import annotations

import asyncio
from collections import deque
from typing import Any
from uuid import uuid4

from multi_agent_research.llm import elapsed_ms
from multi_agent_research.models import (
    AgentSpec,
    Message,
    ModelCallRecord,
    PromptReference,
    UsageStats,
    utc_now,
)


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque(responses)
        self.requests: list[list[Message]] = []

    async def complete(
        self,
        *,
        sequence: int,
        run_id: str,
        task_id: str,
        workflow: str,
        step: str,
        agent: AgentSpec,
        messages: list[Message],
        prompt_references: list[PromptReference] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelCallRecord:
        if not self.responses:
            raise AssertionError("FakeLLMClient has no responses left")
        started_at = utc_now()
        output = self.responses.popleft()
        ended_at = utc_now()
        self.requests.append(messages)
        return ModelCallRecord(
            id=str(uuid4()),
            sequence=sequence,
            run_id=run_id,
            task_id=task_id,
            workflow=workflow,
            step=step,
            agent_id=agent.id,
            requested_model=agent.model,
            response_model=agent.model,
            response_service_tier=agent.service_tier,
            messages=messages,
            prompt_references=prompt_references or [],
            output=Message(role="assistant", content=output),
            usage=UsageStats(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            cost_usd=0.01,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=elapsed_ms(started_at, ended_at),
            status="success",
            metadata=metadata or {},
            raw_response={"fake": True},
        )


class DelayedFakeLLMClient(FakeLLMClient):
    def __init__(
        self,
        responses: list[str],
        delays: dict[str, float],
    ) -> None:
        super().__init__(responses)
        self.delays = delays
        self.active_calls = 0
        self.max_active_calls = 0

    async def complete(self, **kwargs) -> ModelCallRecord:
        agent = kwargs["agent"]
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delays.get(agent.id, 0))
            return await super().complete(**kwargs)
        finally:
            self.active_calls -= 1
