from __future__ import annotations

from typing import Any
from uuid import uuid4

import litellm

from multi_agent_research.llm import LLMCallError, elapsed_ms
from multi_agent_research.models import (
    AgentSpec,
    CallError,
    Message,
    ModelCallRecord,
    PromptReference,
    UsageStats,
    utc_now,
)


class LiteLLMClient:
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
        started_at = utc_now()
        request_messages = [
            message.model_dump(exclude_none=True) for message in messages
        ]

        try:
            response = await litellm.acompletion(
                model=agent.model,
                messages=request_messages,
                **agent.completion_parameters(),
            )
            ended_at = utc_now()
            raw = self._to_dict(response)
            output_text = self._extract_output(response)

            return ModelCallRecord(
                id=str(getattr(response, "id", None) or uuid4()),
                sequence=sequence,
                run_id=run_id,
                task_id=task_id,
                workflow=workflow,
                step=step,
                agent_id=agent.id,
                requested_model=agent.model,
                response_model=getattr(response, "model", None),
                response_service_tier=self._extract_service_tier(response, raw),
                messages=messages,
                prompt_references=prompt_references or [],
                output=Message(role="assistant", content=output_text),
                usage=self._extract_usage(raw),
                cost_usd=self._extract_cost(response),
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=elapsed_ms(started_at, ended_at),
                status="success",
                metadata=metadata or {},
                raw_response=raw,
            )
        except Exception as exc:
            ended_at = utc_now()
            record = ModelCallRecord(
                sequence=sequence,
                run_id=run_id,
                task_id=task_id,
                workflow=workflow,
                step=step,
                agent_id=agent.id,
                requested_model=agent.model,
                messages=messages,
                prompt_references=prompt_references or [],
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=elapsed_ms(started_at, ended_at),
                status="failed",
                error=CallError(type=type(exc).__name__, message=str(exc)),
                metadata=metadata or {},
            )
            raise LLMCallError(record) from exc

    @staticmethod
    def _to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return dict(response)

    @staticmethod
    def _extract_output(response: Any) -> str:
        content = response.choices[0].message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _extract_usage(raw: dict[str, Any]) -> UsageStats:
        usage = raw.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return UsageStats(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            cached_input_tokens=prompt_details.get("cached_tokens"),
        )

    @staticmethod
    def _extract_cost(response: Any) -> float | None:
        hidden = getattr(response, "_hidden_params", {}) or {}
        response_cost = hidden.get("response_cost")
        if response_cost is not None:
            return float(response_cost)
        try:
            return float(litellm.completion_cost(completion_response=response))
        except Exception:
            return None

    @staticmethod
    def _extract_service_tier(
        response: Any,
        raw: dict[str, Any],
    ) -> str | None:
        tier = getattr(response, "service_tier", None)
        if tier is None:
            tier = raw.get("service_tier")
        if tier is None:
            hidden = getattr(response, "_hidden_params", {}) or {}
            tier = hidden.get("service_tier")
        return str(tier) if tier is not None else None
