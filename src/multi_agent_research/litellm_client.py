from __future__ import annotations

import json
import traceback
from typing import Any
from urllib.parse import urlsplit, urlunsplit
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
    DISALLOWED_HIDDEN_RELIABILITY_PARAMS = {
        "fallbacks",
        "context_window_fallback_dict",
        "context_window_fallbacks",
    }

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
        request_parameters = self._request_parameters(agent)

        try:
            response = await litellm.acompletion(
                model=agent.model,
                messages=request_messages,
                **request_parameters,
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
                request_parameters=request_parameters,
                response_model=getattr(response, "model", None),
                response_service_tier=self._extract_service_tier(response, raw),
                provider_metadata=self._provider_metadata(response),
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
                request_parameters=request_parameters,
                messages=messages,
                prompt_references=prompt_references or [],
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=elapsed_ms(started_at, ended_at),
                status="failed",
                error=CallError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    details=self._error_details(exc),
                ),
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

    @staticmethod
    def _error_details(exc: Exception) -> dict[str, Any]:
        details = {}
        for name in ("status_code", "code", "param", "type", "body"):
            value = getattr(exc, name, None)
            if value is None:
                continue
            try:
                json.dumps(value)
                details[name] = value
            except (TypeError, ValueError):
                details[name] = str(value)
        return details

    @classmethod
    def _request_parameters(cls, agent: AgentSpec) -> dict[str, Any]:
        parameters = agent.completion_parameters()
        hidden = cls.DISALLOWED_HIDDEN_RELIABILITY_PARAMS.intersection(parameters)
        if hidden:
            raise ValueError(
                "Hidden LiteLLM fallbacks are not allowed in auditable runs: "
                + ", ".join(sorted(hidden))
            )
        for name in ("num_retries", "max_retries"):
            value = parameters.get(name, 0)
            if value not in (None, 0):
                raise ValueError(
                    f"{name} must be 0; retry through the experiment orchestrator"
                )
            parameters[name] = 0
        return parameters

    @staticmethod
    def _provider_metadata(response: Any) -> dict[str, Any]:
        hidden = getattr(response, "_hidden_params", {}) or {}
        allowed = {
            "api_base",
            "custom_llm_provider",
            "litellm_call_id",
            "model_id",
            "region_name",
            "response_cost",
            "service_tier",
        }
        metadata = {}
        for name in allowed:
            value = hidden.get(name)
            if value is None:
                continue
            if name == "api_base":
                value = LiteLLMClient._sanitized_url(str(value))
            try:
                json.dumps(value)
                metadata[name] = value
            except (TypeError, ValueError):
                metadata[name] = str(value)
        return metadata

    @staticmethod
    def _sanitized_url(value: str) -> str:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if parsed.port is not None:
            hostname += f":{parsed.port}"
        return urlunsplit(
            (
                parsed.scheme,
                hostname,
                parsed.path,
                "",
                "",
            )
        )
