from __future__ import annotations

from types import SimpleNamespace

import pytest

from multi_agent_research.llm import LLMCallError
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import (
    AgentSpec,
    ImageContent,
    ImageURL,
    Message,
    TextContent,
)


class FakeResponse:
    id = "call-1"
    model = "provider/model-version"
    service_tier = "flex"
    choices = [SimpleNamespace(message=SimpleNamespace(content="normalized answer"))]
    _hidden_params = {"response_cost": 0.0123}

    def model_dump(self):
        return {
            "id": self.id,
            "model": self.model,
            "choices": [{"message": {"content": "normalized answer"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }


@pytest.mark.asyncio
async def test_litellm_response_is_normalized(monkeypatch):
    async def fake_completion(**kwargs):
        assert kwargs["model"] == "provider/model"
        return FakeResponse()

    monkeypatch.setattr(
        "multi_agent_research.litellm_client.litellm.acompletion",
        fake_completion,
    )

    record = await LiteLLMClient().complete(
        sequence=0,
        run_id="run-1",
        task_id="task-1",
        workflow="solo",
        step="answer",
        agent=AgentSpec(id="agent-1", model="provider/model"),
        messages=[Message(role="user", content="Question")],
    )

    assert record.output.content == "normalized answer"
    assert record.response_model == "provider/model-version"
    assert record.response_service_tier == "flex"
    assert record.usage.input_tokens == 12
    assert record.usage.output_tokens == 8
    assert record.usage.reasoning_tokens == 4
    assert record.usage.cached_input_tokens == 3
    assert record.cost_usd == pytest.approx(0.0123)
    assert record.request_parameters == {
        "num_retries": 0,
        "max_retries": 0,
    }
    assert record.provider_metadata["response_cost"] == pytest.approx(0.0123)


@pytest.mark.asyncio
async def test_litellm_receives_multimodal_message_parts(monkeypatch):
    captured = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        "multi_agent_research.litellm_client.litellm.acompletion",
        fake_completion,
    )
    message = Message(
        role="user",
        content=[
            TextContent(text="Inspect the image"),
            ImageContent(image_url=ImageURL(url="https://example.com/image.png")),
        ],
    )

    await LiteLLMClient().complete(
        sequence=0,
        run_id="run-1",
        task_id="task-1",
        workflow="solo",
        step="answer",
        agent=AgentSpec(id="agent-1", model="provider/model"),
        messages=[message],
    )

    assert captured["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Inspect the image",
    }
    assert captured["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.png",
            "detail": "auto",
        },
    }


@pytest.mark.asyncio
async def test_reasoning_effort_is_sent_as_first_class_agent_setting(monkeypatch):
    captured = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        "multi_agent_research.litellm_client.litellm.acompletion",
        fake_completion,
    )

    record = await LiteLLMClient().complete(
        sequence=0,
        run_id="run-1",
        task_id="task-1",
        workflow="solo",
        step="answer",
        agent=AgentSpec(
            id="agent-1",
            model="provider/model",
            reasoning_effort="high",
        ),
        messages=[Message(role="user", content="Question")],
    )

    assert captured["reasoning_effort"] == "high"
    assert record.request_parameters == {
        "reasoning_effort": "high",
        "num_retries": 0,
        "max_retries": 0,
    }


def test_legacy_nested_reasoning_effort_is_normalized():
    agent = AgentSpec(
        id="agent-1",
        model="provider/model",
        parameters={"reasoning_effort": "medium", "temperature": 0.5},
    )

    assert agent.reasoning_effort == "medium"
    assert agent.parameters == {"temperature": 0.5}
    assert agent.completion_parameters() == {
        "temperature": 0.5,
        "reasoning_effort": "medium",
    }


@pytest.mark.asyncio
async def test_service_tier_is_sent_as_first_class_agent_setting(monkeypatch):
    captured = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        "multi_agent_research.litellm_client.litellm.acompletion",
        fake_completion,
    )

    record = await LiteLLMClient().complete(
        sequence=0,
        run_id="run-1",
        task_id="task-1",
        workflow="solo",
        step="answer",
        agent=AgentSpec(
            id="agent-1",
            model="provider/model",
            service_tier="priority",
        ),
        messages=[Message(role="user", content="Question")],
    )

    assert captured["service_tier"] == "priority"
    assert record.response_service_tier == "flex"


def test_legacy_nested_service_tier_is_normalized():
    agent = AgentSpec(
        id="agent-1",
        model="provider/model",
        parameters={"service_tier": "flex", "temperature": 0.5},
    )

    assert agent.service_tier == "flex"
    assert agent.parameters == {"temperature": 0.5}
    assert agent.completion_parameters() == {
        "temperature": 0.5,
        "service_tier": "flex",
    }


@pytest.mark.asyncio
async def test_litellm_failure_records_request_and_provider_details(monkeypatch):
    class ProviderError(RuntimeError):
        status_code = 429
        code = "rate_limit"
        body = {"message": "slow down"}

    async def fake_completion(**kwargs):
        raise ProviderError("request failed")

    monkeypatch.setattr(
        "multi_agent_research.litellm_client.litellm.acompletion",
        fake_completion,
    )

    with pytest.raises(LLMCallError) as caught:
        await LiteLLMClient().complete(
            sequence=0,
            run_id="run-1",
            task_id="task-1",
            workflow="solo",
            step="answer",
            agent=AgentSpec(
                id="agent-1",
                model="provider/model",
                parameters={"temperature": 0.2},
            ),
            messages=[Message(role="user", content="Question")],
        )

    record = caught.value.record
    assert record.request_parameters == {
        "temperature": 0.2,
        "num_retries": 0,
        "max_retries": 0,
    }
    assert record.error.details == {
        "status_code": 429,
        "code": "rate_limit",
        "body": {"message": "slow down"},
    }
    assert "ProviderError" in record.error.traceback


def test_hidden_litellm_retries_and_fallbacks_are_rejected():
    with pytest.raises(ValueError, match="num_retries must be 0"):
        LiteLLMClient._request_parameters(
            AgentSpec(
                id="agent-1",
                model="provider/model",
                parameters={"num_retries": 2},
            )
        )

    with pytest.raises(ValueError, match="Hidden LiteLLM fallbacks"):
        LiteLLMClient._request_parameters(
            AgentSpec(
                id="agent-1",
                model="provider/model",
                parameters={"fallbacks": ["provider/other-model"]},
            )
        )
