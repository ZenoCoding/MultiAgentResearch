from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        run_id="run-1",
        task_id="task-1",
        workflow="solo",
        step="answer",
        agent=AgentSpec(id="agent-1", model="provider/model"),
        messages=[Message(role="user", content="Question")],
    )

    assert record.output.content == "normalized answer"
    assert record.response_model == "provider/model-version"
    assert record.usage.input_tokens == 12
    assert record.usage.output_tokens == 8
    assert record.usage.reasoning_tokens == 4
    assert record.usage.cached_input_tokens == 3
    assert record.cost_usd == pytest.approx(0.0123)


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
