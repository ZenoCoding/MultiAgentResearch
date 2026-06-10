from __future__ import annotations

import json

import pytest

from multi_agent_research.prompt_overrides import load_prompt_overrides
from multi_agent_research.prompt_overrides import overridden
from multi_agent_research.prompts import DEBATE_REVIEW_PROMPT


def test_prompt_overrides_load_versioned_templates(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "workflow.debate.peer_review": {
                    "version": "2.0.0",
                    "template": "Compare these answers:\n$peer_answers",
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_prompt_overrides(path)
    prompt = overrides["workflow.debate.peer_review"]

    assert prompt.version == "2.0.0"
    assert prompt.render(peer_answers="A and B") == ("Compare these answers:\nA and B")


def test_prompt_override_must_preserve_required_variables(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "workflow.debate.peer_review": {
                    "version": "2.0.0",
                    "template": "Compare the answers.",
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_prompt_overrides(path)

    with pytest.raises(ValueError, match="peer_answers"):
        overridden(DEBATE_REVIEW_PROMPT, overrides)
