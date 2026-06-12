from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkChoice:
    label: str
    text: str | None = None


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    prompt: str
    answer: str
    answer_type: str = "multiple_choice"
    choices: tuple[BenchmarkChoice, ...] = ()
    category: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    public_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    id: str
    workflow: str
    agents: int = 1
    rounds: int = 1
    aggregation: str = "judge"
    debate_peer_view: str = "full_response"
    vote_tie_break: str = "inconclusive"
    include_confidence: bool = False
    reasoning_effort: str | None = None
    judge_reasoning_effort: str | None = None
    supervisor_reasoning_effort: str | None = None
    service_tier: str | None = None
    judge_service_tier: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    sequential: bool = False
    cross_exam_claim_max_tokens: int = 240
    cross_exam_challenge_max_tokens: int = 120
    cross_exam_response_max_tokens: int = 160
    cross_exam_verdict_max_tokens: int = 80
