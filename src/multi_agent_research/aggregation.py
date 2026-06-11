from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import random
import re
from typing import Literal

from pydantic import Field, model_validator

from multi_agent_research.models import (
    AnswerSpec,
    HarnessModel,
    TaskInput,
    WorkflowOutput,
)


AggregationMode = Literal["judge", "majority_vote", "plurality_vote"]
VALID_AGGREGATION_MODES = {"judge", "majority_vote", "plurality_vote"}


class AggregationInconclusive(Exception):
    def __init__(self, message: str, *, details: dict) -> None:
        super().__init__(message)
        self.details = details


class JudgeTieBreakRequired(Exception):
    def __init__(self, *, details: dict) -> None:
        super().__init__("Voting tie requires judge resolution")
        self.details = details


class VotingConfig(HarnessModel):
    tie_break: Literal["inconclusive", "first", "random", "judge"] = "inconclusive"
    random_seed: int = 0
    invalid_ballot_policy: Literal["exclude", "error"] = "exclude"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_error_policy(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("tie_break") == "error":
            return {**data, "tie_break": "inconclusive"}
        return data


class VoteBallot(HarnessModel):
    candidate_id: str
    raw_response: str
    answer: str
    normalized_answer: str
    contract_valid: bool
    included: bool
    validation_errors: list[str] = Field(default_factory=list)


class VoteResult(HarnessModel):
    mode: Literal["majority_vote", "plurality_vote"]
    winner: str
    winner_normalized: str
    tally: dict[str, int]
    valid_ballots: int
    total_ballots: int
    tied_answers: list[str] = Field(default_factory=list)
    tie_break_applied: str | None = None
    ballots: list[VoteBallot]

    def response(self, answer_spec: AnswerSpec) -> str:
        response = f"<final_answer>{self.winner}</final_answer>"
        if answer_spec.include_confidence:
            vote_share = 100 * self.tally[self.winner_normalized] / self.valid_ballots
            response += f"\n<confidence>{vote_share:.2f}</confidence>"
        return response


def aggregate_votes(
    *,
    task: TaskInput,
    candidates: list[tuple[str, str]],
    mode: Literal["majority_vote", "plurality_vote"],
    config: VotingConfig,
) -> VoteResult:
    ballots = prepare_vote_ballots(
        candidates=candidates,
        answer_spec=task.answer_spec,
        config=config,
    )
    included = [ballot for ballot in ballots if ballot.included]
    if not included:
        raise ValueError("Voting produced no valid ballots")

    tally = Counter(ballot.normalized_answer for ballot in included)
    top_count = max(tally.values())
    tied_answers = sorted(
        answer for answer, count in tally.items() if count == top_count
    )

    if (
        mode == "majority_vote"
        and top_count <= len(included) / 2
        and not (len(tied_answers) > 1 and config.tie_break == "judge")
    ):
        raise AggregationInconclusive(
            "No strict majority: "
            + ", ".join(
                f"{answer}={count}" for answer, count in sorted(tally.items())
            ),
            details=_inconclusive_details(
                mode=mode,
                tally=tally,
                ballots=ballots,
                included=included,
                tied_answers=tied_answers,
                reason="no_strict_majority",
            ),
        )

    tie_break_applied: str | None = None
    if len(tied_answers) == 1:
        winner_normalized = tied_answers[0]
    elif config.tie_break == "judge":
        raise JudgeTieBreakRequired(
            details=_inconclusive_details(
                mode=mode,
                tally=tally,
                ballots=ballots,
                included=included,
                tied_answers=tied_answers,
                reason="tie",
            )
        )
    elif config.tie_break == "inconclusive":
        raise AggregationInconclusive(
            f"Voting tie between: {', '.join(tied_answers)}",
            details=_inconclusive_details(
                mode=mode,
                tally=tally,
                ballots=ballots,
                included=included,
                tied_answers=tied_answers,
                reason="tie",
            ),
        )
    else:
        winner_normalized = _break_tie(
            task_id=task.id,
            tied_answers=tied_answers,
            ballots=included,
            config=config,
        )
        tie_break_applied = config.tie_break

    winner = next(
        ballot.answer
        for ballot in included
        if ballot.normalized_answer == winner_normalized
    )
    return VoteResult(
        mode=mode,
        winner=winner,
        winner_normalized=winner_normalized,
        tally=dict(sorted(tally.items())),
        valid_ballots=len(included),
        total_ballots=len(ballots),
        tied_answers=tied_answers if len(tied_answers) > 1 else [],
        tie_break_applied=tie_break_applied,
        ballots=ballots,
    )


def _inconclusive_details(
    *,
    mode: Literal["majority_vote", "plurality_vote"],
    tally: Counter[str],
    ballots: list[VoteBallot],
    included: list[VoteBallot],
    tied_answers: list[str],
    reason: Literal["no_strict_majority", "tie"],
) -> dict:
    return {
        "aggregation": mode,
        "reason": reason,
        "tally": dict(sorted(tally.items())),
        "valid_ballots": len(included),
        "total_ballots": len(ballots),
        "tied_answers": tied_answers,
        "ballots": [ballot.model_dump() for ballot in ballots],
    }


def prepare_vote_ballots(
    *,
    candidates: list[tuple[str, str]],
    answer_spec: AnswerSpec,
    config: VotingConfig,
) -> list[VoteBallot]:
    ballots: list[VoteBallot] = []
    for candidate_id, response in candidates:
        output = WorkflowOutput.from_response(response, answer_spec)
        if not output.contract_valid and config.invalid_ballot_policy == "error":
            raise ValueError(
                f"Invalid ballot from {candidate_id}: "
                + ", ".join(output.validation_errors)
            )
        ballots.append(
            VoteBallot(
                candidate_id=candidate_id,
                raw_response=response,
                answer=output.answer,
                normalized_answer=normalize_answer(output.answer, answer_spec),
                contract_valid=output.contract_valid,
                included=output.contract_valid,
                validation_errors=output.validation_errors,
            )
        )
    return ballots


def normalize_answer(answer: str, answer_spec: AnswerSpec) -> str:
    normalized = re.sub(r"\s+", " ", answer.strip())
    if answer_spec.type == "multiple_choice":
        labels = {
            choice.label.casefold(): choice.label for choice in answer_spec.choices
        }
        return labels.get(normalized.casefold(), normalized).casefold()
    if answer_spec.type == "number":
        try:
            return format(float(normalized.replace(",", "")), ".15g")
        except ValueError:
            return normalized.casefold()
    if answer_spec.type == "json":
        try:
            return json.dumps(
                json.loads(normalized),
                sort_keys=True,
                separators=(",", ":"),
            )
        except json.JSONDecodeError:
            return normalized
    return normalized.casefold()


def _break_tie(
    *,
    task_id: str,
    tied_answers: list[str],
    ballots: list[VoteBallot],
    config: VotingConfig,
) -> str:
    if config.tie_break == "first":
        return next(
            ballot.normalized_answer
            for ballot in ballots
            if ballot.normalized_answer in tied_answers
        )

    seed_material = f"{config.random_seed}:{task_id}".encode("utf-8")
    seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
    return random.Random(seed).choice(tied_answers)
