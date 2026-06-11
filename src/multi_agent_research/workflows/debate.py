from __future__ import annotations

from typing import Any, Literal

from multi_agent_research.aggregation import (
    aggregate_votes,
    AggregationMode,
    JudgeTieBreakRequired,
    normalize_answer,
    VALID_AGGREGATION_MODES,
    VotingConfig,
)
from multi_agent_research.context import CompletionSpec, RunContext, task_messages
from multi_agent_research.models import (
    AgentSpec,
    Message,
    PromptReference,
    PromptTemplate,
    TaskInput,
    WorkflowOutput,
)
from multi_agent_research.prompts import (
    DEBATE_ADVERSARIAL_CHALLENGE_PROMPT,
    DEBATE_ADVERSARIAL_RESOLUTION_PROMPT,
    DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT,
    DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
    DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
    DEBATE_DERIVATION_ROLE_PROMPT,
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    SHORT_ANSWER_SEMANTIC_VOTE_PROMPT,
    TIE_BREAK_JUDGE_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow
from multi_agent_research.workflows.sample import (
    _judge_prompt,
    _judge_semantic_vote,
    _judge_vote_tie,
)


PeerView = Literal[
    "full_response",
    "answer_only",
    "answer_and_confidence",
]
VALID_PEER_VIEWS = {
    "full_response",
    "answer_only",
    "answer_and_confidence",
}
DebateMode = Literal["standard", "adversarial"]
VALID_DEBATE_MODES = {"standard", "adversarial"}


class DebateWorkflow(Workflow):
    name = "debate"
    version = "2.7.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec | None = None,
        rounds: int = 1,
        debate_prompt: PromptTemplate = DEBATE_REVIEW_PROMPT,
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
        semantic_vote_prompt: PromptTemplate = (
            SHORT_ANSWER_SEMANTIC_VOTE_PROMPT
        ),
        tie_break_judge_prompt: PromptTemplate = TIE_BREAK_JUDGE_PROMPT,
        parallel: bool = True,
        aggregation: AggregationMode = "judge",
        voting: VotingConfig | None = None,
        peer_view: PeerView = "full_response",
        mode: DebateMode = "standard",
        adversarial_role_prompts: tuple[PromptTemplate, ...] = (
            DEBATE_DERIVATION_ROLE_PROMPT,
            DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT,
            DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT,
        ),
        adversarial_challenge_prompt: PromptTemplate = (
            DEBATE_ADVERSARIAL_CHALLENGE_PROMPT
        ),
        adversarial_unanimous_prompt: PromptTemplate = (
            DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT
        ),
        adversarial_resolution_prompt: PromptTemplate = (
            DEBATE_ADVERSARIAL_RESOLUTION_PROMPT
        ),
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Debate requires at least two agents")
        if rounds < 1:
            raise ValueError("Debate rounds must be at least 1")
        if aggregation not in VALID_AGGREGATION_MODES:
            raise ValueError(f"Unsupported aggregation mode: {aggregation}")
        if (
            aggregation == "judge"
            or (voting is not None and voting.tie_break == "judge")
        ) and judge is None:
            raise ValueError("Judge aggregation or tie-breaking requires a judge")
        if peer_view not in VALID_PEER_VIEWS:
            raise ValueError(f"Unsupported peer view: {peer_view}")
        if mode not in VALID_DEBATE_MODES:
            raise ValueError(f"Unsupported debate mode: {mode}")
        if mode == "adversarial" and not adversarial_role_prompts:
            raise ValueError("Adversarial debate requires at least one role prompt")
        self.agents = agents
        self.judge = judge
        self.rounds = rounds
        self.debate_prompt = debate_prompt
        self.judge_prompt = judge_prompt
        self.semantic_vote_prompt = semantic_vote_prompt
        self.tie_break_judge_prompt = tie_break_judge_prompt
        self.parallel = parallel
        self.aggregation = aggregation
        self.voting = voting or VotingConfig()
        self.peer_view = peer_view
        self.mode = mode
        self.adversarial_role_prompts = adversarial_role_prompts
        self.adversarial_challenge_prompt = adversarial_challenge_prompt
        self.adversarial_unanimous_prompt = adversarial_unanimous_prompt
        self.adversarial_resolution_prompt = adversarial_resolution_prompt

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        initial_responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"initial_{index}",
                    agent=agent,
                    messages=task_messages(
                        task,
                        agent,
                        self._role_messages(index),
                    ),
                    prompt_references=[
                        *self._role_references(index),
                        task.answer_spec.prompt_reference(),
                    ],
                    metadata={
                        "phase": "initial",
                        "agent_index": index,
                        **self._role_metadata(index),
                    },
                    track_answer=True,
                )
                for index, agent in enumerate(self.agents)
            ],
            parallel=self.parallel,
        )
        answers = {
            agent.id: response
            for agent, response in zip(
                self.agents,
                initial_responses,
                strict=True,
            )
        }

        for round_index in range(self.rounds):
            previous_answers = dict(answers)
            round_prompt, round_strategy, answer_tally = self._round_prompt(
                task,
                previous_answers,
                round_index,
            )
            if self.mode == "adversarial":
                context.emit(
                    "debate_round_strategy",
                    round=round_index + 1,
                    mode=self.mode,
                    strategy=round_strategy,
                    prompt_name=round_prompt.name,
                    answer_tally=answer_tally,
                )
            round_specs: list[CompletionSpec] = []
            for index, agent in enumerate(self.agents):
                peer_text = self._peer_text(task, previous_answers, agent.id)
                followups = [
                    *self._role_messages(index),
                    Message(
                        role="assistant",
                        content=previous_answers[agent.id],
                    ),
                    Message(
                        role="user",
                        content=round_prompt.render(peer_answers=peer_text),
                    ),
                ]
                messages = task_messages(
                    task,
                    agent,
                    followups,
                )
                round_specs.append(
                    CompletionSpec(
                        step=f"debate_{round_index + 1}_{index}",
                        agent=agent,
                        messages=messages,
                        prompt_references=[
                            round_prompt.reference(),
                            *self._role_references(index),
                            task.answer_spec.prompt_reference(),
                        ],
                        metadata={
                            "phase": "debate",
                            "round": round_index + 1,
                            "agent_index": index,
                            "strategy": round_strategy,
                            **self._role_metadata(index),
                        },
                        track_answer=True,
                    )
                )
            round_responses = await context.complete_many(
                round_specs,
                parallel=self.parallel,
            )
            answers = {
                agent.id: response
                for agent, response in zip(
                    self.agents,
                    round_responses,
                    strict=True,
                )
            }

        if self.aggregation != "judge":
            if task.answer_spec.type == "short_answer":
                return await _judge_semantic_vote(
                    task=task,
                    context=context,
                    judge=self.judge,
                    prompt=self.semantic_vote_prompt,
                    candidates=list(answers.items()),
                    mode=self.aggregation,
                    voting=self.voting,
                )
            try:
                vote = aggregate_votes(
                    task=task,
                    candidates=list(answers.items()),
                    mode=self.aggregation,
                    config=self.voting,
                )
            except JudgeTieBreakRequired as tie:
                return await _judge_vote_tie(
                    task=task,
                    context=context,
                    judge=self.judge,
                    prompt=self.tie_break_judge_prompt,
                    details=tie.details,
                )
            context.emit("votes_aggregated", **vote.model_dump())
            context.record_stage_answer(
                step="aggregation",
                response=vote.response(task.answer_spec),
                kind="aggregate",
                metadata={"aggregation": self.aggregation},
            )
            context.emit("workflow_completed", workflow=self.name)
            return vote.response(task.answer_spec)

        assert self.judge is not None
        final_answer = await context.complete(
            step="judge",
            agent=self.judge,
            messages=task_messages(
                task,
                self.judge,
                [
                    Message(
                        role="user",
                        content=_judge_prompt(
                            self.judge_prompt,
                            list(answers.items()),
                        ),
                    ),
                ],
            ),
            prompt_references=[
                self.judge_prompt.reference(),
                task.answer_spec.prompt_reference(),
            ],
            metadata={"phase": "judge", "aggregation": "judge"},
            track_answer=True,
            answer_kind="aggregate",
        )
        context.emit("workflow_completed", workflow=self.name)
        return final_answer

    def config(self) -> dict[str, Any]:
        config = {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump() if self.judge else None,
            "rounds": self.rounds,
            "parallel": self.parallel,
            "aggregation": self.aggregation,
            "voting": self.voting.model_dump(),
            "peer_view": self.peer_view,
        }
        if self.mode == "adversarial":
            config.update(
                {
                    "mode": self.mode,
                    "adversarial_roles": [
                    self._role_prompt(index).name
                    for index in range(len(self.agents))
                    ],
                }
            )
        return config

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge) if self.judge else None,
                (
                    self.debate_prompt
                    if self.mode == "standard"
                    else None
                ),
                *(
                    self.adversarial_role_prompts
                    if self.mode == "adversarial"
                    else ()
                ),
                (
                    self.adversarial_challenge_prompt
                    if self.mode == "adversarial"
                    else None
                ),
                (
                    self.adversarial_unanimous_prompt
                    if self.mode == "adversarial"
                    else None
                ),
                (
                    self.adversarial_resolution_prompt
                    if self.mode == "adversarial"
                    else None
                ),
                self.judge_prompt if self.aggregation == "judge" else None,
                (
                    self.semantic_vote_prompt
                    if self.aggregation != "judge"
                    else None
                ),
                (
                    self.tie_break_judge_prompt
                    if self.voting.tie_break == "judge"
                    else None
                ),
            ]
        )

    def _role_prompt(self, agent_index: int) -> PromptTemplate:
        return self.adversarial_role_prompts[
            agent_index % len(self.adversarial_role_prompts)
        ]

    def _role_messages(self, agent_index: int) -> list[Message]:
        if self.mode != "adversarial":
            return []
        return [
            Message(
                role="user",
                content=self._role_prompt(agent_index).template,
            )
        ]

    def _role_references(self, agent_index: int) -> list[PromptReference]:
        if self.mode != "adversarial":
            return []
        return [self._role_prompt(agent_index).reference()]

    def _role_metadata(self, agent_index: int) -> dict[str, str]:
        if self.mode != "adversarial":
            return {}
        return {"debate_role": self._role_prompt(agent_index).name}

    def _round_prompt(
        self,
        task: TaskInput,
        answers: dict[str, str],
        round_index: int,
    ) -> tuple[PromptTemplate, str, dict[str, int]]:
        if self.mode == "standard":
            return self.debate_prompt, "peer_review", {}

        answer_tally = self._answer_tally(task, answers)
        if round_index == 0 and len(answer_tally) == 1:
            return (
                self.adversarial_unanimous_prompt,
                "unanimous_challenge",
                answer_tally,
            )
        if round_index == 0:
            return (
                self.adversarial_challenge_prompt,
                "adversarial_challenge",
                answer_tally,
            )
        return (
            self.adversarial_resolution_prompt,
            "evidence_resolution",
            answer_tally,
        )

    @staticmethod
    def _answer_tally(
        task: TaskInput,
        answers: dict[str, str],
    ) -> dict[str, int]:
        tally: dict[str, int] = {}
        for response in answers.values():
            output = WorkflowOutput.from_response(response, task.answer_spec)
            if not output.contract_valid:
                continue
            normalized = normalize_answer(output.answer, task.answer_spec)
            tally[normalized] = tally.get(normalized, 0) + 1
        return dict(sorted(tally.items()))

    def _peer_text(
        self,
        task: TaskInput,
        answers: dict[str, str],
        current_agent_id: str,
    ) -> str:
        peers: list[str] = []
        for peer_id, response in answers.items():
            if peer_id == current_agent_id:
                continue
            if self.peer_view == "full_response":
                visible = response
            else:
                output = WorkflowOutput.from_response(response, task.answer_spec)
                visible = f"Final answer: {output.answer}"
                if (
                    self.peer_view == "answer_and_confidence"
                    and output.confidence is not None
                ):
                    visible += f"\nConfidence: {output.confidence:g}"
            peers.append(f"{peer_id}:\n{visible}")
        return "\n\n".join(peers)


class AdversarialDebateWorkflow(DebateWorkflow):
    name = "adversarial_debate"
    version = "1.2.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["mode"] = "adversarial"
        super().__init__(*args, **kwargs)
