from __future__ import annotations

from typing import Any

from multi_agent_research.aggregation import (
    aggregate_votes,
    AggregationMode,
    JudgeTieBreakRequired,
    VALID_AGGREGATION_MODES,
    VotingConfig,
)
from multi_agent_research.context import (
    benchmark_messages,
    CompletionSpec,
    RunContext,
    task_messages,
)
from multi_agent_research.models import (
    AgentSpec,
    Message,
    PromptTemplate,
    TaskInput,
)
from multi_agent_research.prompts import (
    CROSS_EXAMINATION_CHALLENGE_PROMPT,
    CROSS_EXAMINATION_CLAIM_PROMPT,
    CROSS_EXAMINATION_FINAL_REVISION_PROMPT,
    CROSS_EXAMINATION_RESPONSE_PROMPT,
    CROSS_EXAMINATION_VERDICT_PROMPT,
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


class CrossExaminationDebateWorkflow(Workflow):
    name = "cross_examination_debate"
    version = "1.1.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec | None = None,
        rounds: int = 1,
        claim_prompt: PromptTemplate = CROSS_EXAMINATION_CLAIM_PROMPT,
        challenge_prompt: PromptTemplate = CROSS_EXAMINATION_CHALLENGE_PROMPT,
        response_prompt: PromptTemplate = CROSS_EXAMINATION_RESPONSE_PROMPT,
        verdict_prompt: PromptTemplate = CROSS_EXAMINATION_VERDICT_PROMPT,
        final_revision_prompt: PromptTemplate = (
            CROSS_EXAMINATION_FINAL_REVISION_PROMPT
        ),
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
        semantic_vote_prompt: PromptTemplate = (SHORT_ANSWER_SEMANTIC_VOTE_PROMPT),
        tie_break_judge_prompt: PromptTemplate = TIE_BREAK_JUDGE_PROMPT,
        parallel: bool = True,
        aggregation: AggregationMode = "judge",
        voting: VotingConfig | None = None,
        claim_max_tokens: int = 240,
        challenge_max_tokens: int = 120,
        response_max_tokens: int = 160,
        verdict_max_tokens: int = 80,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Cross-examination debate requires at least two agents")
        if rounds < 1:
            raise ValueError("Cross-examination rounds must be at least 1")
        if aggregation not in VALID_AGGREGATION_MODES:
            raise ValueError(f"Unsupported aggregation mode: {aggregation}")
        if (
            aggregation == "judge"
            or (voting is not None and voting.tie_break == "judge")
        ) and judge is None:
            raise ValueError("Judge aggregation or tie-breaking requires a judge")
        caps = {
            "claim_max_tokens": claim_max_tokens,
            "challenge_max_tokens": challenge_max_tokens,
            "response_max_tokens": response_max_tokens,
            "verdict_max_tokens": verdict_max_tokens,
        }
        if any(value < 1 for value in caps.values()):
            raise ValueError("Cross-examination token caps must be positive")

        self.agents = agents
        self.judge = judge
        self.rounds = rounds
        self.claim_prompt = claim_prompt
        self.challenge_prompt = challenge_prompt
        self.response_prompt = response_prompt
        self.verdict_prompt = verdict_prompt
        self.final_revision_prompt = final_revision_prompt
        self.judge_prompt = judge_prompt
        self.semantic_vote_prompt = semantic_vote_prompt
        self.tie_break_judge_prompt = tie_break_judge_prompt
        self.parallel = parallel
        self.aggregation = aggregation
        self.voting = voting or VotingConfig()
        self.claim_max_tokens = claim_max_tokens
        self.challenge_max_tokens = challenge_max_tokens
        self.response_max_tokens = response_max_tokens
        self.verdict_max_tokens = verdict_max_tokens

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        initial_responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"initial_{index}",
                    agent=agent,
                    messages=task_messages(task, agent),
                    prompt_references=[task.answer_spec.prompt_reference()],
                    metadata={
                        "phase": "initial",
                        "phase_id": "initial",
                        "interaction_id": f"initial:{agent.id}",
                        "agent_index": index,
                        "depends_on_call_ids": [],
                        "visible_call_ids": [],
                    },
                    track_answer=True,
                )
                for index, agent in enumerate(self.agents)
            ],
            parallel=self.parallel,
        )
        initial = dict(
            zip(
                (agent.id for agent in self.agents),
                initial_responses,
                strict=True,
            )
        )
        initial_call_ids = self._call_ids_by_step(
            context,
            [f"initial_{index}" for index in range(len(self.agents))],
        )

        claim_responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"claims_{index}",
                    agent=self._capped_agent(agent, self.claim_max_tokens),
                    messages=benchmark_messages(
                        task,
                        agent,
                        [
                            Message(role="assistant", content=initial[agent.id]),
                            Message(
                                role="user",
                                content=(
                                    self.claim_prompt.template
                                    + f"\n\nYour response should be under {self.claim_max_tokens} tokens."
                                ),
                            ),
                        ],
                    ),
                    prompt_references=[self.claim_prompt.reference()],
                    metadata={
                        "phase": "claim_extraction",
                        "phase_id": "claim_extraction",
                        "interaction_id": f"claims:{agent.id}",
                        "agent_index": index,
                        "max_tokens": self.claim_max_tokens,
                        "depends_on_call_ids": [initial_call_ids[f"initial_{index}"]],
                        "visible_call_ids": [initial_call_ids[f"initial_{index}"]],
                    },
                )
                for index, agent in enumerate(self.agents)
            ],
            parallel=self.parallel,
        )
        claims = dict(
            zip(
                (agent.id for agent in self.agents),
                claim_responses,
                strict=True,
            )
        )
        claim_call_ids = self._call_ids_by_step(
            context,
            [f"claims_{index}" for index in range(len(self.agents))],
        )

        transcripts: dict[str, list[str]] = {agent.id: [] for agent in self.agents}
        all_exchanges: list[str] = []
        agent_count = len(self.agents)
        for round_index in range(self.rounds):
            prior_visible_call_ids = {
                agent.id: self._visible_exchange_call_ids(context, agent.id)
                for agent in self.agents
            }
            exchange_specs: list[tuple[int, AgentSpec, AgentSpec]] = []
            target_offset = (round_index % (agent_count - 1)) + 1
            for challenger_index, challenger in enumerate(self.agents):
                target_index = (challenger_index + target_offset) % agent_count
                target = self.agents[target_index]
                exchange_specs.append((challenger_index, challenger, target))

            challenge_responses = await context.complete_many(
                [
                    CompletionSpec(
                        step=f"cross_exam_{round_index + 1}_challenge_{index}",
                        agent=self._capped_agent(
                            challenger,
                            self.challenge_max_tokens,
                        ),
                        messages=benchmark_messages(
                            task,
                            challenger,
                            [
                                Message(
                                    role="assistant",
                                    content=(
                                        "My initial solution:\n"
                                        f"{initial[challenger.id]}\n\n"
                                        "My claim dossier:\n"
                                        f"{claims[challenger.id]}"
                                    ),
                                ),
                                Message(
                                    role="user",
                                    content=(
                                        self.challenge_prompt.render(
                                            target_agent=target.id,
                                            target_initial=initial[target.id],
                                            target_claims=claims[target.id],
                                            prior_transcript=self._transcript(
                                                transcripts[challenger.id]
                                            ),
                                        )
                                        + f"\n\nYour response should be under {self.challenge_max_tokens} tokens."
                                    ),
                                ),
                            ],
                        ),
                        prompt_references=[self.challenge_prompt.reference()],
                        metadata={
                            "phase": "challenge",
                            "phase_id": f"cross_examination_round_{round_index + 1}",
                            "interaction_id": (
                                f"cross_exam:{round_index + 1}:{index}"
                            ),
                            "round": round_index + 1,
                            "exchange_index": index,
                            "challenger_id": challenger.id,
                            "target_id": target.id,
                            "max_tokens": self.challenge_max_tokens,
                            "depends_on_call_ids": [
                                initial_call_ids[
                                    f"initial_{self.agents.index(challenger)}"
                                ],
                                claim_call_ids[
                                    f"claims_{self.agents.index(challenger)}"
                                ],
                                initial_call_ids[
                                    f"initial_{self.agents.index(target)}"
                                ],
                                claim_call_ids[
                                    f"claims_{self.agents.index(target)}"
                                ],
                            ],
                            "visible_call_ids": [
                                initial_call_ids[
                                    f"initial_{self.agents.index(challenger)}"
                                ],
                                claim_call_ids[
                                    f"claims_{self.agents.index(challenger)}"
                                ],
                                initial_call_ids[
                                    f"initial_{self.agents.index(target)}"
                                ],
                                claim_call_ids[
                                    f"claims_{self.agents.index(target)}"
                                ],
                                *prior_visible_call_ids[challenger.id],
                            ],
                        },
                    )
                    for index, (_, challenger, target) in enumerate(exchange_specs)
                ],
                parallel=self.parallel,
            )
            challenge_call_ids = self._call_ids_by_step(
                context,
                [
                    f"cross_exam_{round_index + 1}_challenge_{index}"
                    for index in range(len(exchange_specs))
                ],
            )

            response_responses = await context.complete_many(
                [
                    CompletionSpec(
                        step=f"cross_exam_{round_index + 1}_response_{index}",
                        agent=self._capped_agent(
                            target,
                            self.response_max_tokens,
                        ),
                        messages=benchmark_messages(
                            task,
                            target,
                            [
                                Message(
                                    role="assistant",
                                    content=(
                                        "My initial solution:\n"
                                        f"{initial[target.id]}\n\n"
                                        "My claim dossier:\n"
                                        f"{claims[target.id]}"
                                    ),
                                ),
                                Message(
                                    role="user",
                                    content=(
                                        self.response_prompt.render(
                                            challenger_agent=challenger.id,
                                            challenge=challenge_responses[index],
                                            prior_transcript=self._transcript(
                                                transcripts[target.id]
                                            ),
                                        )
                                        + f"\n\nYour response should be under {self.response_max_tokens} tokens."
                                    ),
                                ),
                            ],
                        ),
                        prompt_references=[self.response_prompt.reference()],
                        metadata={
                            "phase": "response",
                            "phase_id": f"cross_examination_round_{round_index + 1}",
                            "interaction_id": (
                                f"cross_exam:{round_index + 1}:{index}"
                            ),
                            "round": round_index + 1,
                            "exchange_index": index,
                            "challenger_id": challenger.id,
                            "target_id": target.id,
                            "max_tokens": self.response_max_tokens,
                            "depends_on_call_ids": [
                                challenge_call_ids[
                                    f"cross_exam_{round_index + 1}_challenge_{index}"
                                ]
                            ],
                            "visible_call_ids": [
                                initial_call_ids[
                                    f"initial_{self.agents.index(target)}"
                                ],
                                claim_call_ids[
                                    f"claims_{self.agents.index(target)}"
                                ],
                                *prior_visible_call_ids[target.id],
                                challenge_call_ids[
                                    f"cross_exam_{round_index + 1}_challenge_{index}"
                                ],
                            ],
                        },
                    )
                    for index, (_, challenger, target) in enumerate(exchange_specs)
                ],
                parallel=self.parallel,
            )
            response_call_ids = self._call_ids_by_step(
                context,
                [
                    f"cross_exam_{round_index + 1}_response_{index}"
                    for index in range(len(exchange_specs))
                ],
            )

            verdict_responses = await context.complete_many(
                [
                    CompletionSpec(
                        step=f"cross_exam_{round_index + 1}_verdict_{index}",
                        agent=self._capped_agent(
                            challenger,
                            self.verdict_max_tokens,
                        ),
                        messages=benchmark_messages(
                            task,
                            challenger,
                            [
                                Message(
                                    role="user",
                                    content=(
                                        self.verdict_prompt.render(
                                            target_agent=target.id,
                                            challenge=challenge_responses[index],
                                            response=response_responses[index],
                                        )
                                        + f"\n\nYour response should be under {self.verdict_max_tokens} tokens."
                                    ),
                                )
                            ],
                        ),
                        prompt_references=[self.verdict_prompt.reference()],
                        metadata={
                            "phase": "verdict",
                            "phase_id": f"cross_examination_round_{round_index + 1}",
                            "interaction_id": (
                                f"cross_exam:{round_index + 1}:{index}"
                            ),
                            "round": round_index + 1,
                            "exchange_index": index,
                            "challenger_id": challenger.id,
                            "target_id": target.id,
                            "max_tokens": self.verdict_max_tokens,
                            "depends_on_call_ids": [
                                challenge_call_ids[
                                    f"cross_exam_{round_index + 1}_challenge_{index}"
                                ],
                                response_call_ids[
                                    f"cross_exam_{round_index + 1}_response_{index}"
                                ],
                            ],
                            "visible_call_ids": [
                                challenge_call_ids[
                                    f"cross_exam_{round_index + 1}_challenge_{index}"
                                ],
                                response_call_ids[
                                    f"cross_exam_{round_index + 1}_response_{index}"
                                ],
                            ],
                        },
                    )
                    for index, (_, challenger, target) in enumerate(exchange_specs)
                ],
                parallel=self.parallel,
            )
            verdict_call_ids = self._call_ids_by_step(
                context,
                [
                    f"cross_exam_{round_index + 1}_verdict_{index}"
                    for index in range(len(exchange_specs))
                ],
            )

            for index, (_, challenger, target) in enumerate(exchange_specs):
                exchange = self._format_exchange(
                    round_index=round_index,
                    challenger_id=challenger.id,
                    target_id=target.id,
                    challenge=challenge_responses[index],
                    response=response_responses[index],
                    verdict=verdict_responses[index],
                )
                transcripts[challenger.id].append(exchange)
                transcripts[target.id].append(exchange)
                all_exchanges.append(exchange)
                context.emit(
                    "cross_examination_exchange",
                    round=round_index + 1,
                    exchange_index=index,
                    challenger_id=challenger.id,
                    target_id=target.id,
                    challenge=challenge_responses[index],
                    response=response_responses[index],
                    verdict=verdict_responses[index],
                    challenge_call_id=challenge_call_ids[
                        f"cross_exam_{round_index + 1}_challenge_{index}"
                    ],
                    response_call_id=response_call_ids[
                        f"cross_exam_{round_index + 1}_response_{index}"
                    ],
                    verdict_call_id=verdict_call_ids[
                        f"cross_exam_{round_index + 1}_verdict_{index}"
                    ],
                )

        final_responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"final_revision_{index}",
                    agent=agent,
                    messages=task_messages(
                        task,
                        agent,
                        [
                            Message(
                                role="user",
                                content=self.final_revision_prompt.render(
                                    initial_response=initial[agent.id],
                                    claims=claims[agent.id],
                                    transcript=self._transcript(transcripts[agent.id]),
                                ),
                            )
                        ],
                    ),
                    prompt_references=[
                        self.final_revision_prompt.reference(),
                        task.answer_spec.prompt_reference(),
                    ],
                    metadata={
                        "phase": "final_revision",
                        "phase_id": "final_revision",
                        "interaction_id": f"final_revision:{agent.id}",
                        "agent_index": index,
                        "exchange_count": len(transcripts[agent.id]),
                        "depends_on_call_ids": [
                            initial_call_ids[f"initial_{index}"],
                            claim_call_ids[f"claims_{index}"],
                            *self._visible_exchange_call_ids(context, agent.id),
                        ],
                        "visible_call_ids": [
                            initial_call_ids[f"initial_{index}"],
                            claim_call_ids[f"claims_{index}"],
                            *self._visible_exchange_call_ids(context, agent.id),
                        ],
                    },
                    track_answer=True,
                )
                for index, agent in enumerate(self.agents)
            ],
            parallel=self.parallel,
        )
        final_answers = list(
            zip(
                (agent.id for agent in self.agents),
                final_responses,
                strict=True,
            )
        )
        context.emit(
            "cross_examination_completed",
            rounds=self.rounds,
            exchanges=len(all_exchanges),
        )
        return await self._aggregate(task, context, final_answers)

    async def _aggregate(
        self,
        task: TaskInput,
        context: RunContext,
        answers: list[tuple[str, str]],
    ) -> str:
        if self.aggregation != "judge":
            if task.answer_spec.type == "short_answer":
                return await _judge_semantic_vote(
                    task=task,
                    context=context,
                    judge=self.judge,
                    prompt=self.semantic_vote_prompt,
                    candidates=answers,
                    mode=self.aggregation,
                    voting=self.voting,
                )
            try:
                vote = aggregate_votes(
                    task=task,
                    candidates=answers,
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
            response = vote.response(task.answer_spec)
            context.record_stage_answer(
                step="aggregation",
                response=response,
                kind="aggregate",
                metadata={"aggregation": self.aggregation},
            )
            context.emit("workflow_completed", workflow=self.name)
            return response

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
                        content=_judge_prompt(self.judge_prompt, answers),
                    )
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
        return {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump() if self.judge else None,
            "rounds": self.rounds,
            "routing": "rotating_ring",
            "parallel": self.parallel,
            "aggregation": self.aggregation,
            "voting": self.voting.model_dump(),
            "phase_max_tokens": {
                "claims": self.claim_max_tokens,
                "challenge": self.challenge_max_tokens,
                "response": self.response_max_tokens,
                "verdict": self.verdict_max_tokens,
            },
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge) if self.judge else None,
                self.claim_prompt,
                self.challenge_prompt,
                self.response_prompt,
                self.verdict_prompt,
                self.final_revision_prompt,
                self.judge_prompt if self.aggregation == "judge" else None,
                (self.semantic_vote_prompt if self.aggregation != "judge" else None),
                (
                    self.tie_break_judge_prompt
                    if self.voting.tie_break == "judge"
                    else None
                ),
            ]
        )

    @staticmethod
    def _transcript(exchanges: list[str]) -> str:
        return "\n\n".join(exchanges) if exchanges else "(none)"

    @staticmethod
    def _format_exchange(
        *,
        round_index: int,
        challenger_id: str,
        target_id: str,
        challenge: str,
        response: str,
        verdict: str,
    ) -> str:
        return (
            f"Round {round_index + 1}: {challenger_id} -> {target_id}\n"
            f"Challenge: {challenge}\n"
            f"Response: {response}\n"
            f"Verdict: {verdict}"
        )

    @staticmethod
    def _capped_agent(agent: AgentSpec, max_tokens: int) -> AgentSpec:
        # We no longer pass the transient phase limit as a hard max_tokens limit 
        # to the API request parameter, to ensure reasoning models have the 
        # necessary token budget to perform internal reasoning. We only preserve 
        # the user-configured max_tokens if explicitly set in parameters.
        parameters = dict(agent.parameters)
        configured = parameters.get("max_tokens")
        if isinstance(configured, int):
            parameters["max_tokens"] = configured
        else:
            parameters.pop("max_tokens", None)
        return agent.model_copy(update={"parameters": parameters})

    @staticmethod
    def _call_ids_by_step(
        context: RunContext,
        steps: list[str],
    ) -> dict[str, str]:
        wanted = set(steps)
        call_ids = {
            call.step: call.id
            for call in context.calls
            if call.step in wanted
        }
        missing = wanted - call_ids.keys()
        if missing:
            raise RuntimeError(
                "Missing recorded calls for steps: " + ", ".join(sorted(missing))
            )
        return call_ids

    @staticmethod
    def _visible_exchange_call_ids(
        context: RunContext,
        agent_id: str,
    ) -> list[str]:
        visible: list[str] = []
        for call in context.calls:
            metadata = call.metadata
            if metadata.get("phase") not in {"challenge", "response", "verdict"}:
                continue
            if agent_id in {
                metadata.get("challenger_id"),
                metadata.get("target_id"),
            }:
                visible.append(call.id)
        return visible
