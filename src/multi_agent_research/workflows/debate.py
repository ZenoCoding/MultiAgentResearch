from __future__ import annotations

from typing import Any, Literal

from multi_agent_research.aggregation import (
    aggregate_votes,
    AggregationMode,
    JudgeTieBreakRequired,
    VALID_AGGREGATION_MODES,
    VotingConfig,
)
from multi_agent_research.context import CompletionSpec, RunContext, task_messages
from multi_agent_research.models import (
    AgentSpec,
    Message,
    PromptTemplate,
    TaskInput,
    WorkflowOutput,
)
from multi_agent_research.prompts import (
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    TIE_BREAK_JUDGE_PROMPT,
    system_prompt_template,
    unique_prompts,
)
from multi_agent_research.workflows.base import Workflow
from multi_agent_research.workflows.sample import _judge_prompt, _judge_vote_tie


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


class DebateWorkflow(Workflow):
    name = "debate"
    version = "2.5.0"

    def __init__(
        self,
        agents: list[AgentSpec],
        judge: AgentSpec | None = None,
        rounds: int = 1,
        debate_prompt: PromptTemplate = DEBATE_REVIEW_PROMPT,
        judge_prompt: PromptTemplate = JUDGE_SELECTION_PROMPT,
        tie_break_judge_prompt: PromptTemplate = TIE_BREAK_JUDGE_PROMPT,
        parallel: bool = True,
        aggregation: AggregationMode = "judge",
        voting: VotingConfig | None = None,
        peer_view: PeerView = "full_response",
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
        self.agents = agents
        self.judge = judge
        self.rounds = rounds
        self.debate_prompt = debate_prompt
        self.judge_prompt = judge_prompt
        self.tie_break_judge_prompt = tie_break_judge_prompt
        self.parallel = parallel
        self.aggregation = aggregation
        self.voting = voting or VotingConfig()
        self.peer_view = peer_view

    async def run(self, task: TaskInput, context: RunContext) -> str:
        context.emit("workflow_started", workflow=self.name)
        initial_responses = await context.complete_many(
            [
                CompletionSpec(
                    step=f"initial_{index}",
                    agent=agent,
                    messages=task_messages(task, agent),
                    prompt_references=[task.answer_spec.prompt_reference()],
                    metadata={"phase": "initial", "agent_index": index},
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
            round_specs: list[CompletionSpec] = []
            for index, agent in enumerate(self.agents):
                peer_text = self._peer_text(task, previous_answers, agent.id)
                messages = task_messages(
                    task,
                    agent,
                    [
                        Message(
                            role="assistant",
                            content=previous_answers[agent.id],
                        ),
                        Message(
                            role="user",
                            content=self.debate_prompt.render(peer_answers=peer_text),
                        ),
                    ],
                )
                round_specs.append(
                    CompletionSpec(
                        step=f"debate_{round_index + 1}_{index}",
                        agent=agent,
                        messages=messages,
                        prompt_references=[
                            self.debate_prompt.reference(),
                            task.answer_spec.prompt_reference(),
                        ],
                        metadata={
                            "phase": "debate",
                            "round": round_index + 1,
                            "agent_index": index,
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
        return {
            "agents": [agent.model_dump() for agent in self.agents],
            "judge": self.judge.model_dump() if self.judge else None,
            "rounds": self.rounds,
            "parallel": self.parallel,
            "aggregation": self.aggregation,
            "voting": self.voting.model_dump(),
            "peer_view": self.peer_view,
        }

    def prompt_templates(self) -> list[PromptTemplate]:
        return unique_prompts(
            [
                *(system_prompt_template(agent) for agent in self.agents),
                system_prompt_template(self.judge) if self.judge else None,
                self.debate_prompt,
                self.judge_prompt if self.aggregation == "judge" else None,
                (
                    self.tie_break_judge_prompt
                    if self.voting.tie_break == "judge"
                    else None
                ),
            ]
        )

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
