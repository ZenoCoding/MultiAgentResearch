from __future__ import annotations

from hashlib import sha256
import json
import tarfile

import pytest

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.models import (
    AgentSpec,
    AnswerChoice,
    AnswerSpec,
    ImageContent,
    ImageURL,
    Message,
    PromptTemplate,
    TaskInput,
    TextContent,
    WorkflowOutput,
)
from multi_agent_research.runner import ExperimentRunner
from multi_agent_research.storage import FileRunStore
from multi_agent_research.workflows import (
    AdversarialDebateWorkflow,
    CrossExaminationDebateWorkflow,
    DebateWorkflow,
    IndependentSampleWorkflow,
    SelfCriticWorkflow,
    SoloWorkflow,
    SupervisorWorkflow,
)
from tests.fakes import DelayedFakeLLMClient, FakeLLMClient


def agent(agent_id: str) -> AgentSpec:
    return AgentSpec(id=agent_id, model="fake/model")


@pytest.mark.asyncio
async def test_solo_run_is_standardized_and_persisted(tmp_path):
    llm = FakeLLMClient(["final answer"])
    runner = ExperimentRunner(llm=llm, store=FileRunStore(tmp_path))

    result = await runner.run(
        task=TaskInput.from_prompt(id="task-1", prompt="Solve this"),
        workflow=SoloWorkflow(agent("solo")),
        experiment_id="experiment-1",
    )

    assert result.status == "success"
    assert result.final_answer == "final answer"
    assert result.metrics.model_calls == 1
    assert result.metrics.total_tokens == 15
    assert result.metrics.cost_usd == pytest.approx(0.01)
    assert [stage.output.answer for stage in result.stage_answers] == ["final answer"]

    run_dir = tmp_path / "experiment-1" / result.run_id
    saved_result = json.loads((run_dir / "result.json").read_text())
    assert saved_result["final_answer"] == "final answer"
    assert saved_result["stage_answers"][0]["step"] == "answer"
    provenance = json.loads((run_dir / "provenance.json").read_text())
    artifact_manifest = json.loads(
        (run_dir / "artifact-manifest.json").read_text()
    )
    source_reference = json.loads(
        (run_dir / "source-reference.json").read_text()
    )
    source_path = tmp_path / source_reference["path"]
    snapshot = source_path.read_bytes()
    assert provenance["source_snapshot_sha256"] == sha256(snapshot).hexdigest()
    assert source_reference["sha256"] == sha256(snapshot).hexdigest()
    assert artifact_manifest["source-reference.json"]["sha256"]
    assert provenance["git"]["commit"]
    assert provenance["python_version"]
    assert provenance["dependency_versions"]["pydantic"]
    with tarfile.open(source_path, "r:gz") as archive:
        names = archive.getnames()
    assert "src/multi_agent_research/runner.py" in names
    assert not any(name == ".env" or name.startswith("results/") for name in names)
    assert len((run_dir / "calls.jsonl").read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_source_snapshot_is_cached_across_runs(tmp_path):
    runner = ExperimentRunner(
        llm=FakeLLMClient(["first", "second"]),
        store=FileRunStore(tmp_path),
    )

    first = await runner.run(
        task=TaskInput.from_prompt(id="task-1", prompt="First"),
        workflow=SoloWorkflow(agent("solo")),
        experiment_id="experiment-1",
    )
    second = await runner.run(
        task=TaskInput.from_prompt(id="task-2", prompt="Second"),
        workflow=SoloWorkflow(agent("solo")),
        experiment_id="experiment-1",
    )

    source_files = list((tmp_path / "_artifacts" / "sources").glob("*.tar.gz"))
    assert len(source_files) == 1
    for run_id in (first.run_id, second.run_id):
        run_dir = tmp_path / "experiment-1" / run_id
        reference = json.loads((run_dir / "source-reference.json").read_text())
        assert reference["path"] == source_files[0].relative_to(tmp_path).as_posix()
        assert not (run_dir / "source.tar.gz").exists()


@pytest.mark.asyncio
async def test_independent_samples_are_judged():
    llm = FakeLLMClient(["answer a", "answer b", "answer b"])
    workflow = IndependentSampleWorkflow(
        [agent("a"), agent("b")],
        agent("judge"),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=workflow,
        experiment_id="experiment",
    )

    assert result.final_answer == "answer b"
    assert [call.step for call in result.calls] == [
        "sample_0",
        "sample_1",
        "judge",
    ]
    assert [stage.step for stage in result.stage_answers] == [
        "sample_0",
        "sample_1",
        "judge",
    ]
    assert [stage.kind for stage in result.stage_answers] == [
        "candidate",
        "candidate",
        "aggregate",
    ]
    assert result.workflow.version == "2.6.0"
    assert result.workflow.fingerprint
    assert result.calls[-1].prompt_references[0].name == "workflow.judge.selection"


@pytest.mark.asyncio
async def test_independent_samples_run_in_parallel_with_stable_call_order():
    llm = DelayedFakeLLMClient(
        ["fast sample", "slow sample", "judged answer"],
        delays={"a": 0.05, "b": 0.01},
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
        ),
        experiment_id="experiment",
    )

    assert llm.max_active_calls == 2
    assert result.final_answer == "judged answer"
    assert [call.step for call in result.calls] == [
        "sample_0",
        "sample_1",
        "judge",
    ]
    assert [call.sequence for call in result.calls] == [0, 1, 2]


@pytest.mark.asyncio
async def test_short_answer_sampling_plurality_uses_semantic_vote_judge():
    llm = FakeLLMClient(
        [
            "First candidate\n<final_answer>NYC</final_answer>",
            "Second candidate\n<final_answer>New York City</final_answer>",
            "Third candidate\n<final_answer>Boston</final_answer>",
            "<vote_status>winner</vote_status>\n"
            "<final_answer>New York City</final_answer>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            agent("judge"),
            aggregation="plurality_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "New York City"
    assert result.calls[-1].step == "semantic_vote_judge"
    assert result.calls[-1].metadata == {
        "aggregation": "plurality_vote",
        "judge_objective": "semantic_vote",
        "candidate_count": 3,
        "total_ballots": 3,
        "tie_break": "inconclusive",
    }
    assert result.calls[-1].prompt_references[0].name == (
        "workflow.vote.short_answer_semantic"
    )
    judge_instruction = result.calls[-1].messages[-2].content
    assert isinstance(judge_instruction, str)
    assert "Group final answers that are semantically equivalent" in (
        judge_instruction
    )
    assert "<final_answer>NYC</final_answer>" in judge_instruction
    assert "<final_answer>New York City</final_answer>" in judge_instruction


@pytest.mark.asyncio
async def test_short_answer_majority_vote_can_be_inconclusive():
    llm = FakeLLMClient(
        [
            "<final_answer>NYC</final_answer>",
            "<final_answer>Boston</final_answer>",
            "<final_answer>Chicago</final_answer>",
            "<vote_status>inconclusive</vote_status>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            agent("judge"),
            aggregation="majority_vote",
        ),
        experiment_id="experiment",
    )

    assert result.status == "inconclusive"
    assert result.final_answer is None
    assert result.inconclusive is not None
    assert result.inconclusive.details["reason"] == "no_strict_majority"
    assert result.calls[-1].step == "semantic_vote_judge"


@pytest.mark.asyncio
async def test_short_answer_semantic_vote_rejects_random_tie_break():
    result = await ExperimentRunner(
        llm=FakeLLMClient(
            [
                "<final_answer>NYC</final_answer>",
                "<final_answer>Boston</final_answer>",
            ]
        )
    ).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="random"),
        ),
        experiment_id="experiment",
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "does not support random tie-breaking" in result.error.message


@pytest.mark.asyncio
async def test_short_answer_judge_aggregation_keeps_best_answer_behavior():
    llm = FakeLLMClient(
        [
            "<final_answer>NYC</final_answer>",
            "<final_answer>New York City</final_answer>",
            "<final_answer>Boston</final_answer>",
            "<final_answer>Boston</final_answer>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            agent("judge"),
            aggregation="judge",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "Boston"
    assert result.calls[-1].step == "judge"
    assert result.calls[-1].prompt_references[0].name == "workflow.judge.selection"


@pytest.mark.asyncio
async def test_short_answer_semantic_vote_excludes_invalid_ballots():
    llm = FakeLLMClient(
        [
            "<final_answer>NYC</final_answer>",
            "missing answer tag",
            "<final_answer>New York City</final_answer>",
            "<vote_status>winner</vote_status>\n"
            "<final_answer>NYC</final_answer>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            agent("judge"),
            aggregation="majority_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "NYC"
    assert result.calls[-1].metadata["candidate_count"] == 2
    assert result.calls[-1].metadata["total_ballots"] == 3
    semantic_vote_prompt = result.calls[-1].messages[-2].content
    assert isinstance(semantic_vote_prompt, str)
    assert "missing answer tag" not in semantic_vote_prompt


@pytest.mark.asyncio
async def test_multiple_choice_sampling_keeps_best_answer_judge():
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput(
            id="task-1",
            messages=[Message(role="user", content="Choose A or B")],
            answer_spec=AnswerSpec(
                type="multiple_choice",
                choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
            ),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
        ),
        experiment_id="experiment",
    )

    assert result.calls[-1].step == "judge"
    assert result.calls[-1].prompt_references[0].name == "workflow.judge.selection"


@pytest.mark.asyncio
async def test_parallel_phases_can_be_disabled():
    llm = DelayedFakeLLMClient(
        ["sample a", "sample b", "judged answer"],
        delays={"a": 0.01, "b": 0.01},
    )

    await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
            parallel=False,
        ),
        experiment_id="experiment",
    )

    assert llm.max_active_calls == 1


@pytest.mark.asyncio
async def test_self_critic_revises_answer():
    llm = FakeLLMClient(["draft", "revision one", "revision two"])

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=SelfCriticWorkflow(agent("a"), rounds=2),
        experiment_id="experiment",
    )

    assert result.final_answer == "revision two"
    assert result.metrics.model_calls == 3
    assert [stage.output.answer for stage in result.stage_answers] == [
        "draft",
        "revision one",
        "revision two",
    ]


@pytest.mark.asyncio
async def test_debate_uses_peer_answers_then_judges():
    llm = FakeLLMClient(
        ["initial a", "initial b", "revised a", "revised b", "judged answer"]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=DebateWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
            rounds=1,
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "judged answer"
    assert result.metrics.model_calls == 5
    assert [stage.step for stage in result.stage_answers] == [
        "initial_0",
        "initial_1",
        "debate_1_0",
        "debate_1_1",
        "judge",
    ]
    assert result.stage_answers[-1].kind == "aggregate"
    assert "initial b" in llm.requests[2][-2].content
    assert "initial a" in llm.requests[3][-2].content
    assert result.calls[2].prompt_references[0].name == ("workflow.debate.peer_review")
    assert result.calls[-1].prompt_references[0].name == "workflow.judge.selection"


@pytest.mark.asyncio
async def test_debate_initial_answers_and_each_round_run_in_parallel():
    llm = DelayedFakeLLMClient(
        [
            "fast initial",
            "slow initial",
            "fast revision",
            "slow revision",
            "judged answer",
        ],
        delays={"a": 0.05, "b": 0.01},
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=DebateWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
            rounds=1,
        ),
        experiment_id="experiment",
    )

    assert llm.max_active_calls == 2
    assert result.final_answer == "judged answer"
    assert [call.step for call in result.calls] == [
        "initial_0",
        "initial_1",
        "debate_1_0",
        "debate_1_1",
        "judge",
    ]
    assert [call.sequence for call in result.calls] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_short_answer_debate_plurality_uses_semantic_vote_judge():
    llm = FakeLLMClient(
        [
            "<final_answer>NYC</final_answer>",
            "<final_answer>New York City</final_answer>",
            "<final_answer>Boston</final_answer>",
            "<final_answer>NYC</final_answer>",
            "<final_answer>New York City</final_answer>",
            "<final_answer>Boston</final_answer>",
            "<vote_status>winner</vote_status>\n"
            "<final_answer>New York City</final_answer>",
        ]
    )
    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(
            id="task-1",
            prompt="Name the city.",
            answer_spec=AnswerSpec(type="short_answer"),
        ),
        workflow=DebateWorkflow(
            [agent("a"), agent("b"), agent("c")],
            agent("judge"),
            rounds=1,
            aggregation="plurality_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "New York City"
    assert result.calls[-1].step == "semantic_vote_judge"
    assert result.calls[-1].prompt_references[0].name == (
        "workflow.vote.short_answer_semantic"
    )
    assert result.calls[-1].metadata["judge_objective"] == "semantic_vote"


@pytest.mark.asyncio
async def test_adversarial_debate_diversifies_and_challenges_unanimity():
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=AdversarialDebateWorkflow(
            [agent("a"), agent("b"), agent("c")],
            rounds=2,
            aggregation="plurality_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "B"
    assert result.workflow.name == "adversarial_debate"
    assert result.workflow.version == "1.2.0"
    assert result.workflow.config["mode"] == "adversarial"
    assert result.workflow.config["adversarial_roles"] == [
        "workflow.debate.role.derivation",
        "workflow.debate.role.assumption_auditor",
        "workflow.debate.role.alternative_method",
    ]
    assert [
        result.calls[index].prompt_references[0].name for index in range(3)
    ] == result.workflow.config["adversarial_roles"]
    assert {
        result.calls[index].messages[-2].content for index in range(3)
    } == {
        prompt.template
        for prompt in result.workflow.prompts
        if prompt.name.startswith("workflow.debate.role.")
    }
    assert all(
        result.calls[index].prompt_references[0].name
        == "workflow.debate.adversarial.unanimous_challenge"
        for index in range(3, 6)
    )
    assert all(
        result.calls[index].prompt_references[0].name
        == "workflow.debate.adversarial.resolution"
        for index in range(6, 9)
    )
    strategy_events = [
        event for event in result.events if event.type == "debate_round_strategy"
    ]
    assert [event.data["strategy"] for event in strategy_events] == [
        "unanimous_challenge",
        "evidence_resolution",
    ]
    assert strategy_events[0].data["answer_tally"] == {"a": 3}


def test_adversarial_debate_has_separate_workflow_identity():
    agents = [agent("a"), agent("b"), agent("c")]

    standard = DebateWorkflow(agents, aggregation="plurality_vote")
    adversarial = AdversarialDebateWorkflow(
        agents,
        aggregation="plurality_vote",
    )

    assert standard.spec().name == "debate"
    assert adversarial.spec().name == "adversarial_debate"
    assert "mode" not in standard.config()
    assert adversarial.config()["mode"] == "adversarial"
    assert standard.spec().fingerprint != adversarial.spec().fingerprint


@pytest.mark.asyncio
async def test_cross_examination_uses_directed_short_exchanges_then_revises():
    llm = FakeLLMClient(
        [
            "Initial A\n<final_answer>A</final_answer>",
            "Initial B\n<final_answer>B</final_answer>",
            "Initial C\n<final_answer>C</final_answer>",
            "<claim id=\"C1\">A1</claim>",
            "<claim id=\"C1\">B1</claim>",
            "<claim id=\"C1\">C1</claim>",
            "A challenges B1",
            "B challenges C1",
            "C challenges A1",
            "B answers A",
            "C answers B",
            "A answers C",
            "UNRESOLVED B did not justify B1.",
            "RESOLVED C justified C1.",
            "CONCEDED A withdrew A1.",
            "Final A\n<final_answer>A</final_answer>",
            "Final B\n<final_answer>B</final_answer>",
            "Final B\n<final_answer>B</final_answer>",
        ]
    )
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A, B, or C")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[
                AnswerChoice(label="A"),
                AnswerChoice(label="B"),
                AnswerChoice(label="C"),
            ],
        ),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=CrossExaminationDebateWorkflow(
            [agent("a"), agent("b"), agent("c")],
            rounds=1,
            aggregation="plurality_vote",
        ),
        experiment_id="experiment",
    )

    assert result.status == "success"
    assert result.final_answer == "B"
    assert result.workflow.name == "cross_examination_debate"
    assert result.workflow.version == "1.1.0"
    assert result.workflow.config["routing"] == "rotating_ring"
    assert result.metrics.model_calls == 18
    assert [call.step for call in result.calls] == [
        "initial_0",
        "initial_1",
        "initial_2",
        "claims_0",
        "claims_1",
        "claims_2",
        "cross_exam_1_challenge_0",
        "cross_exam_1_challenge_1",
        "cross_exam_1_challenge_2",
        "cross_exam_1_response_0",
        "cross_exam_1_response_1",
        "cross_exam_1_response_2",
        "cross_exam_1_verdict_0",
        "cross_exam_1_verdict_1",
        "cross_exam_1_verdict_2",
        "final_revision_0",
        "final_revision_1",
        "final_revision_2",
    ]
    assert [stage.step for stage in result.stage_answers] == [
        "initial_0",
        "initial_1",
        "initial_2",
        "final_revision_0",
        "final_revision_1",
        "final_revision_2",
        "aggregation",
    ]

    challenge = result.calls[6]
    assert challenge.agent_id == "a"
    assert challenge.metadata["target_id"] == "b"
    assert "max_tokens" not in challenge.request_parameters
    challenge_text = challenge.messages[-1].content
    assert isinstance(challenge_text, str)
    assert "Cross-examine b" in challenge_text
    assert "Initial B" in challenge_text
    assert "Your response should be under 120 tokens." in challenge_text
    challenger_context = challenge.messages[-2].content
    assert isinstance(challenger_context, str)
    assert "Initial A" in challenger_context
    assert "A1" in challenger_context

    response = result.calls[9]
    assert response.agent_id == "b"
    assert response.metadata["challenger_id"] == "a"
    assert "max_tokens" not in response.request_parameters
    assert "Your response should be under 160 tokens." in response.messages[-1].content
    assert "max_tokens" not in result.calls[12].request_parameters
    assert "Your response should be under 80 tokens." in result.calls[12].messages[-1].content

    final_a = result.calls[15].messages[-2].content
    assert isinstance(final_a, str)
    assert "A challenges B1" in final_a
    assert "C challenges A1" in final_a
    exchange_events = [
        event
        for event in result.events
        if event.type == "cross_examination_exchange"
    ]
    assert len(exchange_events) == 3
    assert exchange_events[0].data["challenge_call_id"] == result.calls[6].id
    assert result.calls[9].metadata["depends_on_call_ids"] == [result.calls[6].id]
    assert result.calls[12].metadata["depends_on_call_ids"] == [
        result.calls[6].id,
        result.calls[9].id,
    ]
    assert set(result.calls[15].metadata["visible_call_ids"]) == {
        result.calls[0].id,
        result.calls[3].id,
        result.calls[6].id,
        result.calls[8].id,
        result.calls[9].id,
        result.calls[11].id,
        result.calls[12].id,
        result.calls[14].id,
    }


@pytest.mark.asyncio
async def test_cross_examination_second_round_rotates_and_keeps_transcript():
    llm = FakeLLMClient(
        [
            *[
                f"Initial {label}\n<final_answer>{label}</final_answer>"
                for label in ("A", "B", "C")
            ],
            *[f"<claim id=\"C1\">{label}1</claim>" for label in ("A", "B", "C")],
            *["A->B challenge", "B->C challenge", "C->A challenge"],
            *["B response", "C response", "A response"],
            *["RESOLVED one", "RESOLVED two", "RESOLVED three"],
            *["A->C challenge", "B->A challenge", "C->B challenge"],
            *["C second response", "A second response", "B second response"],
            *["RESOLVED four", "RESOLVED five", "RESOLVED six"],
            *[
                f"Final {label}\n<final_answer>{label}</final_answer>"
                for label in ("A", "B", "C")
            ],
        ]
    )
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A, B, or C")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[
                AnswerChoice(label="A"),
                AnswerChoice(label="B"),
                AnswerChoice(label="C"),
            ],
        ),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=CrossExaminationDebateWorkflow(
            [agent("a"), agent("b"), agent("c")],
            rounds=2,
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="first"),
        ),
        experiment_id="experiment",
    )

    second_challenge = next(
        call for call in result.calls if call.step == "cross_exam_2_challenge_0"
    )
    assert second_challenge.metadata["challenger_id"] == "a"
    assert second_challenge.metadata["target_id"] == "c"
    second_challenge_text = second_challenge.messages[-1].content
    assert isinstance(second_challenge_text, str)
    assert "Round 1: a -> b" in second_challenge_text

    second_response = next(
        call for call in result.calls if call.step == "cross_exam_2_response_0"
    )
    second_response_text = second_response.messages[-1].content
    assert isinstance(second_response_text, str)
    assert "Round 1: b -> c" in second_response_text
    assert "Round 1: c -> a" in second_response_text


def test_cross_examination_rounds_rotate_targets_and_change_identity():
    agents = [agent("a"), agent("b"), agent("c")]
    one_round = CrossExaminationDebateWorkflow(
        agents,
        aggregation="plurality_vote",
        rounds=1,
    )
    two_rounds = CrossExaminationDebateWorkflow(
        agents,
        aggregation="plurality_vote",
        rounds=2,
    )

    assert one_round.spec().fingerprint != two_rounds.spec().fingerprint
    assert two_rounds.config()["rounds"] == 2
    assert two_rounds.config()["phase_max_tokens"] == {
        "claims": 240,
        "challenge": 120,
        "response": 160,
        "verdict": 80,
    }


@pytest.mark.asyncio
async def test_supervisor_can_request_revision_then_approve():
    llm = FakeLLMClient(["draft", "REVISE fix the arithmetic", "fixed", "APPROVE"])

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput.from_prompt(id="task-1", prompt="Question"),
        workflow=SupervisorWorkflow(
            worker=agent("worker"),
            supervisor=agent("supervisor"),
            max_revisions=2,
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "fixed"
    assert result.metrics.model_calls == 4
    assert [stage.step for stage in result.stage_answers] == [
        "worker_initial",
        "worker_revision_1",
    ]


@pytest.mark.asyncio
async def test_multimodal_benchmark_task_reaches_agent_and_judge():
    task = TaskInput(
        id="hle-image-1",
        messages=[
            Message(
                role="user",
                content=[
                    TextContent(text="Which option matches the image?"),
                    ImageContent(
                        image_url=ImageURL(
                            url="data:image/png;base64,example",
                        )
                    ),
                ],
            )
        ],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[
                AnswerChoice(label="A", text="First"),
                AnswerChoice(label="B", text="Second"),
            ],
            include_confidence=True,
        ),
    )
    llm = FakeLLMClient(
        [
            "Reasoning\n<final_answer>A</final_answer><confidence>70</confidence>",
            "Reasoning\n<final_answer>B</final_answer><confidence>80</confidence>",
            "Best\n<final_answer>B</final_answer><confidence>85</confidence>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            agent("judge"),
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "B"
    assert result.output.raw_response.startswith("Best")
    assert result.output.confidence == 85
    assert result.output.parse_status == "parsed"
    assert result.output.contract_valid is True
    assert llm.requests[0][0].content == task.messages[0].content
    assert llm.requests[-1][0].content == task.messages[0].content
    assert "final_answer" in llm.requests[-1][-1].content


def test_task_input_has_no_gold_answer_field():
    assert "expected_answer" not in TaskInput.model_fields


def test_output_contract_reports_malformed_choice():
    output = WorkflowOutput.from_response(
        "I think it is C.",
        AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )

    assert output.answer == "I think it is C."
    assert output.contract_valid is False
    assert "missing final_answer block" in output.validation_errors
    assert "answer is not an allowed choice label" in output.validation_errors


def test_workflow_fingerprint_is_stable_and_content_sensitive():
    base = SelfCriticWorkflow(agent("a"), rounds=1)
    identical = SelfCriticWorkflow(agent("a"), rounds=1)
    changed_prompt = SelfCriticWorkflow(
        agent("a"),
        rounds=1,
        revision_prompt=PromptTemplate(
            name="workflow.self_critic.revision",
            version="1.1.0",
            template="Find errors and return a corrected answer only.",
        ),
    )
    changed_config = SelfCriticWorkflow(agent("a"), rounds=2)

    assert base.spec().fingerprint == identical.spec().fingerprint
    assert base.spec().fingerprint != changed_prompt.spec().fingerprint
    assert base.spec().fingerprint != changed_config.spec().fingerprint


def test_prompt_hash_prevents_version_label_from_hiding_content_change():
    first = PromptTemplate(
        name="test.prompt",
        version="1.0.0",
        template="First text",
    )
    second = PromptTemplate(
        name="test.prompt",
        version="1.0.0",
        template="Changed text",
    )

    assert first.content_sha256 != second.content_sha256


def test_service_tier_changes_workflow_fingerprint():
    default_tier = SoloWorkflow(
        AgentSpec(
            id="a",
            model="fake/model",
            service_tier="default",
        )
    )
    flex_tier = SoloWorkflow(
        AgentSpec(
            id="a",
            model="fake/model",
            service_tier="flex",
        )
    )

    assert default_tier.spec().fingerprint != flex_tier.spec().fingerprint


@pytest.mark.asyncio
async def test_independent_sample_can_use_majority_vote_without_judge():
    llm = FakeLLMClient(
        [
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput(
            id="task-1",
            messages=[Message(role="user", content="Choose A or B")],
            answer_spec=AnswerSpec(
                type="multiple_choice",
                choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
            ),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            aggregation="majority_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "B"
    assert result.metrics.model_calls == 3
    assert [call.step for call in result.calls] == [
        "sample_0",
        "sample_1",
        "sample_2",
    ]
    assert result.workflow.config["judge"] is None
    assert result.workflow.config["aggregation"] == "majority_vote"
    vote_event = next(
        event for event in result.events if event.type == "votes_aggregated"
    )
    assert vote_event.data["tally"] == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_debate_can_use_plurality_vote_without_judge():
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput(
            id="task-1",
            messages=[Message(role="user", content="Choose A or B")],
            answer_spec=AnswerSpec(
                type="multiple_choice",
                choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
            ),
        ),
        workflow=DebateWorkflow(
            [agent("a"), agent("b"), agent("c")],
            rounds=1,
            aggregation="plurality_vote",
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "B"
    assert result.metrics.model_calls == 6
    assert all(call.agent_id != "judge" for call in result.calls)
    assert [stage.step for stage in result.stage_answers] == [
        "initial_0",
        "initial_1",
        "initial_2",
        "debate_1_0",
        "debate_1_1",
        "debate_1_2",
        "aggregation",
    ]
    assert result.stage_answers[-1].kind == "aggregate"


@pytest.mark.asyncio
async def test_debate_answer_only_hides_peer_reasoning():
    llm = FakeLLMClient(
        [
            "Reasoning A\n<final_answer>A</final_answer>",
            "Reasoning B\n<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=DebateWorkflow(
            [agent("a"), agent("b")],
            rounds=1,
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="first"),
            peer_view="answer_only",
        ),
        experiment_id="experiment",
    )

    debate_calls = [call for call in result.calls if call.step.startswith("debate_")]
    visible_messages = [
        message.content
        for call in debate_calls
        for message in call.messages
        if message.role == "user" and isinstance(message.content, str)
    ]
    assert any("Final answer: B" in message for message in visible_messages)
    assert any("Final answer: A" in message for message in visible_messages)
    assert all("Reasoning A" not in message for message in visible_messages)
    assert all("Reasoning B" not in message for message in visible_messages)
    assert result.workflow.config["peer_view"] == "answer_only"


@pytest.mark.asyncio
async def test_debate_answer_and_confidence_exposes_confidence():
    llm = FakeLLMClient(
        [
            "Reasoning A\n<final_answer>A</final_answer>\n"
            "<confidence>35</confidence>",
            "Reasoning B\n<final_answer>B</final_answer>\n"
            "<confidence>80</confidence>",
            "<final_answer>A</final_answer>\n<confidence>55</confidence>",
            "<final_answer>B</final_answer>\n<confidence>85</confidence>",
        ]
    )
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
            include_confidence=True,
        ),
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=DebateWorkflow(
            [agent("a"), agent("b")],
            rounds=1,
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="first"),
            peer_view="answer_and_confidence",
        ),
        experiment_id="experiment",
    )

    debate_calls = [call for call in result.calls if call.step.startswith("debate_")]
    visible_messages = [
        message.content
        for call in debate_calls
        for message in call.messages
        if message.role == "user" and isinstance(message.content, str)
    ]
    assert any("Confidence: 80" in message for message in visible_messages)
    assert any("Confidence: 35" in message for message in visible_messages)


def test_debate_peer_view_changes_fingerprint():
    agents = [agent("a"), agent("b")]

    full_response = DebateWorkflow(
        agents,
        aggregation="plurality_vote",
        peer_view="full_response",
    )
    answer_only = DebateWorkflow(
        agents,
        aggregation="plurality_vote",
        peer_view="answer_only",
    )

    assert full_response.spec().fingerprint != answer_only.spec().fingerprint


def test_unknown_debate_peer_view_is_rejected():
    with pytest.raises(ValueError, match="Unsupported peer view"):
        DebateWorkflow(
            [agent("a"), agent("b")],
            aggregation="plurality_vote",
            peer_view="unknown",
        )


@pytest.mark.asyncio
async def test_voting_tie_policy_is_explicit():
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="inconclusive"),
        ),
        experiment_id="experiment",
    )

    assert result.status == "inconclusive"
    assert result.final_answer is None
    assert result.error is None
    assert result.inconclusive is not None
    assert result.inconclusive.details["reason"] == "tie"
    assert result.inconclusive.details["tally"] == {"a": 1, "b": 1}
    assert len(result.inconclusive.details["ballots"]) == 2
    event = next(
        event for event in result.events if event.type == "run_inconclusive"
    )
    assert event.data["inconclusive"]["details"]["tied_answers"] == ["a", "b"]


def test_legacy_error_tie_policy_normalizes_to_inconclusive():
    voting = VotingConfig.model_validate({"tie_break": "error"})

    assert voting.tie_break == "inconclusive"


@pytest.mark.asyncio
async def test_sampling_can_use_judge_only_to_break_a_tie():
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )
    llm = FakeLLMClient(
        [
            "Reasoning A\n<final_answer>A</final_answer>",
            "Reasoning B\n<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            judge=agent("judge"),
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="judge"),
        ),
        experiment_id="experiment",
    )

    assert result.status == "success"
    assert result.final_answer == "B"
    assert [call.step for call in result.calls] == [
        "sample_0",
        "sample_1",
        "tie_break_judge",
    ]
    assert result.calls[-1].prompt_references[0].name == "workflow.judge.tie_break"
    vote_event = next(
        event for event in result.events if event.type == "votes_aggregated"
    )
    assert vote_event.data["tie_break_applied"] == "judge"
    assert vote_event.data["tied_answers"] == ["a", "b"]


@pytest.mark.asyncio
async def test_sampling_does_not_call_tie_break_judge_when_vote_has_winner():
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            judge=agent("judge"),
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="judge"),
        ),
        experiment_id="experiment",
    )

    assert result.status == "success"
    assert result.final_answer == "B"
    assert result.metrics.model_calls == 3
    assert all(call.step != "tie_break_judge" for call in result.calls)


@pytest.mark.asyncio
async def test_debate_can_use_judge_only_to_break_a_tie():
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=DebateWorkflow(
            [agent("a"), agent("b")],
            judge=agent("judge"),
            rounds=1,
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="judge"),
        ),
        experiment_id="experiment",
    )

    assert result.status == "success"
    assert result.final_answer == "A"
    assert result.calls[-1].step == "tie_break_judge"
    assert result.metrics.model_calls == 5


def test_judge_tie_break_requires_a_judge():
    with pytest.raises(
        ValueError,
        match="Judge aggregation or tie-breaking requires a judge",
    ):
        IndependentSampleWorkflow(
            [agent("a"), agent("b")],
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="judge"),
        )


@pytest.mark.asyncio
async def test_majority_without_consensus_is_inconclusive():
    task = TaskInput(
        id="task-1",
        messages=[Message(role="user", content="Choose A or B")],
        answer_spec=AnswerSpec(
            type="multiple_choice",
            choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
        ),
    )
    llm = FakeLLMClient(
        [
            "<final_answer>A</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
            "<final_answer>A</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=task,
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c"), agent("d")],
            aggregation="majority_vote",
        ),
        experiment_id="experiment",
    )

    assert result.status == "inconclusive"
    assert result.inconclusive is not None
    assert result.inconclusive.details["reason"] == "no_strict_majority"
    assert result.inconclusive.details["tally"] == {"a": 2, "b": 2}


@pytest.mark.asyncio
async def test_invalid_ballots_can_be_excluded():
    llm = FakeLLMClient(
        [
            "unformatted answer",
            "<final_answer>B</final_answer>",
            "<final_answer>B</final_answer>",
        ]
    )

    result = await ExperimentRunner(llm=llm).run(
        task=TaskInput(
            id="task-1",
            messages=[Message(role="user", content="Choose A or B")],
            answer_spec=AnswerSpec(
                type="multiple_choice",
                choices=[AnswerChoice(label="A"), AnswerChoice(label="B")],
            ),
        ),
        workflow=IndependentSampleWorkflow(
            [agent("a"), agent("b"), agent("c")],
            aggregation="majority_vote",
            voting=VotingConfig(invalid_ballot_policy="exclude"),
        ),
        experiment_id="experiment",
    )

    assert result.final_answer == "B"
    vote_event = next(
        event for event in result.events if event.type == "votes_aggregated"
    )
    assert vote_event.data["valid_ballots"] == 2
    assert vote_event.data["total_ballots"] == 3


def test_unknown_aggregation_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        IndependentSampleWorkflow(
            [agent("a")],
            aggregation="unknown",
        )
