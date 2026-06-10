from __future__ import annotations

import json

import pytest

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
    DebateWorkflow,
    IndependentSampleWorkflow,
    SelfCriticWorkflow,
    SoloWorkflow,
    SupervisorWorkflow,
)
from tests.fakes import FakeLLMClient


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

    run_dir = tmp_path / "experiment-1" / result.run_id
    assert json.loads((run_dir / "result.json").read_text())["final_answer"] == (
        "final answer"
    )
    assert len((run_dir / "calls.jsonl").read_text().splitlines()) == 1


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
    assert [call.step for call in result.calls] == ["sample_0", "sample_1", "judge"]
    assert result.workflow.version == "2.0.0"
    assert result.workflow.fingerprint
    assert result.calls[-1].prompt_references[0].name == ("workflow.judge.selection")


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
    assert "initial b" in llm.requests[2][-2].content
    assert "initial a" in llm.requests[3][-2].content
    assert result.calls[2].prompt_references[0].name == ("workflow.debate.peer_review")


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
