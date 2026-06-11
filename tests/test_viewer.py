from __future__ import annotations

from copy import deepcopy

import pytest

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.models import (
    AgentSpec,
    AnswerChoice,
    AnswerSpec,
    Message,
    TaskInput,
)
from multi_agent_research.runner import ExperimentRunner
from multi_agent_research.viewer import RunArtifact, normalize_run
from multi_agent_research.workflows import CrossExaminationDebateWorkflow
from tests.fakes import FakeLLMClient


def agent(agent_id: str) -> AgentSpec:
    return AgentSpec(id=agent_id, model="fake/model")


def test_viewer_groups_independent_samples_into_one_phase(tmp_path):
    result = {
        "run_id": "sample-run",
        "experiment_id": "sample-experiment",
        "task_id": "sample-task",
        "workflow": {
            "name": "independent_sample",
            "version": "2.0.0",
            "config": {
                "agents": [
                    {"id": "a", "model": "fake/model"},
                    {"id": "b", "model": "fake/model"},
                ],
                "judge": {"id": "judge", "model": "fake/model"},
            },
        },
        "status": "success",
        "final_answer": "B",
        "started_at": "2026-06-11T00:00:00Z",
        "metrics": {"total_tokens": 45, "wall_time_ms": 10, "model_calls": 3},
        "calls": [
            _viewer_call("call-a", 0, "sample_0", "a"),
            _viewer_call("call-b", 1, "sample_1", "b"),
            _viewer_call("call-judge", 2, "judge", "judge"),
        ],
        "stage_answers": [
            _viewer_stage(0, "sample_0", "a", "call-a", "A", "candidate"),
            _viewer_stage(1, "sample_1", "b", "call-b", "B", "candidate"),
            _viewer_stage(
                2,
                "judge",
                "judge",
                "call-judge",
                "B",
                "aggregate",
            ),
        ],
        "events": [],
    }

    normalized = normalize_run(
        RunArtifact(path=tmp_path, result=result, request=None)
    )

    assert [phase["id"] for phase in normalized["phase_order"]] == [
        "samples",
        "aggregate",
    ]
    assert {
        card["agent_id"]: card["phase"]
        for card in normalized["stage_cards"]
        if card["kind"] == "candidate"
    } == {"a": "samples", "b": "samples"}
    assert [
        card["phase"]
        for card in normalized["stage_cards"]
        if card["kind"] == "aggregate"
    ] == ["aggregate"]


def _viewer_call(
    call_id: str,
    sequence: int,
    step: str,
    agent_id: str,
) -> dict:
    return {
        "id": call_id,
        "sequence": sequence,
        "step": step,
        "agent_id": agent_id,
        "usage": {"total_tokens": 15},
        "metadata": {},
        "messages": [],
        "output": {"role": "assistant", "content": ""},
    }


def _viewer_stage(
    sequence: int,
    step: str,
    agent_id: str,
    call_id: str,
    answer: str,
    kind: str,
) -> dict:
    return {
        "sequence": sequence,
        "step": step,
        "kind": kind,
        "agent_id": agent_id,
        "call_id": call_id,
        "output": {
            "answer": answer,
            "confidence": None,
            "contract_valid": True,
            "raw_response": f"<final_answer>{answer}</final_answer>",
        },
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_viewer_normalizes_cross_examination_as_directed_exchanges(tmp_path):
    llm = FakeLLMClient(
        [
            *[
                f"Initial {label}\n<final_answer>{label}</final_answer>"
                for label in ("A", "B", "C")
            ],
            *[f"<claim id=\"C1\">{label}1</claim>" for label in ("A", "B", "C")],
            "A challenges B1",
            "B challenges C1",
            "C challenges A1",
            "B answers A",
            "C answers B",
            "A answers C",
            "UNRESOLVED B did not justify B1.",
            "RESOLVED C justified C1.",
            "CONCEDED A withdrew A1.",
            *[
                f"Final {label}\n<final_answer>{label}</final_answer>"
                for label in ("A", "B", "B")
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
            rounds=1,
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="first"),
        ),
        experiment_id="experiment",
    )
    artifact = RunArtifact(
        path=tmp_path,
        result=result.model_dump(mode="json"),
        request={
            "task": task.model_dump(mode="json"),
        },
    )

    normalized = normalize_run(artifact)

    assert normalized["summary"]["cross_examination"] is True
    assert [phase["id"] for phase in normalized["phase_order"]] == [
        "initial",
        "cross_examination",
        "final_revision",
        "aggregate",
    ]
    assert len(normalized["exchanges"]) == 3
    first = normalized["exchanges"][0]
    assert first["challenger_id"] == "a"
    assert first["target_id"] == "b"
    assert first["verdict_status"] == "unresolved"
    assert first["calls"]["challenge"]["step"] == "cross_exam_1_challenge_0"
    assert first["calls"]["response"]["step"] == "cross_exam_1_response_0"
    assert first["calls"]["verdict"]["step"] == "cross_exam_1_verdict_0"
    assert first["total_tokens"] == 45


@pytest.mark.asyncio
async def test_viewer_supports_cross_examination_v1_events_without_call_ids(
    tmp_path,
):
    llm = FakeLLMClient(
        [
            "Initial A\n<final_answer>A</final_answer>",
            "Initial B\n<final_answer>B</final_answer>",
            "<claim id=\"C1\">A1</claim>",
            "<claim id=\"C1\">B1</claim>",
            "A challenges B",
            "B challenges A",
            "B responds",
            "A responds",
            "RESOLVED",
            "CONCEDED",
            "Final A\n<final_answer>A</final_answer>",
            "Final B\n<final_answer>B</final_answer>",
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
        workflow=CrossExaminationDebateWorkflow(
            [agent("a"), agent("b")],
            aggregation="plurality_vote",
            voting=VotingConfig(tie_break="first"),
        ),
        experiment_id="experiment",
    )
    legacy = deepcopy(result.model_dump(mode="json"))
    legacy["workflow"]["version"] = "1.0.0"
    for event in legacy["events"]:
        if event["type"] == "cross_examination_exchange":
            event["data"].pop("challenge_call_id", None)
            event["data"].pop("response_call_id", None)
            event["data"].pop("verdict_call_id", None)
    for call in legacy["calls"]:
        metadata = call["metadata"]
        metadata.pop("phase_id", None)
        metadata.pop("interaction_id", None)
        metadata.pop("depends_on_call_ids", None)
        metadata.pop("visible_call_ids", None)

    normalized = normalize_run(
        RunArtifact(path=tmp_path, result=legacy, request=None)
    )

    assert len(normalized["exchanges"]) == 2
    assert normalized["exchanges"][0]["calls"]["challenge"]["agent_id"] == "a"
    assert normalized["exchanges"][0]["calls"]["response"]["agent_id"] == "b"


def test_viewer_normalizes_solo_workflow(tmp_path):
    result = {
        "run_id": "solo-run",
        "experiment_id": "solo-experiment",
        "task_id": "solo-task",
        "workflow": {
            "name": "solo",
            "version": "2.0.0",
            "config": {
                "agent": {"id": "a", "model": "fake/model"},
            },
        },
        "status": "success",
        "final_answer": "B",
        "started_at": "2026-06-11T00:00:00Z",
        "metrics": {"total_tokens": 15, "wall_time_ms": 10, "model_calls": 1},
        "calls": [_viewer_call("call-a", 0, "answer", "a")],
        "stage_answers": [_viewer_stage(0, "answer", "a", "call-a", "B", "candidate")],
        "events": [],
    }

    normalized = normalize_run(
        RunArtifact(path=tmp_path, result=result, request=None)
    )

    assert [agent["id"] for agent in normalized["agents"]] == ["a"]
    assert normalized["agents"][0]["role"] is None
    assert [phase["id"] for phase in normalized["phase_order"]] == ["initial"]
    assert normalized["stage_cards"][0]["phase"] == "initial"


def test_viewer_normalizes_self_critic_workflow(tmp_path):
    result = {
        "run_id": "self-critic-run",
        "experiment_id": "self-critic-experiment",
        "task_id": "self-critic-task",
        "workflow": {
            "name": "self_critic",
            "version": "2.0.0",
            "config": {
                "agent": {"id": "a", "model": "fake/model"},
                "rounds": 2,
            },
        },
        "status": "success",
        "final_answer": "C",
        "started_at": "2026-06-11T00:00:00Z",
        "metrics": {"total_tokens": 45, "wall_time_ms": 30, "model_calls": 3},
        "calls": [
            _viewer_call("call-initial", 0, "initial_answer", "a"),
            _viewer_call("call-rev1", 1, "revision_1", "a"),
            _viewer_call("call-rev2", 2, "revision_2", "a"),
        ],
        "stage_answers": [
            _viewer_stage(0, "initial_answer", "a", "call-initial", "A", "candidate"),
            _viewer_stage(1, "revision_1", "a", "call-rev1", "B", "candidate"),
            _viewer_stage(2, "revision_2", "a", "call-rev2", "C", "candidate"),
        ],
        "events": [],
    }

    normalized = normalize_run(
        RunArtifact(path=tmp_path, result=result, request=None)
    )

    assert [agent["id"] for agent in normalized["agents"]] == ["a"]
    assert [phase["id"] for phase in normalized["phase_order"]] == [
        "initial",
        "revision_1",
        "revision_2",
    ]
    assert [card["phase"] for card in normalized["stage_cards"]] == [
        "initial",
        "revision_1",
        "revision_2",
    ]


def test_viewer_normalizes_supervisor_workflow(tmp_path):
    result = {
        "run_id": "supervisor-run",
        "experiment_id": "supervisor-experiment",
        "task_id": "supervisor-task",
        "workflow": {
            "name": "supervisor",
            "version": "2.0.0",
            "config": {
                "worker": {"id": "worker-1", "model": "fake/model"},
                "supervisor": {"id": "supervisor-1", "model": "fake/model"},
                "max_revisions": 1,
            },
        },
        "status": "success",
        "final_answer": "B",
        "started_at": "2026-06-11T00:00:00Z",
        "metrics": {"total_tokens": 30, "wall_time_ms": 20, "model_calls": 2},
        "calls": [
            _viewer_call("call-init", 0, "worker_initial", "worker-1"),
            _viewer_call("call-rev", 1, "worker_revision_1", "worker-1"),
        ],
        "stage_answers": [
            _viewer_stage(0, "worker_initial", "worker-1", "call-init", "A", "candidate"),
            _viewer_stage(1, "worker_revision_1", "worker-1", "call-rev", "B", "candidate"),
        ],
        "events": [],
    }

    normalized = normalize_run(
        RunArtifact(path=tmp_path, result=result, request=None)
    )

    assert [agent["id"] for agent in normalized["agents"]] == [
        "worker-1",
        "supervisor-1",
    ]
    assert [agent["role"] for agent in normalized["agents"]] == [
        "worker",
        "supervisor",
    ]
    assert [phase["id"] for phase in normalized["phase_order"]] == [
        "initial",
        "worker_revision_1",
    ]
    assert [card["phase"] for card in normalized["stage_cards"]] == [
        "initial",
        "worker_revision_1",
    ]

