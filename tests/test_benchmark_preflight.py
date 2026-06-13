from __future__ import annotations

import json
import sys

import pytest

from extensions.benchmark_tools import cli
from extensions.benchmark_tools.preflight import (
    plan_preflight_checks,
    preflight_experiment,
)
from extensions.benchmark_tools.config import load_experiment_config
from tests.fakes import FakeLLMClient


def _write_config(tmp_path) -> object:  # type: ignore[no-untyped-def]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "synthetic-task",
                "prompt": "Synthetic task",
                "answer": "unused",
                "answer_type": "short_answer",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "preflight-test",
                "tasks": str(tasks),
                "aggregation_judge_model": "fake/judge",
                "defaults": {
                    "aggregation": "plurality_vote",
                    "vote_tie_break": "judge",
                    "judge_reasoning_effort": "low",
                },
                "families": {
                    "solo": {"efforts": ["none", "low"]},
                    "sampling": [
                        {"effort": "low", "agents": [3, 6]},
                    ],
                    "supervisor_worker": [
                        {
                            "worker_effort": "low",
                            "supervisor_effort": "high",
                            "max_rounds": [2],
                        }
                    ],
                },
                "metadata": {
                    "grader_model": "fake/grader",
                    "grader_reasoning_effort": "low",
                },
            }
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.asyncio
async def test_preflight_success_reports_calls_tokens_and_cost(tmp_path) -> None:
    config = _write_config(tmp_path)
    llm = FakeLLMClient(
        [
            "READY",
            "READY",
            "READY",
            "<final_answer>beta</final_answer>",
            (
                "<vote_status>winner</vote_status>\n"
                "<final_answer>one half</final_answer>"
            ),
            json.dumps(
                {
                    "extracted_final_answer": "4",
                    "reasoning": "The answers match.",
                    "correct": "yes",
                    "confidence": 100,
                    "strict": True,
                }
            ),
        ]
    )

    summary = await preflight_experiment(
        config_path=config,
        primary_model="fake/primary",
        llm=llm,
        max_attempts=1,
    )

    assert summary["status"] == "passed"
    assert summary["required_checks"] == 6
    assert summary["call_count"] == 6
    assert summary["tokens"] == 90
    assert summary["cost"] == pytest.approx(0.06)
    assert all(check["status"] == "passed" for check in summary["checks"])


@pytest.mark.asyncio
async def test_preflight_dry_run_is_deduplicated_and_makes_zero_calls(
    tmp_path,
) -> None:
    config_path = _write_config(tmp_path)
    config = load_experiment_config(config_path)
    checks = plan_preflight_checks(
        config=config,
        config_path=config_path,
        primary_model="fake/primary",
    )

    summary = await preflight_experiment(
        config_path=config_path,
        primary_model="fake/primary",
        dry_run=True,
        llm=FakeLLMClient([]),
    )

    assert [check.check_id for check in checks] == [
        "primary.reasoning_effort.none",
        "primary.reasoning_effort.low",
        "primary.reasoning_effort.high",
        "aggregation_judge.final_answer.low",
        "aggregation_judge.semantic_vote.low",
        "hle_grader.strict_json_schema.low",
    ]
    assert summary["status"] == "planned"
    assert summary["call_count"] == 0
    assert all(check["status"] == "planned" for check in summary["checks"])
    assert all("prompt" in check for check in summary["checks"])


@pytest.mark.asyncio
async def test_preflight_fails_invalid_strict_schema_output(tmp_path) -> None:
    config = _write_config(tmp_path)
    llm = FakeLLMClient(
        [
            "READY",
            "READY",
            "READY",
            "<final_answer>beta</final_answer>",
            ("<vote_status>winner</vote_status>\n<final_answer>0.5</final_answer>"),
            json.dumps(
                {
                    "extracted_final_answer": "4",
                    "reasoning": "The answers match.",
                    "correct": "yes",
                    "confidence": 100,
                }
            ),
        ]
    )

    summary = await preflight_experiment(
        config_path=config,
        primary_model="fake/primary",
        llm=llm,
        max_attempts=1,
    )

    assert summary["status"] == "failed"
    assert summary["failed_checks"] == 1
    failed = [check for check in summary["checks"] if check["status"] == "failed"]
    assert failed[0]["check_id"] == "hle_grader.strict_json_schema.low"
    assert "validation error for HLEGrade" in failed[0]["error"]
    assert failed[0]["response_preview"]


@pytest.mark.asyncio
async def test_preflight_retries_invalid_model_contract(tmp_path) -> None:
    config = _write_config(tmp_path)
    llm = FakeLLMClient(
        [
            "READY",
            "READY",
            "READY",
            "<final_answer>beta</final_answer>",
            "<vote_status>winner</vote_status>\n0.5",
            "<vote_status>winner</vote_status>\n<final_answer>0.5</final_answer>",
            json.dumps(
                {
                    "extracted_final_answer": "4",
                    "reasoning": "The answers match.",
                    "correct": "yes",
                    "confidence": 100,
                    "strict": True,
                }
            ),
        ]
    )

    summary = await preflight_experiment(
        config_path=config,
        primary_model="fake/primary",
        llm=llm,
        max_attempts=2,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
    )

    semantic = next(
        check for check in summary["checks"] if check["kind"] == "semantic_vote"
    )
    assert summary["status"] == "passed"
    assert semantic["call_count"] == 2
    assert "End your response with exactly one" in llm.requests[4][-1].content


def test_preflight_cli_exits_nonzero_on_required_failure(
    monkeypatch,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    async def fake_preflight(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "dry_run": False,
            "status": "failed",
            "required_checks": 1,
            "passed_checks": 0,
            "failed_checks": 1,
            "call_count": 1,
            "tokens": 0,
            "cost": 0.0,
            "checks": [],
        }

    monkeypatch.setattr(cli, "preflight_experiment", fake_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-tools",
            "preflight",
            "--config",
            "config.json",
            "--model",
            "fake/primary",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
