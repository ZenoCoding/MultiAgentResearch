from __future__ import annotations

from io import StringIO

from extensions.benchmark_tools.progress import TerminalProgress


def test_progress_reports_completion_cost_tokens_retries_and_eta() -> None:
    output = StringIO()
    now = [0.0]
    progress = TerminalProgress(
        stream=output,
        clock=lambda: now[0],
        interactive=False,
    )

    progress.handle(
        {
            "event": "benchmark_started",
            "experiment_id": "hle-screen-v1",
            "purpose": "Compare workflow scaling",
            "model": "openai/test-model",
            "manifest_schema_version": 1,
            "task_count": 2,
            "condition_count": 2,
            "scope_condition_count": 2,
            "repetitions": 1,
            "concurrency": 2,
            "max_in_flight_requests": 2,
            "estimated_minimum_model_calls": 12,
            "total_jobs": 4,
            "scheduled_jobs": 3,
            "completed_jobs": 1,
            "deferred_jobs": 2,
        }
    )
    now[0] = 5.0
    progress.handle(
        {
            "event": "attempt_started",
            "job_id": "job-1",
            "attempt": 1,
            "condition": "debate-e-medium-a3-r2",
            "workflow": "debate",
            "workflow_version": "2.7.0",
            "agents": 3,
            "rounds": 2,
            "reasoning_effort": "medium",
            "task_id": "hle-1",
            "task_category": "Mathematics",
            "task_prompt": "Which answer is correct?",
            "repetition": 1,
        }
    )
    progress.handle(
        {
            "event": "model_call_completed",
            "total_tokens": 100,
            "output_tokens": 50,
            "latency_ms": 2000,
        }
    )
    now[0] = 10.0
    progress.handle(
        {
            "event": "attempt_completed",
            "job_id": "job-1",
            "attempt": 1,
            "status": "failed",
            "will_retry": True,
            "cost_usd": 0.01,
            "total_tokens": 100,
            "output_tokens": 40,
        }
    )
    progress.handle(
        {
            "event": "attempt_completed",
            "job_id": "job-1",
            "attempt": 2,
            "status": "success",
            "will_retry": False,
            "cost_usd": 0.02,
            "total_tokens": 200,
            "output_tokens": 60,
        }
    )

    line = output.getvalue().splitlines()[-1]
    assert "2/4  50.0%" in line
    assert "active 0" in line
    assert "deferred 2" in line
    assert "ok 1 inconclusive 0 failed 0" in line
    assert "attempts 2 retries 1" in line
    assert "session $0.0300" in line
    assert "300 tok" in line
    assert "TPM 100 RPM 1" in line
    assert "TPS 5.0" in line
    assert "ETA 00:20" in line
    assert "Experiment hle-screen-v1 | harness v" in output.getvalue()
    assert "Purpose Compare workflow scaling" in output.getvalue()


def test_progress_finishes_cleanly_when_every_job_was_already_done() -> None:
    output = StringIO()
    progress = TerminalProgress(
        stream=output,
        clock=lambda: 0.0,
        interactive=False,
    )

    progress.handle(
        {
            "event": "benchmark_started",
            "total_jobs": 2,
            "scheduled_jobs": 0,
            "completed_jobs": 2,
        }
    )
    progress.handle({"event": "benchmark_finished"})

    assert "2/2 100.0%" in output.getvalue().splitlines()[-1]
    assert progress.closed


def test_progress_uses_a_rolling_sixty_second_tpm_and_rpm_window() -> None:
    output = StringIO()
    now = [0.0]
    progress = TerminalProgress(
        stream=output,
        clock=lambda: now[0],
        interactive=False,
    )
    progress.handle(
        {
            "event": "benchmark_started",
            "total_jobs": 1,
            "scheduled_jobs": 1,
            "completed_jobs": 0,
        }
    )
    progress.handle(
        {
            "event": "model_call_completed",
            "total_tokens": 1_000,
            "output_tokens": 100,
        }
    )
    now[0] = 30.0
    progress.handle(
        {
            "event": "model_call_completed",
            "total_tokens": 2_000,
            "output_tokens": 200,
        }
    )

    assert "TPM 3,000 RPM 2" in output.getvalue().splitlines()[-1]

    now[0] = 61.0
    progress.render()

    assert "TPM 2,000 RPM 1" in output.getvalue().splitlines()[-1]


def test_progress_reports_partial_model_call_progress_for_active_job() -> None:
    output = StringIO()
    now = [0.0]
    progress = TerminalProgress(
        stream=output,
        clock=lambda: now[0],
        interactive=True,
    )
    progress.handle(
        {
            "event": "benchmark_started",
            "total_jobs": 1,
            "scheduled_jobs": 1,
            "completed_jobs": 0,
            "max_in_flight_requests": 20,
        }
    )
    progress.handle(
        {
            "event": "attempt_started",
            "job_id": "job-1",
            "attempt": 1,
            "condition": "sample-e-medium-a6",
            "workflow": "sample",
            "agents": 6,
            "rounds": 1,
            "reasoning_effort": "medium",
            "task_id": "hle-1",
            "repetition": 1,
            "estimated_model_calls": 7,
        }
    )
    for sequence in range(6):
        progress.handle(
            {
                "event": "model_call_started",
                "job_id": "job-1",
                "run_id": "run-1",
                "sequence": sequence,
                "request_attempt": 1,
                "step": f"sample_{sequence}",
                "agent_id": f"agent-{sequence + 1}",
            }
        )
    for sequence in range(3):
        progress.handle(
            {
                "event": "model_call_completed",
                "job_id": "job-1",
                "run_id": "run-1",
                "sequence": sequence,
                "request_attempt": 1,
                "total_tokens": 10_000,
                "output_tokens": 9_000,
                "checkpoint_reused": sequence < 2,
            }
        )
    now[0] = 65.0
    progress.render()

    rendered = output.getvalue()
    assert "30,000 returned tokens" in rendered
    assert (
        "Calls      3/~7 done  |  0 failed  |  3 running  |  "
        "2 checkpointed  |  1 queued"
    ) in rendered
    assert "oldest sample_3/agent-4 01:05" in rendered
    progress.close()


def test_interactive_progress_comet_sweeps_through_unfinished_work() -> None:
    output = StringIO()
    now = [0.0]
    progress = TerminalProgress(
        stream=output,
        clock=lambda: now[0],
        interactive=True,
    )
    progress.total = 1
    progress.started_at = 0.0

    progress.render()
    now[0] = 0.25
    progress.render()
    now[0] = 1.25
    progress.render()

    rendered = output.getvalue()
    assert "\033[2K[>---------------]" in rendered
    assert "\033[2K[-.=>------------]" in rendered
    assert "\033[2K[-------------.=>]" in rendered
    progress.close()


def test_interactive_progress_shows_experiment_and_human_active_jobs() -> None:
    output = StringIO()
    progress = TerminalProgress(
        stream=output,
        clock=lambda: 0.0,
        interactive=True,
        harness_version="0.1.0",
    )
    progress.handle(
        {
            "event": "benchmark_started",
            "experiment_id": "hle-screen-v1",
            "purpose": "Compare workflow scaling",
            "model": "openai/test-model",
            "judge_model": "openai/judge-model",
            "manifest_schema_version": 1,
            "task_count": 40,
            "condition_count": 6,
            "scope_condition_count": 5,
            "repetitions": 1,
            "concurrency": 3,
            "max_in_flight_requests": 6,
            "estimated_minimum_model_calls": 480,
            "total_jobs": 200,
            "scheduled_jobs": 200,
            "completed_jobs": 0,
            "deferred_jobs": 40,
        }
    )
    progress.handle(
        {
            "event": "attempt_started",
            "job_id": "job-1",
            "attempt": 1,
            "condition": "debate-e-medium-a3-r2",
            "workflow": "debate",
            "workflow_version": "2.7.0",
            "agents": 3,
            "rounds": 2,
            "reasoning_effort": "medium",
            "task_id": "hle-1",
            "task_category": "Mathematics",
            "task_prompt": "Which answer is correct?",
            "repetition": 1,
        }
    )

    rendered = output.getvalue()
    assert "MULTI-AGENT RESEARCH  RUNNING" in rendered
    assert "Experiment  hle-screen-v1  |  harness v0.1.0  |  manifest v1" in rendered
    assert "Purpose     Compare workflow scaling" in rendered
    assert "40 tasks x 5 conditions x 1 repetition = 200 jobs" in rendered
    assert "Debate with 3 agents over 2 rounds  |  medium reasoning" in rendered
    assert "workflow v2.7.0" in rendered
    assert "hle-1 (Mathematics)" in rendered
    assert "Which answer is correct?" in rendered
    progress.close()


def test_interactive_comet_preserves_completed_portion_of_bar() -> None:
    bar = TerminalProgress._animated_bar(50.0, elapsed=0.25, width=16)

    assert bar.startswith("[########")
    assert bar == "[########-.=>----]"


def test_interactive_comet_rebounds_at_end_of_pending_rail() -> None:
    outbound = TerminalProgress._animated_bar(0.0, elapsed=1.25, width=16)
    returning = TerminalProgress._animated_bar(0.0, elapsed=1.5, width=16)

    assert outbound == "[-------------.=>]"
    assert returning == "[------------<=.-]"


def test_noninteractive_and_final_progress_bars_are_static() -> None:
    output = StringIO()
    progress = TerminalProgress(
        stream=output,
        clock=lambda: 0.25,
        interactive=False,
    )
    progress.total = 2
    progress.completed = 1
    progress.started_at = 0.0

    progress.render()

    assert "[########--------] 1/2" in output.getvalue()
    assert ".=>" not in output.getvalue()


def test_close_stops_animation_thread() -> None:
    progress = TerminalProgress(stream=StringIO(), interactive=True)
    progress._start_animation()
    thread = progress._animation_thread

    assert thread is not None
    assert thread.is_alive()

    progress.close()

    assert not thread.is_alive()
    assert progress._animation_thread is None
