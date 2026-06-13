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
            "total_jobs": 4,
            "scheduled_jobs": 3,
            "completed_jobs": 1,
        }
    )
    now[0] = 5.0
    progress.handle(
        {"event": "attempt_started", "job_id": "job-1", "attempt": 1}
    )
    progress.handle(
        {
            "event": "model_call_completed",
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
    assert "ok 1 inconclusive 0 failed 0" in line
    assert "attempts 2 retries 1" in line
    assert "session $0.0300" in line
    assert "300 tok" in line
    assert "TPS 25.0" in line
    assert "ETA 00:20" in line


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
