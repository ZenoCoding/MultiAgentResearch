from __future__ import annotations

from dataclasses import dataclass, field
import shutil
import sys
import time
from typing import Any, Callable, TextIO


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class TerminalProgress:
    stream: TextIO = sys.stderr
    clock: Callable[[], float] = time.monotonic
    interactive: bool | None = None
    total: int = 0
    completed: int = 0
    scheduled: int = 0
    session_completed: int = 0
    active: int = 0
    attempts: int = 0
    retries: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    current_tps: float = 0.0
    statuses: dict[str, int] = field(default_factory=dict)
    active_jobs: set[str] = field(default_factory=set, repr=False)
    started_at: float | None = None
    rendered: bool = False
    closed: bool = False

    def __post_init__(self) -> None:
        if self.interactive is None:
            self.interactive = self.stream.isatty()

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "benchmark_started":
            self.total = event["total_jobs"]
            self.scheduled = event["scheduled_jobs"]
            self.completed = event["completed_jobs"]
            self.started_at = self.clock()
        elif event_type == "attempt_started":
            job_id = str(event["job_id"])
            if job_id not in self.active_jobs:
                self.active_jobs.add(job_id)
                self.active += 1
        elif event_type == "model_call_completed":
            latency_seconds = float(event.get("latency_ms") or 0.0) / 1000
            output_tokens = int(event.get("output_tokens") or 0)
            self.current_tps = (
                output_tokens / latency_seconds if latency_seconds > 0 else 0.0
            )
        elif event_type == "attempt_completed":
            self.attempts += 1
            self.cost_usd += float(event.get("cost_usd") or 0.0)
            self.total_tokens += int(event.get("total_tokens") or 0)
            if event["will_retry"]:
                self.retries += 1
            else:
                self.completed += 1
                self.session_completed += 1
                self.active_jobs.discard(str(event["job_id"]))
                self.active = len(self.active_jobs)
                status = str(event["status"])
                self.statuses[status] = self.statuses.get(status, 0) + 1
        elif event_type == "benchmark_finished":
            self.active = 0
        else:
            return

        self.render(final=event_type == "benchmark_finished")

    def render(self, *, final: bool = False) -> None:
        if self.closed:
            return
        elapsed = self._elapsed()
        remaining = max(0, self.total - self.completed)
        eta = None
        if self.session_completed:
            eta = elapsed * remaining / self.session_completed
        percent = 100.0 if not self.total else 100 * self.completed / self.total
        success = self.statuses.get("success", 0)
        inconclusive = self.statuses.get("inconclusive", 0)
        failed = self.statuses.get("failed", 0)
        line = (
            f"{self._bar(percent)} {self.completed}/{self.total} {percent:5.1f}%"
            f" | session ${self.cost_usd:.4f}"
            f" | {self.total_tokens:,} tok"
            f" | TPS {self.current_tps:.1f}"
            f" | elapsed {_duration(elapsed)} ETA {_duration(eta)}"
            f" | active {self.active}"
            f" | ok {success} inconclusive {inconclusive} failed {failed}"
            f" | attempts {self.attempts} retries {self.retries}"
        )
        if self.interactive:
            width = shutil.get_terminal_size(fallback=(120, 24)).columns
            line = line[: max(1, width - 1)]
            self.stream.write(f"\r\033[2K{line}")
            if final:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self.rendered = True
        if final:
            self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        if self.rendered and self.interactive:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    @staticmethod
    def _bar(percent: float, width: int = 16) -> str:
        filled = min(width, max(0, round(width * percent / 100)))
        return f"[{'#' * filled}{'-' * (width - filled)}]"
