from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import shutil
import sys
import threading
import time
from typing import Any, Callable, TextIO


ANIMATION_INTERVAL_SECONDS = 0.08
ANIMATION_FRAMES_PER_SECOND = 12


def _harness_version() -> str:
    try:
        return version("multi-agent-research")
    except PackageNotFoundError:
        return "dev"


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
    harness_version: str = field(default_factory=_harness_version)
    experiment_id: str = "unknown"
    purpose: str | None = None
    model: str = "unknown"
    judge_model: str | None = None
    manifest_schema_version: int | None = None
    task_count: int = 0
    condition_count: int = 0
    scope_condition_count: int = 0
    repetitions: int = 1
    concurrency: int = 1
    max_in_flight_requests: int = 1
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    estimated_minimum_model_calls: int = 0
    total: int = 0
    completed: int = 0
    scheduled: int = 0
    deferred: int = 0
    session_completed: int = 0
    active: int = 0
    attempts: int = 0
    retries: int = 0
    request_retries: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    total_output_tokens: int = 0
    completed_call_tokens: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    active_jobs: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    recent_calls: deque[tuple[float, int]] = field(
        default_factory=deque,
        repr=False,
    )
    started_at: float | None = None
    rendered: bool = False
    closed: bool = False
    _render_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    _animation_stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    _animation_thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _rendered_line_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interactive is None:
            self.interactive = self.stream.isatty()

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        with self._render_lock:
            if event_type == "benchmark_started":
                self.experiment_id = str(event.get("experiment_id") or "unknown")
                self.purpose = event.get("purpose")
                self.model = str(event.get("model") or "unknown")
                self.judge_model = event.get("judge_model")
                self.manifest_schema_version = event.get("manifest_schema_version")
                self.task_count = int(event.get("task_count") or 0)
                self.condition_count = int(event.get("condition_count") or 0)
                self.scope_condition_count = int(
                    event.get("scope_condition_count") or self.condition_count
                )
                self.repetitions = int(event.get("repetitions") or 1)
                self.concurrency = int(event.get("concurrency") or 1)
                self.max_in_flight_requests = int(
                    event.get("max_in_flight_requests") or 1
                )
                self.requests_per_minute = event.get("requests_per_minute")
                self.tokens_per_minute = event.get("tokens_per_minute")
                self.estimated_minimum_model_calls = int(
                    event.get("estimated_minimum_model_calls") or 0
                )
                self.total = event["total_jobs"]
                self.scheduled = event["scheduled_jobs"]
                self.completed = event["completed_jobs"]
                self.deferred = event.get("deferred_jobs", 0)
                self.started_at = self.clock()
            elif event_type == "attempt_started":
                job_id = str(event["job_id"])
                self.active_jobs[job_id] = {
                    **event,
                    "calls_started": 0,
                    "calls_completed": 0,
                    "calls_failed": 0,
                    "calls_reused": 0,
                    "active_calls": {},
                }
                self.active = len(self.active_jobs)
            elif event_type == "model_call_started":
                job = self.active_jobs.get(str(event.get("job_id") or ""))
                if job is not None:
                    key = self._call_key(event)
                    job["calls_started"] += 1
                    job["active_calls"][key] = {
                        **event,
                        "started_at_clock": self.clock(),
                    }
            elif event_type == "model_call_completed":
                total_tokens = int(event.get("total_tokens") or 0)
                output_tokens = int(event.get("output_tokens") or 0)
                self.completed_call_tokens += total_tokens
                self.total_output_tokens += output_tokens
                self.recent_calls.append((self.clock(), total_tokens))
                job = self.active_jobs.get(str(event.get("job_id") or ""))
                if job is not None:
                    job["calls_completed"] += 1
                    if event.get("checkpoint_reused"):
                        job["calls_reused"] += 1
                    job["active_calls"].pop(self._call_key(event), None)
            elif event_type == "model_call_failed":
                job = self.active_jobs.get(str(event.get("job_id") or ""))
                if job is not None:
                    job["calls_failed"] += 1
                    job["active_calls"].pop(self._call_key(event), None)
            elif event_type == "model_call_retry_scheduled":
                self.request_retries += 1
            elif event_type == "attempt_completed":
                self.attempts += 1
                self.cost_usd += float(event.get("cost_usd") or 0.0)
                self.total_tokens += int(event.get("total_tokens") or 0)
                if event["will_retry"]:
                    self.retries += 1
                else:
                    self.completed += 1
                    self.session_completed += 1
                    self.active_jobs.pop(str(event["job_id"]), None)
                    self.active = len(self.active_jobs)
                    status = str(event["status"])
                    self.statuses[status] = self.statuses.get(status, 0) + 1
            elif event_type == "benchmark_finished":
                self.active = 0
                self.active_jobs.clear()
            else:
                return

        self.render(final=event_type == "benchmark_finished")
        if event_type == "benchmark_started" and self.scheduled:
            self._start_animation()
        elif event_type == "benchmark_finished":
            self._stop_animation()

    def render(self, *, final: bool = False) -> None:
        with self._render_lock:
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
            aggregate_tps = (
                self.total_output_tokens / elapsed if elapsed > 0 else 0.0
            )
            visible_tokens = max(self.total_tokens, self.completed_call_tokens)
            rolling_tpm, rolling_rpm = self._rolling_rates()
            bar = (
                self._animated_bar(percent, elapsed)
                if self.interactive and not final
                else self._bar(percent)
            )
            summary = (
                f"{bar} "
                f"{self.completed}/{self.total} {percent:5.1f}%"
                f" | session ${self.cost_usd:.4f}"
                f" | {visible_tokens:,} tok"
                f" | TPM {rolling_tpm:,}"
                f" RPM {rolling_rpm}"
                f" | TPS {aggregate_tps:.1f}"
                f" | elapsed {_duration(elapsed)} ETA {_duration(eta)}"
                f" | active {self.active}"
                f" | deferred {self.deferred}"
                f" | ok {success} inconclusive {inconclusive} failed {failed}"
                f" | attempts {self.attempts}"
                f" retries {self.retries + self.request_retries}"
            )
            if self.interactive:
                size = shutil.get_terminal_size(fallback=(120, 24))
                progress_line = (
                    f"{bar}  {self.completed:,}/{self.total:,} jobs "
                    f"({percent:.1f}%)  |  elapsed {_duration(elapsed)}  |  "
                    f"ETA {_duration(eta)}"
                )
                session_line = (
                    f"Session     ${self.cost_usd:.4f}  |  "
                    f"{visible_tokens:,} returned tokens  |  TPM {rolling_tpm:,}  |  "
                    f"RPM {rolling_rpm}  |  output TPS {aggregate_tps:.1f}"
                )
                results_line = (
                    f"This run    ok {success}  |  inconclusive {inconclusive}  |  "
                    f"failed {failed}  |  attempts {self.attempts}  |  "
                    f"retries {self.retries + self.request_retries}  |  "
                    f"active {self.active}  |  deferred {self.deferred}"
                )
                lines = self._dashboard_lines(
                    progress_lines=[progress_line, session_line, results_line],
                    final=final,
                    width=max(40, size.columns - 1),
                    height=size.lines,
                )
                self._write_dashboard(lines)
                if final:
                    self.stream.write("\n")
            else:
                if not self.rendered:
                    for line in self._plain_header_lines():
                        self.stream.write(f"{line}\n")
                active_text = self._plain_active_text()
                self.stream.write(f"{summary}{active_text}\n")
            self.stream.flush()
            self.rendered = True
            if final:
                self.closed = True

    def close(self) -> None:
        self._stop_animation()
        with self._render_lock:
            if self.closed:
                return
            if self.rendered and self.interactive:
                self.stream.write("\n")
                self.stream.flush()
            self.closed = True

    def _start_animation(self) -> None:
        if not self.interactive or self.closed or self._animation_thread is not None:
            return
        self._animation_stop.clear()
        self._animation_thread = threading.Thread(
            target=self._animate,
            name="benchmark-progress-animation",
            daemon=True,
        )
        self._animation_thread.start()

    def _stop_animation(self) -> None:
        self._animation_stop.set()
        thread = self._animation_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            self._animation_thread = None

    def _animate(self) -> None:
        while not self._animation_stop.wait(ANIMATION_INTERVAL_SECONDS):
            self.render()

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    def _rolling_rates(self) -> tuple[int, int]:
        now = self.clock()
        cutoff = now - 60.0
        while self.recent_calls and self.recent_calls[0][0] <= cutoff:
            self.recent_calls.popleft()
        tokens = sum(total_tokens for _, total_tokens in self.recent_calls)
        return tokens, len(self.recent_calls)

    def _dashboard_lines(
        self,
        *,
        progress_lines: list[str],
        final: bool,
        width: int,
        height: int,
    ) -> list[str]:
        state = "COMPLETE" if final else "RUNNING"
        schema = (
            f"manifest v{self.manifest_schema_version}"
            if self.manifest_schema_version is not None
            else "manifest version unknown"
        )
        lines = [
            f"MULTI-AGENT RESEARCH  {state}",
            (
                f"Experiment  {self.experiment_id}  |  "
                f"harness v{self.harness_version}  |  {schema}"
            ),
        ]
        if self.purpose:
            lines.append(f"Purpose     {self.purpose}")
        model_text = f"Model       {self.model}"
        if self.judge_model:
            model_text += f"  |  judge {self.judge_model}"
        lines.append(model_text)
        lines.append(
            f"Workload    {self.task_count} tasks x "
            f"{self.scope_condition_count} conditions x "
            f"{self.repetitions} "
            f"repetition{'s' if self.repetitions != 1 else ''} = "
            f"{self.total:,} jobs"
            f"  |  ~{self.estimated_minimum_model_calls:,} model calls"
        )
        limits = (
            f"Concurrency {self.concurrency} jobs  |  "
            f"{self.max_in_flight_requests} model calls in flight"
        )
        if self.requests_per_minute:
            limits += f"  |  {self.requests_per_minute:,} RPM limit"
        if self.tokens_per_minute:
            limits += f"  |  {self.tokens_per_minute:,} TPM limit"
        lines.extend([limits, "", *progress_lines])

        if self.active_jobs and not final:
            available_rows = max(1, height - len(lines) - 3)
            max_jobs = max(1, min(4, available_rows // 3))
            visible = list(self.active_jobs.values())[:max_jobs]
            lines.append(f"Working now ({len(self.active_jobs)})")
            for job in visible:
                lines.extend(self._active_job_lines(job))
            hidden = len(self.active_jobs) - len(visible)
            if hidden:
                lines.append(f"  ... and {hidden} more active job{'s' if hidden != 1 else ''}")
        elif not final:
            lines.append("Working now  waiting for the next job")

        return [self._fit(line, width) for line in lines]

    def _plain_header_lines(self) -> list[str]:
        schema = (
            f"manifest v{self.manifest_schema_version}"
            if self.manifest_schema_version is not None
            else "manifest version unknown"
        )
        lines = [
            (
                f"Experiment {self.experiment_id} | harness v{self.harness_version} | "
                f"{schema} | model {self.model}"
            ),
            (
                f"Workload {self.task_count} tasks x {self.scope_condition_count} "
                f"conditions x {self.repetitions} "
                f"repetition{'s' if self.repetitions != 1 else ''} = "
                f"{self.total:,} jobs"
            ),
        ]
        if self.purpose:
            lines.insert(1, f"Purpose {self.purpose}")
        return lines

    def _plain_active_text(self) -> str:
        if not self.active_jobs:
            return ""
        job = next(iter(self.active_jobs.values()))
        hidden = len(self.active_jobs) - 1
        suffix = f" (+{hidden} more)" if hidden else ""
        return f" | working: {self._workflow_description(job)}{suffix}"

    def _active_job_lines(self, job: dict[str, Any]) -> list[str]:
        workflow_version = job.get("workflow_version")
        version_text = f"  |  workflow v{workflow_version}" if workflow_version else ""
        category = job.get("task_category")
        task_label = str(job.get("task_id") or "unknown task")
        if category:
            task_label += f" ({category})"
        prompt = " ".join(str(job.get("task_prompt") or "").split())
        attempt = int(job.get("attempt") or 1)
        repetition = int(job.get("repetition") or 1)
        return [
            f"  * {self._workflow_description(job)}{version_text}",
            (
                f"    {task_label}  |  repetition {repetition}, attempt {attempt}"
                + (f"  |  {prompt}" if prompt else "")
            ),
            f"    {self._call_progress(job)}",
        ]

    def _call_progress(self, job: dict[str, Any]) -> str:
        completed = int(job.get("calls_completed") or 0)
        failed = int(job.get("calls_failed") or 0)
        active_calls = job.get("active_calls") or {}
        expected = int(job.get("estimated_model_calls") or 0)
        expected_text = f"/~{expected}" if expected else ""
        not_started = max(
            0,
            expected - completed - failed - len(active_calls),
        )
        text = (
            f"Calls      {completed}{expected_text} done  |  "
            f"{failed} failed  |  {len(active_calls)} running"
        )
        reused = int(job.get("calls_reused") or 0)
        if reused:
            text += f"  |  {reused} checkpointed"
        if expected:
            text += f"  |  {not_started} queued"
        if active_calls:
            oldest = min(
                active_calls.values(),
                key=lambda call: call["started_at_clock"],
            )
            age = max(0.0, self.clock() - oldest["started_at_clock"])
            text += (
                f"  |  oldest {oldest.get('step', 'call')}/"
                f"{oldest.get('agent_id', 'agent')} {_duration(age)}"
            )
        return text

    @staticmethod
    def _call_key(event: dict[str, Any]) -> str:
        return (
            f"{event.get('run_id')}:{event.get('sequence')}:"
            f"{event.get('request_attempt', 1)}"
        )

    @staticmethod
    def _workflow_description(job: dict[str, Any]) -> str:
        workflow = str(job.get("workflow") or "workflow")
        agents = int(job.get("agents") or 1)
        rounds = int(job.get("rounds") or 1)
        effort = job.get("reasoning_effort")
        if workflow == "solo":
            description = "Solo response"
        elif workflow == "self-critic":
            description = (
                f"Self-critique with {rounds} revision "
                f"round{'s' if rounds != 1 else ''}"
            )
        elif workflow == "sample":
            description = (
                f"Independent sampling with {agents} "
                f"agent{'s' if agents != 1 else ''}"
            )
        elif workflow in {"debate", "adversarial-debate"}:
            label = "Adversarial debate" if workflow.startswith("adversarial") else "Debate"
            description = (
                f"{label} with {agents} agents over {rounds} "
                f"round{'s' if rounds != 1 else ''}"
            )
        elif workflow == "cross-examination-debate":
            description = (
                f"Cross-examination debate with {agents} agents over {rounds} "
                f"round{'s' if rounds != 1 else ''}"
            )
        elif workflow == "supervisor":
            description = (
                f"Supervisor and worker with up to {rounds} "
                f"revision{'s' if rounds != 1 else ''}"
            )
        else:
            description = workflow.replace("-", " ").title()
        if effort:
            reasoning = "no extra reasoning" if effort == "none" else f"{effort} reasoning"
            description += f"  |  {reasoning}"
        condition = job.get("condition")
        if condition:
            description += f"  |  {condition}"
        return description

    def _write_dashboard(self, lines: list[str]) -> None:
        if self._rendered_line_count:
            self.stream.write("\r")
            if self._rendered_line_count > 1:
                self.stream.write(f"\033[{self._rendered_line_count - 1}A")
        for index, line in enumerate(lines):
            self.stream.write(f"\033[2K{line}")
            if index < len(lines) - 1:
                self.stream.write("\n")
        if self._rendered_line_count > len(lines):
            for _ in range(self._rendered_line_count - len(lines)):
                self.stream.write("\n\033[2K")
            self.stream.write(f"\033[{self._rendered_line_count - len(lines)}A")
        self._rendered_line_count = len(lines)

    @staticmethod
    def _fit(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return f"{text[: width - 3]}..."

    @staticmethod
    def _animated_bar(percent: float, elapsed: float, width: int = 16) -> str:
        """Sweep a small ASCII comet through work that has not completed yet."""
        filled = min(width, max(0, round(width * percent / 100)))
        pending = width - filled
        if pending <= 0:
            return f"[{'#' * width}]"

        cells = ["#"] * filled + ["-"] * pending
        if pending == 1:
            cells[filled] = ">"
            return f"[{''.join(cells)}]"

        cycle = 2 * (pending - 1)
        step = int(elapsed * ANIMATION_FRAMES_PER_SECOND) % cycle
        moving_right = step < pending
        head = step if moving_right else cycle - step
        head += filled
        cells[head] = ">" if moving_right else "<"

        direction = 1 if moving_right else -1
        for distance, symbol in ((1, "="), (2, ".")):
            tail = head - direction * distance
            if filled <= tail < width:
                cells[tail] = symbol
        return f"[{''.join(cells)}]"

    @staticmethod
    def _bar(percent: float, width: int = 16) -> str:
        filled = min(width, max(0, round(width * percent / 100)))
        return f"[{'#' * filled}{'-' * (width - filled)}]"
