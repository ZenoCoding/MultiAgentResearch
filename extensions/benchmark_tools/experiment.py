from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1


class ExperimentValidationError(ValueError):
    """Raised when persisted experiment data does not match its schema."""


class ManifestMismatchError(ValueError):
    """Raised when resume would mix materially different experiment inputs."""


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionPolicy:
    concurrency: int = 1
    max_in_flight_requests: int = 1
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    estimated_tokens_per_request: int = 4096
    max_attempts: int = 1
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 30.0
    retry_jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ExperimentValidationError("concurrency must be at least 1")
        if self.max_in_flight_requests < 1:
            raise ExperimentValidationError(
                "max_in_flight_requests must be at least 1"
            )
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ExperimentValidationError("requests_per_minute must be positive")
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise ExperimentValidationError("tokens_per_minute must be positive")
        if self.estimated_tokens_per_request < 0:
            raise ExperimentValidationError(
                "estimated_tokens_per_request cannot be negative"
            )
        if (
            self.tokens_per_minute is not None
            and self.estimated_tokens_per_request > self.tokens_per_minute
        ):
            raise ExperimentValidationError(
                "estimated_tokens_per_request cannot exceed tokens_per_minute"
            )
        if self.max_attempts < 1:
            raise ExperimentValidationError("max_attempts must be at least 1")
        if self.retry_base_delay_seconds < 0 or self.retry_max_delay_seconds < 0:
            raise ExperimentValidationError("retry delays cannot be negative")
        if not 0 <= self.retry_jitter_ratio <= 1:
            raise ExperimentValidationError(
                "retry_jitter_ratio must be between 0 and 1"
            )


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    task_set_path: str
    task_set_sha256: str
    task_count: int
    conditions: tuple[dict[str, Any], ...]
    model: str
    judge_model: str | None = None
    generation_settings: dict[str, Any] = field(default_factory=dict)
    system_settings: dict[str, Any] = field(default_factory=dict)
    repetitions: int = 1
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    created_at: str = field(default_factory=lambda: _utc_now())
    created_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ExperimentValidationError(
                f"unsupported manifest schema_version: {self.schema_version}"
            )
        if not self.experiment_id:
            raise ExperimentValidationError("experiment_id cannot be empty")
        if not self.task_set_path:
            raise ExperimentValidationError("task_set_path cannot be empty")
        if len(self.task_set_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.task_set_sha256
        ):
            raise ExperimentValidationError(
                "task_set_sha256 must be a lowercase SHA-256 hex digest"
            )
        if self.task_count < 0:
            raise ExperimentValidationError("task_count cannot be negative")
        if self.repetitions < 1:
            raise ExperimentValidationError("repetitions must be at least 1")
        if not self.model:
            raise ExperimentValidationError("model cannot be empty")
        if not isinstance(self.policy, ExecutionPolicy):
            raise ExperimentValidationError("policy must be an ExecutionPolicy")
        for name, value in (
            ("generation_settings", self.generation_settings),
            ("system_settings", self.system_settings),
            ("metadata", self.metadata),
        ):
            if not isinstance(value, dict):
                raise ExperimentValidationError(f"{name} must be an object")
        condition_ids = [condition.get("id") for condition in self.conditions]
        if not self.conditions or any(
            not isinstance(condition_id, str) or not condition_id
            for condition_id in condition_ids
        ):
            raise ExperimentValidationError(
                "each condition snapshot must have a non-empty string id"
            )
        if len(set(condition_ids)) != len(condition_ids):
            raise ExperimentValidationError("condition ids must be unique")
        _validate_json_value(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentManifest:
        values = _strict_fields(data, cls, "manifest")
        policy_data = values.get("policy", {})
        if not isinstance(policy_data, Mapping):
            raise ExperimentValidationError("manifest policy must be an object")
        values["policy"] = ExecutionPolicy(
            **_strict_fields(policy_data, ExecutionPolicy, "execution policy")
        )
        conditions = values.get("conditions")
        if not isinstance(conditions, list):
            raise ExperimentValidationError("manifest conditions must be an array")
        values["conditions"] = tuple(
            _json_object(row, "condition") for row in conditions
        )
        return cls(**values)

    @property
    def compatibility_fingerprint(self) -> str:
        material = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "task_set_sha256": self.task_set_sha256,
            "task_count": self.task_count,
            "conditions": self.conditions,
            "model": self.model,
            "judge_model": self.judge_model,
            "generation_settings": self.generation_settings,
            "system_settings": self.system_settings,
            "repetitions": self.repetitions,
            "policy": asdict(self.policy),
        }
        return _digest(material)

    def assert_compatible(self, other: ExperimentManifest) -> None:
        if self.compatibility_fingerprint != other.compatibility_fingerprint:
            raise ManifestMismatchError(
                "experiment manifest is incompatible with the persisted manifest"
            )


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    experiment_id: str
    condition_id: str
    task_id: str
    repetition: int


@dataclass
class AttemptRecord:
    attempt_id: str
    job_id: str
    number: int
    state: JobState
    started_at: str
    finished_at: str | None = None
    run_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttemptRecord:
        values = _strict_fields(data, cls, "attempt")
        values["state"] = _job_state(values["state"])
        return cls(**values)


@dataclass
class JobRecord:
    spec: JobSpec
    state: JobState = JobState.PENDING
    attempt_count: int = 0
    latest_attempt_id: str | None = None
    latest_run_id: str | None = None
    latest_error: str | None = None
    selected_attempt_id: str | None = None
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    attempts: list[AttemptRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobRecord:
        values = _strict_fields(data, cls, "job")
        spec_data = values["spec"]
        if not isinstance(spec_data, Mapping):
            raise ExperimentValidationError("job spec must be an object")
        values["spec"] = JobSpec(**_strict_fields(spec_data, JobSpec, "job spec"))
        values["state"] = _job_state(values["state"])
        attempts = values["attempts"]
        if not isinstance(attempts, list):
            raise ExperimentValidationError("job attempts must be an array")
        values["attempts"] = [AttemptRecord.from_dict(item) for item in attempts]
        record = cls(**values)
        record.validate()
        return record

    def validate(self) -> None:
        if self.attempt_count != len(self.attempts):
            raise ExperimentValidationError("job attempt_count does not match attempts")
        for number, attempt in enumerate(self.attempts, start=1):
            if attempt.job_id != self.spec.job_id or attempt.number != number:
                raise ExperimentValidationError("attempt does not match its job")
        if self.latest_attempt_id != (
            self.attempts[-1].attempt_id if self.attempts else None
        ):
            raise ExperimentValidationError("job latest_attempt_id is inconsistent")
        if self.state == JobState.SUCCESS:
            selected = {attempt.attempt_id: attempt for attempt in self.attempts}.get(
                self.selected_attempt_id
            )
            if selected is None or selected.state != JobState.SUCCESS:
                raise ExperimentValidationError(
                    "successful job must select a successful attempt"
                )


@dataclass
class ExperimentLedger:
    experiment_id: str
    manifest_fingerprint: str
    jobs: dict[str, JobRecord]
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    schema_version: int = LEDGER_SCHEMA_VERSION

    @classmethod
    def create(
        cls, manifest: ExperimentManifest, task_ids: Iterable[str]
    ) -> ExperimentLedger:
        specs = expand_jobs(manifest, task_ids)
        return cls(
            experiment_id=manifest.experiment_id,
            manifest_fingerprint=manifest.compatibility_fingerprint,
            jobs={spec.job_id: JobRecord(spec=spec) for spec in specs},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentLedger:
        values = _strict_fields(data, cls, "ledger")
        if values["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise ExperimentValidationError(
                f"unsupported ledger schema_version: {values['schema_version']}"
            )
        jobs_data = values["jobs"]
        if not isinstance(jobs_data, Mapping):
            raise ExperimentValidationError("ledger jobs must be an object")
        values["jobs"] = {
            job_id: JobRecord.from_dict(record) for job_id, record in jobs_data.items()
        }
        ledger = cls(**values)
        ledger.validate()
        return ledger

    def validate(self) -> None:
        for job_id, record in self.jobs.items():
            if job_id != record.spec.job_id:
                raise ExperimentValidationError("ledger job key does not match job_id")
            if record.spec.experiment_id != self.experiment_id:
                raise ExperimentValidationError(
                    "ledger job belongs to another experiment"
                )

    def assert_manifest(self, manifest: ExperimentManifest) -> None:
        if (
            self.experiment_id != manifest.experiment_id
            or self.manifest_fingerprint != manifest.compatibility_fingerprint
        ):
            raise ManifestMismatchError("ledger does not match experiment manifest")

    def start_attempt(
        self, job_id: str, *, run_id: str | None = None, at: str | None = None
    ) -> AttemptRecord:
        job = self.jobs[job_id]
        if job.state == JobState.SUCCESS:
            raise ExperimentValidationError("cannot retry a successful job")
        number = job.attempt_count + 1
        timestamp = at or _utc_now()
        if job.attempts and job.attempts[-1].state == JobState.RUNNING:
            interrupted = job.attempts[-1]
            interrupted.state = JobState.FAILED
            interrupted.finished_at = timestamp
            interrupted.error = "interrupted before completion"
        attempt = AttemptRecord(
            attempt_id=attempt_id(job_id, number),
            job_id=job_id,
            number=number,
            state=JobState.RUNNING,
            started_at=timestamp,
            run_id=run_id,
        )
        job.attempts.append(attempt)
        job.attempt_count = number
        job.latest_attempt_id = attempt.attempt_id
        job.latest_run_id = run_id
        job.latest_error = None
        job.state = JobState.RUNNING
        job.updated_at = timestamp
        self.updated_at = timestamp
        return attempt

    def finish_attempt(
        self,
        job_id: str,
        state: JobState,
        *,
        run_id: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        at: str | None = None,
    ) -> AttemptRecord:
        if state not in {
            JobState.SUCCESS,
            JobState.INCONCLUSIVE,
            JobState.FAILED,
        }:
            raise ExperimentValidationError("attempt must finish in a terminal state")
        job = self.jobs[job_id]
        if not job.attempts or job.attempts[-1].state != JobState.RUNNING:
            raise ExperimentValidationError("job has no running attempt")
        timestamp = at or _utc_now()
        attempt = job.attempts[-1]
        attempt.state = state
        attempt.finished_at = timestamp
        attempt.run_id = run_id or attempt.run_id
        attempt.error = error
        attempt.metadata = dict(metadata or {})
        job.state = state
        job.latest_run_id = attempt.run_id
        job.latest_error = error
        job.selected_attempt_id = (
            attempt.attempt_id if state == JobState.SUCCESS else None
        )
        job.updated_at = timestamp
        self.updated_at = timestamp
        return attempt


def job_id(experiment_id: str, condition_id: str, task_id: str, repetition: int) -> str:
    if repetition < 1:
        raise ExperimentValidationError("repetition must be at least 1")
    digest = _digest(
        {
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "task_id": task_id,
            "repetition": repetition,
        }
    )
    return f"job_{digest[:24]}"


def attempt_id(stable_job_id: str, number: int) -> str:
    if number < 1:
        raise ExperimentValidationError("attempt number must be at least 1")
    return f"attempt_{_digest({'job_id': stable_job_id, 'number': number})[:24]}"


def expand_jobs(manifest: ExperimentManifest, task_ids: Iterable[str]) -> list[JobSpec]:
    ordered_task_ids = list(task_ids)
    if len(ordered_task_ids) != manifest.task_count:
        raise ExperimentValidationError(
            "task id count does not match manifest task_count"
        )
    if len(set(ordered_task_ids)) != len(ordered_task_ids):
        raise ExperimentValidationError("task ids must be unique")
    return [
        JobSpec(
            job_id=job_id(
                manifest.experiment_id,
                condition["id"],
                task_id,
                repetition,
            ),
            experiment_id=manifest.experiment_id,
            condition_id=condition["id"],
            task_id=task_id,
            repetition=repetition,
        )
        for condition in manifest.conditions
        for task_id in ordered_task_ids
        for repetition in range(1, manifest.repetitions + 1)
    ]


def select_resume_jobs(
    ledger: ExperimentLedger,
    *,
    max_attempts: int,
    retry_inconclusive: bool = True,
) -> list[JobSpec]:
    if max_attempts < 1:
        raise ExperimentValidationError("max_attempts must be at least 1")
    retryable_states = {JobState.PENDING, JobState.RUNNING, JobState.FAILED}
    if retry_inconclusive:
        retryable_states.add(JobState.INCONCLUSIVE)
    return [
        record.spec
        for record in ledger.jobs.values()
        if record.state in retryable_states and record.attempt_count < max_attempts
    ]


def write_json_atomic(path: Path | str, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_json_value(data)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_manifest(path: Path | str, manifest: ExperimentManifest) -> None:
    write_json_atomic(path, manifest.to_dict())


def load_manifest(path: Path | str) -> ExperimentManifest:
    return ExperimentManifest.from_dict(_load_json_object(path))


def save_ledger(path: Path | str, ledger: ExperimentLedger) -> None:
    ledger.validate()
    write_json_atomic(path, ledger.to_dict())


def load_ledger(
    path: Path | str, *, manifest: ExperimentManifest | None = None
) -> ExperimentLedger:
    ledger = ExperimentLedger.from_dict(_load_json_object(path))
    if manifest is not None:
        ledger.assert_manifest(manifest)
    return ledger


def _load_json_object(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentValidationError(f"cannot load {path}: {exc}") from exc
    return _json_object(data, "JSON document")


def _strict_fields(
    data: Mapping[str, Any], model: type[Any], label: str
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ExperimentValidationError(f"{label} must be an object")
    field_names = set(model.__dataclass_fields__)
    unknown = set(data) - field_names
    missing = {
        name
        for name, definition in model.__dataclass_fields__.items()
        if name not in data
        and definition.default is MISSING
        and definition.default_factory is MISSING
    }
    if unknown:
        raise ExperimentValidationError(
            f"{label} has unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ExperimentValidationError(
            f"{label} is missing fields: {', '.join(sorted(missing))}"
        )
    return dict(data)


def _job_state(value: Any) -> JobState:
    try:
        return JobState(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(f"invalid job state: {value!r}") from exc


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExperimentValidationError(f"{label} must be a string-keyed object")
    _validate_json_value(value)
    return dict(value)


def _validate_json_value(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError("value is not strict JSON data") from exc


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
