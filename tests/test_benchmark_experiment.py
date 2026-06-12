from __future__ import annotations

from dataclasses import replace

import pytest

from extensions.benchmark_tools.experiment import (
    ExperimentLedger,
    ExperimentManifest,
    ExecutionPolicy,
    JobState,
    ManifestMismatchError,
    attempt_id,
    expand_jobs,
    job_id,
    load_ledger,
    load_manifest,
    save_ledger,
    save_manifest,
    select_resume_jobs,
)


def manifest(**overrides: object) -> ExperimentManifest:
    values = {
        "experiment_id": "pilot-1",
        "task_set_path": "tasks/pilot.jsonl",
        "task_set_sha256": "a" * 64,
        "task_count": 2,
        "conditions": (
            {"id": "solo", "workflow": "solo"},
            {"id": "sample-3", "workflow": "sample", "agents": 3},
        ),
        "model": "openai/model",
        "judge_model": "openai/judge",
        "generation_settings": {"temperature": 0},
        "system_settings": {"system_prompt": "Be precise."},
        "repetitions": 2,
        "policy": ExecutionPolicy(max_attempts=3),
        "created_at": "2026-06-12T00:00:00+00:00",
        "created_by": "test",
    }
    values.update(overrides)
    return ExperimentManifest(**values)


def test_job_and_attempt_ids_are_stable_and_attempts_are_distinct() -> None:
    first = job_id("exp", "solo", "task-1", 1)
    assert first == job_id("exp", "solo", "task-1", 1)
    assert first != job_id("exp", "solo", "task-1", 2)
    assert attempt_id(first, 1) != attempt_id(first, 2)


def test_expand_jobs_includes_every_repetition() -> None:
    jobs = expand_jobs(manifest(), ["task-a", "task-b"])

    assert len(jobs) == 8
    assert {(job.condition_id, job.task_id, job.repetition) for job in jobs} == {
        (condition, task, repetition)
        for condition in ("solo", "sample-3")
        for task in ("task-a", "task-b")
        for repetition in (1, 2)
    }


def test_manifest_and_ledger_atomic_round_trip(tmp_path) -> None:
    current = manifest()
    ledger = ExperimentLedger.create(current, ["task-a", "task-b"])
    selected = next(iter(ledger.jobs))
    running = ledger.start_attempt(
        selected, run_id="run-1", at="2026-06-12T00:01:00+00:00"
    )
    ledger.finish_attempt(
        selected,
        JobState.SUCCESS,
        at="2026-06-12T00:02:00+00:00",
    )
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.json"

    save_manifest(manifest_path, current)
    save_ledger(ledger_path, ledger)

    assert load_manifest(manifest_path) == current
    loaded = load_ledger(ledger_path, manifest=current)
    assert loaded == ledger
    assert loaded.jobs[selected].selected_attempt_id == running.attempt_id
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_selection_keeps_jobs_and_attempts_separate() -> None:
    current = manifest(repetitions=1)
    ledger = ExperimentLedger.create(current, ["task-a", "task-b"])
    job_ids = list(ledger.jobs)
    ledger.start_attempt(job_ids[0], run_id="interrupted")
    ledger.start_attempt(job_ids[1])
    ledger.finish_attempt(job_ids[1], JobState.FAILED, error="timeout")
    ledger.start_attempt(job_ids[2])
    ledger.finish_attempt(job_ids[2], JobState.INCONCLUSIVE)
    ledger.start_attempt(job_ids[3], run_id="good")
    ledger.finish_attempt(job_ids[3], JobState.SUCCESS)

    selected = select_resume_jobs(ledger, max_attempts=3)

    assert [job.job_id for job in selected] == job_ids[:3]
    second_attempt = ledger.start_attempt(job_ids[0], run_id="retry")
    assert second_attempt.job_id == job_ids[0]
    assert second_attempt.attempt_id != ledger.jobs[job_ids[0]].attempts[0].attempt_id
    assert ledger.jobs[job_ids[0]].attempts[0].state == JobState.FAILED
    assert ledger.jobs[job_ids[0]].attempts[0].error == "interrupted before completion"


def test_retry_exhaustion_and_inconclusive_policy() -> None:
    current = manifest(repetitions=1)
    ledger = ExperimentLedger.create(current, ["task-a", "task-b"])
    failed_id, inconclusive_id, *_ = ledger.jobs
    for _ in range(2):
        ledger.start_attempt(failed_id)
        ledger.finish_attempt(failed_id, JobState.FAILED, error="provider error")
    ledger.start_attempt(inconclusive_id)
    ledger.finish_attempt(inconclusive_id, JobState.INCONCLUSIVE)

    selected = select_resume_jobs(ledger, max_attempts=2, retry_inconclusive=False)

    assert failed_id not in {job.job_id for job in selected}
    assert inconclusive_id not in {job.job_id for job in selected}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_set_sha256", "b" * 64),
        ("conditions", ({"id": "other", "workflow": "solo"},)),
        ("model", "openai/other"),
        ("judge_model", "openai/other-judge"),
        ("repetitions", 3),
        ("generation_settings", {"temperature": 0.5}),
        ("system_settings", {"system_prompt": "Different."}),
        ("policy", ExecutionPolicy(concurrency=8, max_attempts=5)),
    ],
)
def test_manifest_mismatch_refuses_material_changes(field: str, value: object) -> None:
    original = manifest()
    changed = replace(original, **{field: value})

    with pytest.raises(ManifestMismatchError):
        original.assert_compatible(changed)


def test_manifest_allows_non_material_resume_changes() -> None:
    original = manifest()
    relocated = replace(
        original,
        task_set_path="/mounted/tasks/pilot.jsonl",
        metadata={"resume_note": "new machine"},
    )

    original.assert_compatible(relocated)
