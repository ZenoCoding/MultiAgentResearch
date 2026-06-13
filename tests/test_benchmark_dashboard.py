from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from pathlib import Path

from extensions.benchmark_tools.dashboard import app, get_experiment_status


client = TestClient(app)


def test_get_experiment_status() -> None:
    # Complete
    assert get_experiment_status(10, 10, 10, False, None) == "complete"
    # Partially graded
    assert get_experiment_status(10, 10, 5, False, None) == "partially graded"
    # Running due to active run
    assert get_experiment_status(10, 5, 5, True, None) == "running"
    # Incomplete
    assert get_experiment_status(10, 5, 5, False, None) == "incomplete"
    # Incomplete (no jobs done)
    assert get_experiment_status(10, 0, 0, False, None) == "incomplete"


def test_api_experiments_list_empty(monkeypatch, tmp_path) -> None:
    # Mock RESULTS_DIR to an empty folder
    monkeypatch.setattr("extensions.benchmark_tools.dashboard.RESULTS_DIR", tmp_path)
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert response.json() == []


def test_api_experiments_not_found() -> None:
    response = client.get("/api/experiments/does-not-exist")
    assert response.status_code == 404


def test_api_run_trace_not_found() -> None:
    response = client.get("/api/experiments/does-not-exist/runs/some-run")
    assert response.status_code == 404


def test_api_compare_bad_request() -> None:
    # Less than two experiments
    response = client.post("/api/compare", json={"experiment_ids": ["one"]})
    assert response.status_code == 400  # explicitly raised by comparison handler


def test_smoke_pilot_experiment_metrics() -> None:
    from pathlib import Path
    results_dir = Path("/Users/tycho/Projects/MultiAgentResearch/results")
    
    # Check if the smoke experiment folder exists (otherwise skip in non-local environments)
    if not (results_dir / "hle-smoke-5-paid-e2e-v1").exists():
        pytest.skip("Smoke experiment folder not found")
        
    response = client.get("/api/experiments")
    assert response.status_code == 200
    experiments = response.json()
    
    smoke_exp = next((e for e in experiments if e["experiment_id"] == "hle-smoke-5-paid-e2e-v1"), None)
    assert smoke_exp is not None
    assert smoke_exp["expected_jobs"] == 25
    assert smoke_exp["completed_jobs"] == 25
    assert smoke_exp["graded_jobs"] == 25
    
    assert abs(smoke_exp["workflow_cost"] - 0.0809) < 0.005
    assert abs(smoke_exp["grading_cost"] - 0.0228) < 0.005
    assert abs(smoke_exp["total_cost"] - 0.1037) < 0.005
    
    # Details API
    response = client.get("/api/experiments/hle-smoke-5-paid-e2e-v1")
    assert response.status_code == 200
    data = response.json()
    
    # 6/25 correct overall
    total_correct = sum(cond["correct_valid_completed"] for cond in data["summary"]["conditions"])
    assert total_correct == 6
    
    # Supervisor-worker: 2/5 correct, others: 1/5 correct
    conditions = {cond["condition"]: cond["correct_valid_completed"] for cond in data["summary"]["conditions"]}
    assert conditions["supervisor-w-low-s-low-r1"] == 2
    for name, correct in conditions.items():
        if name != "supervisor-w-low-s-low-r1":
            assert correct == 1
            
    # Sample size warning
    assert any("Small sample size" in w for w in data["warnings"])


def test_api_compare_success() -> None:
    results_dir = Path("/Users/tycho/Projects/MultiAgentResearch/results")
    if not (results_dir / "hle-smoke-5-paid-e2e-v1").exists():
        pytest.skip("Smoke experiment folder not found")
        
    response = client.post(
        "/api/compare",
        json={"experiment_ids": ["hle-smoke-5-paid-e2e-v1", "hle-smoke-5-paid-e2e-v1"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["comparable"] is True
    assert len(data["experiments"]) == 2
    assert "paired_deltas" in data


def test_no_missing_jobs_warning_for_smoke() -> None:
    results_dir = Path("/Users/tycho/Projects/MultiAgentResearch/results")
    if not (results_dir / "hle-smoke-5-paid-e2e-v1").exists():
        pytest.skip("Smoke experiment folder not found")
        
    response = client.get("/api/experiments/hle-smoke-5-paid-e2e-v1")
    assert response.status_code == 200
    data = response.json()
    
    # Verify no missing jobs warning is present
    warnings = data.get("warnings", [])
    assert not any("Missing jobs" in w for w in warnings), f"Expected no Missing jobs warning, got: {warnings}"



