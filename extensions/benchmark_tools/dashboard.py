from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import webbrowser
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from extensions.benchmark_tools.analysis import (
    analyze_attempts,
    wilson_interval,
    task_clustered_bootstrap_ci,
)
from extensions.benchmark_tools.connector import load_jsonl
from multi_agent_research.models import AnswerChoice, AnswerSpec
from extensions.benchmark_tools.experiment import (
    load_ledger,
    load_manifest,
    JobState,
)
from extensions.benchmark_tools.grading import load_grade_set
from src.multi_agent_research.viewer import RunArtifact, normalize_run


app = FastAPI(title="MART Local Results Dashboard")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results")

def get_experiment_status(
    expected_jobs: int,
    completed_jobs: int,
    graded_jobs: int,
    has_running: bool,
    updated_at_str: str | None
) -> str:
    if expected_jobs == 0:
        return "incomplete"
    if completed_jobs == expected_jobs:
        if graded_jobs == expected_jobs:
            return "complete"
        else:
            return "partially graded"
    
    is_recent = False
    if updated_at_str:
        try:
            dt = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - dt).total_seconds() < 300: # 5 minutes
                is_recent = True
        except Exception:
            pass
            
    if has_running or is_recent:
        return "running"
    
    if completed_jobs == 0:
        return "incomplete"
        
    return "incomplete"


def _preprocess_results_with_ledger(results: list[dict[str, Any]], ledger: Any) -> None:
    if ledger is None:
        return
    run_to_job = {
        attempt.run_id: record
        for record in ledger.jobs.values()
        for attempt in record.attempts
        if attempt.run_id
    }
    for result in results:
        record = run_to_job.get(result.get("run_id"))
        if record is not None:
            result["job_id"] = record.spec.job_id
            result["repetition"] = record.spec.repetition


def get_experiments_list(results_path: Path) -> list[dict[str, Any]]:
    experiments = []
    if not results_path.exists():
        return []
        
    for path in results_path.iterdir():
        if not path.is_dir() or path.name.startswith("_") or path.name.startswith("."):
            continue
        manifest_path = path / "experiment-manifest.json"
        if not manifest_path.exists():
            continue
            
        try:
            manifest = load_manifest(manifest_path)
            
            ledger_path = path / "experiment-ledger.json"
            ledger = None
            if ledger_path.exists():
                try:
                    ledger = load_ledger(ledger_path, manifest=manifest)
                except Exception:
                    pass
            
            expected_jobs = 0
            completed_jobs = 0
            graded_jobs = 0
            has_running = False
            updated_at = manifest.created_at
            
            grade_set = None
            try:
                grade_set = load_grade_set(results_dir=results_path, experiment_id=manifest.experiment_id)
            except Exception:
                pass
                
            workflow_cost = 0.0
            grading_cost = 0.0
            total_tokens = 0
            
            if ledger is not None:
                expected_jobs = len(ledger.jobs)
                updated_at = ledger.updated_at
                for job in ledger.jobs.values():
                    if job.state in {JobState.SUCCESS, JobState.FAILED, JobState.INCONCLUSIVE}:
                        completed_jobs += 1
                    elif job.state == JobState.RUNNING:
                        has_running = True
                        
                    for attempt in job.attempts:
                        if attempt.metadata:
                            workflow_cost += float(attempt.metadata.get("cost_usd") or 0.0)
                            total_tokens += int(attempt.metadata.get("total_tokens") or 0)
            else:
                expected_jobs = manifest.task_count * len(manifest.conditions) * manifest.repetitions
                run_dirs = [p for p in path.iterdir() if p.is_dir() and p.name != "grades"]
                completed_jobs = len(run_dirs)
            
            if grade_set is not None:
                grading_cost = sum(
                    float((attempt.get("call") or {}).get("cost_usd") or 0.0)
                    for record in grade_set.records.values()
                    for attempt in record.attempts
                )
                
            best_condition = None
            best_accuracy = -1.0
            best_ci = (0.0, 0.0)
            
            tasks_path = Path(manifest.task_set_path)
            if tasks_path.exists():
                try:
                    examples = load_jsonl(tasks_path)
                    results = []
                    for run_path in sorted(path.glob("*/result.json")):
                        try:
                            results.append(json.loads(run_path.read_text(encoding="utf-8")))
                        except Exception:
                            pass
                            
                    _preprocess_results_with_ledger(results, ledger)
                    expected_jobs_list = None
                    if ledger is not None:
                        expected_jobs_list = [
                            {
                                "job_id": record.spec.job_id,
                                "condition": record.spec.condition_id,
                                "task_id": record.spec.task_id,
                                "repetition": record.spec.repetition,
                                "attempt_count": record.attempt_count,
                            }
                            for record in ledger.jobs.values()
                        ]
                        
                    summary, run_rows, _ = analyze_attempts(
                        results,
                        expected_answers={ex.id: ex.answer for ex in examples},
                        answer_specs={
                            ex.id: AnswerSpec(
                                type=ex.answer_type,  # type: ignore[arg-type]
                                choices=[
                                    AnswerChoice(label=choice.label, text=choice.text)
                                    for choice in ex.choices
                                ] if getattr(ex, "choices", None) else [],
                                include_confidence=True,
                            )
                            for ex in examples
                        },
                        categories={ex.id: ex.category or "unknown" for ex in examples},
                        answer_types={ex.id: ex.answer_type for ex in examples},
                        experiment_id=manifest.experiment_id,
                        expected_jobs=expected_jobs_list,
                        semantic_grades=grade_set,
                    )
                    
                    # Accurate count of graded logical jobs
                    graded_jobs = sum(cond.get("graded_jobs", 0) for cond in summary.get("conditions", []))
                    
                    for cond in summary.get("conditions", []):
                        acc = cond.get("planned_job_accuracy", 0.0)
                        if acc > best_accuracy:
                            best_accuracy = acc
                            best_condition = cond["condition"]
                            best_ci = (cond.get("planned_job_accuracy_ci_lower", 0.0), cond.get("planned_job_accuracy_ci_upper", 0.0))
                            
                    workflow_cost = sum(float(cond.get("cost_usd") or 0.0) for cond in summary.get("conditions", []))
                    total_tokens = sum(int(cond.get("total_tokens") or 0) for cond in summary.get("conditions", []))
                except Exception:
                    pass
            
            if graded_jobs == 0 and grade_set is not None:
                graded_jobs = sum(
                    1 for record in grade_set.records.values() if record.status == "success"
                )

            status = get_experiment_status(
                expected_jobs=expected_jobs,
                completed_jobs=completed_jobs,
                graded_jobs=graded_jobs,
                has_running=has_running,
                updated_at_str=updated_at
            )
            
            experiments.append({
                "experiment_id": manifest.experiment_id,
                "purpose": manifest.metadata.get("purpose") or "n/a",
                "status": status,
                "model": manifest.model,
                "judge_model": manifest.judge_model or "n/a",
                "grader_model": manifest.metadata.get("grader_model") or (grade_set.manifest.grader_model if grade_set else "n/a"),
                "task_set": manifest.metadata.get("benchmark") or "n/a",
                "task_set_sha256": manifest.task_set_sha256,
                "task_count": manifest.task_count,
                "conditions_count": len(manifest.conditions),
                "repetitions": manifest.repetitions,
                "expected_jobs": expected_jobs,
                "completed_jobs": completed_jobs,
                "graded_jobs": graded_jobs,
                "workflow_cost": workflow_cost,
                "grading_cost": grading_cost,
                "total_cost": workflow_cost + grading_cost,
                "total_tokens": total_tokens,
                "created_at": manifest.created_at,
                "updated_at": updated_at,
                "best_condition": best_condition,
                "best_accuracy": best_accuracy if best_condition else None,
                "best_ci_lower": best_ci[0] if best_condition else None,
                "best_ci_upper": best_ci[1] if best_condition else None,
            })
        except Exception as exc:
            print(f"Error loading experiment {path.name}: {exc}")
            
    experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return experiments


@app.get("/api/experiments")
def get_experiments():
    return get_experiments_list(RESULTS_DIR)


@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    experiment_root = RESULTS_DIR / experiment_id
    manifest_path = experiment_root / "experiment-manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    try:
        manifest = load_manifest(manifest_path)
        
        ledger_path = experiment_root / "experiment-ledger.json"
        ledger = None
        if ledger_path.exists():
            try:
                ledger = load_ledger(ledger_path, manifest=manifest)
            except Exception:
                pass
                
        tasks_path = Path(manifest.task_set_path)
        if not tasks_path.exists():
            tasks_path = Path(manifest.task_set_path)
            
        if not tasks_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Task set file not found: {manifest.task_set_path}"
            )
            
        examples = load_jsonl(tasks_path)
        
        grade_set = None
        try:
            grade_set = load_grade_set(results_dir=RESULTS_DIR, experiment_id=experiment_id)
        except Exception:
            pass
            
        results = []
        for run_path in sorted(experiment_root.glob("*/result.json")):
            try:
                results.append(json.loads(run_path.read_text(encoding="utf-8")))
            except Exception:
                pass
                
        _preprocess_results_with_ledger(results, ledger)
        expected_jobs_list = None
        if ledger is not None:
            expected_jobs_list = [
                {
                    "job_id": record.spec.job_id,
                    "condition": record.spec.condition_id,
                    "task_id": record.spec.task_id,
                    "repetition": record.spec.repetition,
                    "attempt_count": record.attempt_count,
                }
                for record in ledger.jobs.values()
            ]
            
        summary, run_rows, stage_rows = analyze_attempts(
            results,
            expected_answers={ex.id: ex.answer for ex in examples},
            answer_specs={
                ex.id: AnswerSpec(
                    type=ex.answer_type,  # type: ignore[arg-type]
                    choices=[
                        AnswerChoice(label=choice.label, text=choice.text)
                        for choice in ex.choices
                    ] if getattr(ex, "choices", None) else [],
                    include_confidence=True,
                )
                for ex in examples
            },
            categories={ex.id: ex.category or "unknown" for ex in examples},
            answer_types={ex.id: ex.answer_type for ex in examples},
            experiment_id=experiment_id,
            expected_jobs=expected_jobs_list,
            semantic_grades=grade_set,
        )
        
        warnings = []
        total_tasks = len(examples)
        if total_tasks < 10:
            warnings.append(
                f"Small sample size: This experiment uses only {total_tasks} tasks. "
                "Confidence intervals will be wide and accuracy differences are exploratory."
            )
        missing_count = sum(cond.get("missing_jobs", 0) for cond in summary.get("conditions", []))
        if missing_count > 0:
            warnings.append(f"Missing jobs: {missing_count} jobs were planned but have no run outputs.")
        invalid_count = sum(cond.get("contract_invalid_outputs", 0) for cond in summary.get("conditions", []))
        if invalid_count > 0:
            warnings.append(f"Invalid contract outputs: {invalid_count} jobs generated responses violating the output schema.")
        grade_fail_count = sum(cond.get("grading_failures", 0) for cond in summary.get("conditions", []))
        if grade_fail_count > 0:
            warnings.append(f"Grading failures: {grade_fail_count} semantic grading attempts failed.")
            
        tasks_details = [
            {
                "id": ex.id,
                "prompt": ex.prompt,
                "choices": [{"label": choice.label, "text": choice.text} for choice in ex.choices] if getattr(ex, "choices", None) else [],
                "answer": ex.answer,
                "answer_type": ex.answer_type,
                "category": ex.category or "unknown"
            }
            for ex in examples
        ]
        
        return {
            "manifest": manifest.to_dict(),
            "ledger": ledger.to_dict() if ledger else None,
            "summary": summary,
            "runs": run_rows,
            "stages": stage_rows,
            "tasks": tasks_details,
            "warnings": warnings,
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing experiment: {exc}")


@app.get("/api/experiments/{experiment_id}/runs/{run_id}")
def get_run_trace(experiment_id: str, run_id: str):
    try:
        run_dir = RESULTS_DIR / experiment_id / run_id
        result_path = run_dir / "result.json"
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Run result not found")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        
        request_path = run_dir / "request.json"
        request = None
        if request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except Exception:
                pass
                
        artifact = RunArtifact(path=run_dir, result=result, request=request)
        normalized = normalize_run(artifact)
        return normalized
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading run detail: {exc}")


class ComparisonRequest(BaseModel):
    experiment_ids: list[str]

@app.post("/api/compare")
def compare_experiments(req: ComparisonRequest):
    if len(req.experiment_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least two experiments to compare")
        
    experiments_data = []
    for exp_id in req.experiment_ids:
        path = RESULTS_DIR / exp_id
        manifest_path = path / "experiment-manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
        manifest = load_manifest(manifest_path)
        
        try:
            tasks_path = Path(manifest.task_set_path)
            examples = load_jsonl(tasks_path)
            results = []
            for run_path in sorted(path.glob("*/result.json")):
                try:
                    results.append(json.loads(run_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            grade_set = None
            try:
                grade_set = load_grade_set(results_dir=RESULTS_DIR, experiment_id=exp_id)
            except Exception:
                pass
            ledger_path = path / "experiment-ledger.json"
            expected_jobs_list = None
            ledger = None
            if ledger_path.exists():
                ledger = load_ledger(ledger_path, manifest=manifest)
                expected_jobs_list = [
                    {
                        "job_id": record.spec.job_id,
                        "condition": record.spec.condition_id,
                        "task_id": record.spec.task_id,
                        "repetition": record.spec.repetition,
                        "attempt_count": record.attempt_count,
                    }
                    for record in ledger.jobs.values()
                ]
            _preprocess_results_with_ledger(results, ledger)
            summary, run_rows, _ = analyze_attempts(
                results,
                expected_answers={ex.id: ex.answer for ex in examples},
                answer_specs={
                    ex.id: AnswerSpec(
                        type=ex.answer_type,  # type: ignore[arg-type]
                        choices=[
                            AnswerChoice(label=choice.label, text=choice.text)
                            for choice in ex.choices
                        ] if getattr(ex, "choices", None) else [],
                        include_confidence=True,
                    )
                    for ex in examples
                },
                experiment_id=exp_id,
                expected_jobs=expected_jobs_list,
                semantic_grades=grade_set,
            )
            experiments_data.append({
                "id": exp_id,
                "manifest": manifest,
                "summary": summary,
                "runs": run_rows,
                "grader_model": summary.get("metadata", {}).get("grader_model") or (grade_set.manifest.grader_model if grade_set else None)
            })
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error loading experiment {exp_id} for comparison: {exc}")
            
    ref = experiments_data[0]
    comparable = True
    reasons_incomparable = []
    
    for other in experiments_data[1:]:
        if other["manifest"].task_set_sha256 != ref["manifest"].task_set_sha256:
            comparable = False
            reasons_incomparable.append(
                f"Experiments use different task sets (SHA mismatch: {ref['id']} vs {other['id']})"
            )
        if other["manifest"].model != ref["manifest"].model:
            comparable = False
            reasons_incomparable.append(
                f"Experiments use different primary models ({ref['manifest'].model} vs {other['manifest'].model})"
            )
        if other["manifest"].repetitions != ref["manifest"].repetitions:
            comparable = False
            reasons_incomparable.append(
                f"Experiments use different repetition policies ({ref['manifest'].repetitions} vs {other['manifest'].repetitions})"
            )
        if other["grader_model"] != ref["grader_model"]:
            comparable = False
            reasons_incomparable.append(
                f"Experiments use different grading models ({ref['grader_model']} vs {other['grader_model']})"
            )
            
    paired_deltas = []
    if comparable:
        for idx_a, exp_a in enumerate(experiments_data):
            for idx_b, exp_b in enumerate(experiments_data):
                if idx_a >= idx_b:
                    continue
                runs_by_key_a = defaultdict(dict)
                for r in exp_a["runs"]:
                    if r["correct"] is not None:
                        runs_by_key_a[r["condition"]][(r["task_id"], r["repetition"])] = r
                        
                runs_by_key_b = defaultdict(dict)
                for r in exp_b["runs"]:
                    if r["correct"] is not None:
                        runs_by_key_b[r["condition"]][(r["task_id"], r["repetition"])] = r
                        
                all_conditions = sorted(set(runs_by_key_a.keys()) | set(runs_by_key_b.keys()))
                for cond in all_conditions:
                    pairs_a = runs_by_key_a.get(cond, {})
                    pairs_b = runs_by_key_b.get(cond, {})
                    matched_keys = sorted(set(pairs_a.keys()) & set(pairs_b.keys()))
                    if not matched_keys:
                        continue
                    correct_a = sum(1 for k in matched_keys if pairs_a[k]["correct"] is True)
                    correct_b = sum(1 for k in matched_keys if pairs_b[k]["correct"] is True)
                    count = len(matched_keys)
                    
                    diffs_by_task = defaultdict(list)
                    for k in matched_keys:
                        val_a = 1 if pairs_a[k]["correct"] is True else 0
                        val_b = 1 if pairs_b[k]["correct"] is True else 0
                        diffs_by_task[k[0]].append(val_b - val_a)
                        
                    ci_lower, ci_upper = task_clustered_bootstrap_ci(diffs_by_task)
                    
                    paired_deltas.append({
                        "experiment_a": exp_a["id"],
                        "experiment_b": exp_b["id"],
                        "condition": cond,
                        "matched_pairs": count,
                        "accuracy_a": correct_a / count,
                        "accuracy_b": correct_b / count,
                        "accuracy_delta": (correct_b - correct_a) / count,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper
                    })
                    
    return {
        "experiments": [
            {
                "id": exp["id"],
                "purpose": exp["manifest"].metadata.get("purpose") or "n/a",
                "model": exp["manifest"].model,
                "task_set": exp["manifest"].metadata.get("benchmark") or "n/a",
                "conditions": exp["summary"].get("conditions", [])
            }
            for exp in experiments_data
        ],
        "comparable": comparable,
        "warnings": reasons_incomparable,
        "paired_deltas": paired_deltas
    }


STATIC_DIR = Path("extensions/benchmark_tools/dashboard_static")
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def serve(
    results_dir: Path | str = "results",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    global RESULTS_DIR
    RESULTS_DIR = Path(results_dir)
    
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"MART Local Results Dashboard starting...")
    print(f"Discovered experiments will be loaded from: {RESULTS_DIR.resolve()}")
    print(f"Open your browser to: {url}")
    print(f"=======================================================\n")
    
    if open_browser:
        loop = asyncio.get_event_loop()
        def open_tab():
            try:
                webbrowser.open(url)
            except Exception:
                pass
        loop.call_later(1.0, open_tab)
        
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
