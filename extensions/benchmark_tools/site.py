from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from extensions.benchmark_tools.charts import write_charts


def build_site(*, analysis_dir: Path | str, output_dir: Path | str) -> Path:
    analysis = Path(analysis_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart_dir = output / "charts"
    write_charts(analysis_dir=analysis, output_dir=chart_dir)
    for name in (
        "summary.json",
        "runs.json",
        "stage_answers.json",
        "runs.csv",
        "stage_answers.csv",
        "summary.csv",
    ):
        source = analysis / name
        if source.exists():
            shutil.copy2(source, output / name)
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    runs = json.loads((analysis / "runs.json").read_text(encoding="utf-8"))
    (output / "index.html").write_text(_html(summary, runs), encoding="utf-8")
    return output / "index.html"


def _html(summary: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    condition_rows = "\n".join(
        "<tr>"
        f"<td>{_e(row['condition'])}</td>"
        f"<td>{row['completed_answer_jobs']}/{row['expected_jobs']} "
        f"({row['coverage_rate']:.1%})</td>"
        f"<td>{row['graded_jobs']}/{row['expected_jobs']}</td>"
        f"<td>{row['grading_failures']}</td>"
        f"<td>{row['provider_execution_failures']}</td>"
        f"<td>{row['contract_invalid_outputs']}</td>"
        f"<td>{row['inconclusive_jobs']}</td>"
        f"<td>{row['missing_jobs']}</td>"
        f"<td>{row['planned_job_accuracy']:.1%}</td>"
        f"<td>{_percent(row['graded_accuracy'])}</td>"
        f"<td>{int(row['total_tokens']):,}</td>"
        f"<td>${float(row['cost_usd']):.4f}</td>"
        "</tr>"
        for row in summary["conditions"]
    )
    run_rows = "\n".join(
        "<tr>"
        f"<td>{_e(row['task_id'])}</td>"
        f"<td>{_e(row['repetition'])}</td>"
        f"<td>{_e(row['condition'])}</td>"
        f"<td>{_e(row['outcome'])}</td>"
        f"<td>{row['attempt_count']}</td>"
        f"<td>{_score_label(row['correct'])}</td>"
        f"<td>{_e(row['grading_status'])}</td>"
        f"<td>{_e(row['final_answer'])}</td>"
        f"<td>{_e(row['expected_answer'])}</td>"
        f"<td>{_e(row['error_reason'])}</td>"
        f"<td>{int(row['total_tokens']):,}</td>"
        f"<td>${float(row['cost_usd']):.4f}</td>"
        "</tr>"
        for row in runs
    )
    answer_type_rows = "\n".join(
        "<tr>"
        f"<td>{_e(row['condition'])}</td>"
        f"<td>{_e(row['answer_type'])}</td>"
        f"<td>{row['completed_answer_jobs']}/{row['expected_jobs']}</td>"
        f"<td>{row['graded_jobs']}/{row['expected_jobs']}</td>"
        f"<td>{row['correct']}</td>"
        f"<td>{row['planned_job_accuracy']:.1%}</td>"
        f"<td>{_percent(row['graded_accuracy'])}</td>"
        "</tr>"
        for row in summary.get("answer_type_breakdown", [])
    )
    metadata = _metadata(summary.get("metadata") or {})
    cards = _cards(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Agent Benchmark Report</title>
  <style>
    :root {{ color-scheme:light; --ink:#111827; --muted:#6b7280; --line:#d1d5db; --bg:#f8fafc; --panel:#fff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:28px 32px 18px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    h2 {{ margin:0 0 14px; font-size:18px; }}
    p {{ color:var(--muted); margin:0; }}
    main {{ padding:24px 32px 40px; display:grid; gap:22px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
    .card, section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .card {{ padding:14px; }}
    .metric {{ font-size:26px; font-weight:750; }}
    .label {{ color:var(--muted); font-size:13px; margin-top:4px; }}
    section {{ padding:18px; overflow:auto; }}
    .metadata {{ display:flex; flex-wrap:wrap; gap:8px 18px; color:var(--muted); font-size:13px; margin-top:10px; }}
    .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
    .charts img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:white; }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#f9fafb; font-size:12px; color:#374151; }}
    .filter {{ margin-bottom:10px; width:340px; max-width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Multi-Agent Benchmark Report</h1>
    <p>Coverage and execution health precede accuracy. Each planned logical job is counted once.</p>
    {metadata}
  </header>
  <main>
    <div class="cards">{cards}</div>
    <section>
      <h2>Coverage And Execution Health</h2>
      <table><thead><tr><th>Condition</th><th>Valid completed / planned</th><th>Semantically graded / planned</th><th>Grading failures</th><th>Execution failures</th><th>Contract invalid</th><th>Inconclusive</th><th>Missing</th><th>Planned-job accuracy</th><th>Accuracy among graded</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>{condition_rows}</tbody></table>
    </section>
    <section>
      <h2>Accuracy And Cost Charts</h2>
      <div class="charts">
        <img src="charts/accuracy_by_condition.svg" alt="Planned-job accuracy by condition">
        <img src="charts/cost_vs_accuracy.svg" alt="Cost versus planned-job accuracy">
        <img src="charts/tokens_vs_accuracy.svg" alt="Tokens versus planned-job accuracy">
        <img src="charts/error_reasons.svg" alt="Outcome and error reasons">
        <img src="charts/revision_transitions.svg" alt="Revision transitions">
        <img src="charts/question_heatmap.svg" alt="Question heatmap">
      </div>
    </section>
    <section>
      <h2>Answer-Type Breakdown</h2>
      <table><thead><tr><th>Condition</th><th>Answer type</th><th>Valid completed / planned</th><th>Graded / planned</th><th>Correct</th><th>Planned-job accuracy</th><th>Accuracy among graded</th></tr></thead><tbody>{answer_type_rows}</tbody></table>
    </section>
    <section>
      <h2>Canonical Job Details</h2>
      <input id="filter" class="filter" placeholder="Filter task, condition, outcome, answer, reason">
      <table id="runs"><thead><tr><th>Task</th><th>Repetition</th><th>Condition</th><th>Outcome</th><th>Attempts</th><th>Score</th><th>Grading</th><th>Answer</th><th>Expected</th><th>Reason</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>{run_rows}</tbody></table>
    </section>
  </main>
  <script>
    const filter = document.getElementById('filter');
    const rows = Array.from(document.querySelectorAll('#runs tbody tr'));
    filter.addEventListener('input', () => {{
      const q = filter.value.toLowerCase();
      rows.forEach(row => row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none');
    }});
  </script>
</body>
</html>
"""


def _cards(summary: dict[str, Any]) -> str:
    rows = summary["conditions"]
    expected = sum(int(row["expected_jobs"]) for row in rows)
    completed = sum(int(row["completed_answer_jobs"]) for row in rows)
    graded = sum(int(row["graded_jobs"]) for row in rows)
    grading_failures = sum(int(row["grading_failures"]) for row in rows)
    failures = sum(int(row["provider_execution_failures"]) for row in rows)
    invalid = sum(int(row["contract_invalid_outputs"]) for row in rows)
    missing = sum(int(row["missing_jobs"]) for row in rows)
    total_cost = sum(float(row["cost_usd"]) for row in rows)
    grading_cost = float(
        (summary.get("metadata") or {}).get("grading_cost_usd") or 0.0
    )
    return (
        _card(
            "Valid coverage",
            f"{completed / expected:.1%}" if expected else "n/a",
            f"{completed}/{expected} planned jobs",
        )
        + _card(
            "Semantic grading",
            f"{graded / expected:.1%}" if expected else "n/a",
            f"{graded}/{expected} graded · {grading_failures} failures",
        )
        + _card(
            "Execution health",
            f"{failures + invalid + missing:,}",
            f"{failures} failed · {invalid} invalid · {missing} missing",
        )
        + _card(
            "Canonical attempts",
            f"{sum(int(row['attempts']) for row in rows):,}",
            f"{sum(int(row['retried_jobs']) for row in rows)} retried jobs",
        )
        + _card("Total cost", f"${total_cost:.4f}", "selected attempts")
        + _card("Grading cost", f"${grading_cost:.4f}", "semantic judge calls")
    )


def _metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    items = "".join(
        f"<span><strong>{_e(key.replace('_', ' ').title())}:</strong> {_e(value)}</span>"
        for key, value in sorted(metadata.items())
    )
    return f'<div class="metadata">{items}</div>'


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _score_label(value: bool | None) -> str:
    if value is None:
        return "ungraded"
    return "correct" if value else "incorrect"


def _card(label: str, value: str, detail: str) -> str:
    return (
        f'<div class="card"><div class="metric">{_e(value)}</div>'
        f'<div class="label">{_e(label)} · {_e(detail)}</div></div>'
    )


def _e(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
