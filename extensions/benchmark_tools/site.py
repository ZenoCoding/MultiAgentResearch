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
    for name in ("summary.json", "runs.json", "stage_answers.json", "runs.csv", "stage_answers.csv", "summary.csv"):
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
        f"<td>{row['accuracy']:.1%}</td>"
        f"<td>{row['correct']}/{row['runs']}</td>"
        f"<td>{int(row['total_tokens']):,}</td>"
        f"<td>${float(row['cost_usd']):.4f}</td>"
        f"<td>{row['contract_valid_rate']:.1%}</td>"
        "</tr>"
        for row in summary["conditions"]
    )
    run_rows = "\n".join(
        "<tr>"
        f"<td>{_e(row['task_id'])}</td>"
        f"<td>{_e(row['condition'])}</td>"
        f"<td>{'correct' if row['correct'] else 'wrong'}</td>"
        f"<td>{_e(row['final_answer'])}</td>"
        f"<td>{_e(row['expected_answer'])}</td>"
        f"<td>{_e(row['error_reason'])}</td>"
        f"<td>{int(row['total_tokens']):,}</td>"
        f"<td>${float(row['cost_usd']):.4f}</td>"
        "</tr>"
        for row in runs
    )
    cards = _cards(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Agent Benchmark Report</title>
  <style>
    :root {{ color-scheme: light; --ink:#111827; --muted:#6b7280; --line:#d1d5db; --bg:#f8fafc; --panel:#ffffff; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:28px 32px 18px; background:#ffffff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:18px; }}
    p {{ color:var(--muted); margin:0; }}
    main {{ padding:24px 32px 40px; display:grid; gap:22px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .metric {{ font-size:26px; font-weight:750; }}
    .label {{ color:var(--muted); font-size:13px; margin-top:4px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; overflow:auto; }}
    .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }}
    .charts img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:white; }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#f9fafb; font-size:12px; color:#374151; }}
    .filter {{ margin-bottom:10px; width:320px; max-width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Multi-Agent Benchmark Report</h1>
    <p>Static report regenerated from saved run artifacts and benchmark references.</p>
  </header>
  <main>
    <div class="cards">{cards}</div>
    <section>
      <h2>Condition Summary</h2>
      <table><thead><tr><th>Condition</th><th>Accuracy</th><th>Correct</th><th>Tokens</th><th>Cost</th><th>Contract Valid</th></tr></thead><tbody>{condition_rows}</tbody></table>
    </section>
    <section>
      <h2>Charts</h2>
      <div class="charts">
        <img src="charts/accuracy_by_condition.svg" alt="Accuracy by condition">
        <img src="charts/cost_vs_accuracy.svg" alt="Cost versus accuracy">
        <img src="charts/tokens_vs_accuracy.svg" alt="Tokens versus accuracy">
        <img src="charts/error_reasons.svg" alt="Error reasons">
        <img src="charts/revision_transitions.svg" alt="Revision transitions">
        <img src="charts/question_heatmap.svg" alt="Question heatmap">
      </div>
    </section>
    <section>
      <h2>Run Details</h2>
      <input id="filter" class="filter" placeholder="Filter task, condition, answer, reason">
      <table id="runs"><thead><tr><th>Task</th><th>Condition</th><th>Result</th><th>Answer</th><th>Expected</th><th>Reason</th><th>Tokens</th><th>Cost</th></tr></thead><tbody>{run_rows}</tbody></table>
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
    total_cost = sum(float(row["cost_usd"]) for row in rows)
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    best = max(rows, key=lambda row: row["accuracy"], default={"condition": "n/a", "accuracy": 0})
    total_runs = sum(int(row["runs"]) for row in rows)
    return (
        _card("Best accuracy", f"{best['accuracy']:.1%}", str(best["condition"]))
        + _card("Total runs", f"{total_runs:,}", "scored result files")
        + _card("Total tokens", f"{total_tokens:,}", "all conditions")
        + _card("Total cost", f"${total_cost:.4f}", "provider estimate")
    )


def _card(label: str, value: str, detail: str) -> str:
    return f'<div class="card"><div class="metric">{_e(value)}</div><div class="label">{_e(label)} · {_e(detail)}</div></div>'


def _e(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

