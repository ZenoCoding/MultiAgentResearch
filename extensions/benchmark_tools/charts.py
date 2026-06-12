from __future__ import annotations

import json
from pathlib import Path
from typing import Any


COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def write_charts(*, analysis_dir: Path | str, output_dir: Path | str) -> list[Path]:
    analysis = Path(analysis_dir)
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    runs = json.loads((analysis / "runs.json").read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "accuracy_by_condition.svg",
        output / "cost_vs_accuracy.svg",
        output / "tokens_vs_accuracy.svg",
        output / "error_reasons.svg",
        output / "revision_transitions.svg",
        output / "question_heatmap.svg",
    ]
    paths[0].write_text(bar_chart(summary["conditions"], "condition", "accuracy", "Accuracy by condition", percent=True), encoding="utf-8")
    paths[1].write_text(scatter_chart(summary["conditions"], "cost_usd", "accuracy", "Cost vs accuracy", x_label="Cost USD", y_label="Accuracy"), encoding="utf-8")
    paths[2].write_text(scatter_chart(summary["conditions"], "total_tokens", "accuracy", "Tokens vs accuracy", x_label="Total tokens", y_label="Accuracy"), encoding="utf-8")
    paths[3].write_text(counter_chart(summary["error_reasons"], "Incorrect answer reasons"), encoding="utf-8")
    paths[4].write_text(grouped_counter_chart(summary["revision_transitions"], "transition", "Revision transitions"), encoding="utf-8")
    paths[5].write_text(heatmap(summary["question_heatmap"], runs), encoding="utf-8")
    return paths


def bar_chart(rows: list[dict[str, Any]], label_key: str, value_key: str, title: str, *, percent: bool = False) -> str:
    width, height = 920, 420
    margin = 70
    chart_h = height - 2 * margin
    max_value = max([float(row[value_key] or 0) for row in rows] + [1])
    bar_w = (width - 2 * margin) / max(len(rows), 1)
    bars = []
    labels = []
    for index, row in enumerate(rows):
        value = float(row[value_key] or 0)
        h = chart_h * value / max_value
        x = margin + index * bar_w + 12
        y = height - margin - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 24:.1f}" height="{h:.1f}" fill="{COLORS[index % len(COLORS)]}"/>')
        display = f"{value:.0%}" if percent else f"{value:.2f}"
        labels.append(f'<text x="{x + (bar_w - 24) / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{display}</text>')
        labels.append(f'<text x="{x + (bar_w - 24) / 2:.1f}" y="{height - 30}" text-anchor="middle" class="tick">{_escape(str(row[label_key]))}</text>')
    return _svg(width, height, title, "".join(bars + labels) + _axis(width, height, margin))


def scatter_chart(rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, *, x_label: str, y_label: str) -> str:
    width, height = 920, 460
    margin = 80
    xs = [float(row[x_key] or 0) for row in rows]
    ys = [float(row[y_key] or 0) for row in rows]
    max_x = max(xs + [1])
    max_y = max(ys + [1])
    parts = [_axis(width, height, margin)]
    for index, row in enumerate(rows):
        x = margin + (width - 2 * margin) * float(row[x_key] or 0) / max_x
        y = height - margin - (height - 2 * margin) * float(row[y_key] or 0) / max_y
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLORS[index % len(COLORS)]}"><title>{_escape(row["condition"])}</title></circle>')
        parts.append(f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" class="tick">{_escape(row["condition"])}</text>')
    parts.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" class="axis-label">{_escape(x_label)}</text>')
    parts.append(f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" class="axis-label">{_escape(y_label)}</text>')
    return _svg(width, height, title, "".join(parts))


def counter_chart(counter: dict[str, int], title: str) -> str:
    rows = [{"label": key, "value": value} for key, value in sorted(counter.items())]
    return bar_chart(rows, "label", "value", title)


def grouped_counter_chart(rows: list[dict[str, Any]], key: str, title: str) -> str:
    counter: dict[str, int] = {}
    for row in rows:
        label = f'{row["condition"]}: {row[key]}'
        counter[label] = int(row["count"])
    return counter_chart(counter, title)


def heatmap(rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> str:
    conditions = sorted({row["condition"] for row in rows})
    tasks = sorted({row["task_id"] for row in rows})
    lookup = {(row["task_id"], row["condition"]): row["correct"] for row in rows}
    cell = 20
    left = 180
    top = 80
    width = left + cell * len(conditions) + 40
    height = top + cell * len(tasks) + 60
    parts = []
    for col, condition in enumerate(conditions):
        x = left + col * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="66" transform="rotate(-45 {x:.1f} 66)" class="tick">{_escape(condition)}</text>')
    for row_index, task in enumerate(tasks):
        y = top + row_index * cell
        parts.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="tick">{_escape(task)}</text>')
        for col, condition in enumerate(conditions):
            value = lookup.get((task, condition))
            color = "#16a34a" if value else "#dc2626"
            if value is None:
                color = "#e5e7eb"
            parts.append(f'<rect x="{left + col * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{color}"><title>{_escape(task)} / {_escape(condition)}</title></rect>')
    return _svg(width, height, "Per-question outcome heatmap", "".join(parts))


def _axis(width: int, height: int, margin: int) -> str:
    return (
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827"/>'
    )


def _svg(width: int, height: int, title: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
.title {{ font-size: 22px; font-weight: 700; }}
.tick {{ font-size: 11px; }}
.value {{ font-size: 12px; font-weight: 650; }}
.axis-label {{ font-size: 13px; font-weight: 650; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="24" y="34" class="title">{_escape(title)}</text>
{body}
</svg>
'''


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

