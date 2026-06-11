from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any
import webbrowser


STATIC_ROOT = files("multi_agent_research").joinpath("viewer_static")
VERDICT_PATTERN = re.compile(r"\b(RESOLVED|UNRESOLVED|CONCEDED)\b", re.I)


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    result: dict[str, Any]
    request: dict[str, Any] | None


def discover_runs(root: Path) -> list[RunArtifact]:
    artifacts: list[RunArtifact] = []
    if not root.exists():
        return artifacts
    for result_path in root.rglob("result.json"):
        if "_artifacts" in result_path.parts:
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        request_path = result_path.with_name("request.json")
        request = None
        if request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        artifacts.append(
            RunArtifact(path=result_path.parent, result=result, request=request)
        )
    return sorted(
        artifacts,
        key=lambda artifact: artifact.result.get("started_at", ""),
        reverse=True,
    )


def run_summary(artifact: RunArtifact) -> dict[str, Any]:
    result = artifact.result
    metrics = result.get("metrics") or {}
    workflow = result.get("workflow") or {}
    return {
        "run_id": result.get("run_id"),
        "experiment_id": result.get("experiment_id"),
        "task_id": result.get("task_id"),
        "workflow": workflow.get("name", "unknown"),
        "workflow_version": workflow.get("version"),
        "status": result.get("status"),
        "final_answer": result.get("final_answer"),
        "started_at": result.get("started_at"),
        "total_tokens": metrics.get("total_tokens", 0),
        "wall_time_ms": metrics.get("wall_time_ms", 0),
        "model_calls": metrics.get("model_calls", 0),
        "cross_examination": workflow.get("name") == "cross_examination_debate",
        "relative_path": artifact.path.as_posix(),
    }


def normalize_run(artifact: RunArtifact) -> dict[str, Any]:
    result = artifact.result
    calls = result.get("calls") or []
    stages = result.get("stage_answers") or []
    workflow = result.get("workflow") or {}
    config = workflow.get("config") or {}
    call_by_id = {call.get("id"): call for call in calls}
    call_by_step = {call.get("step"): call for call in calls}

    agents = []
    agents_config = []
    if "agents" in config:
        for agent in config["agents"] or []:
            agents_config.append((agent, None))
    elif "agent" in config:
        agents_config.append((config["agent"], None))
    else:
        if "worker" in config:
            agents_config.append((config["worker"], "worker"))
        if "supervisor" in config:
            agents_config.append((config["supervisor"], "supervisor"))

    for index, (agent, default_role) in enumerate(agents_config):
        agents.append(
            {
                "id": agent.get("id", f"agent-{index + 1}"),
                "model": agent.get("model"),
                "role": _agent_role(calls, agent.get("id")) or default_role,
                "index": index,
            }
        )
    judge = config.get("judge")
    if judge:
        agents.append(
            {
                "id": judge.get("id", "judge"),
                "model": judge.get("model"),
                "role": "judge",
                "index": len(agents),
            }
        )

    stage_cards = []
    for stage in stages:
        call = call_by_id.get(stage.get("call_id")) or call_by_step.get(
            stage.get("step")
        )
        stage_cards.append(_stage_card(stage, call))

    exchanges = _cross_examination_exchanges(result, calls)
    phase_order = _phase_order(stage_cards, calls)
    task = (artifact.request or {}).get("task")

    return {
        "summary": run_summary(artifact),
        "workflow": workflow,
        "task": task,
        "agents": agents,
        "phase_order": phase_order,
        "stage_cards": stage_cards,
        "exchanges": exchanges,
        "calls": [_call_detail(call) for call in calls],
        "events": result.get("events") or [],
        "output": result.get("output"),
        "error": result.get("error"),
        "inconclusive": result.get("inconclusive"),
        "provenance": result.get("provenance"),
    }


def _agent_role(calls: list[dict[str, Any]], agent_id: str | None) -> str | None:
    for call in calls:
        if call.get("agent_id") != agent_id:
            continue
        role = (call.get("metadata") or {}).get("debate_role")
        if role:
            return str(role).rsplit(".", 1)[-1]
    return None


def _stage_card(
    stage: dict[str, Any], call: dict[str, Any] | None
) -> dict[str, Any]:
    metadata = stage.get("metadata") or {}
    output = stage.get("output") or {}
    phase = _stage_phase(stage.get("step", ""), metadata)
    usage = (call or {}).get("usage") or {}
    return {
        "sequence": stage.get("sequence"),
        "step": stage.get("step"),
        "phase": phase,
        "phase_label": _phase_label(phase),
        "round": metadata.get("round"),
        "kind": stage.get("kind"),
        "agent_id": stage.get("agent_id") or "aggregate",
        "call_id": stage.get("call_id"),
        "answer": output.get("answer"),
        "confidence": output.get("confidence"),
        "contract_valid": output.get("contract_valid"),
        "raw_response": output.get("raw_response"),
        "total_tokens": usage.get("total_tokens", 0),
        "latency_ms": (call or {}).get("latency_ms", 0),
        "metadata": metadata,
    }


def _stage_phase(step: str, metadata: dict[str, Any]) -> str:
    explicit = metadata.get("phase")
    if explicit == "debate":
        return f"round_{metadata.get('round', 1)}"
    if explicit in {"judge", "aggregation", "semantic_vote_judge", "tie_break_judge"}:
        return "aggregate"
    if step in {"judge", "aggregation", "semantic_vote_judge", "tie_break_judge"}:
        return "aggregate"
    if explicit:
        return str(explicit)
    if step.startswith("initial_") or step in {"answer", "worker_initial"}:
        return "initial"
    if step.startswith("sample_"):
        return "samples"
    if step.startswith("debate_"):
        parts = step.split("_")
        return f"round_{parts[1]}" if len(parts) > 1 else "debate"
    if step.startswith("final_revision"):
        return "final_revision"
    if step.startswith("revision_") or step.startswith("worker_revision_"):
        return step
    if "revision" in step:
        return step.rsplit("_", 1)[0]
    return step


def _phase_label(phase: str) -> str:
    if phase == "initial":
        return "Initial"
    if phase.startswith("round_"):
        return f"Round {phase.split('_', 1)[1]}"
    return {
        "samples": "Samples",
        "final_revision": "Final",
        "aggregate": "Judge / Aggregate",
        "answer": "Answer",
    }.get(phase, phase.replace("_", " ").title())


def _phase_order(
    cards: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> list[dict[str, str]]:
    phases: list[str] = []
    for card in cards:
        phase = card["phase"]
        if phase not in phases:
            phases.append(phase)
    if any((call.get("metadata") or {}).get("phase") == "claim_extraction" for call in calls):
        insertion = phases.index("final_revision") if "final_revision" in phases else len(phases)
        phases.insert(insertion, "cross_examination")
    aggregate = next(
        (card for card in reversed(cards) if card["kind"] == "aggregate"),
        None,
    )
    ordered = []
    for phase in phases:
        label = _phase_label(phase)
        if phase == "aggregate" and aggregate:
            label = {
                "judge": "Judge",
                "semantic_vote_judge": "Vote Judge",
                "aggregation": "Aggregate",
            }.get(aggregate["step"], "Aggregate")
        ordered.append({"id": phase, "label": label})
    return ordered


def _cross_examination_exchanges(
    result: dict[str, Any], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    calls_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for call in calls:
        metadata = call.get("metadata") or {}
        phase = metadata.get("phase")
        if phase not in {"challenge", "response", "verdict"}:
            continue
        key = (
            int(metadata.get("round", 0)),
            int(metadata.get("exchange_index", 0)),
            phase,
        )
        calls_by_key[key] = call

    exchanges = []
    for event in result.get("events") or []:
        if event.get("type") != "cross_examination_exchange":
            continue
        data = event.get("data") or {}
        round_number = int(data.get("round", 0))
        exchange_index = int(data.get("exchange_index", 0))
        phase_calls = {
            phase: calls_by_key.get((round_number, exchange_index, phase))
            for phase in ("challenge", "response", "verdict")
        }
        verdict = str(data.get("verdict", ""))
        match = VERDICT_PATTERN.search(verdict)
        exchanges.append(
            {
                "id": f"r{round_number}-e{exchange_index}",
                "round": round_number,
                "exchange_index": exchange_index,
                "challenger_id": data.get("challenger_id"),
                "target_id": data.get("target_id"),
                "challenge": data.get("challenge"),
                "response": data.get("response"),
                "verdict": verdict,
                "verdict_status": match.group(1).lower() if match else "unknown",
                "calls": {
                    phase: _call_detail(call) if call else None
                    for phase, call in phase_calls.items()
                },
                "total_tokens": sum(
                    ((call or {}).get("usage") or {}).get("total_tokens", 0)
                    for call in phase_calls.values()
                ),
                "latency_ms": sum(
                    (call or {}).get("latency_ms", 0)
                    for call in phase_calls.values()
                ),
            }
        )
    return exchanges


def _call_detail(call: dict[str, Any]) -> dict[str, Any]:
    output = call.get("output") or {}
    return {
        "id": call.get("id"),
        "sequence": call.get("sequence"),
        "step": call.get("step"),
        "agent_id": call.get("agent_id"),
        "requested_model": call.get("requested_model"),
        "response_model": call.get("response_model"),
        "status": call.get("status"),
        "messages": call.get("messages") or [],
        "output": output,
        "usage": call.get("usage") or {},
        "cost_usd": call.get("cost_usd"),
        "latency_ms": call.get("latency_ms"),
        "started_at": call.get("started_at"),
        "ended_at": call.get("ended_at"),
        "metadata": call.get("metadata") or {},
        "request_parameters": call.get("request_parameters") or {},
        "prompt_references": call.get("prompt_references") or [],
        "error": call.get("error"),
    }


class ViewerApplication:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def summaries(self) -> list[dict[str, Any]]:
        return [run_summary(artifact) for artifact in discover_runs(self.root)]

    def detail(self, run_id: str) -> dict[str, Any] | None:
        for artifact in discover_runs(self.root):
            if artifact.result.get("run_id") == run_id:
                return normalize_run(artifact)
        return None


def handler_for(application: ViewerApplication) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/runs":
                self._json(application.summaries())
                return
            if path.startswith("/api/runs/"):
                detail = application.detail(path.rsplit("/", 1)[-1])
                if detail is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._json(detail)
                return
            asset = "index.html" if path in {"", "/"} else path.lstrip("/")
            allowed_assets = {
                "index.html",
                "app.js",
                "style.css",
            }
            if asset not in allowed_assets:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            resource = STATIC_ROOT.joinpath(asset)
            content = resource.read_bytes()
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
            }[Path(asset).suffix]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, value: Any) -> None:
            content = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return ViewerHandler


def serve(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    application = ViewerApplication(root)
    server = ThreadingHTTPServer((host, port), handler_for(application))
    url = f"http://{host}:{server.server_port}"
    print(f"M.A.R.T. viewing {application.root}")
    print(url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mar view",
        description="Browse persisted multi-agent run traces.",
    )
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(
        Path(args.results_dir),
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
