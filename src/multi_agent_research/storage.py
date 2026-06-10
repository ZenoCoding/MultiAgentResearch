from __future__ import annotations

from pathlib import Path

from multi_agent_research.models import RunRequest, RunResult


class FileRunStore:
    def __init__(self, root: Path | str = "results") -> None:
        self.root = Path(root)

    def save(self, request: RunRequest, result: RunResult) -> Path:
        run_dir = self.root / request.experiment_id / request.id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "request.json").write_text(
            request.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / "result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._write_jsonl(
            run_dir / "calls.jsonl",
            [call.model_dump_json() for call in result.calls],
        )
        self._write_jsonl(
            run_dir / "events.jsonl",
            [event.model_dump_json() for event in result.events],
        )
        return run_dir

    @staticmethod
    def _write_jsonl(path: Path, rows: list[str]) -> None:
        content = "\n".join(rows)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
