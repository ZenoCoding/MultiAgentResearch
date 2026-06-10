from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from multi_agent_research.models import RunRequest, RunResult


class FileRunStore:
    def __init__(self, root: Path | str = "results") -> None:
        self.root = Path(root)

    def save(
        self,
        request: RunRequest,
        result: RunResult,
        *,
        source_snapshot: bytes,
    ) -> Path:
        run_dir = self.root / request.experiment_id / request.id
        run_dir.mkdir(parents=True, exist_ok=True)
        source_path = self._cache_source_snapshot(
            request.provenance.source_snapshot_sha256,
            source_snapshot,
        )

        (run_dir / "request.json").write_text(
            request.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / "result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / "provenance.json").write_text(
            request.provenance.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / "source-reference.json").write_text(
            json.dumps(
                {
                    "sha256": request.provenance.source_snapshot_sha256,
                    "path": source_path.relative_to(self.root).as_posix(),
                    "size_bytes": source_path.stat().st_size,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
        artifact_paths = [
            run_dir / "request.json",
            run_dir / "result.json",
            run_dir / "provenance.json",
            run_dir / "source-reference.json",
            run_dir / "calls.jsonl",
            run_dir / "events.jsonl",
        ]
        manifest = {
            path.name: {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        }
        (run_dir / "artifact-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_dir

    def _cache_source_snapshot(
        self,
        expected_sha256: str,
        source_snapshot: bytes,
    ) -> Path:
        actual_sha256 = sha256(source_snapshot).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Source snapshot hash does not match provenance")

        cache_dir = self.root / "_artifacts" / "sources"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{expected_sha256}.tar.gz"
        if cache_path.exists():
            cached_sha256 = sha256(cache_path.read_bytes()).hexdigest()
            if cached_sha256 != expected_sha256:
                raise ValueError(f"Corrupt cached source snapshot: {cache_path}")
            return cache_path

        temporary_path = cache_dir / f".{expected_sha256}.{uuid4()}.tmp"
        try:
            temporary_path.write_bytes(source_snapshot)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return cache_path

    @staticmethod
    def _write_jsonl(path: Path, rows: list[str]) -> None:
        content = "\n".join(rows)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
