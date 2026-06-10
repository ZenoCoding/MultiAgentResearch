from __future__ import annotations

import gzip
from hashlib import sha256
import importlib.metadata
from io import BytesIO
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit, urlunsplit

from multi_agent_research.models import GitProvenance, RunProvenance


EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "results",
}
EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
}
RELEVANT_ENVIRONMENT = {
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
    "GOOGLE_CLOUD_PROJECT",
    "LITELLM_DROP_PARAMS",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "LITELLM_LOG",
    "OPENAI_API_BASE",
    "OPENAI_API_VERSION",
    "OPENAI_BASE_URL",
    "VERTEXAI_LOCATION",
    "VERTEXAI_PROJECT",
}
URL_ENVIRONMENT = {
    "AZURE_API_BASE",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
}
SECRET_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_PASSWORD",
    "_SECRET",
    "_SECRET_KEY",
    "_TOKEN",
)


def capture_run_provenance(
    cwd: Path | str | None = None,
) -> tuple[RunProvenance, bytes]:
    working_directory = Path(cwd or Path.cwd()).resolve()
    repository_root = _git_output(
        working_directory,
        "rev-parse",
        "--show-toplevel",
    )
    source_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else working_directory
    )
    source_snapshot, source_files = _source_snapshot(source_root)
    lockfile = source_root / "uv.lock"
    lockfile_sha256 = (
        sha256(lockfile.read_bytes()).hexdigest() if lockfile.is_file() else None
    )
    environment, credential_fingerprints = _environment_provenance()
    secret_file_sha256 = _secret_file_hashes(source_root)

    git = None
    if repository_root is not None:
        status = _git_output(source_root, "status", "--short")
        git = GitProvenance(
            repository_root=str(source_root),
            commit=_git_output(source_root, "rev-parse", "HEAD"),
            branch=_git_output(source_root, "branch", "--show-current"),
            dirty=bool(status),
            status_short=status or "",
            remote_url=_git_output(
                source_root,
                "config",
                "--get",
                "remote.origin.url",
            ),
        )

    provenance = RunProvenance(
        working_directory=str(working_directory),
        argv=list(sys.argv),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        harness_version=_package_version("multi-agent-research"),
        litellm_version=_package_version("litellm"),
        dependency_versions=_dependency_versions(),
        environment=environment,
        credential_fingerprints=credential_fingerprints,
        secret_file_sha256=secret_file_sha256,
        lockfile_sha256=lockfile_sha256,
        source_snapshot_sha256=sha256(source_snapshot).hexdigest(),
        source_files=source_files,
        git=git,
    )
    return provenance, source_snapshot


def _source_snapshot(root: Path) -> tuple[bytes, dict[str, str]]:
    files = _source_files(root)
    manifest = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    compressed = BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        compresslevel=9,
        mtime=0,
    ) as output:
        output.write(tar_buffer.getvalue())
    return compressed.getvalue(), dict(sorted(manifest.items()))


def _source_files(root: Path) -> list[Path]:
    git_files = _git_output(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if git_files is not None:
        candidates = [
            root / relative
            for relative in git_files.split("\0")
            if relative
        ]
    else:
        candidates = list(root.rglob("*"))
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and not _excluded(path, root)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    return path.name.startswith(".env.") and path.name != ".env.example"


def _git_output(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace").rstrip("\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dependency_versions() -> dict[str, str]:
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return dict(sorted(versions.items(), key=lambda item: item[0].casefold()))


def _environment_provenance() -> tuple[dict[str, str], dict[str, str]]:
    environment = {}
    credentials = {}
    for name, value in os.environ.items():
        if name in RELEVANT_ENVIRONMENT:
            environment[name] = (
                _sanitized_url(value) if name in URL_ENVIRONMENT else value
            )
        if name.endswith(SECRET_SUFFIXES):
            credentials[name] = sha256(value.encode("utf-8")).hexdigest()
    return dict(sorted(environment.items())), dict(sorted(credentials.items()))


def _secret_file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _sanitized_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname += f":{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
