from __future__ import annotations

from hashlib import sha256

from multi_agent_research.provenance import _environment_provenance


def test_environment_provenance_sanitizes_urls_and_hashes_credentials(
    monkeypatch,
):
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://user:password@example.com/v1?token=secret",
    )
    monkeypatch.setenv("OPENAI_API_VERSION", "2026-01-01")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    environment, credentials = _environment_provenance()

    assert environment["OPENAI_BASE_URL"] == "https://example.com/v1"
    assert environment["OPENAI_API_VERSION"] == "2026-01-01"
    assert credentials["OPENAI_API_KEY"] == sha256(
        b"secret-key"
    ).hexdigest()
