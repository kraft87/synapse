"""The first-device bootstrap CLI (`synapse-admin`).

Fresh installs have no IdP and no trusted device, so this is the ONLY path to a working
credential. It ships as a console script inside the image (pyproject [project.scripts])
because the Dockerfile copies packages, not scripts/ — a break-glass tool that isn't in
the image is not a break-glass tool.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ingestion import surface_admin


@pytest.fixture
def minted(monkeypatch):
    """Capture what the CLI asks surfaces.mint_surface for, and hand back a fake grant."""
    calls: list[tuple] = []

    def _mint(db_url, label, trust, projects):
        calls.append((db_url, label, trust, projects))
        return {
            "surface": {"surface_id": "dev-abc123", "trust": trust, "label": label},
            "token": "sk-not-a-real-token",
        }

    monkeypatch.setattr(surface_admin, "mint_surface", _mint)
    return calls


def test_bootstrap_prints_the_token_and_where_to_paste_it(minted, capsys):
    assert surface_admin.main(["--db-url", "postgresql://x/y", "bootstrap", "audit-host"]) == 0
    out = capsys.readouterr().out
    assert "dev-abc123" in out
    assert "sk-not-a-real-token" in out
    assert "Shown once" in out
    # The token is useless without this line — it is the step a new user is stuck on.
    assert "SYNAPSE_INGEST_TOKEN" in out
    assert minted == [("postgresql://x/y", "audit-host", "full", [])]


def test_mint_still_defaults_to_restricted(minted, capsys):
    assert surface_admin.main(["--db-url", "postgresql://x/y", "mint", "laptop"]) == 0
    assert minted == [("postgresql://x/y", "laptop", "restricted", None)]
    assert "SYNAPSE_INGEST_TOKEN" in capsys.readouterr().out


def test_mint_full_with_projects(minted):
    assert (
        surface_admin.main(
            ["--db-url", "postgresql://x/y", "mint", "box", "--full", "--projects", "a, b"]
        )
        == 0
    )
    assert minted == [("postgresql://x/y", "box", "full", ["a", "b"])]


def test_no_dsn_is_an_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.delenv("SYNAPSE_DB_URL", raising=False)
    assert surface_admin.main(["list"]) == 2
    assert "SYNAPSE_DB_URL" in capsys.readouterr().err


def test_console_script_is_declared():
    """Without this entry there is no `synapse-admin` on PATH in the image, and the
    documented one-command bootstrap silently isn't a command."""
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["scripts"]["synapse-admin"] == "ingestion.surface_admin:main"
