"""Tests for scripts/import_notes.py — the notes-store seed importer.

Two halves, mirroring test_notes_store.py:
  * Parser + type-mapping tests: pure Python, no database, always run.
  * Import-flow tests against the shared Postgres test DB (skipped cleanly when
    it's down): dry-run writes nothing, --apply creates rows keyed on
    source_ref='import:<name>', re-runs skip, edits update in place, and a
    high-sim collision with an existing note routes through reconcile_note
    (embedder stubbed, LLM confirm disabled via its kill switch).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "import_notes.py"

_spec = importlib.util.spec_from_file_location("import_notes_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imp)

from ingestion.db import Database  # noqa: E402
from ingestion.embedding import embed_dims  # noqa: E402
from ingestion.notes import _OWNER  # noqa: E402

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
    _DB_UP = True
except Exception:  # pragma: no cover - environment dependent
    _DB_UP = False

needs_db = pytest.mark.skipif(not _DB_UP, reason="no test DB reachable")

_DIMS = embed_dims()


# ---------------------------------------------------------------------------
# Parser — pure Python
# ---------------------------------------------------------------------------

_FENCED = """---
name: gpu_inventory
description: Only local GPU is a 12GB card
metadata:
  type: user
  project: homelab
---
Body first line.
More body.
"""


def test_fenced_frontmatter_parsed():
    parsed = imp.parse_memory_file(_FENCED, "some_file")
    assert parsed["name"] == "gpu_inventory"
    assert parsed["hook"] == "Only local GPU is a 12GB card"
    assert parsed["body"] == "Body first line.\nMore body."
    assert parsed["meta_type"] == "user"
    assert parsed["meta_project"] == "homelab"


def test_nested_metadata_block_and_quotes():
    text = "---\nname: \"quoted name\"\nmetadata:\n  type: 'reference'\n---\nBody.\n"
    parsed = imp.parse_memory_file(text, "stem")
    assert parsed["name"] == "quoted name"
    assert parsed["meta_type"] == "reference"
    assert parsed["meta_project"] is None


def test_missing_description_falls_back_to_first_sentence():
    text = "---\nname: n\n---\nThe quick brown fox jumps. Second sentence here.\nMore.\n"
    parsed = imp.parse_memory_file(text, "stem")
    assert parsed["hook"] == "The quick brown fox jumps."


def test_description_truncated_to_200():
    text = f"---\nname: n\ndescription: {'x' * 300}\n---\nBody.\n"
    parsed = imp.parse_memory_file(text, "stem")
    assert len(parsed["hook"]) == 200


def test_no_frontmatter_at_all():
    text = "Just plain content here. And a second sentence.\nAnother line.\n"
    parsed = imp.parse_memory_file(text, "my_note")
    assert parsed["name"] == "my_note"
    assert parsed["hook"] == "Just plain content here."
    assert parsed["body"] == text.strip()
    assert parsed["meta_type"] is None


def test_unclosed_fence_treated_as_body():
    text = "---\nname: never closed\nBody-ish content follows.\n"
    parsed = imp.parse_memory_file(text, "stem")
    assert parsed["name"] == "stem"
    assert "never closed" in parsed["body"]


def test_crlf_tolerated():
    unix = imp.parse_memory_file(_FENCED, "stem")
    dos = imp.parse_memory_file(_FENCED.replace("\n", "\r\n"), "stem")
    assert dos == unix


# ---------------------------------------------------------------------------
# Type mapping — metadata.type > filename prefix > --type-default > 'project'
# ---------------------------------------------------------------------------


def test_explicit_valid_metadata_type_wins():
    assert imp.resolve_type("project", "user_x", "reference") == "project"
    assert imp.resolve_type("user", "anything", None) == "user"


def test_invalid_metadata_type_falls_to_prefix():
    assert imp.resolve_type("bogus", "feedback_x", None) == "feedback"


def test_prefix_heuristic():
    assert imp.resolve_type(None, "user_gpu", None) == "user"
    assert imp.resolve_type(None, "feedback_rule", None) == "feedback"
    assert imp.resolve_type(None, "reference_doc", None) == "reference"
    assert imp.resolve_type(None, "plain_note", None) == "project"


def test_type_default_replaces_only_the_fallback():
    assert imp.resolve_type(None, "plain_note", "reference") == "reference"
    # A prefix match is NOT overridden by --type-default.
    assert imp.resolve_type(None, "user_gpu", "reference") == "user"


# ---------------------------------------------------------------------------
# Import flow — DB-backed
# ---------------------------------------------------------------------------


class _StubEmb:
    """One-hot embedder: every hook lands on the same slot -> cosine sim 1.0
    against a pre-inserted note on that slot."""

    model_name = "stub"

    def __init__(self, slot: int = 1) -> None:
        self.slot = slot

    def embed(self, texts, task):
        v = [0.0] * _DIMS
        v[self.slot] = 1.0
        return [list(v) for _ in texts]


def _vec(slot: int) -> list[float]:
    v = [0.0] * _DIMS
    v[slot] = 1.0
    return v


def _wipe(conn):
    conn.execute("DELETE FROM notes")


def _no_network(monkeypatch, embedder=None):
    """Apply-mode runs must never construct real Voyage/LLM clients."""
    monkeypatch.setattr(imp, "_make_embedder", lambda db_url: embedder)
    monkeypatch.setattr(imp, "_make_llm", lambda: None)
    monkeypatch.setenv("SYNAPSE_NOTES_CONFIRM", "0")


def _write(d: Path, filename: str, text: str) -> Path:
    p = d / filename
    p.write_text(text, encoding="utf-8")
    return p


@needs_db
def test_dry_run_writes_zero_rows(conn, db_url, tmp_path):
    _wipe(conn)
    _write(tmp_path, "feedback_no_tables.md", "Never use tables in replies. Ever.\n")
    _write(tmp_path, "some_project.md", "---\nname: proj\n---\nProject body.\n")
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=False)
    assert counts == {"create": 2, "update": 0, "skip": 0, "error": 0}
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 0
    _wipe(conn)


@needs_db
def test_apply_creates_rows_with_source_ref(conn, db_url, tmp_path, monkeypatch):
    _wipe(conn)
    _no_network(monkeypatch)
    _write(tmp_path, "feedback_no_tables.md", "Never use tables in replies. Ever.\n")
    _write(
        tmp_path,
        "widget.md",
        "---\nname: widget_notes\ndescription: Widget uses provider X\n"
        "metadata:\n  type: project\n  project: widget\n---\nDetails here.\n",
    )
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 2, "update": 0, "skip": 0, "error": 0}

    rows = {
        r[0]: r
        for r in conn.execute(
            "SELECT source_ref, type, hook, body, project, embedding FROM notes"
        ).fetchall()
    }
    fb = rows["import:feedback_no_tables"]
    assert fb[1] == "feedback"  # filename-prefix heuristic
    assert fb[2] == "Never use tables in replies."
    assert fb[5] is None  # keyless -> NULL embedding
    w = rows["import:widget_notes"]
    assert w[1] == "project" and w[4] == "widget" and w[3] == "Details here."
    _wipe(conn)


@needs_db
def test_rerun_same_dir_all_skips(conn, db_url, tmp_path, monkeypatch):
    _wipe(conn)
    _no_network(monkeypatch)
    _write(tmp_path, "user_pref.md", "The user prefers short replies. Always.\n")
    _write(tmp_path, "reference_doc.md", "See the runbook for details. It exists.\n")
    first = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert first["create"] == 2
    second = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert second == {"create": 0, "update": 0, "skip": 2, "error": 0}
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 2
    _wipe(conn)


@needs_db
def test_edited_file_updates_in_place(conn, db_url, tmp_path, monkeypatch):
    _wipe(conn)
    _no_network(monkeypatch)
    p = _write(tmp_path, "user_pref.md", "The user prefers short replies. Always.\n")
    assert imp.run_import(directory=tmp_path, db_url=db_url, apply=True)["create"] == 1
    nid, before = conn.execute(
        "SELECT id, updated_at FROM notes WHERE source_ref = 'import:user_pref'"
    ).fetchone()

    p.write_text("The user prefers short replies. And bullet lists.\n", encoding="utf-8")
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 0, "update": 1, "skip": 0, "error": 0}

    rows = conn.execute(
        "SELECT id, body, updated_at FROM notes WHERE source_ref = 'import:user_pref'"
    ).fetchall()
    assert len(rows) == 1  # updated in place, no second row
    assert rows[0][0] == nid
    assert rows[0][1] == "The user prefers short replies. And bullet lists."
    assert rows[0][2] > before
    _wipe(conn)


@needs_db
def test_superseded_source_ref_skip_and_reconcile(conn, db_url, tmp_path, monkeypatch):
    """Newest row for the source_ref is retired: unchanged content stays skipped
    (never resurrect); changed content reconciles into a NEW live row instead of
    touching the retired one."""
    _wipe(conn)
    _no_network(monkeypatch)
    db = Database(db_url)
    old = db.insert_note(
        owner_id=_OWNER,
        group_id="technical",
        project=None,
        type="user",
        hook="Old seeded hook.",
        body="Old seeded body.",
        embedding=None,
        embed_model=None,
        source_ref="import:foo",
    )
    winner = db.insert_note(
        owner_id=_OWNER,
        group_id="technical",
        project=None,
        type="user",
        hook="The contradicting note.",
        body="Newer truth.",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )
    db.supersede_note(old, winner)
    db.close()

    # Same content as the retired row -> skip; nothing resurrected.
    p = _write(
        tmp_path, "foo.md", "---\nname: foo\ndescription: Old seeded hook.\n---\nOld seeded body.\n"
    )
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 0, "update": 0, "skip": 1, "error": 0}
    assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 2

    # Edited content -> a fresh assertion: new live row, retired row untouched.
    p.write_text(
        "---\nname: foo\ndescription: Re-seeded hook.\n---\nRe-seeded body.\n", encoding="utf-8"
    )
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 1, "update": 0, "skip": 0, "error": 0}
    fresh = conn.execute(
        "SELECT id, superseded_by FROM notes WHERE source_ref = 'import:foo' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert fresh[0] != old and fresh[1] is None  # new live row
    retired = conn.execute("SELECT superseded_by, body FROM notes WHERE id = %s", (old,)).fetchone()
    assert retired[0] == winner and retired[1] == "Old seeded body."  # untouched
    _wipe(conn)


@needs_db
def test_high_sim_collision_routes_through_reconcile(conn, db_url, tmp_path, monkeypatch):
    """A seed whose hook collides (cosine ~1.0, same type) with an existing
    remember-written note must UPDATE that note via reconcile_note — not blind-insert
    a duplicate. Embedder stubbed one-hot; the LLM confirm is disabled via its kill
    switch, which collapses to 'same' -> update."""
    _wipe(conn)
    _no_network(monkeypatch, embedder=_StubEmb(slot=1))
    db = Database(db_url)
    existing = db.insert_note(
        owner_id=_OWNER,
        group_id="technical",
        project=None,
        type="user",
        hook="User prefers dark mode",
        body="Set everything to dark.",
        embedding=_vec(1),
        embed_model="stub",
        source_ref=None,  # written by remember(), no import provenance
    )
    db.close()

    _write(
        tmp_path,
        "user_display_pref.md",
        "---\ndescription: User prefers light mode\n---\nSwitched with the redesign.\n",
    )
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 0, "update": 1, "skip": 0, "error": 0}

    rows = conn.execute("SELECT id, hook, body, source_ref FROM notes").fetchall()
    assert len(rows) == 1  # no duplicate row
    assert rows[0][0] == existing
    assert rows[0][1] == "User prefers light mode"
    assert rows[0][2] == "Switched with the redesign."
    # The winning row (remember-written, source_ref NULL) must now carry the seed's
    # provenance — without the stamp, find_note_by_source_ref would miss on every
    # subsequent run and this file would reconcile (embed + LLM confirm) forever.
    assert rows[0][3] == "import:user_display_pref"

    # Re-run: the stamped source_ref makes the idempotency probe hit -> pure skip.
    counts = imp.run_import(directory=tmp_path, db_url=db_url, apply=True)
    assert counts == {"create": 0, "update": 0, "skip": 1, "error": 0}
    _wipe(conn)
