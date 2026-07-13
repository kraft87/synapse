#!/usr/bin/env python3
"""Seed the notes store (schema 041) from a directory of markdown memory files.

Each ``*.md`` file becomes one explicit note: an optional YAML-ish frontmatter
block (fenced by ``---`` lines) supplies name/description/metadata, the rest of
the file is the note body. The parser is deliberately hand-rolled (stdlib only,
no pyyaml): top-level ``key: value`` pairs plus ONE nesting level — a block key
with an empty value (e.g. ``metadata:``) whose indented children are collected
into a dict. Files with no frontmatter at all still import: name from the
filename stem, whole file as body, hook from the first body sentence.

Idempotency is keyed on ``source_ref = "import:<name>"``:
  * source_ref hit, identical hook+body  -> skip (re-running a dir is all skips)
  * source_ref hit, content changed      -> update the row in place
  * source_ref hit but the newest row was SUPERSEDED -> the seed was deliberately
    retired by a later contradicting note. Unchanged content stays skipped (never
    resurrect); changed content is a NEW assertion and goes through reconcile_note
    against the live set instead of touching the retired row.
  * source_ref miss -> reconcile_note(), so a seed colliding with an existing
    remember-written note (high-sim hook) updates/supersedes instead of duplicating.

Dry-run is the DEFAULT — per-file verdict lines + a summary, zero writes (the
database is still read for the idempotency probe). Pass ``--apply`` to execute.

Keyless operation (no Voyage credentials on the default backend) degrades the
same way ingestion/notes.py does: notes land with NULL embeddings and a warning
— dedup KNN is skipped, source_ref idempotency still holds. Never a hard failure.

Usage:
    python scripts/import_notes.py --dir /path/to/memory [--apply]
        [--project X] [--type-default T] [--db-url DSN]

DSN falls back to $SYNAPSE_DB_URL.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.db import Database
from ingestion.embedding import create_embedder, embed_provider
from ingestion.llm_client import create_llm_client
from ingestion.notes import reconcile_note

logger = logging.getLogger("import_notes")

_VALID_TYPES = ("user", "feedback", "project", "reference")
_HOOK_MAX = 200

# Filename-prefix heuristic: "user_gpu_inventory.md" -> user, etc. Anything
# unprefixed falls through to --type-default (or 'project').
_PREFIX_TYPES = ("user", "feedback", "reference")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


# ---------------------------------------------------------------------------
# Parsing — pure functions, unit-tested without a database
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    """Split off a ``---``-fenced frontmatter block. Returns (frontmatter lines,
    body). CRLF is normalized first. An unclosed fence is treated as no
    frontmatter (the whole file becomes the body) — fail toward importing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], "\n".join(lines[i + 1 :])
    return None, text


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    """YAML-ish, hand-rolled: top-level ``key: value`` pairs + ONE nesting level
    (a key with an empty value opens a block; its indented children are collected
    into a dict). Comment lines and anything unparseable are skipped silently —
    tolerance over strictness for human-maintained memory files."""
    fm: dict[str, Any] = {}
    open_block: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw[:1] in (" ", "\t")
        key, colon, value = raw.strip().partition(":")
        key, value = key.strip(), _unquote(value.strip())
        if not colon or not key:
            continue
        if indented:
            if open_block is not None:
                fm[open_block][key] = value
            continue
        if value == "":
            fm[key] = {}
            open_block = key
        else:
            fm[key] = value
            open_block = None
    return fm


def _first_sentence(body: str) -> str:
    """Hook fallback: the first sentence of the first non-empty body line,
    markdown heading markers stripped."""
    for line in body.split("\n"):
        line = line.strip().lstrip("#").strip()
        if line:
            return _SENTENCE_END.split(line, 1)[0]
    return ""


def parse_memory_file(text: str, stem: str) -> dict[str, Any]:
    """Parse one memory file into note fields.

    Returns ``{"name", "hook", "body", "meta_type", "meta_project"}``.
    ``hook`` is the frontmatter description (truncated to 200 chars), else the
    first body sentence, else the name — never empty (hook is NOT NULL)."""
    fm_lines, body = _split_frontmatter(text)
    fm = _parse_frontmatter(fm_lines) if fm_lines is not None else {}
    meta = fm.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    name_raw = fm.get("name")
    name = name_raw if isinstance(name_raw, str) and name_raw.strip() else stem
    desc = fm.get("description")
    hook = desc if isinstance(desc, str) and desc.strip() else _first_sentence(body)
    hook = hook.strip()[:_HOOK_MAX] or name
    meta_type = meta.get("type")
    meta_project = meta.get("project")
    return {
        "name": name,
        "hook": hook,
        "body": body.strip(),
        "meta_type": meta_type if isinstance(meta_type, str) else None,
        "meta_project": meta_project if isinstance(meta_project, str) and meta_project else None,
    }


def resolve_type(meta_type: str | None, stem: str, type_default: str | None) -> str:
    """Note-type precedence: explicit valid metadata.type > filename-prefix
    heuristic > --type-default > 'project'. --type-default replaces only the
    final fallback — never an explicit valid metadata.type or a prefix match."""
    if meta_type in _VALID_TYPES:
        return meta_type  # type: ignore[return-value]
    for prefix in _PREFIX_TYPES:
        if stem.startswith(prefix + "_"):
            return prefix
    return type_default if type_default in _VALID_TYPES else "project"


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def _make_embedder(db_url: str) -> Any | None:
    """Keyless degrade, same shape as the timeline push route: no Voyage
    credentials on the default backend -> None (NULL-embedding inserts, dedup
    KNN skipped). Any construction failure also degrades — never a hard fail."""
    if embed_provider() == "voyage" and not os.environ.get("VOYAGE_API_KEY", ""):
        logger.warning(
            "no Voyage credentials — notes will store NULL embeddings; "
            "dedup KNN skipped (source_ref idempotency still holds)"
        )
        return None
    try:
        return create_embedder(db_url=db_url)
    except Exception as e:
        logger.warning("embedder unavailable (%s); storing NULL embeddings, dedup skipped", e)
        return None


def _make_llm() -> Any | None:
    """The LLM is only consulted by reconcile_note's confirm call, which already
    fails open to 'same' -> update; a client that can't even be constructed
    degrades the same way."""
    try:
        return create_llm_client()
    except Exception as e:
        logger.warning("LLM client unavailable (%s); reconcile confirms fail open to update", e)
        return None


def _embed_hook(embedder: Any | None, hook: str) -> tuple[list[float] | None, str | None]:
    """Embed the hook for the direct-update path, mirroring reconcile_note's
    degrade: any failure -> NULL embedding, never a raise."""
    if embedder is None:
        return None, None
    try:
        vec = list(embedder.embed([hook], task="document")[0])
        return vec, getattr(embedder, "model_name", None) or "voyage-4-large"
    except Exception as e:
        logger.warning("hook embed failed (%s); updating with NULL embedding", e)
        return None, None


_RECONCILE_VERDICT = {"created": "create", "updated": "update", "superseded": "update"}


def _import_one(
    db: Database,
    embedder: Any | None,
    llm: Any | None,
    parsed: dict[str, Any],
    *,
    note_type: str,
    project: str | None,
    apply: bool,
) -> tuple[str, str]:
    """Import a single parsed file. Returns (verdict, detail) where verdict is
    one of create / update / skip."""
    source_ref = f"import:{parsed['name']}"
    hook, body = parsed["hook"], parsed["body"]

    existing = db.find_note_by_source_ref(source_ref)
    if existing is not None:
        unchanged = existing["hook"] == hook and (existing["body"] or "") == body
        if unchanged:
            # Even when the newest row is superseded: identical content means the
            # seed was deliberately retired by a contradicting note — don't resurrect.
            return "skip", ""
        if existing.get("superseded_by") is not None:
            # The seed's newest row was retired; the edited file is a fresh
            # assertion — reconcile against the LIVE set, don't touch the retired row.
            if not apply:
                return "create", "prior import superseded; will reconcile"
            result = reconcile_note(
                db,
                embedder,
                llm,
                hook=hook,
                body=body,
                type=note_type,
                project=project,
                source_ref=source_ref,
            )
            return _RECONCILE_VERDICT[result["outcome"]], f"note #{result['note_id']} (reconciled)"
        if not apply:
            return "update", f"note #{existing['id']}"
        vec, embed_model = _embed_hook(embedder, hook)
        db.update_note(existing["id"], hook=hook, body=body, embedding=vec, embed_model=embed_model)
        return "update", f"note #{existing['id']}"

    if not apply:
        return "create", ""
    result = reconcile_note(
        db,
        embedder,
        llm,
        hook=hook,
        body=body,
        type=note_type,
        project=project,
        source_ref=source_ref,
    )
    detail = f"note #{result['note_id']}"
    if result["outcome"] != "created":
        detail += f" ({result['outcome']} via reconcile)"
    return _RECONCILE_VERDICT[result["outcome"]], detail


def run_import(
    *,
    directory: Path,
    db_url: str,
    apply: bool = False,
    project: str | None = None,
    type_default: str | None = None,
) -> dict[str, int]:
    """Import every ``*.md`` in ``directory``. Returns verdict counts."""
    files = sorted(directory.glob("*.md"))
    counts = {"create": 0, "update": 0, "skip": 0, "error": 0}
    if not files:
        print(f"no *.md files in {directory}")
        return counts

    db = Database(db_url)
    embedder = _make_embedder(db_url) if apply else None
    llm = _make_llm() if apply else None
    try:
        for path in files:
            try:
                parsed = parse_memory_file(path.read_text(encoding="utf-8"), path.stem)
                note_type = resolve_type(parsed["meta_type"], path.stem, type_default)
                note_project = parsed["meta_project"] or project
                verdict, detail = _import_one(
                    db,
                    embedder,
                    llm,
                    parsed,
                    note_type=note_type,
                    project=note_project,
                    apply=apply,
                )
            except Exception as e:
                counts["error"] += 1
                print(f"error   {path.name}: {e}")
                continue
            counts[verdict] += 1
            suffix = f"  ({detail})" if detail else ""
            label = "skip (unchanged)" if verdict == "skip" else verdict
            print(f"{label:<17} {note_type:<9} {parsed['name']}{suffix}")
    finally:
        db.close()

    mode = "" if apply else "  [dry run — pass --apply to execute]"
    print(
        f"\n{len(files)} file(s): {counts['create']} create, {counts['update']} update, "
        f"{counts['skip']} skip, {counts['error']} error{mode}"
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed the notes store from a directory of markdown memory files."
    )
    parser.add_argument("--dir", required=True, help="directory of *.md memory files")
    parser.add_argument(
        "--apply", action="store_true", help="execute writes (default is a dry-run report)"
    )
    parser.add_argument("--project", default=None, help="project tag when metadata has none")
    parser.add_argument(
        "--type-default",
        default=None,
        choices=_VALID_TYPES,
        help="note type when neither metadata.type nor the filename prefix decides (default"
        " 'project')",
    )
    parser.add_argument("--db-url", default=None, help="DSN (falls back to $SYNAPSE_DB_URL)")
    args = parser.parse_args(argv)

    db_url = args.db_url or os.environ.get("SYNAPSE_DB_URL", "")
    if not db_url:
        parser.error("no database DSN — pass --db-url or set SYNAPSE_DB_URL")
    directory = Path(args.dir)
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    counts = run_import(
        directory=directory,
        db_url=db_url,
        apply=args.apply,
        project=args.project,
        type_default=args.type_default,
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
