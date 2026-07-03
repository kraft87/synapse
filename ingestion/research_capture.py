"""Capture raw research-source tool results into the cold research_archive.

Design (2026-06-12, superseding the skill-choreography v1): the tool
RESULT is the verbatim record — a `bird thread --json` stdout, a `bsky thread`
dump, a cleaned YouTube transcript print. Capture happens mechanically at the
transcript layer (same JSONL scan as the web lane, including subagent
transcripts under <session>/subagents/), never by prompting an agent to copy
files. The LLM can't forget, truncate, or reformat what it never touches.

Strict isolation, on purpose (oracle + Gemini review):
  * own table `research_archive` — no FKs, nothing else reads or writes it
  * NEVER chunked, embedded, or KG-extracted; invisible to recall()
  * kill switch: `DROP TABLE research_archive;` + drop the cron line

What gets captured (Bash tool calls, matched on the command line):
  * `bird thread <url-or-id> --json`   -> kind=x_thread,    thread:x:<id>
  * `bsky thread <uri-or-url> --json`  -> kind=bsky_thread, thread:bsky:<id>
  * the yt-VTT cleaning step (its command globs /tmp/yt_<video_id>*.vtt)
                                       -> kind=yt_transcript, transcript:<video_id>
Reddit threads and plain web pages are deliberately NOT archived: noise-prone /
re-fetchable, and their distilled form lives in the research brief.

On re-capture of the same source, the longer content wins (a thread re-fetched
later may have grown; a truncated tool result never overwrites a fuller one).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

DDL = """
CREATE TABLE IF NOT EXISTS research_archive (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind        text        NOT NULL,
    source_id   text        NOT NULL UNIQUE,
    url         text        NOT NULL,
    title       text,
    author      text,
    brief_slug  text,
    captured_from text,
    content     text        NOT NULL,
    archived_at timestamptz NOT NULL DEFAULT now()
)
"""

_MIN_CHARS = 200

# bird thread <arg> / bsky thread <arg> — arg may be a bare id, status URL, or
# at:// URI; commands may prefix the binary ("$BSKY" thread ..., /path/bsky thread ...).
_X_CMD = re.compile(r"(?:^|[;&|]\s*|\s)bird\s+thread\s+(\S+)")
_BSKY_CMD = re.compile(r"(?:bsky[\"']?|\$BSKY[\"']?)\s+thread\s+(\S+)")
_YT_GLOB = re.compile(r"/tmp/yt_([A-Za-z0-9_-]{6,})")  # nosec B108 — matches command TEXT; no temp file is created


def _strip_quotes(s: str) -> str:
    return s.strip().strip("'\"")


def _x_thread_id(arg: str) -> str | None:
    arg = _strip_quotes(arg)
    m = re.search(r"/status/(\d+)", arg)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{8,}", arg):
        return arg
    return None


def _bsky_thread_id(arg: str) -> str | None:
    arg = _strip_quotes(arg)
    # at://did:plc:xxx/app.bsky.feed.post/<rkey> or https://bsky.app/profile/<h>/post/<rkey>
    m = re.search(r"/post/([a-z0-9]+)", arg) or re.search(r"app\.bsky\.feed\.post/([a-z0-9]+)", arg)
    if m:
        return m.group(1)
    return None


def classify_command(command: str) -> tuple[str, str, str] | None:
    """Map a Bash command to (kind, source_id, url) or None if not archivable."""
    m = _X_CMD.search(command)
    if m:
        tid = _x_thread_id(m.group(1))
        if tid:
            return ("x_thread", f"thread:x:{tid}", f"https://x.com/i/status/{tid}")
    m = _BSKY_CMD.search(command)
    if m:
        rkey = _bsky_thread_id(m.group(1))
        if rkey:
            arg = _strip_quotes(m.group(1))
            url = arg if arg.startswith("http") else f"https://bsky.app/search?q={rkey}"
            return ("bsky_thread", f"thread:bsky:{rkey}", url)
    # The transcript CLEANING step (python glob over the VTT) — not the yt-dlp
    # download itself, whose stdout is empty. Requires the WEBVTT marker so a
    # mere `ls /tmp/yt_*` doesn't archive garbage.
    if "WEBVTT" in command:
        m = _YT_GLOB.search(command)
        if m:
            vid = m.group(1)
            return ("yt_transcript", f"transcript:{vid}", f"https://www.youtube.com/watch?v={vid}")
    return None


def _first_author(kind: str, content: str) -> str | None:
    """Best-effort deterministic author pull from thread JSON. Never raises."""
    if kind not in ("x_thread", "bsky_thread"):
        return None
    try:
        data = json.loads(content)
        posts = data if isinstance(data, list) else None
        if isinstance(data, dict):
            for key in ("tweets", "posts", "thread"):
                if isinstance(data.get(key), list):
                    posts = data[key]
                    break
        if not posts:
            return None
        p = posts[0]
        handle = (
            (p.get("author") or {}).get("username")
            or (p.get("author") or {}).get("handle")
            or p.get("username")
            or p.get("handle")
        )
        return f"@{handle}" if handle else None
    except Exception:
        return None


@dataclass
class CaptureStats:
    files_scanned: int = 0
    files_skipped_unchanged: int = 0
    matched: int = 0
    inserted: int = 0
    updated_longer: int = 0
    skipped_existing: int = 0
    skipped_short: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"scanned={self.files_scanned} unchanged={self.files_skipped_unchanged} "
            f"matched={self.matched} inserted={self.inserted} grown={self.updated_longer} "
            f"dup={self.skipped_existing} short={self.skipped_short} kinds={self.by_kind}"
        )


def iter_archive_events(jsonl_path: Path) -> Iterator[tuple[str, str, str, str]]:
    """Yield (kind, source_id, url, result_text) for archivable Bash tool calls."""
    pending: dict[str, tuple[str, str, str]] = {}
    try:
        fh = jsonl_path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    if block.get("name") != "Bash":
                        continue
                    cmd = (block.get("input") or {}).get("command")
                    tu_id = block.get("id")
                    if not isinstance(cmd, str) or not tu_id:
                        continue
                    hit = classify_command(cmd)
                    if hit:
                        pending[tu_id] = hit
                elif btype == "tool_result":
                    tu_id_raw = block.get("tool_use_id")
                    if not isinstance(tu_id_raw, str):
                        continue
                    hit = pending.pop(tu_id_raw, None)
                    if not hit:
                        continue
                    text = _coerce_result_text(block.get("content"))
                    if text:
                        yield (*hit, text)


def _coerce_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


class ResearchArchiveCapture:
    """Walks transcript JSONLs and cold-archives matched tool results.

    Mirrors WebArtifactsIngester's per-file mtime checkpoint (source key
    prefix 'research-archive:') so 5-minute cron re-runs are cheap no-ops.
    """

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn
        self._conn.execute(DDL)

    def _checkpoint_get(self, source: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT last_ingested_at FROM ingestion_state WHERE source = %s", (source,)
        ).fetchone()
        if not row:
            return None
        val = row["last_ingested_at"] if isinstance(row, dict) else row[0]
        if not isinstance(val, datetime):
            return None
        return val.replace(tzinfo=UTC) if val.tzinfo is None else val

    def _checkpoint_set(self, source: str, ts: datetime) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_state (source, last_ingested_at)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE SET last_ingested_at = EXCLUDED.last_ingested_at
            """,
            (source, ts),
        )

    def capture_one(self, jsonl_path: Path, stats: CaptureStats | None = None) -> CaptureStats:
        stats = stats or CaptureStats()
        source_key = f"research-archive:{jsonl_path}"
        try:
            mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime, tz=UTC)
        except OSError:
            return stats
        last = self._checkpoint_get(source_key)
        if last and mtime <= last:
            stats.files_skipped_unchanged += 1
            return stats
        stats.files_scanned += 1

        for kind, source_id, url, text in iter_archive_events(jsonl_path):
            stats.matched += 1
            if len(text.strip()) < _MIN_CHARS:
                stats.skipped_short += 1
                continue
            cur = self._conn.execute(
                """
                INSERT INTO research_archive (kind, source_id, url, author, captured_from, content)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE
                    SET content = EXCLUDED.content, captured_from = EXCLUDED.captured_from
                    WHERE length(EXCLUDED.content) > length(research_archive.content)
                RETURNING (xmax = 0) AS inserted
                """,
                (kind, source_id, url, _first_author(kind, text), str(jsonl_path), text.strip()),
            )
            row = cur.fetchone()
            if row is None:
                stats.skipped_existing += 1
            elif row[0]:
                stats.inserted += 1
                stats.by_kind[kind] = stats.by_kind.get(kind, 0) + 1
            else:
                stats.updated_longer += 1

        self._checkpoint_set(source_key, mtime)
        self._conn.commit()
        return stats
