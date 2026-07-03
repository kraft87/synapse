"""Parse Claude Code transcript JSONL files into Episodes.

Claude Code writes one JSONL per session at
~/.claude/projects/<dash-encoded-cwd>/<sessionId>.jsonl

Each line is a record. Types we care about:
- "user"      — human prompt OR tool_result
- "assistant" — LLM response with content blocks (thinking/text/tool_use)
Other types ("last-prompt", "permission-mode", "file-history-snapshot", ...) are skipped.

Records are grouped into user-turn exchanges. A new turn starts when a "user"
record carries text content (not tool_result). Each group becomes one Episode
with the same shape produced by the Logfire ingestion path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from ingestion.models import Episode

logger = logging.getLogger(__name__)


# User-record content prefixes that mean "Claude Code internal machinery" rather
# than a real human turn (slash commands, command echoes, local-command output, etc).
_MACHINERY_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
    "<system-reminder>",
)


def _is_machinery_text(text: str) -> bool:
    return text.lstrip().startswith(_MACHINERY_PREFIXES)


def _is_user_turn(record: dict[str, Any]) -> bool:
    """True if this user record is a fresh human input (not a tool_result wrapper
    or Claude Code slash-command machinery)."""
    if record.get("type") != "user":
        return False
    content = record.get("message", {}).get("content")
    if isinstance(content, str):
        text = content.strip()
        return bool(text) and not _is_machinery_text(text)
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "text":
                continue
            text = str(b.get("text") or b.get("content") or "").strip()
            if text and not _is_machinery_text(text):
                return True
        return False
    return False


def _tool_input_detail(inp: Any) -> str:
    if not isinstance(inp, dict):
        return str(inp)[:120] if inp else ""
    return (
        inp.get("command")
        or inp.get("file_path")
        or inp.get("query")
        or inp.get("prompt")
        or inp.get("url")
        or (str(inp)[:120] if inp else "")
    )


def _strip_nulls(text: str) -> str:
    """Postgres TEXT columns reject NUL bytes — strip them defensively."""
    return text.replace("\x00", "") if "\x00" in text else text


def _tool_result_text(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(
            str(c.get("text", "")) for c in raw if isinstance(c, dict) and c.get("type") == "text"
        ).strip()
    return str(raw).strip()


def _extract_blocks(
    message: dict[str, Any], rtype: str
) -> tuple[str | None, list[str], list[str], str | None]:
    """Return (text, tool_use_lines, tool_result_lines, model) from a message."""
    text: str | None = None
    tool_uses: list[str] = []
    tool_results: list[str] = []
    model = message.get("model") if rtype == "assistant" else None

    content = message.get("content")
    if isinstance(content, str):
        t = content.strip()
        return (t or None), tool_uses, tool_results, model

    if not isinstance(content, list):
        return text, tool_uses, tool_results, model

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = str(block.get("text") or block.get("content") or "").strip()
            if t:
                text = t
        elif btype == "tool_use":
            name = block.get("name", "tool")
            detail = _tool_input_detail(block.get("input"))
            if detail:
                tool_uses.append(f"[tool:{name}] {str(detail)[:300]}")
        elif btype == "tool_result":
            txt = _tool_result_text(block.get("content", ""))
            if len(txt) > 20:
                tool_results.append(f"[result] {txt[:500]}")

    return text, tool_uses, tool_results, model


def _cwd_to_project(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return cwd.rstrip("/").rsplit("/", 1)[-1] or None


class JSONLParser:
    """Turns a single Claude Code transcript .jsonl into a list of Episodes."""

    _MEANINGFUL_TYPES = ("user", "assistant")

    def parse_file(self, path: Path, project_override: str | None = None) -> list[Episode]:
        records = self._load_records(path)
        if not records:
            return []
        # records are already filtered by _load_records
        return self.parse_records(records, str(path), project_override, _filtered=True)

    def parse_records(
        self,
        records: list[dict[str, Any]],
        source_label: str,
        project_override: str | None = None,
        *,
        _filtered: bool = False,
    ) -> list[Episode]:
        """Parse already-loaded transcript records into Episodes.

        The seam shared by the disk sweep (``parse_file``) and the real-time
        ``/ingest`` push: both feed the SAME records through the SAME grouping
        + episode construction, so they produce identical (session_id,
        sequence) and span_id keys and converge idempotently. ``source_label``
        is cosmetic (lands in metadata.jsonl_path only) — identity is derived
        from the record fields, not the path. Callers that pass raw, unfiltered
        records (the push path) leave ``_filtered=False`` so the same
        type/sidechain/machinery filtering the file loader applies runs here too.
        """
        recs = records if _filtered else self._filter_records(records)
        if not recs:
            return []
        groups = self._group_into_turns(recs)
        return self._groups_to_episodes(groups, source_label, project_override)

    @classmethod
    def _load_records(cls, path: Path) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    raw.append(rec)
        return cls._filter_records(raw)

    @classmethod
    def _filter_records(cls, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            if rec.get("type") not in cls._MEANINGFUL_TYPES:
                continue
            if rec.get("isSidechain", False):
                continue
            if cls._is_pure_machinery(rec):
                continue
            out.append(rec)
        return out

    @staticmethod
    def _is_pure_machinery(record: dict[str, Any]) -> bool:
        """Drop user records that are entirely Claude Code slash-command machinery.
        Tool-result records and real user prompts are kept."""
        if record.get("type") != "user":
            return False
        content = record.get("message", {}).get("content")
        if isinstance(content, str):
            return _is_machinery_text(content)
        if isinstance(content, list):
            has_real_part = False
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "tool_result":
                    return False  # keep — real tool output
                if btype == "text":
                    text = str(b.get("text") or b.get("content") or "").strip()
                    if text and not _is_machinery_text(text):
                        has_real_part = True
            return not has_real_part
        return False

    @staticmethod
    def _group_into_turns(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for rec in records:
            if _is_user_turn(rec) and current:
                groups.append(current)
                current = [rec]
            else:
                current.append(rec)
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _groups_to_episodes(
        groups: list[list[dict[str, Any]]],
        source_path: str,
        project_override: str | None,
    ) -> list[Episode]:
        episodes: list[Episode] = []
        prev_assistant: str | None = None
        seq = 0
        # Claude Code transcripts physically repeat records across compaction/resume
        # (one file can carry the same uuid up to ~5x). The span_id is a group's last
        # uuid, so a repeated turn yields duplicate span_ids — which collide on the
        # episodes partial-unique span_id index and 500 the /ingest push mid-batch
        # (dropping the session's tail). A repeated uuid IS the same turn, so keep only
        # its first occurrence. Lives at the shared seam, so push and sweep stay identical.
        seen_span_ids: set[str] = set()

        for group in groups:
            if not group:
                continue
            seq += 1

            content_parts: list[str] = []
            human_turn: str | None = None
            assistant_turn: str | None = None
            model: str | None = None
            session_id: str | None = None
            cwd: str | None = None
            last_uuid: str | None = None
            ts: str | None = None
            git_branch: str | None = None

            if prev_assistant:
                content_parts.append(f"[context] {prev_assistant.strip()[:300]}")

            for rec in group:
                session_id = session_id or rec.get("sessionId")
                cwd = cwd or rec.get("cwd")
                git_branch = git_branch or rec.get("gitBranch")
                last_uuid = rec.get("uuid", last_uuid)
                ts = ts or rec.get("timestamp")
                rtype = rec.get("type", "")
                msg = rec.get("message") or {}
                text, tuses, tresults, mdl = _extract_blocks(msg, rtype)
                model = model or mdl

                if rtype == "user":
                    if text and human_turn is None and len(text) > 0:
                        human_turn = text[:3000]
                        content_parts.append(f"[user] {human_turn}")
                    for tr in tresults:
                        content_parts.append(tr)
                else:  # assistant
                    for tu in tuses:
                        content_parts.append(tu)
                    if text:
                        assistant_turn = text
                        content_parts.append(f"[assistant] {text[:3000]}")

            if not session_id or not content_parts:
                continue

            span_id = f"jsonl:{last_uuid}" if last_uuid else None
            if span_id is not None:
                if span_id in seen_span_ids:
                    continue  # repeated turn (compaction re-dump) — already emitted
                seen_span_ids.add(span_id)

            try:
                created_at = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            except (ValueError, TypeError):
                created_at = None

            project = project_override or _cwd_to_project(cwd)

            episodes.append(
                Episode(
                    session_id=session_id,
                    sequence=seq,
                    project=project,
                    platform="claude_code",
                    model=model,
                    human_turn=_strip_nulls(human_turn) if human_turn else None,
                    assistant_turn=_strip_nulls(assistant_turn) if assistant_turn else None,
                    content=_strip_nulls("\n\n".join(content_parts)),
                    span_id=span_id,
                    source="jsonl",
                    metadata={
                        "jsonl_path": source_path,
                        "ts": ts,
                        "cwd": cwd,
                        "git_branch": git_branch,
                    },
                    created_at=created_at,
                )
            )
            prev_assistant = assistant_turn

        return episodes
