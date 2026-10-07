#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print the board into context.

The board (schema 041) is a small always-injected index of explicit memories: curated
note hooks, the last week's milestones, and a banner saying what memory exists at all.
It replaces the timeline-milestones block — the server renders the milestones INSIDE
the board now, so one block covers both. Server-rendered and hard-capped server-side;
this hook just fetches and prints, so caps and layout evolve without a plugin release.

Reads the machine-token-gated ``GET /context`` route (thin client, no DSN). The project
scope comes from the hook payload's ``cwd``, labeled the same way the ingest path labels
episodes (mirror of ``ingestion.jsonl_client._cwd_to_project`` — basename of cwd).

No ``surface`` param is sent any more (schema 054): the token identifies the caller, and
a hostname the client asserts is no longer evidence of anything. Enrollment itself is
interactive — it prints a sign-in code and waits for a human — so this hook never runs
it; it only says so when this machine holds no device credential, because an empty
session start with no explanation is the one outcome worse than a restricted one.

It also carries the one-line credential notice (config.token_override_notice): when an
enrolled machine's device token overrode a SYNAPSE_INGEST_TOKEN from the environment,
say so and where to remove it, to the user (systemMessage) and the model. That line is
local, needs no server, and is printed even with the board disabled or the server down,
since a silent override is the bug it exists to prevent.

Disable the board with SYNAPSE_BOARD=0. Fail-open: any error prints nothing beyond that
notice and exits 0 — a broken board must never break a session start.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enroll
from config import _cfg, get_json, token_override_notice


def _emit_context(parts: list[str], user_notice: str = "") -> None:
    """``user_notice`` also goes out as ``systemMessage``, which Claude Code shows to the
    user directly rather than only to the model."""
    if not parts:
        return
    out: dict = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join(parts),
        }
    }
    if user_notice:
        out["systemMessage"] = user_notice
    print(json.dumps(out))


def _cwd_to_project(cwd: str | None) -> str | None:
    """Mirror of ``ingestion.jsonl_client._cwd_to_project`` — kept inline so the hook
    stays dependency-free (it runs under the CLI's bare Python, off the repo path).
    Must stay in lockstep: the board's project scope has to match how episodes are
    labeled, or the project section goes empty."""
    if not cwd:
        return None
    return cwd.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or None


def _project_label() -> str | None:
    """Project label from the hook payload's cwd; falls back to the process cwd."""
    try:
        cwd = json.loads(sys.stdin.read() or "{}").get("cwd")
    except Exception:
        cwd = None
    return _cwd_to_project(cwd) or _cwd_to_project(os.getcwd())


def _board_parts() -> list[str]:
    """The server-rendered board, plus the enrollment explainer when one applies."""
    parts = []
    project = _project_label()
    params = {"project": project} if project else {}
    try:
        r = get_json("/context", params, timeout=10)
    except urllib.error.HTTPError as e:
        # 401 with no device credential is the one error worth explaining: this
        # machine has not enrolled (or its token was revoked), so it is served
        # nothing and will go on being served nothing until someone signs in.
        if e.code == 401 and not enroll.is_enrolled():
            parts.append(enroll.not_enrolled_block())
        return parts
    ok = r.get("status") == "ok"
    # 200 + restricted + no device credential is the SILENT version of the 401 above:
    # an open server, or a caller holding the shared root token, is served an empty
    # board and no reason for it. Say why here, or a fresh install reads as "Synapse
    # is just empty" and the user never learns there is a credential to get.
    if ok and r.get("trust") == "restricted" and not enroll.is_enrolled():
        parts.append(enroll.restricted_block())
    text = r.get("text") if ok else None
    if text:
        parts.append(text)
    return parts


def main() -> None:
    parts = []
    try:
        notice = token_override_notice()
    except Exception:
        notice = ""
    if notice:
        parts.append(notice)
    if _cfg("SYNAPSE_BOARD", "1") != "0":
        try:
            parts += _board_parts()
        except Exception:
            pass  # fail-open: no board, no noise
    try:
        # guarded: a print that raises must not break the session
        _emit_context(parts, user_notice=notice)
    except Exception:
        return


if __name__ == "__main__":
    main()
