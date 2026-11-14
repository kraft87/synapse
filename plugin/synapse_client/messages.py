"""A conversation in the ``/ingest`` ``messages`` format, cut into turns.

A harness that can hand over its chat messages uploads them as ``messages`` records: no
server parser of its own. Its adapter builds the records from the harness's own shapes
(harnesses/hermes/messages.py is one); this module cuts them into turns and says which
are finished, and ``turn_upload.TurnUploader`` sends those.

A record is a dict: ``role`` ("user", "assistant" or "tool"), ``content`` (text), and
where they apply an assistant row's ``tool_calls`` (``{"name", "arguments"}`` each), a
tool row's ``name``, ``ts``, an ``id`` the harness keeps stable for the row, and
``synthetic`` on a user row the harness wrote in the user's place. The server skips a
synthetic row: it neither opens a turn nor renders.

A turn starts at a real user row (not synthetic, not empty) and runs to the next one.
It is complete when its last record that is not synthetic is an assistant row with text
and no tool calls: an assistant row that calls tools is followed by their results, so a
turn ending in one is still running (a harness's list may be read mid-turn). Only
complete turns are uploaded, each with the complete turn before it as its context.
Synthetic rows do not count towards a turn's key either, so one added after a finished
turn neither reopens it nor makes it look new. The reply to one does: it lands on the
turn before, which then has a new key, as does the reply to a user row with no text.
The row that opens a turn stays its own however the turn goes on (``opening_key``), and
``joined`` says a turn holds more than its own exchange. For that an adapter keeps
every user row, also one that opens no turn: left out, the reply to it would read as the
reply of the turn before. And a row its harness folded two user messages into is
followed by a synthetic row with no text: one row cannot show that it holds two.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from typing import Any

Record = dict[str, Any]
Turn = list[Record]


def _opens_turn(record: Record) -> bool:
    return record["role"] == "user" and not record.get("synthetic") and bool(record["content"])


def turns(records: Sequence[Record]) -> list[Turn]:
    """``records`` cut into turns, each starting at a real user row. Records before the
    first real user row belong to no turn and are left out."""
    out: list[Turn] = []
    for record in records:
        if _opens_turn(record):
            out.append([record])
        elif out:
            out[-1].append(record)
    return out


def _real(turn: Sequence[Record]) -> list[Record]:
    return [record for record in turn if not record.get("synthetic")]


def _final(record: Record) -> bool:
    """Whether ``record`` ends a turn: an assistant row with text and no tool calls."""
    return (
        record["role"] == "assistant" and bool(record["content"]) and not record.get("tool_calls")
    )


def complete(turn: Sequence[Record]) -> bool:
    """Whether ``turn`` has finished: its last record that is not synthetic is an
    assistant row with text and no tool calls."""
    real = _real(turn)
    return bool(real) and _final(real[-1])


def joined(turn: Sequence[Record]) -> bool:
    """Whether ``turn`` holds more than the exchange its first row opens: a user row
    besides that one, or any row after its reply. Such a row began something later
    than the turn did, which landed on it: a row written in the user's place, or a
    user row with no text. A synthetic row counts too, also when nothing follows it:
    the harness may have begun a turn with it that left no other row."""
    replied = False
    for record in turn[1:]:
        if replied or record["role"] == "user":
            return True
        replied = _final(record)
    return False


def opening_key(turn: Sequence[Record]) -> str:
    """A stable key for the row that opens ``turn``: its ``id``, else a hash of the row
    (two rows that read the same are then one). ``turn_key`` changes when a turn goes
    on; this one does not, so it says the turn is the same one, however it ends."""
    return turn_key(turn[:1])


def turn_key(turn: Sequence[Record]) -> str:
    """A stable key for ``turn``: the ``id`` of its last record that is not synthetic,
    else a hash of those records."""
    real = _real(turn)
    last_id = real[-1].get("id") if real else None
    if isinstance(last_id, str) and last_id:
        return last_id
    blob = json.dumps(real, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def pending(
    all_turns: Sequence[Turn],
    sent: Collection[str],
    *,
    lookback: int | None = None,
) -> list[tuple[Turn | None, Turn]]:
    """The complete turns whose key is not in ``sent``, oldest first, each paired with
    the complete turn before it (its context, or None). With ``lookback``, only the
    last ``lookback`` complete turns are considered."""
    done = [turn for turn in all_turns if complete(turn)]
    start = max(0, len(done) - lookback) if lookback is not None else 0
    return [
        (done[i - 1] if i > 0 else None, done[i])
        for i in range(start, len(done))
        if turn_key(done[i]) not in sent
    ]


def batches(
    selected: Sequence[tuple[Turn | None, Turn]],
    *,
    max_turns: int = 20,
    max_bytes: int = 256_000,
) -> list[tuple[list[Record], list[Turn]]]:
    """Group ``(context, turn)`` pairs into upload batches of at most ``max_turns`` new
    turns and roughly ``max_bytes`` of records. ``context`` is the turn before ``turn``
    (or None): it leads each batch so the server's ``[context]`` line is right, and is
    not counted. Returns ``(records, new_turns)`` per batch. A single turn larger than
    ``max_bytes`` still goes, alone."""
    out: list[tuple[list[Record], list[Turn]]] = []
    records: list[Record] = []
    new: list[Turn] = []
    size = 0
    for context, turn in selected:
        turn_size = _size(turn)
        if new and (len(new) >= max_turns or size + turn_size > max_bytes):
            out.append((records, new))
            records, new, size = [], [], 0
        if not new and context:
            records.extend(context)
            size += _size(context)
        records.extend(turn)
        new.append(turn)
        size += turn_size
    if new:
        out.append((records, new))
    return out


def _size(turn: Sequence[Record]) -> int:
    return len(json.dumps(list(turn), ensure_ascii=False, default=str).encode("utf-8", "replace"))
