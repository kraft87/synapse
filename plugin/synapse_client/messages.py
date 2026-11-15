"""A conversation in the ``/ingest`` ``messages`` format, cut into turns.

A harness that can hand over its chat messages uploads them as ``messages`` records: no
server parser of its own. Its adapter builds the records from the harness's own shapes
(harnesses/hermes/messages.py is one); this module cuts them into turns and says which
are finished, and ``turn_upload.TurnUploader`` sends those.

A record is a dict: ``role`` ("user", "assistant" or "tool"), ``content`` (text), and
where they apply an assistant row's ``tool_calls`` (``{"name", "arguments"}`` each), a
tool row's ``name``, ``ts``, an ``id`` the harness keeps stable for the row,
``synthetic`` on a user row the harness wrote in the user's place, and ``begins`` (with
``kind``) on a synthetic row the harness wrote to begin a run of its own: a background
task's report, an event, an auto-continue. The server skips a synthetic row: it neither
opens a turn nor renders. ``begins`` and ``kind`` are the client's alone.

A turn starts at a real user row (not synthetic, not empty) and runs to the next one.
A row that ``begins`` a run starts a turn of its own too, when the turn before is
complete (or there is none): a continuation. It is complete when its last record that
is not synthetic is an assistant row with text and no tool calls: an assistant row that
calls tools is followed by their results, so a turn ending in one is still running.
Only complete turns are uploaded, each with the complete turn before it as its context.
Synthetic rows do not count towards a turn's key, so one added after a finished turn
neither reopens it nor makes it look new. The row that opens a turn stays its own
however the turn goes on (``opening_key``), and ``joined`` says a turn holds more than
its own exchange.

Rows are shipped once (``pending`` with ``shipped``): a turn some of whose rows were
shipped already sends only the rows after the last of them. Such a part, like a
continuation, does not open at a real user row, so the server would join it to any rows
before it in a batch: it goes at the head of a batch of its own, with no context, led
by a one-line label (``label``) that says what it continues and when, so two parts with
the same reply are two episodes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import Any

Record = dict[str, Any]
Turn = list[Record]


def _opens_turn(record: Record) -> bool:
    return record["role"] == "user" and not record.get("synthetic") and bool(record["content"])


def _begins(record: Record) -> bool:
    return record["role"] == "user" and bool(record.get("begins"))


def turns(records: Sequence[Record]) -> list[Turn]:
    """``records`` cut into turns, each starting at a real user row or at a row that
    ``begins`` a run after a complete turn, or before any (a continuation). Records
    before the first of either belong to no turn and are left out."""
    out: list[Turn] = []
    for record in records:
        if _opens_turn(record) or (_begins(record) and (not out or complete(out[-1]))):
            out.append([record])
        elif out:
            out[-1].append(record)
    return out


def continues(turn: Sequence[Record]) -> bool:
    """Whether ``turn`` does not open at a real user row: a continuation, or the rest of
    a turn shipped in part. It goes at the head of a batch of its own, with no context."""
    return bool(turn) and not _opens_turn(turn[0])


def label(turn: Sequence[Record]) -> Record:
    """The record that leads a part that ``continues``: a tool row naming what it
    continues (the run's kind, or the turn before) and when, and nothing of its text."""
    first = turn[0] if turn else {}
    kind = first.get("kind") if first.get("begins") else None
    what = (
        f"a run the harness began ({kind})"
        if isinstance(kind, str) and kind
        else "more of the turn before"
    )
    when = _when(next((r.get("ts") for r in turn if r.get("ts") not in (None, "")), None))
    text = f"{what}, at {when}" if when else what
    return {"role": "tool", "content": text, "name": "continued"}


def _when(ts: Any) -> str:
    if isinstance(ts, int | float) and not isinstance(ts, bool):
        try:
            return datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return ""
    return ts if isinstance(ts, str) else ""


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
    (two rows that read the same are then one), a continuation's synthetic row too.
    ``turn_key`` changes when a turn goes on; this one does not, so it says the turn is
    the same one, however it ends."""
    first_id = turn[0].get("id") if turn else None
    if isinstance(first_id, str) and first_id:
        return first_id
    return _hash(turn[:1])


def turn_key(turn: Sequence[Record]) -> str:
    """A stable key for ``turn``: the ``id`` of its last record that is not synthetic,
    else a hash of those records."""
    real = _real(turn)
    last_id = real[-1].get("id") if real else None
    if isinstance(last_id, str) and last_id:
        return last_id
    return _hash(real)


def _hash(records: Sequence[Record]) -> str:
    blob = json.dumps(list(records), ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def ids(records: Sequence[Record]) -> list[str]:
    """The ``id`` of every record that has one, in order."""
    return [r["id"] for r in records if isinstance(r.get("id"), str) and r["id"]]


def unshipped(turn: Turn, shipped: Collection[str]) -> Turn | None:
    """What of ``turn`` is still to be shipped: all of it when none of its rows is in
    ``shipped``; the rows after the last one that is, when they hold a record that is
    not synthetic; None when nothing is left."""
    last = -1
    for i, record in enumerate(turn):
        if record.get("id") in shipped:
            last = i
    if last < 0:
        return turn
    rest = turn[last + 1 :]
    return rest if _real(rest) else None


def pending(
    all_turns: Sequence[Turn],
    sent: Collection[str],
    *,
    lookback: int | None = None,
    shipped: Collection[str] = (),
) -> list[tuple[Turn | None, Turn]]:
    """The complete turns not sent yet, oldest first, each paired with the complete turn
    before it (its context, or None). A turn is sent when its key is in ``sent``, or
    when its rows are in ``shipped`` (row ids): of a turn shipped in part, what is left
    (``unshipped``) is the turn here. A part that ``continues`` has no context. With
    ``lookback``, only the last ``lookback`` complete turns are considered."""
    done = [turn for turn in all_turns if complete(turn)]
    start = max(0, len(done) - lookback) if lookback is not None else 0
    out: list[tuple[Turn | None, Turn]] = []
    for i in range(start, len(done)):
        if turn_key(done[i]) in sent:
            continue
        part = unshipped(done[i], shipped) if shipped else done[i]
        if part is None:
            continue
        before = done[i - 1] if i > 0 and not continues(part) else None
        out.append((before, part))
    return out


def batches(
    selected: Sequence[tuple[Turn | None, Turn]],
    *,
    max_turns: int = 20,
    max_bytes: int = 256_000,
) -> list[tuple[list[Record], list[Turn]]]:
    """Group ``(context, turn)`` pairs into upload batches of at most ``max_turns`` new
    turns and roughly ``max_bytes`` of records. ``context`` is the turn before ``turn``
    (or None): it leads each batch so the server's ``[context]`` line is right, and is
    not counted. A part that ``continues`` starts a batch of its own, led by its
    ``label``: the server would join it to any rows before it. Returns
    ``(records, new_turns)`` per batch. A single turn larger than ``max_bytes`` still
    goes, alone."""
    out: list[tuple[list[Record], list[Turn]]] = []
    records: list[Record] = []
    new: list[Turn] = []
    size = 0
    for context, turn in selected:
        turn_size = _size(turn)
        follows = continues(turn)
        full = len(new) >= max_turns or size + turn_size > max_bytes
        if new and (full or follows):
            out.append((records, new))
            records, new, size = [], [], 0
        if follows:
            records.append(label(turn))
        elif not new and context:
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
