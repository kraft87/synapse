#!/usr/bin/env python3
"""Rewrite stored facts that call the owner "User" so they say the owner's name.

Until the extraction prompt was told the owner's name (``ingestion.extraction_llm``,
``_OWNER_NAME_RULE``) it wrote "User ..." whenever a conversation never said the name and
the name whenever it did, so one memory holds both wordings for the same person. Recall
searches the fact text. A query that names the owner then ranks the named facts and loses
the "User" ones, and a first-person query does the opposite, so neither wording finds
everything. This script makes the stored text agree with what the prompt now writes.

Which "user" is the owner is a judgement, so a model makes it; the edit itself is not. Each
"user" word in a fact (``TOKEN``: "User", "the user", "user's", any case, never part of a
longer word such as "users", "user_id" or "end-user") is numbered, a classifier names the
numbers that mean the memory's owner, and the script swaps exactly those for the owner's
name ("the user's" -> "<name>'s"). No other character of a fact can change. Each fact is
classified twice, among different neighbours, and a word is renamed only when both passes
name it; a word either pass leaves out, or a fact a reply does not cover, stays as it is.

Three passes, each its own run:

  --propose      Read the live, non-web facts that hold such a word, classify them, and
                 write a PLAN FILE (JSON lines: uuid, old, new) outside the repository.
                 Writes nothing to the database.
  --apply PLAN   Copy every row the plan names into the backup table
                 ``kg_owner_rename_<YYYYMMDD>`` (uuid, fact, fact_embedding), embed the new
                 text the way ingest does (``task="document"``), and update ``fact`` and
                 ``fact_embedding`` where the stored text still equals the planned ``old``.
                 A row that changed since the plan was written is skipped and counted.
                 ``--dry-run`` reports what it would write.
  --restore DAY  Put back the text and embedding of every row in that day's backup table
                 whose stored text is still the one this script wrote.

Not changed: retired facts, facts extracted from web pages, entity names and summaries,
edges, dates, labels, counts. ``fact`` is not a column the ``source_projects`` trigger
watches, and the BM25 and HNSW indexes follow the row.

The plan file holds private fact text. It defaults to ``~/data/synapse/`` and must stay out
of this repository, which is public.

    scripts/rename_owner_in_facts.py --propose [--out PLAN] [--limit N]
    scripts/rename_owner_in_facts.py --apply PLAN [--dry-run] [--date YYYYMMDD]
    scripts/rename_owner_in_facts.py --restore YYYYMMDD

The DSN comes from --db-url, else $SYNAPSE_DB_URL. --propose needs the LLM settings the
pipeline runs with (``SYNAPSE_OWNER_RENAME_MODEL`` beats ``SYNAPSE_LLM_MODEL``); --apply
needs the embedder's.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.dedup import owner_name
from ingestion.llm_client import UsageLimitError, stage_model, structured_call

logger = logging.getLogger("rename_owner_in_facts")

#: Pipeline stage name for :func:`ingestion.llm_client.stage_model`.
STAGE = "OWNER_RENAME"

DEFAULT_PLAN = Path("~/data/synapse/owner-rename-plan.jsonl").expanduser()

#: Facts per classifier call, and per embedding call.
BATCH = 40
EMBED_BATCH = 64

#: A fact is one sentence; the cap only keeps a malformed row from swamping a batch. A
#: longer fact is left alone: its marks would not all be in the prompt.
FACT_CHARS = 1000

# One "user" word, with its article and possessive: "User", "the user", "user's". Not part
# of a longer word, identifier, path, address or hyphenated compound ("users", "user_id",
# "USER.md", "@user", "end-user", "user-facing"); a sentence-final "user." still counts.
TOKEN = re.compile(r"(?<![\w\-/.@])(?:the\s+)?user(?:['\u2019]s)?(?![\w\-/]|\.\w)", re.IGNORECASE)

PROMPT = """A personal AI memory stores one-sentence facts extracted from the conversations \
between ONE person, the memory's owner, and their assistant. The owner's name is {owner}. \
Older facts call the owner "User", "the user" or "user" in place of the name. Each line below \
is one fact. Every place a fact uses that word is marked [[k: word]], numbered from 1 within \
the fact.

For each fact, list the numbers of the marks that mean {owner}: the person whose memory this \
is, who talks to the assistant and whose life, work, things, opinions and requests the facts \
record. A listed mark is replaced by the name "{owner}", so list it only when the sentence \
still reads right and means the same with the name in its place.

Leave a mark out when the word means anyone else or is not a person:
- the people who use a product, service, app, tool or system, in general or as an example \
("a user who signs in", "the user's device token is checked", "shown to the user of the \
plugin", "when the user scrolls")
- an account, role, login or field in a system ("the user role", "the postgres user", "logs \
in as user sam")
- a kind of thing named with the word ("user input", "user message", "user prompt", "user \
turn", "user agent", "User Application", "a Pro-plan user")
- part of a name or title ("User Rules", "User Guide")
If you cannot tell from the line alone, leave the mark out.

You see one fact per line with no other context. Judge only what the line says.

Facts:
{facts}

Reply with ONLY a JSON object that names every line number exactly once, with an empty \
list when no mark means the owner:
{"facts": [{"n": 1, "owner": [1, 2]}, {"n": 2, "owner": []}, ...]}"""


class FactMarks(BaseModel):
    """One fact's verdict, by its 1-based line number in the prompt."""

    n: int
    owner: list[int]


class OwnerMarksReply(BaseModel):
    """The classifier's reply. An object root: strict structured output needs one."""

    facts: list[FactMarks]


def marks(fact: str) -> list[re.Match[str]]:
    """The "user" words of ``fact``, in order."""
    return list(TOKEN.finditer(fact))


def marked(fact: str) -> str:
    """``fact`` with each "user" word wrapped as ``[[k: word]]``."""
    out, last = [], 0
    for k, m in enumerate(marks(fact), start=1):
        out += [fact[last : m.start()], f"[[{k}: {m.group(0)}]]"]
        last = m.end()
    return "".join([*out, fact[last:]])


def rename(fact: str, picked: Iterable[int], owner: str) -> str:
    """``fact`` with the marks numbered in ``picked`` replaced by ``owner``, a possessive
    kept ("the user's" -> "<owner>'s"). A picked mark that stands right before the name
    ("User Sam asked") is dropped, not turned into a second copy of it. Every other
    character is as it was; a number that is not a mark of this fact is ignored."""
    chosen = set(picked)
    named = re.compile(rf"\s+(?={re.escape(owner)}\b)", re.IGNORECASE)
    out, last = [], 0
    for k, m in enumerate(marks(fact), start=1):
        if k not in chosen:
            continue
        possessive = m.group(0)[-2:].lower() in ("'s", "\u2019s")
        before_name = None if possessive else named.match(fact, m.end())
        out += [fact[last : m.start()], "" if before_name else owner + ("'s" if possessive else "")]
        last = before_name.end() if before_name else m.end()
    return "".join([*out, fact[last:]])


def render(rows: list[dict[str, Any]]) -> str:
    """The prompt's fact lines, numbered from 1, whitespace folded so a fact stays one line."""
    return "\n".join(f"{i}|{' '.join(marked(r['fact']).split())}" for i, r in enumerate(rows, 1))


def classify_batch(
    llm: Any, rows: list[dict[str, Any]], *, owner: str, model: str | None = None
) -> dict[str, list[int]]:
    """The marks of ``rows`` (``uuid`` and ``fact``) that mean ``owner``, by uuid. A usage limit
    propagates. A call that fails, or a reply that does not name each line exactly once, is
    split in halves and each half classified on its own, down to single facts; a single
    fact that still fails is left out, so it is not renamed."""
    if not rows:
        return {}
    reply: OwnerMarksReply | None
    try:
        reply = structured_call(
            llm,
            output_model=OwnerMarksReply,
            base_prompt=PROMPT.replace("{owner}", owner).replace("{facts}", render(rows)),
            model=model or stage_model(STAGE),
            max_tokens=4096,
            max_attempts=2,
        )
    except UsageLimitError:
        raise
    except Exception as e:
        logger.warning("classifier call over %d facts failed: %s", len(rows), str(e)[:160])
        reply = None
    if reply is not None and sorted(f.n for f in reply.facts) == list(range(1, len(rows) + 1)):
        return {rows[f.n - 1]["uuid"]: sorted(set(f.owner)) for f in reply.facts}
    if len(rows) == 1:
        logger.warning("fact %s did not classify: left as it is", rows[0]["uuid"])
        return {}
    half = len(rows) // 2
    return {
        **classify_batch(llm, rows[:half], owner=owner, model=model),
        **classify_batch(llm, rows[half:], owner=owner, model=model),
    }


def candidates(conn: Any, owner_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    """The live, non-web facts of ``owner_id`` that hold a "user" word, oldest first."""
    rows = conn.execute(
        "SELECT uuid, fact FROM kg_relationships "
        "WHERE owner_id = %s AND t_invalid IS NULL AND web_artifact_id IS NULL "
        "  AND fact ~* 'user' ORDER BY id",
        (owner_id,),
    ).fetchall()
    out = [
        {"uuid": r["uuid"], "fact": r["fact"]}
        for r in rows
        if len(r["fact"]) <= FACT_CHARS and marks(r["fact"])
    ]
    return out[:limit] if limit else out


def propose(
    rows: list[dict[str, Any]],
    classify: Callable[[list[dict[str, Any]]], dict[str, list[int]]],
    owner: str,
    batch: int = BATCH,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """The plan for ``rows``: one ``{uuid, old, new}`` per fact whose text changes, and the
    counts. ``classify`` is :func:`classify_batch` bound to a client.

    Every fact is classified twice, in two different orders, and a mark is renamed only
    when both passes name it. A verdict leans toward its neighbours' in the batch (a
    "user input" in a run of facts about the owner gets named too), and which facts are
    neighbours is the one thing the second order changes. ``disagreed`` counts the facts
    the two passes read differently."""
    passes: list[dict[str, list[int]]] = []
    for order in (rows, random.Random(0).sample(rows, len(rows))):
        verdicts: dict[str, list[int]] = {}
        for i in range(0, len(order), batch):
            verdicts.update(classify(order[i : i + batch]))
            logger.info(
                "pass %d: classified %d of %d",
                len(passes) + 1,
                min(i + batch, len(order)),
                len(order),
            )
        passes.append(verdicts)
    first, second = passes
    plan: list[dict[str, str]] = []
    counts = {"facts": len(rows), "renamed": 0, "unchanged": 0, "unclassified": 0, "disagreed": 0}
    for r in rows:
        if r["uuid"] not in first or r["uuid"] not in second:
            counts["unclassified"] += 1
            continue
        one, two = set(first[r["uuid"]]), set(second[r["uuid"]])
        counts["disagreed"] += one != two
        new = rename(r["fact"], one & two, owner)
        if new == r["fact"]:
            counts["unchanged"] += 1
            continue
        counts["renamed"] += 1
        plan.append({"uuid": r["uuid"], "old": r["fact"], "new": new})
    return plan, counts


def read_plan(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def backup_table(day: str) -> str:
    if not re.fullmatch(r"\d{8}", day):
        raise ValueError(f"not a YYYYMMDD date: {day!r}")
    return f"kg_owner_rename_{day}"


def apply(
    conn: Any,
    plan: list[dict[str, str]],
    embed: Callable[[list[str]], list[list[float]]],
    day: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write ``plan``. Rows whose stored text is no longer the planned ``old`` are skipped.
    The backup is one transaction and commits before the first update; each embedding
    batch is then its own transaction, so an interrupted run has written whole batches and
    a re-run finishes the rest (a written row no longer matches its ``old``)."""
    table = backup_table(day)
    by_uuid = {p["uuid"]: p for p in plan}
    current = conn.execute(
        "SELECT uuid, fact FROM kg_relationships WHERE uuid = ANY(%s)", (list(by_uuid),)
    ).fetchall()
    todo = [by_uuid[r["uuid"]] for r in current if r["fact"] == by_uuid[r["uuid"]]["old"]]
    counts = {"planned": len(plan), "matching": len(todo), "written": 0}
    counts["skipped"] = len(plan) - len(todo)
    if dry_run or not todo:
        return counts
    with conn.transaction():
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{table}" (uuid text PRIMARY KEY, fact text, '
            "fact_embedding vector, written text, backed_up_at timestamptz NOT NULL DEFAULT now())"
        )
        for p in todo:
            # The earliest copy wins: a second run on the same day never overwrites the
            # original text with text this script wrote.
            conn.execute(
                f'INSERT INTO "{table}" (uuid, fact, fact_embedding, written) '
                "SELECT uuid, fact, fact_embedding, %s FROM kg_relationships "
                "WHERE uuid = %s AND fact = %s ON CONFLICT (uuid) DO NOTHING",
                (p["new"], p["uuid"], p["old"]),
            )
    for i in range(0, len(todo), EMBED_BATCH):
        chunk = todo[i : i + EMBED_BATCH]
        vectors = embed([p["new"] for p in chunk])
        with conn.transaction():
            for p, vec in zip(chunk, vectors, strict=True):
                cur = conn.execute(
                    "UPDATE kg_relationships SET fact = %s, fact_embedding = %s::vector "
                    "WHERE uuid = %s AND fact = %s",
                    (
                        p["new"],
                        "[" + ",".join(repr(float(x)) for x in vec) + "]",
                        p["uuid"],
                        p["old"],
                    ),
                )
                counts["written"] += cur.rowcount
        logger.info("wrote %d of %d", min(i + EMBED_BATCH, len(todo)), len(todo))
    counts["skipped"] = len(plan) - counts["written"]
    return counts


def restore(conn: Any, day: str) -> dict[str, int]:
    """Put back every backed-up row whose stored text is still the text this script wrote.
    A row edited since keeps its newer text."""
    table = backup_table(day)
    with conn.transaction():
        total = conn.execute(f'SELECT count(*) AS n FROM "{table}"').fetchone()["n"]
        cur = conn.execute(
            f"UPDATE kg_relationships r SET fact = b.fact, fact_embedding = b.fact_embedding "
            f'FROM "{table}" b WHERE r.uuid = b.uuid AND r.fact = b.written'
        )
    return {"backed_up": total, "restored": cur.rowcount}


def _connect(db_url: str, read_only: bool) -> Any:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    if read_only:
        conn.execute("SET default_transaction_read_only = on")
    return conn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propose", action="store_true", help="classify and write the plan file")
    mode.add_argument("--apply", metavar="PLAN", help="write a reviewed plan file")
    mode.add_argument("--restore", metavar="YYYYMMDD", help="undo a run from its backup table")
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    ap.add_argument("--owner-id", default=os.environ.get("SYNAPSE_KG_OWNER_ID", "default"))
    ap.add_argument("--out", type=Path, default=DEFAULT_PLAN, help="plan file (--propose)")
    ap.add_argument("--limit", type=int, help="classify only the first N facts (--propose)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing (--apply)")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="backup table date")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if not args.db_url:
        ap.error("no DSN: pass --db-url or set SYNAPSE_DB_URL")

    if args.restore:
        print(json.dumps(restore(_connect(args.db_url, read_only=False), args.restore)))
        return 0

    if args.apply:
        from ingestion.embedding import create_embedder

        embedder = create_embedder()
        counts = apply(
            _connect(args.db_url, read_only=args.dry_run),
            read_plan(Path(args.apply)),
            lambda texts: embedder.embed(texts, task="document"),
            args.date,
            dry_run=args.dry_run,
        )
        print(json.dumps(counts))
        return 0

    owner = owner_name()
    if owner == "User":
        ap.error('SYNAPSE_OWNER_NAME is unset: the owner is already written "User"')
    from ingestion.llm_client import create_llm_client

    model = stage_model(STAGE)
    llm = create_llm_client(model=model)
    rows = candidates(_connect(args.db_url, read_only=True), args.owner_id, args.limit)
    plan, counts = propose(
        rows, lambda chunk: classify_batch(llm, chunk, owner=owner, model=model), owner
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan))
    print(json.dumps({**counts, "owner": owner, "model": model, "plan": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
