#!/usr/bin/env python3
"""Split the "magnet" entities the old entity resolver built out of distinct numbered IDs.

Until the resolver guard (``ingestion.dedup.ids_conflict``) shipped, a mention of
``PR #269`` could resolve onto the existing ``PR #290`` (name-embedding cosine 0.97, over
the 0.95 autoconfirm), and every merge renamed the node to the newest mention, so the
next near-identical number matched it again. One node ended up holding the facts of
dozens of PRs, Jira-style ticket keys collected the same way, and because supersession
compares facts that share an endpoint pair, a fact about one PR could retire a fact about
another. The guard stops new merges; this script repairs what is already stored.

Families handled (the entity's whole name must be the identifier, so a decorated name
such as "widget refactor (PR #12)" or a version string is never touched):

    PR #N, draft PR #N, pull request #N    issue #N    task #N    ABC-123 (ticket key)

Rules (``plan`` below; pinned in tests/test_repair_numbered_entity_magnets*.py):

1. **Re-point an endpoint to the ID its fact names.** For each edge (live or retired) with
   an endpoint of one of the shapes above, the fact text is read for identifiers of the
   SAME kind (a PR entity reads "PR #N", "pull request N", ".../pull/N" and a bare "#N";
   a ticket key reads keys with the same prefix). If the fact names exactly one such
   identifier and it is not the endpoint's own number, the endpoint moves to the entity
   named for that identifier in the same group (an exact-ID name such as "PR #218"),
   created if missing: name as written in the fact, entity type/project copied from the
   magnet, embedding from ``ingestion.embedding.create_embedder`` with ``task='entity'``.
   An edge stays where it is when the fact names its own number, no identifier, several,
   a range ("#117-119"), only a bare "#N" next to a ticket-free kind word, when the move
   would join an edge to itself, or when the edge and the entity sit in different groups.
   Retired edges move by the same rule, so a supersession about one PR stays on one pair.
2. **Numbers compare the way the shipped guard compares them.** The entity's own number is
   ``ingestion.dedup.identifier_tokens(name)`` (leading zeros dropped) and an entity
   qualifies only when that is a single number. Every move is checked with
   ``ids_conflict(old name, new name)``: the guard would have kept the two apart. The
   fact side is stricter than the guard on purpose: the guard compares every digit run,
   which in fact text would also catch dates, counts and commit hashes.
3. **Wrong supersessions become candidates, not automatic un-retirements.** A retired
   edge whose ``invalidated_by`` successor shares an ID-bearing endpoint with it, where
   both facts name identifiers and the two sets are disjoint, and no other live edge on
   the retired edge's (post-repair) pair names one of its identifiers, is listed. Many of
   these are real corrections whose identifier IS the value that changed ("the epic is
   ABC-1" -> "the epic is ABC-2"), so a rule cannot tell them from the wrong ones: only
   uuids an operator reviewed and listed in ``--unretire-allow FILE`` are un-retired
   (``t_invalid``, its legacy mirror ``invalid_at`` and ``invalidated_by`` cleared, as
   ``KGPostgresWriter.invalidate_edges`` set them). ``t_expired`` is never touched; an
   edge carrying one is not a candidate.
4. **Magnet summaries are cleared.** An entity summary is longer-wins
   (``NodeDeduper.merge_summary``), so a magnet's summary describes whichever foreign
   mention was longest. Every entity that loses an edge gets ``summary = ''``; the next
   mention that resolves to it (now guarded to the same number) writes a fresh one.

What is NOT changed: fact text, fact embeddings, ``episodes`` (so the schema-056
``source_projects`` trigger, which fires only on ``episodes``/``source_projects``, has
nothing to recompute and provenance stays as it was), mention counts, the stale
``kg_entities.degree`` column, and entity names (a magnet keeps its name).

Writes (``--apply``): the plan is recomputed from the live database at apply time, never
replayed, and every write is optimistic (``... WHERE uuid = %s AND src_uuid = <planned
from>``), so a row that changed since the read is skipped and counted. Before the first
write, one transaction copies every row the run will touch into backup tables

    kg_magnet_repair_<YYYYMMDD>_relationships   full kg_relationships rows (moved/un-retired)
    kg_magnet_repair_<YYYYMMDD>_entities        full kg_entities rows (summaries cleared)
    kg_magnet_repair_<YYYYMMDD>_created         uuids of the entities this run created

each with a ``backed_up_at`` column. Then one transaction per magnet entity: create the
targets it needs (recorded in ``_created`` in the same transaction), re-point its edges,
clear its summary. Un-retirements run last, in one transaction. A re-run is a no-op:
moved edges name their own number now.

``--restore YYYYMMDD`` puts back, from the earliest backed-up copy of each row, every
edge's endpoints and supersession state and every entity's summary, then deletes the
created entities nothing references any more (one transaction; dry run unless --apply).

Usage:
    scripts/repair_numbered_entity_magnets.py [--db-url DSN] [--samples N] [--plan-out FILE]
    scripts/repair_numbered_entity_magnets.py --apply [--unretire-allow FILE] [--date YYYYMMDD]
    scripts/repair_numbered_entity_magnets.py --restore YYYYMMDD [--apply]
    scripts/repair_numbered_entity_magnets.py --metrics

The DSN comes from --db-url, else $SYNAPSE_DB_URL; the dry run and --metrics use a READ
ONLY connection. ``--plan-out`` (fact text, entity names) and the allow file hold private
data and must live outside the repository; the script refuses a path inside it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid as uuidlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ingestion.dedup import _normalize_name, identifier_tokens, ids_conflict  # noqa: E402

# ------------------------------------------------------------------- identifiers

HASH_KINDS = ("pr", "issue", "task")

#: Whole-name shapes of an ID-bearing entity. A decorated name is never a candidate: it
#: says more than the number, and the extra words may be what it is really about.
_ENTITY_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pr", re.compile(r"^(?:draft\s+)?(?:pr|pull\s+request)\s*#\s*(\d+)$", re.I)),
    ("issue", re.compile(r"^issue\s*#\s*(\d+)$", re.I)),
    ("task", re.compile(r"^task\s*#\s*(\d+)$", re.I)),
)
#: A ticket key: an upper-case prefix that starts with a letter, a dash, 2+ digits.
_KEY_ENTITY = re.compile(r"^([A-Z][A-Z0-9]{1,9})-(\d{2,})$")

#: Names a move may land on: the bare identifier ("draft PR #12" is not "PR #12").
_TARGET_SHAPES: dict[str, re.Pattern[str]] = {
    "pr": re.compile(r"^(?:pr|pull\s+request)\s*#?\s*(\d+)$", re.I),
    "issue": re.compile(r"^issue\s*#\s*(\d+)$", re.I),
    "task": re.compile(r"^task\s*#\s*(\d+)$", re.I),
}

# A hash-style mention in fact text: a kind word with an optional '#', or a bare '#N'
# (not inside a word or an HTML entity). ``rng`` catches "#117-119" / "#5 to #9".
_HASH_MENTION = re.compile(
    r"""(?:
          \b(?P<kw>pull\s+requests?|prs?|issues?|tasks?)\s*(?P<hash>\#)?\s*
        | (?<![\w&\#])\#
        )
        (?P<num>\d+)\b
        (?P<rng>\s*(?:-|\u2013|\.\.|to)\s*\#?\d+\b)?""",
    re.I | re.X,
)
_URL_MENTION = re.compile(r"/(?P<kw>pull|issues)/(?P<num>\d+)\b")
# A ticket key in fact text; ``rng`` catches "ABC-12/13", "ABC-12-14", "ABC-12 and -14".
_KEY_MENTION = re.compile(
    r"\b(?P<prefix>[A-Za-z][A-Za-z0-9]{1,9})-(?P<num>\d{2,})\b"
    r"(?P<rng>\s*(?:/|-|\u2013|,|and|or)\s*-?\d{2,}\b)?"
)


@dataclass(frozen=True)
class Ident:
    """An identifier: ``kind`` is 'pr' / 'issue' / 'task' / 'key:<PREFIX>'; ``number``
    is normalized the way the resolver guard normalizes it (leading zeros dropped)."""

    kind: str
    number: str

    @property
    def family(self) -> str:
        return "ticket key" if self.kind.startswith("key:") else f"{self.kind} #"


def _num(digits: str) -> str:
    return next(iter(identifier_tokens(digits)))


def entity_ident(name: str) -> Ident | None:
    """The identifier an entity's whole name is, or None. The name must carry exactly one
    number by the guard's own reading (``identifier_tokens``)."""
    name = (name or "").strip()
    toks = identifier_tokens(name)
    if len(toks) != 1:
        return None
    for kind, rx in _ENTITY_SHAPES:
        m = rx.match(name)
        if m:
            return Ident(kind, _num(m.group(1)))
    m = _KEY_ENTITY.match(name)
    if m:
        return Ident(f"key:{m.group(1)}", _num(m.group(2)))
    return None


_KEY_TARGET = re.compile(r"^([A-Za-z][A-Za-z0-9]{1,9})-(\d{2,})$")


def target_ident(name: str) -> Ident | None:
    """The identifier ``name`` is exactly, if a move may land on it: "PR #218", "PR 218",
    "pull request 218", "issue #3", "task #4", "ABC-123" or "abc-123". Wider than
    ``entity_ident`` on purpose: an existing "PR 218" is reused rather than duplicated,
    but what sits on it is not read for other numbers."""
    name = (name or "").strip()
    if len(identifier_tokens(name)) != 1:
        return None
    for kind, rx in _TARGET_SHAPES.items():
        m = rx.match(name)
        if m:
            return Ident(kind, _num(m.group(1)))
    m = _KEY_TARGET.match(name)
    if m and re.search(r"[A-Za-z]", m.group(1)):
        return Ident(f"key:{m.group(1).upper()}", _num(m.group(2)))
    return None


def is_target_name(name: str, ident: Ident) -> bool:
    """True if ``name`` is exactly ``ident`` (an entity a move may land on)."""
    return target_ident(name) == ident


@dataclass(frozen=True)
class Mention:
    kind: str  # 'pr' / 'issue' / 'task' / 'bare' / 'key:<PREFIX>'
    number: str
    written: str  # the identifier as written in the text (range tail excluded)
    multi: bool  # part of a range or a continued list ("#117-119", "ABC-12/13")


def _hash_kind(kw: str) -> str:
    kw = kw.lower()
    if kw.startswith("pull") or kw.startswith("pr"):
        return "pr"
    return "issue" if kw.startswith("issue") else "task"


def mentions(text: str) -> list[Mention]:
    """Every PR / issue / task / bare-# / ticket-key identifier in ``text``."""
    out: list[Mention] = []
    text = text or ""
    for m in _HASH_MENTION.finditer(text):
        kw = m.group("kw")
        if kw:
            kind = _hash_kind(kw)
            # "issue 5" / "task 3" are prose more often than identifiers; "PR 12" is not.
            if kind != "pr" and not m.group("hash"):
                continue
            # As written, singular, with the '#': "PRs #12" / "pr 12" -> "PR #12" / "pr #12".
            word = " ".join(kw.split())
            word = word[:-1] if word[-1] in "sS" else word
            written = f"{word} #{m.group('num')}"
        else:
            kind = "bare"
            written = f"#{m.group('num')}"
        out.append(Mention(kind, _num(m.group("num")), written, bool(m.group("rng"))))
    for m in _URL_MENTION.finditer(text):
        kind = "pr" if m.group("kw") == "pull" else "issue"
        num = _num(m.group("num"))
        written = f"PR #{num}" if kind == "pr" else f"issue #{num}"
        out.append(Mention(kind, num, written, False))
    for m in _KEY_MENTION.finditer(text):
        prefix = m.group("prefix").upper()
        if not re.search(r"[A-Z]", prefix):
            continue
        written = f"{prefix}-{m.group('num')}"
        out.append(Mention(f"key:{prefix}", _num(m.group("num")), written, bool(m.group("rng"))))
    return out


def identifier_set(text: str) -> frozenset[tuple[str, str]]:
    """The identifiers ``text`` names, for comparing two facts. All hash kinds share one
    namespace (a bare '#N' can be any of them), so 'disjoint' is the strict reading; a
    short range is expanded ("#117-119" names 118)."""
    out: set[tuple[str, str]] = set()
    for mt in mentions(text):
        ns = mt.kind if mt.kind.startswith("key:") else "#"
        out.add((ns, mt.number))
    for m in _HASH_MENTION.finditer(text or ""):
        rng = m.group("rng")
        if rng:
            hi = re.search(r"\d+", rng)
            lo_n, hi_n = int(m.group("num")), int(hi.group(0)) if hi else 0
            if 0 < hi_n - lo_n <= 50:
                out.update(("#", str(n)) for n in range(lo_n, hi_n + 1))
    return frozenset(out)


# Decision for one endpoint of one edge.
MOVE = "move"
STAY_OWN = "names own number"
STAY_NONE = "names no identifier"
STAY_SEVERAL = "names several identifiers"
STAY_RANGE = "names a range or list"
STAY_BARE = "names only a bare #N"


def decide(ident: Ident, fact: str) -> tuple[str, Mention | None]:
    """Where the endpoint ``ident`` of an edge with this fact belongs: (MOVE, the
    identifier it names) or (a STAY_* reason, None)."""
    hash_kind = ident.kind in HASH_KINDS
    found = mentions(fact)
    same = [m for m in found if m.kind == ident.kind]
    bare = [m for m in found if hash_kind and m.kind == "bare"]
    numbers = {m.number for m in same} | {m.number for m in bare}
    if ident.number in numbers:
        return STAY_OWN, None
    if any(m.multi for m in same + bare):
        return STAY_RANGE, None
    if not same and not bare:
        return STAY_NONE, None
    same_nums = {m.number for m in same}
    if len(same_nums) == 1 and {m.number for m in bare} <= same_nums:
        return MOVE, same[0]
    if not same:
        return STAY_BARE, None
    return STAY_SEVERAL, None


# ------------------------------------------------------------------- plan


@dataclass
class Entity:
    uuid: str
    owner_id: str
    group_id: str
    name: str
    entity_type: str | None = None
    supertype: str | None = None
    subtype: str | None = None
    project: str | None = None
    summary: str = ""
    created_at: Any = None


@dataclass
class Edge:
    uuid: str
    owner_id: str
    group_id: str
    src: str
    tgt: str
    fact: str
    name: str | None = None
    live: bool = True
    invalidated_by: str | None = None
    expired: bool = False
    created_at: Any = None
    source_projects: list[str] | None = None


@dataclass
class Target:
    """Where moves for one (owner, group, identifier) land."""

    owner_id: str
    group_id: str
    ident: Ident
    uuid: str
    name: str
    existing: bool
    template: Entity  # the magnet the type/project are copied from
    moves: int = 0


@dataclass
class Move:
    edge: str
    side: str  # 'src' | 'tgt'
    from_uuid: str
    to_uuid: str
    family: str
    live: bool


@dataclass
class Candidate:
    """A retired edge whose supersession looks like a magnet artifact (rule 3)."""

    old: str
    new: str
    old_fact: str
    new_fact: str
    family: str


@dataclass
class Plan:
    moves: list[Move] = field(default_factory=list)
    targets: dict[tuple[str, str, Ident], Target] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    losers: list[str] = field(default_factory=list)  # every entity that loses an edge
    clear_summaries: list[str] = field(default_factory=list)
    stats: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def new_targets(self) -> list[Target]:
        return [t for t in self.targets.values() if not t.existing and t.moves]


def plan_repair(entities: Iterable[Entity], edges: Iterable[Edge]) -> Plan:
    """The whole repair, decided from rows already read (no I/O, deterministic)."""
    ents = {e.uuid: e for e in entities}
    edges = sorted(edges, key=lambda e: (str(e.created_at or ""), e.uuid))
    idents = {u: i for u, e in ents.items() if (i := entity_ident(e.name)) is not None}
    live_deg: Counter[str] = Counter()
    for ed in edges:
        if ed.live:
            live_deg[ed.src] += 1
            live_deg[ed.tgt] += 1

    # Existing landing entities per (owner, group, identifier): most live edges, oldest.
    landing: dict[tuple[str, str, Ident], Entity] = {}
    for e in sorted(
        ents.values(), key=lambda e: (-live_deg[e.uuid], str(e.created_at or ""), e.uuid)
    ):
        i = target_ident(e.name)
        if i is not None:
            landing.setdefault((e.owner_id, e.group_id, i), e)

    plan = Plan()
    st = plan.stats
    written_forms: dict[tuple[str, str, Ident], Counter[str]] = defaultdict(Counter)
    pending: list[tuple[Edge, dict[str, tuple[tuple[str, str, Ident], Entity]]]] = []
    for ed in edges:
        wants: dict[str, tuple[tuple[str, str, Ident], Entity]] = {}
        for side in ("src", "tgt"):
            ent = ents.get(getattr(ed, side))
            ident = idents.get(getattr(ed, side))
            if ent is None or ident is None:
                continue
            fam = ident.family
            st[fam]["endpoints examined"] += 1
            if (ent.owner_id, ent.group_id) != (ed.owner_id, ed.group_id):
                st[fam]["stay: edge in another group"] += 1
                continue
            verdict, mention = decide(ident, ed.fact)
            if verdict != MOVE or mention is None:
                st[fam][f"stay: {verdict}"] += 1
                continue
            key = (ed.owner_id, ed.group_id, Ident(ident.kind, mention.number))
            wants[side] = (key, ent)
            written_forms[key][mention.written] += 1
        if wants:
            pending.append((ed, wants))

    def target_for(key: tuple[str, str, Ident], template: Entity) -> Target:
        t = plan.targets.get(key)
        if t is None:
            hit = landing.get(key)
            if hit is not None:
                t = Target(key[0], key[1], key[2], hit.uuid, hit.name, True, template)
            else:
                # Most common spelling among the facts that move there, then first seen.
                forms = written_forms[key]
                name = max(forms, key=lambda f: (forms[f], -list(forms).index(f)))
                t = Target(key[0], key[1], key[2], str(uuidlib.uuid4()), name, False, template)
            plan.targets[key] = t
        return t

    new_end: dict[str, tuple[str, str]] = {}
    losers: set[str] = set()
    for ed, wants in pending:
        ends = {"src": ed.src, "tgt": ed.tgt}
        planned: list[tuple[str, Target, Entity]] = []
        for side, (key, ent) in wants.items():
            t = target_for(key, ent)
            # The shipped guard would have kept these two apart; anything else is a bug.
            assert ids_conflict(ent.name, t.name), (ent.name, t.name)
            ends[side] = t.uuid
            planned.append((side, t, ent))
        if ends["src"] == ends["tgt"]:
            for _side, _t, ent in planned:
                st[idents[ent.uuid].family]["stay: move would join the edge to itself"] += 1
            continue
        new_end[ed.uuid] = (ends["src"], ends["tgt"])
        for side, t, ent in planned:
            t.moves += 1
            fam = idents[ent.uuid].family
            st[fam]["move (live edge)" if ed.live else "move (retired edge)"] += 1
            plan.moves.append(Move(ed.uuid, side, ent.uuid, t.uuid, fam, ed.live))
            losers.add(ent.uuid)
    plan.losers = sorted(losers)
    plan.clear_summaries = [u for u in plan.losers if ents[u].summary]

    # Rule 3: supersession candidates.
    by_uuid = {e.uuid: e for e in edges}

    def pair(e: Edge) -> tuple[str, str]:
        return new_end.get(e.uuid, (e.src, e.tgt))

    on_pair: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        on_pair[pair(e)].append(e)
    for o in edges:
        if o.live or not o.invalidated_by or o.expired:
            continue
        n = by_uuid.get(o.invalidated_by)
        if n is None:
            continue
        shared = [u for u in {o.src, o.tgt} & {n.src, n.tgt} if u in idents]
        if not shared:
            continue
        fam = idents[sorted(shared)[0]].family
        before_same = (o.src, o.tgt) == (n.src, n.tgt)
        if before_same and (o.uuid in new_end or n.uuid in new_end):
            st[fam]["linked pair: " + ("kept together" if pair(o) == pair(n) else "separated")] += 1
        ids_o, ids_n = identifier_set(o.fact), identifier_set(n.fact)
        if not ids_o or not ids_n or ids_o & ids_n:
            continue
        st[fam]["supersession with disjoint identifiers"] += 1
        others = [
            e
            for e in on_pair[pair(o)]
            if e.live and e.uuid not in (o.uuid, n.uuid) and identifier_set(e.fact) & ids_o
        ]
        if others:
            st[fam]["  ...kept: another live edge on its pair names its identifier"] += 1
            continue
        st[fam]["  ...un-retire candidate"] += 1
        plan.candidates.append(Candidate(o.uuid, n.uuid, o.fact, n.fact, fam))
    return plan


# ------------------------------------------------------------------- database

_ENTITY_SQL = """
    SELECT uuid, owner_id, group_id, COALESCE(name, ''), entity_type, entity_supertype,
           entity_subtype, project, COALESCE(summary, ''), created_at
      FROM kg_entities
     WHERE name ~* '^\\s*(draft\\s+)?(pr|pull\\s+request|issue|task)\\s*#?\\s*[0-9]+\\s*$'
        OR name ~* '^[a-z][a-z0-9]{1,9}-[0-9]{2,}$'
"""
_EDGE_COLS = (
    "uuid, owner_id, group_id, src_uuid, tgt_uuid, COALESCE(fact, ''), name, "
    "t_invalid IS NULL, invalidated_by, t_expired IS NOT NULL, created_at, source_projects"
)


def _connect(url: str, *, read_only: bool) -> Any:
    import psycopg

    opts = "-c default_transaction_read_only=on" if read_only else None
    conn = psycopg.connect(url, options=opts) if opts else psycopg.connect(url)
    if read_only:
        conn.read_only = True
    return conn


def load(conn: Any) -> tuple[list[Entity], list[Edge]]:
    """Every ID-shaped entity, and every edge (live or retired) touching one."""
    ents = [Entity(*r) for r in conn.execute(_ENTITY_SQL).fetchall()]
    by_scope: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e in ents:
        by_scope[(e.owner_id, e.group_id)].append(e.uuid)
    seen: set[str] = set()
    edges: list[Edge] = []
    for (owner, group), uuids in sorted(by_scope.items()):
        for col in ("src_uuid", "tgt_uuid"):
            rows = conn.execute(
                f"SELECT {_EDGE_COLS} FROM kg_relationships "  # nosec B608 - fixed columns
                f"WHERE owner_id = %s AND group_id = %s AND {col} = ANY(%s)",
                (owner, group, uuids),
            ).fetchall()
            for r in rows:
                if r[0] not in seen:
                    seen.add(r[0])
                    edges.append(Edge(*r))
    # A successor on another pair (rare) is still needed to judge a supersession.
    missing = sorted({e.invalidated_by for e in edges if e.invalidated_by} - seen)
    if missing:
        for r in conn.execute(
            f"SELECT {_EDGE_COLS} FROM kg_relationships WHERE uuid = ANY(%s)",  # nosec B608
            (missing,),
        ).fetchall():
            if r[0] not in seen:
                seen.add(r[0])
                edges.append(Edge(*r))
    conn.rollback()
    return ents, edges


def _tables(day: str) -> tuple[Any, Any, Any]:
    from psycopg import sql

    if not re.fullmatch(r"\d{8}", day):
        raise SystemExit(f"date must be YYYYMMDD, got {day!r}")
    base = f"kg_magnet_repair_{day}"
    return (
        sql.Identifier(f"{base}_relationships"),
        sql.Identifier(f"{base}_entities"),
        sql.Identifier(f"{base}_created"),
    )


def backup(conn: Any, day: str, edge_uuids: list[str], entity_uuids: list[str]) -> None:
    """Copy every row the run will touch, in one transaction, before any write."""
    from psycopg import sql

    rel, ent, created = _tables(day)
    with conn.transaction():
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} AS "
                "SELECT now() AS backed_up_at, r.* FROM kg_relationships r WHERE false"
            ).format(rel)
        )
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} AS "
                "SELECT now() AS backed_up_at, e.* FROM kg_entities e WHERE false"
            ).format(ent)
        )
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} "
                "(uuid TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            ).format(created)
        )
        conn.execute(
            sql.SQL(
                "INSERT INTO {} SELECT now(), r.* FROM kg_relationships r WHERE r.uuid = ANY(%s)"
            ).format(rel),
            (edge_uuids,),
        )
        conn.execute(
            sql.SQL(
                "INSERT INTO {} SELECT now(), e.* FROM kg_entities e WHERE e.uuid = ANY(%s)"
            ).format(ent),
            (entity_uuids,),
        )


def _create_target(conn: Any, t: Target, embedding: list[float], created_table: Any) -> None:
    from psycopg import sql

    from ingestion.kg_pg_write import _EMBED_DIMS, _vec

    tpl = t.template
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, project, name, normalized_name, "
        "  entity_type, summary, embedding, created_at, t_created, valid_at, t_valid, "
        "  entity_supertype, entity_subtype) "
        f"VALUES (%s,%s,%s,%s,%s,%s,%s,'',%s::vector({_EMBED_DIMS}),now(),now(),now(),now(),%s,%s) "
        "ON CONFLICT (uuid) DO NOTHING",
        (
            t.uuid,
            t.owner_id,
            t.group_id,
            tpl.project or "",
            t.name,
            _normalize_name(t.name),
            tpl.entity_type or "",
            _vec(embedding),
            tpl.supertype,
            tpl.subtype,
        ),
    )
    conn.execute(
        sql.SQL("INSERT INTO {} (uuid) VALUES (%s) ON CONFLICT DO NOTHING").format(created_table),
        (t.uuid,),
    )


def apply(conn: Any, plan: Plan, embedder: Any, day: str, allow: set[str]) -> Counter[str]:
    """Execute ``plan`` (computed on this connection just before). Returns write counts."""
    done: Counter[str] = Counter()
    unretire = [c for c in plan.candidates if c.old in allow]
    moved_edges = sorted({m.edge for m in plan.moves})
    if not moved_edges and not unretire:
        return done
    new = plan.new_targets()
    vectors: dict[str, list[float]] = {}
    if new:
        embs = embedder.embed([t.name for t in new], task="entity")
        if len(embs) != len(new) or any(not v for v in embs):
            raise SystemExit("embedder returned no vector for some new entities; nothing written")
        vectors = {t.uuid: v for t, v in zip(new, embs, strict=True)}
    backed_edges = sorted(set(moved_edges) | {c.old for c in unretire})
    backup(conn, day, backed_edges, plan.losers)
    done["rows backed up (edges)"] = len(backed_edges)
    done["rows backed up (entities)"] = len(plan.losers)
    _rel, _ent, created_table = _tables(day)

    by_magnet: dict[str, list[Move]] = defaultdict(list)
    first_side: dict[str, str] = {}
    for m in plan.moves:
        first_side.setdefault(m.edge, m.from_uuid)
        by_magnet[first_side[m.edge]].append(m)
    clear = set(plan.clear_summaries)
    created: set[str] = set()
    cleared: set[str] = set()
    for magnet in sorted(by_magnet):
        moves = by_magnet[magnet]
        with conn.transaction():
            for m in moves:
                t = next(t for t in plan.targets.values() if t.uuid == m.to_uuid)
                if not t.existing and t.uuid not in created:
                    _create_target(conn, t, vectors[t.uuid], created_table)
                    created.add(t.uuid)
                    done["entities created"] += 1
                col = "src_uuid" if m.side == "src" else "tgt_uuid"
                cur = conn.execute(
                    f"UPDATE kg_relationships SET {col} = %s "  # nosec B608 - col is fixed
                    f"WHERE uuid = %s AND {col} = %s",
                    (m.to_uuid, m.edge, m.from_uuid),
                )
                done[
                    "endpoints moved"
                    if cur.rowcount == 1
                    else "endpoints changed since read (skipped)"
                ] += 1
            for u in sorted(({m.from_uuid for m in moves} & clear) - cleared):
                conn.execute("UPDATE kg_entities SET summary = '' WHERE uuid = %s", (u,))
                cleared.add(u)
                done["summaries cleared"] += 1
    if unretire:
        with conn.transaction():
            for c in unretire:
                cur = conn.execute(
                    "UPDATE kg_relationships SET t_invalid = NULL, invalid_at = NULL, "
                    "  invalidated_by = NULL "
                    "WHERE uuid = %s AND invalidated_by = %s AND t_invalid IS NOT NULL "
                    "  AND t_expired IS NULL",
                    (c.old, c.new),
                )
                done[
                    "supersessions un-retired"
                    if cur.rowcount == 1
                    else "un-retire skipped (changed)"
                ] += 1
    return done


def restore(conn: Any, day: str, *, write: bool) -> dict[str, int]:
    """Undo a run from its backup tables (one transaction; rolled back unless ``write``).

    Reverts what the run changed and nothing written since, from the earliest backed-up
    copy of each row:

    * an entity the run drained of every edge, which the poller's daily orphan sweep then
      deleted, is re-created from its backup row first, so no edge points at nothing;
    * every backed-up edge gets its endpoints back;
    * an edge the run un-retired that is still live is retired again as it was. A
      retirement written after the run (a moved fact superseded on its new pair, an
      un-retired fact retired again) is kept: nothing else ever clears ``t_invalid``;
    * a summary the run emptied gets its text back unless a mention has written one since;
    * the entities the run created are deleted unless an edge references them now.
    """
    from psycopg import sql

    rel, ent, created = _tables(day)
    out: dict[str, int] = {}
    import psycopg

    with conn.transaction():
        cols_sql = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "ORDER BY ordinal_position"
        )
        backed = [r[0] for r in conn.execute(cols_sql, (f"kg_magnet_repair_{day}_entities",))]
        if not backed:
            raise SystemExit(f"no backup tables for {day}")
        now_cols = {r[0] for r in conn.execute(cols_sql, ("kg_entities",))}
        cols = sql.SQL(", ").join(sql.Identifier(c) for c in backed if c in now_cols)
        cur = conn.execute(
            sql.SQL(
                "INSERT INTO kg_entities ({cols}) SELECT {cols} FROM "
                "  (SELECT DISTINCT ON (uuid) * FROM {ent} ORDER BY uuid, backed_up_at) b "
                "WHERE NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.uuid = b.uuid)"
            ).format(cols=cols, ent=ent)
        )
        out["entities re-created (deleted since the run)"] = cur.rowcount
        cur = conn.execute(
            sql.SQL(
                "UPDATE kg_relationships r SET src_uuid = b.src_uuid, tgt_uuid = b.tgt_uuid "
                "FROM (SELECT DISTINCT ON (uuid) uuid, src_uuid, tgt_uuid FROM {} "
                "      ORDER BY uuid, backed_up_at) b "
                "WHERE r.uuid = b.uuid "
                "  AND (r.src_uuid, r.tgt_uuid) IS DISTINCT FROM (b.src_uuid, b.tgt_uuid)"
            ).format(rel)
        )
        out["edges restored"] = cur.rowcount
        cur = conn.execute(
            sql.SQL(
                "UPDATE kg_relationships r SET t_invalid = b.t_invalid, "
                "  invalid_at = b.invalid_at, invalidated_by = b.invalidated_by "
                "FROM (SELECT DISTINCT ON (uuid) uuid, t_invalid, invalid_at, invalidated_by "
                "      FROM {} ORDER BY uuid, backed_up_at) b "
                "WHERE r.uuid = b.uuid AND b.t_invalid IS NOT NULL AND r.t_invalid IS NULL"
            ).format(rel)
        )
        out["un-retirements reverted"] = cur.rowcount
        cur = conn.execute(
            sql.SQL(
                "UPDATE kg_entities e SET summary = b.summary "
                "FROM (SELECT DISTINCT ON (uuid) uuid, summary FROM {} "
                "      ORDER BY uuid, backed_up_at) b "
                "WHERE e.uuid = b.uuid AND COALESCE(e.summary, '') = '' "
                "  AND COALESCE(b.summary, '') <> ''"
            ).format(ent)
        )
        out["summaries restored"] = cur.rowcount
        cur = conn.execute(
            sql.SQL(
                "DELETE FROM kg_entities e USING {} c WHERE e.uuid = c.uuid "
                "  AND NOT EXISTS (SELECT 1 FROM kg_relationships r WHERE r.owner_id = e.owner_id "
                "                  AND r.group_id = e.group_id AND r.src_uuid = e.uuid) "
                "  AND NOT EXISTS (SELECT 1 FROM kg_relationships r WHERE r.owner_id = e.owner_id "
                "                  AND r.group_id = e.group_id AND r.tgt_uuid = e.uuid)"
            ).format(created)
        )
        out["created entities deleted"] = cur.rowcount
        row = conn.execute(
            sql.SQL("SELECT count(*) FROM {} c JOIN kg_entities e ON e.uuid = c.uuid").format(
                created
            )
        ).fetchone()
        out["created entities kept (new edges reference them)"] = int(row[0]) if row else 0
        if not write:
            raise psycopg.Rollback()
    return out


# ------------------------------------------------------------------- metrics

# The audit's two health numbers, with the audit's own (looser) definitions so a before
# and an after compare with what it reported:
#   * live edges on entities whose name contains "PR #N" whose fact lacks that '#N';
#   * linked supersessions with an endpoint whose name carries '#N' or a ticket key, where
#     both facts name identifiers ('#' and 2+ digits, or a key) and the two sets are disjoint.
_HEALTH_PR_SQL = """
    SELECT e.uuid, e.name, r.uuid, COALESCE(r.fact, '') FROM kg_entities e
      JOIN kg_relationships r ON r.owner_id = e.owner_id AND r.group_id = e.group_id
                             AND r.{col} = e.uuid AND r.t_invalid IS NULL
     WHERE e.name ~ 'PR #[0-9]+'
"""
_HEALTH_SUP_SQL = """
    SELECT o.uuid, COALESCE(o.fact, ''), COALESCE(n.fact, '')
      FROM kg_relationships o
      JOIN kg_relationships n ON n.uuid = o.invalidated_by
      JOIN kg_entities se ON se.uuid = o.src_uuid
      JOIN kg_entities te ON te.uuid = o.tgt_uuid
     WHERE o.t_invalid IS NOT NULL
       AND (se.name ~ '#[0-9]+|[A-Z]-[0-9]{2,}' OR te.name ~ '#[0-9]+|[A-Z]-[0-9]{2,}')
"""
_AUDIT_IDS = re.compile(r"#0*(\d{2,})|\b([A-Z][A-Z0-9]{1,9}-\d{2,})\b")


def _audit_ids(text: str) -> set[str]:
    return {a or b for a, b in _AUDIT_IDS.findall(text or "")}


def health(conn: Any, plan: Plan | None = None, unretire: Iterable[str] = ()) -> dict[str, Any]:
    """The two numbers as stored now, or (given ``plan``) as they will be once it ran."""
    rows: list[tuple[str, str, str, str]] = []
    for col in ("src_uuid", "tgt_uuid"):
        rows += conn.execute(_HEALTH_PR_SQL.format(col=col)).fetchall()
    sups = conn.execute(_HEALTH_SUP_SQL).fetchall()
    conn.rollback()
    if plan is not None:
        names = {t.uuid: t.name for t in plan.targets.values()}
        moved = {(m.edge, m.from_uuid): m.to_uuid for m in plan.moves}
        rows = [
            (moved[(edge, ent)], names[moved[(edge, ent)]], edge, fact)
            if (edge, ent) in moved
            else (ent, name, edge, fact)
            for ent, name, edge, fact in rows
        ]
        gone = set(unretire)
        sups = [s for s in sups if s[0] not in gone]
    out: dict[str, Any] = {}
    for label, rx in (("name is PR #N", r"^PR #(\d+)$"), ("name contains PR #N", r"PR #(\d+)")):
        ents: set[str] = set()
        tot = miss = 0
        for ent, name, _edge, fact in rows:
            m = re.search(rx, name)
            if not m:
                continue
            ents.add(ent)
            tot += 1
            own = m.group(1).lstrip("0") or "0"
            miss += not re.search(rf"(?<![0-9])#0*{own}(?![0-9])", fact)
        out[label] = {
            "entities": len(ents),
            "live_edges": tot,
            "missing_own_number": miss,
            "ratio": round(miss / tot, 3) if tot else None,
        }
    pairs = [(_audit_ids(s[1]), _audit_ids(s[2])) for s in sups]
    both = [(a, b) for a, b in pairs if a and b]
    out["linked supersessions on ID-named entities"] = {
        "linked": len(sups),
        "both_name_ids": len(both),
        "disjoint": sum(1 for a, b in both if not a & b),
    }
    return out


# ------------------------------------------------------------------- reporting


def _outside_repo(path: str) -> Path:
    p = Path(path).resolve()
    if p == _REPO or _REPO in p.parents:
        raise SystemExit(f"{path}: holds private data; write it outside the repository")
    return p


def report(plan: Plan, edges: list[Edge]) -> dict[str, Any]:
    fams = sorted(plan.stats)
    summary: dict[str, Any] = {"by_family": {f: dict(sorted(plan.stats[f].items())) for f in fams}}
    magnets = Counter(m.from_uuid for m in plan.moves)
    summary["entities losing edges"] = len(magnets)
    summary["targets"] = {
        "existing": sum(1 for t in plan.targets.values() if t.existing and t.moves),
        "to create": len(plan.new_targets()),
    }
    summary["summaries to clear"] = len(plan.clear_summaries)
    summary["un-retire candidates"] = len(plan.candidates)
    # The same number in two repositories is one target (the guard cannot tell them apart
    # either). Upper bound: targets fed by facts whose single source project differs; a
    # project that discusses several repositories counts too.
    by_target: dict[str, set[str]] = defaultdict(set)
    edge_by = {e.uuid: e for e in edges}
    for m in plan.moves:
        sp = edge_by[m.edge].source_projects or []
        if len(sp) == 1:
            by_target[m.to_uuid].add(sp[0].lower().replace("-", "_"))
    summary["targets fed from 2+ single-project sources (upper bound on cross-repo)"] = sum(
        1 for v in by_target.values() if len(v) > 1
    )
    summary["moves from the 5 largest losers"] = [n for _u, n in magnets.most_common(5)]
    return summary


def print_samples(plan: Plan, entities: list[Entity], edges: list[Edge], n: int, seed: int) -> None:
    names = {e.uuid: e.name for e in entities}
    names.update({t.uuid: t.name + ("" if t.existing else " (new)") for t in plan.targets.values()})
    edge_by = {e.uuid: e for e in edges}
    rnd = random.Random(seed)
    picks = rnd.sample(plan.moves, min(n, len(plan.moves)))
    for m in picks:
        e = edge_by[m.edge]
        state = "live" if e.live else "retired"
        print(
            f"  [{m.family}] {names.get(m.from_uuid)} -> {names.get(m.to_uuid)} ({m.side}, {state})"
        )
        print(f"      {e.fact[:200]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL", ""))
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--unretire-allow", help="file of reviewed old-edge uuids to un-retire")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="backup table date")
    ap.add_argument("--restore", metavar="YYYYMMDD", help="undo a run from its backup tables")
    ap.add_argument("--metrics", action="store_true", help="print the health numbers only")
    ap.add_argument("--samples", type=int, default=0, help="print N planned moves")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plan-out", help="write the plan as JSON (outside the repository)")
    args = ap.parse_args(argv)
    if not args.db_url:
        ap.error("no DSN: pass --db-url or set SYNAPSE_DB_URL")

    if args.restore:
        conn = _connect(args.db_url, read_only=False)
        conn.autocommit = True
        out = restore(conn, args.restore, write=args.apply)
        print(json.dumps({"restore": args.restore, "written": args.apply, **out}, indent=1))
        return 0

    if args.metrics:
        print(json.dumps(health(_connect(args.db_url, read_only=True)), indent=1))
        return 0

    allow: set[str] = set()
    if args.unretire_allow:
        text = _outside_repo(args.unretire_allow).read_text()
        allow = {ln.split()[0] for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}

    conn = _connect(args.db_url, read_only=not args.apply)
    ents, edges = load(conn)
    plan = plan_repair(ents, edges)
    print(json.dumps({"dry_run": not args.apply, **report(plan, edges)}, indent=1))
    unretire = allow & {c.old for c in plan.candidates}
    if allow - unretire:
        print(f"note: {len(allow - unretire)} allow-listed uuid(s) are not candidates now")
    now, after = health(conn), health(conn, plan, unretire)
    print(json.dumps({"health now": now, "health after this plan": after}, indent=1))
    if args.samples:
        print(f"sample of {min(args.samples, len(plan.moves))} planned moves:")
        print_samples(plan, ents, edges, args.samples, args.seed)
    if args.plan_out:
        names = {e.uuid: e.name for e in ents}
        names.update({t.uuid: t.name for t in plan.targets.values()})
        edge_by = {e.uuid: e for e in edges}
        doc = {
            "moves": [
                {
                    "edge": m.edge,
                    "side": m.side,
                    "from": names.get(m.from_uuid),
                    "to": names.get(m.to_uuid),
                    "live": m.live,
                    "fact": edge_by[m.edge].fact,
                }
                for m in plan.moves
            ],
            "unretire_candidates": [vars(c) for c in plan.candidates],
            "new_entities": [t.name for t in plan.new_targets()],
        }
        _outside_repo(args.plan_out).write_text(json.dumps(doc, indent=1, default=str))
        print(f"plan written to {args.plan_out}")
    if not args.apply:
        return 0
    from ingestion.embedding import create_embedder

    conn.autocommit = True
    done = apply(
        conn,
        plan,
        create_embedder(db_url=args.db_url) if plan.new_targets() else None,
        args.date,
        allow,
    )
    print(json.dumps({"applied": dict(done), "backup_date": args.date}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
