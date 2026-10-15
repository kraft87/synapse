# mypy: ignore-errors
# Deliberately untyped route-handler module (FastMCP @custom_route decorators are inherently
# untyped) — already in the mypy pre-commit exclude; the pragma keeps it clean when a typed
# module (mcp_server/dashboard_routes.py) imports its _proposal_* helpers via follow-imports.
"""Plain-HTTP skill sync + review routes — the DB seam that lets the plugin stay DSN-free.

The Claude Code plugin used to reach Postgres directly (SYNAPSE_DB_URL) for two things:
two-way skill sync (the SessionStart hook) and the dream→skills proposal review CLI. Both
now go over these machine-token-gated HTTP routes instead, so a client needs ONE base URL +
an optional bearer — never database access. Mirrors the /ingest + /recall custom routes:
auth via the shared machine token (custom routes bypass FastMCP's auth middleware by design),
PG work in a threadpool, fail-soft JSON.

Routes (all POST):
  /skills/list        {scope, held?}                 -> active skills (name, desc, mtime, file shas,
                                                        content_hash: the whole-skill hash, so a
                                                        client compares it instead of the body)
                                                        + retired ones (name, hashes, retired_at)
                                                        + withdrawn ones among `held` (below)
                                                        + publish: may this caller push here
  /skills/fetch       {name}                         -> body + file contents (base64) for a pull
  /skills/publish     {name, scope, body, ...files}  -> upsert registry + replace files; refused
                                                        when the skill was retired after the edit,
                                                        lives in another scope, or the caller may
                                                        not write that scope
  /skills/retire      {name}                         -> status='retired', stamped with retired_at
  /skills/rescope     {name, scope}                  -> move an active skill to another scope
  /skills/overwrite   {name, scope, body, mtime}     -> record a clobbered local edit in history
  /skills/proposals   {id?}                          -> list proposed candidates, or one's detail
                                                        (detail: + where accept would apply it,
                                                        + why its last draft failed)
  /skills/proposals/act {id, action, reason?, body?, scope?, force?}
                                                     -> accept (applies) | reject

Who is served what (schema 053/054 trust, the board's rule applied to skills): a full-trust
device reads and writes every scope. A restricted device is served only ``project:<p>``
skills whose project is in its allowlist (case-insensitively) and may write only there: no
global skill, no other project's skill, no retire/rescope/review. A caller with no device
credential (the root token included) is served nothing. Before this, any approved device
was served every global skill, which is the leak schema 053 closed for notes.

``withdrawn``: a client sends the names it holds in the folder it is syncing (``held``),
and gets back each one that is still an active skill but is no longer served to it in this
scope, because it was moved to another scope or this device may not read it. It then deletes
its copy exactly when it is one Synapse delivered (the same removal path as a retirement,
``plugin/scripts/skills_sync.py::_settle_unserved``), instead of pushing it back. Only names
the client says it holds are answered, with content hashes and never content; a restricted
device sees another scope's name only as "" and an unknown caller gets no answer at all.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request
from starlette.responses import JSONResponse

from dream.skills.skill_doc import (
    body_hash,
    frontmatter_name,
    skill_description,
    valid_skill_name,
)
from dream.skills.skill_ledger import DRAFT_STATE, _rollup
from ingestion import pg_pool
from ingestion.project_label import label_from_path
from ingestion.schema_check import ColumnProbe
from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust
from mcp_server.http_helpers import _iso, err, guarded_json, status_conflict

logger = logging.getLogger(__name__)


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _utc(dt: datetime) -> datetime:
    """`dt`, with a naive one read as UTC so it compares against a timestamptz."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


#: Schema 060 adds skill_registry.retired_at. The image ships before it is applied, so every
#: statement that names the column asks this probe first; until the column exists, retire
#: reports the missing schema and list and publish behave exactly as before 060.
RETIRED_AT_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('skills_lane.skill_registry') "
    "  AND attname = 'retired_at' AND NOT attisdropped",
    "schema 060 (skill_registry.retired_at)",
)
_NO_RETIRED_AT = "schema 060 (skill_registry.retired_at) not applied"


def _retired_at_ready(c) -> bool:
    return RETIRED_AT_PROBE.ready(c, c.info.dsn)


@dataclass(frozen=True)
class SkillAccess:
    """Which skill scopes one caller is served and may write.

    Full trust: every scope. A restricted surface: the ``project:<p>`` scopes whose project
    is in its allowlist, compared case-insensitively (a label keeps its folder's case), and
    nothing else: not ``global``. Anyone else, an unknown surface, the root token, a wiring
    slip: nothing at all."""

    full: bool = False
    projects: frozenset[str] = frozenset()
    #: A surfaces row vouches for the caller. An unknown one is answered nothing about
    #: held names either, so the shared root token can't probe which skills exist.
    known: bool = False

    @classmethod
    def of(cls, st: SurfaceTrust) -> SkillAccess:
        if st.known and not st.restricted:
            return cls(full=True, known=True)
        return cls(projects=frozenset(p.lower() for p in st.allowed_projects), known=st.known)

    def serves(self, scope) -> bool:
        if not isinstance(scope, str):
            return False
        if self.full:
            return True
        return scope.lower().startswith("project:") and scope[8:].lower() in self.projects


FULL_ACCESS = SkillAccess(full=True, known=True)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def safe_skill_name(name) -> bool:
    """A skill name every machine can use as ONE folder name: no separator, no drive or
    stream colon, no control character, not "." or "..", no outer whitespace. A name is a
    folder on every machine the skill reaches, so a publish of anything else is refused."""
    return (
        isinstance(name, str)
        and 0 < len(name) <= 128
        and name == name.strip()
        and name not in (".", "..")
        and not any(ch in name for ch in "/\\:")
        and not _CONTROL.search(name)
    )


def safe_file_path(path) -> bool:
    """A bundled file path that stays inside its skill's folder on every OS: relative,
    "/"-separated, no empty, "." or ".." segment, no backslash, colon or control char."""
    if not isinstance(path, str) or not 0 < len(path) <= 512 or path.startswith("/"):
        return False
    if any(ch in path for ch in "\\:") or _CONTROL.search(path):
        return False
    return all(seg and seg not in (".", "..") and seg == seg.strip() for seg in path.split("/"))


def _lock_name(c, name: str) -> None:
    """Serialise writers of one skill name for the rest of the transaction: a row that does
    not exist yet has nothing for FOR UPDATE to lock, so two first publishes could race."""
    c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("skills_lane.skill:" + name,))


#: How many held names one listing answers, and how many past versions a withdrawn or
#: retired entry vouches for.
_MAX_HELD = 1000
_MAX_PAST_VERSIONS = 20


# --------------------------------------------------------------------------- sync


def whole_skill_hash(body: str, files) -> str:
    """The client sync's whole-skill hash (plugin/scripts/skills_sync.py ``_whole_hash``),
    byte for byte: the body, then each sorted (path, sha256) pair, NUL-separated. A retired
    entry carries it so a client can tell the copy Synapse delivered from a local edit."""
    h = hashlib.sha256()
    h.update(body.encode("utf-8"))
    for path, sha in sorted(files):
        h.update(b"\0")
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(sha.encode())
    return h.hexdigest()


def _files_by_skill(c, names: list[str]) -> dict[str, list[dict]]:
    """{name: [{path, sha256}, ...] by path} for each of `names`, in one query."""
    out: dict[str, list[dict]] = {n: [] for n in names}
    rows = c.cursor(row_factory=dict_row).execute(  # whatever row factory `c` has
        "SELECT skill_name, path, sha256 FROM skills_lane.skill_files "
        "WHERE skill_name = ANY(%s) ORDER BY skill_name, path",
        (names,),
    )
    for r in rows.fetchall():
        out[r["skill_name"]].append({"path": r["path"], "sha256": r["sha256"]})
    return out


def _file_shas(c, name: str) -> list[dict]:
    return _files_by_skill(c, [name])[name]


def _held_names(held) -> list[str] | None:
    if not isinstance(held, list):
        return None
    return sorted({h for h in held if isinstance(h, str) and h})[:_MAX_HELD]


def _list_skills(db_url: str, scope: str, access: SkillAccess = FULL_ACCESS, held=None) -> dict:
    """The skills in `scope` this caller is served: active ones under "skills" (all an
    older client reads); once schema 060 is applied, retired ones under "retired"; and,
    when the client says which names it holds, the held ones it is no longer served here
    under "withdrawn". "publish" says whether it may push to this scope at all.

    A project scope matches in any case: a project's label keeps its folder's case, and
    the same project can reach the server as ``project:Repo`` from one machine and
    ``project:repo`` from another."""
    served = access.serves(scope)
    held_names = _held_names(held) if access.known else None
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        skills = (
            c.execute(
                "SELECT name, scope, COALESCE(body,'') AS body, "
                "COALESCE(description,'') AS description, content_modified_at "
                "FROM skills_lane.skill_registry "
                "WHERE status='active' AND lower(scope)=lower(%s) ORDER BY name",
                (scope,),
            ).fetchall()
            if served
            else []
        )
        files = _files_by_skill(c, [s["name"] for s in skills])
        listing = {
            "skills": [
                {
                    "name": s["name"],
                    "scope": s["scope"],
                    # body and description stay for a client that predates content_hash (the
                    # plugin updates from the marketplace independently of the server)
                    "body": s["body"],
                    "description": s["description"],
                    "content_modified_at": _iso(s["content_modified_at"]),
                    "files": files[s["name"]],
                    "content_hash": whole_skill_hash(s["body"], _pairs(files[s["name"]])),
                }
                for s in skills
            ],
            "publish": served,
        }
        if _retired_at_ready(c):
            listing["retired"] = _retired_skills(c, scope, access, held_names)
        if held_names is not None:
            listed = {s["name"] for s in skills}
            listing["withdrawn"] = _withdrawn_skills(
                c, scope, access, [n for n in held_names if n not in listed]
            )
    return listing


def _shown_scope(scope: str, asked: str, access: SkillAccess) -> str:
    """The scope a retired/withdrawn entry reports: the real one when this caller is
    served that scope, else "". "" reads to a client as "another scope's skill", so only
    the versions the entry vouches for (Synapse's own) are removed, never one a local
    ledger remembers: in a scope a device may not write, an older client recorded pushes
    the server refused as delivered. "" also keeps a restricted device from learning where
    a skill it may not read lives."""
    return scope if access.serves(scope) else ""


def _pairs(files: list[dict]) -> list[tuple[str, str]]:
    return [(f["path"], f["sha256"]) for f in files]


def _past_bodies(c, names: list[str]) -> dict[str, list[str]]:
    """{name: its recent past registry bodies, newest first} for each of `names`, in one
    query. Only bodies the registry itself held ('superseded' by the history trigger); a
    'disk_overwrite' row is a local edit a pull replaced, which Synapse never delivered."""
    out: dict[str, list[str]] = {n: [] for n in names}
    rows = c.cursor(row_factory=dict_row).execute(
        "SELECT name, body FROM ("
        "  SELECT name, COALESCE(body,'') AS body, "
        "         row_number() OVER (PARTITION BY name ORDER BY recorded_at DESC) AS n "
        "  FROM skills_lane.skill_history WHERE name = ANY(%s) AND op='superseded'"
        ") h WHERE n <= %s ORDER BY name, n",
        (names, _MAX_PAST_VERSIONS),
    )
    for r in rows.fetchall():
        out[r["name"]].append(r["body"])
    return out


def _entry_hashes(body: str, files: list[dict], past_bodies: list[str]) -> tuple[str, list[str]]:
    """The whole-skill hash of the skill as it is, and with it the hashes of its recent past
    bodies under today's files: a copy pulled before the latest edit is still a copy Synapse
    delivered."""
    pairs = _pairs(files)
    current = whole_skill_hash(body, pairs)
    past = dict.fromkeys(whole_skill_hash(b, pairs) for b in past_bodies)
    return current, [current, *(h for h in past if h != current)]


def _retired_skills(c, asked: str, access: SkillAccess, held: list[str] | None) -> list[dict]:
    """Retired skills, each with the whole-skill hash of the content it was retired with
    ("content_hash") and of its recent past versions ("content_hashes"). A name is one
    registry row, and its scope is whichever folder published it last: a copy of the same
    skill in another scope's folder is just as retired. A client removes its copy only
    when that copy is still exactly a version Synapse delivered, whatever its timestamp
    (in another scope's folder: only those exact contents). A row with no retired_at
    (status set by hand) is left out: publish does not enforce it either.

    Full trust is told every retirement, whatever scope it asks about. A restricted caller
    only of the skills it could have been served, and of the names it says it holds."""
    rows = [
        r
        for r in c.execute(
            "SELECT name, scope, COALESCE(body,'') AS body, retired_at "
            "FROM skills_lane.skill_registry "
            "WHERE status='retired' AND retired_at IS NOT NULL ORDER BY name",
        ).fetchall()
        if access.serves(r["scope"]) or (held is not None and r["name"] in held)
    ]
    names = [r["name"] for r in rows]
    files, past = _files_by_skill(c, names), _past_bodies(c, names)
    out = []
    for r in rows:
        current, hashes = _entry_hashes(r["body"], files[r["name"]], past[r["name"]])
        out.append(
            {
                "name": r["name"],
                "scope": _shown_scope(r["scope"], asked, access),
                "content_hash": current,
                "content_hashes": hashes,
                "retired_at": _iso(r["retired_at"]),
            }
        )
    return out


def _withdrawn_skills(c, asked: str, access: SkillAccess, names: list[str]) -> list[dict]:
    """Of the held names the listing did not serve, the ones that are active skills: moved
    to another scope, or not this caller's to read. Each carries the hashes of the versions
    Synapse delivered (see _retired_skills), and no retired_at: a withdrawn copy is removed
    when it is exactly one of them and otherwise left alone, never pushed back."""
    if not names:
        return []
    rows = c.execute(
        "SELECT name, scope, COALESCE(body,'') AS body FROM skills_lane.skill_registry "
        "WHERE status='active' AND name = ANY(%s) ORDER BY name",
        (names,),
    ).fetchall()
    found = [r["name"] for r in rows]
    files, past = _files_by_skill(c, found), _past_bodies(c, found)
    out = []
    for r in rows:
        current, hashes = _entry_hashes(r["body"], files[r["name"]], past[r["name"]])
        out.append(
            {
                "name": r["name"],
                "scope": _shown_scope(r["scope"], asked, access),
                "content_hash": current,
                "content_hashes": hashes,
            }
        )
    return out


def _fetch_skill(db_url: str, name: str, access: SkillAccess = FULL_ACCESS) -> dict:
    """One active skill's body and files, for a pull. A skill this caller is not served is
    "not found", exactly as one that does not exist."""
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        row = c.execute(
            "SELECT name, scope, COALESCE(body,'') AS body, COALESCE(description,'') AS description, "
            "content_modified_at FROM skills_lane.skill_registry WHERE name=%s AND status='active'",
            (name,),
        ).fetchone()
        if not row or not access.serves(row["scope"]):
            return {"found": False}
        files = c.execute(
            "SELECT path, content, is_executable FROM skills_lane.skill_files "
            "WHERE skill_name=%s ORDER BY path",
            (name,),
        ).fetchall()
    return {
        "found": True,
        "name": row["name"],
        "scope": row["scope"],
        "body": row["body"],
        "description": row["description"],
        "content_modified_at": _iso(row["content_modified_at"]),
        "files": [
            {
                "path": f["path"],
                "content_b64": base64.b64encode(bytes(f["content"])).decode(),
                "is_executable": f["is_executable"],
            }
            for f in files
        ],
    }


def _upsert_registry(
    cur, name: str, scope: str, body: str, description: str | None, content_modified_at
) -> None:
    """THE registry write for a skill's SKILL.md — /skills/publish and the review accept both
    go through it, so an accepted proposal lands exactly like a client publish.

    The skill_history trigger fires automatically on a body change, so a superseded body stays
    recoverable. A write is the skill's newest edit, so it ends any retirement (callers refuse
    the writes that must not)."""
    cur.execute(
        """INSERT INTO skills_lane.skill_registry
               (name, scope, body, description, status, content_modified_at)
             VALUES (%s,%s,%s,%s,'active',%s)
           ON CONFLICT (name) DO UPDATE SET
             scope=EXCLUDED.scope, body=EXCLUDED.body, description=EXCLUDED.description,
             status='active', content_modified_at=EXCLUDED.content_modified_at,
             updated_at=now()""",
        (name, scope, body, description, content_modified_at),
    )
    if _retired_at_ready(cur.connection):
        cur.execute(
            "UPDATE skills_lane.skill_registry SET retired_at=NULL "
            "WHERE name=%s AND retired_at IS NOT NULL",
            (name,),
        )


def _retirement_holds(c, p: dict, edited: datetime | None) -> datetime | None:
    """The retired_at that refuses the publish `p` (edited at `edited`), or None.

    Newest edit wins: only an edit made after the retirement brings the skill back, and an
    edit with no time is not newer. Nor is the very content that was retired, whatever its
    timestamp: a clone, a restore, a copy or a fast clock renews an mtime, not the skill.
    Without this a machine that still holds a retired skill republishes it at its next
    session start. Locks the row, so a retire can't land between this check and the
    publish's write."""
    if not _retired_at_ready(c):
        return None
    name = p["name"]
    row = c.execute(
        "SELECT status, retired_at, COALESCE(body,'') FROM skills_lane.skill_registry "
        "WHERE name=%s FOR UPDATE",
        (name,),
    ).fetchone()
    if row is None or row[0] != "retired" or row[1] is None:
        return None
    if edited is None or _utc(edited) <= row[1]:
        return row[1]
    retired = whole_skill_hash(row[2], _pairs(_file_shas(c, name)))
    pushed = whole_skill_hash(
        p.get("body", ""), [(f["path"], f["sha256"]) for f in p.get("files", [])]
    )
    return row[1] if pushed == retired else None


def _scope_holds(c, name: str, scope: str, access: SkillAccess) -> dict | str:
    """The scope a publish of ``name`` to ``scope`` writes, or the refusal (a dict).

    An active skill keeps its scope: a push from another scope's folder is refused
    ("scoped") instead of moving it, because a machine still holding a copy in the folder
    a skill was moved OUT of would otherwise move it back at its next session start (and a
    skill re-scoped for trust would leak again). Moving a skill is /skills/rescope. The
    same scope in another case keeps the registry's spelling. A restricted caller may not
    take over a retired skill of a scope it is not served."""
    row = c.execute(
        "SELECT scope, status FROM skills_lane.skill_registry WHERE name=%s FOR UPDATE",
        (name,),
    ).fetchone()
    if row is None:
        return scope
    if row[1] == "active":
        if row[0].lower() != scope.lower():
            return {
                "status": "scoped",
                "name": name,
                "scope": row[0] if access.serves(row[0]) else None,
            }
        return row[0]
    if not access.serves(row[0]):
        return {"status": "forbidden", "name": name}
    return scope


def _publish_skill(db_url: str, p: dict, access: SkillAccess = FULL_ACCESS) -> dict:
    name = p["name"]
    scope = p.get("scope", "global")
    if not safe_skill_name(name):
        return {"status": "error", "detail": "skill name must be usable as one folder name"}
    if not all(isinstance(f, dict) and safe_file_path(f.get("path")) for f in p.get("files", [])):
        return {"status": "error", "detail": "file paths must stay inside the skill's folder"}
    if not isinstance(scope, str) or not _SCOPE_RE.match(scope):
        return {"status": "error", "detail": "scope must be 'global' or 'project:<name>'"}
    if not access.serves(scope):
        # 200, not 403: an older client treats an HTTP error as the end of its whole sync.
        return {"status": "forbidden", "name": name}
    edited = _dt(p.get("content_modified_at"))
    with pg_pool.connection(db_url, autocommit=False) as c:
        _lock_name(c, name)
        retired_at = _retirement_holds(c, p, edited)
        if retired_at is not None:
            c.rollback()
            return {"status": "retired", "name": name, "retired_at": _iso(retired_at)}
        held = _scope_holds(c, name, scope, access)
        if isinstance(held, dict):
            c.rollback()
            return held
        cur = c.cursor()
        _upsert_registry(
            cur,
            name,
            held,
            p.get("body", ""),
            p.get("description") or None,
            edited,
        )
        cur.execute("DELETE FROM skills_lane.skill_files WHERE skill_name=%s", (name,))
        for f in p.get("files", []):
            content = base64.b64decode(f["content_b64"])
            cur.execute(
                """INSERT INTO skills_lane.skill_files
                       (skill_name, path, content, sha256, size, is_executable)
                     VALUES (%s,%s,%s,%s,%s,%s)""",
                (
                    name,
                    f["path"],
                    content,
                    f["sha256"],
                    f.get("size", len(content)),
                    bool(f.get("is_executable", False)),
                ),
            )
        c.commit()
    return {"status": "ok", "name": name}


def _record_overwrite(db_url: str, p: dict, access: SkillAccess = FULL_ACCESS) -> dict:
    """Record a local body a pull is about to overwrite. A caller may record one only for a
    skill it is served: checked against the skill's stored scope, not the one it sends."""
    with pg_pool.connection(db_url, autocommit=False) as c:
        row = c.execute(
            "SELECT scope FROM skills_lane.skill_registry WHERE name=%s", (p["name"],)
        ).fetchone()
        if not access.serves(row[0] if row else p.get("scope") or "global"):
            return {"status": "forbidden"}
        c.execute(
            "INSERT INTO skills_lane.skill_history (name, scope, body, content_modified_at, op) "
            "VALUES (%s,%s,%s,%s,'disk_overwrite')",
            (p["name"], p.get("scope"), p.get("body", ""), _dt(p.get("content_modified_at"))),
        )
        c.commit()
    return {"status": "ok"}


def _retire_skill(db_url: str, name) -> dict:
    """Retire `name`: status='retired' stamped with retired_at, plus a skill_history row so
    the retirement shows in the change trail. A row retired by hand (no retired_at) gets
    stamped too, so its retirement starts to hold."""
    if not isinstance(name, str) or not name.strip():
        return {"status": "error", "detail": "'name' must be a non-empty string"}
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        if not _retired_at_ready(c):
            return {"status": "error", "detail": _NO_RETIRED_AT}
        row = c.execute(
            "UPDATE skills_lane.skill_registry "
            "SET status='retired', retired_at=now(), updated_at=now() "
            "WHERE name=%s AND (status<>'retired' OR retired_at IS NULL) "
            "RETURNING name, scope, body, content_modified_at, retired_at",
            (name,),
        ).fetchone()
        if row is None:
            prior = c.execute(
                "SELECT retired_at FROM skills_lane.skill_registry WHERE name=%s", (name,)
            ).fetchone()
            if prior is None:
                return {"status": "not_found", "name": name}
            return {
                "status": "already_retired",
                "name": name,
                "retired_at": _iso(prior["retired_at"]),
            }
        c.execute(
            "INSERT INTO skills_lane.skill_history (name, scope, body, content_modified_at, op) "
            "VALUES (%s,%s,%s,%s,'retired')",
            (row["name"], row["scope"], row["body"], row["content_modified_at"]),
        )
        c.commit()
    return {
        "status": "retired",
        "name": row["name"],
        "scope": row["scope"],
        "retired_at": _iso(row["retired_at"]),
    }


def _rescope_skill(db_url: str, name, scope) -> dict:
    """Move an active skill to ``scope`` ('global' or 'project:<label>'), with a
    skill_history row so the move shows in the change trail. Machines holding the skill in
    the folder it left drop their copy at their next session start when it is one Synapse
    delivered (it comes back as "withdrawn"); the new scope's folders pull it."""
    if not isinstance(name, str) or not name.strip():
        return {"status": "error", "detail": "'name' must be a non-empty string"}
    if not isinstance(scope, str) or not _SCOPE_RE.match(scope):
        return {"status": "error", "detail": "scope must be 'global' or 'project:<name>'"}
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        row = c.execute(
            "SELECT scope, status, body, content_modified_at FROM skills_lane.skill_registry "
            "WHERE name=%s FOR UPDATE",
            (name,),
        ).fetchone()
        if row is None:
            return {"status": "not_found", "name": name}
        if row["status"] != "active":
            return {"status": "error", "detail": f"{name} is {row['status']}, not active"}
        if row["scope"] == scope:
            return {"status": "unchanged", "name": name, "scope": scope}
        c.execute(
            "UPDATE skills_lane.skill_registry SET scope=%s, updated_at=now() WHERE name=%s",
            (scope, name),
        )
        c.execute(
            "INSERT INTO skills_lane.skill_history (name, scope, body, content_modified_at, op) "
            "VALUES (%s,%s,%s,%s,'rescoped')",
            (name, row["scope"], row["body"], row["content_modified_at"]),
        )
        c.commit()
    return {"status": "rescoped", "name": name, "scope": scope, "from": row["scope"]}


# ----------------------------------------------------------------------- review
#
# Accept APPLIES a proposal: the drafted (or reviewer-supplied) SKILL.md is written to the
# registry through _upsert_registry — the same write /skills/publish does — and the candidate
# goes straight to 'promoted'. Delivery to the machine that owns the skill is the existing
# two-way sync's job: the registry now carries a different whole-skill hash with a newer
# content_modified_at, so that machine pulls it at its next SessionStart.

# --scope values: 'global' or 'project:<dir basename>' (what the client sync keys on).
_SCOPE_RE = re.compile(r"^(global|project:[^/\\\r\n]+)$")
# A session started in a home directory carries that directory's basename as its project,
# but the client sync never syncs project:<home> (it would sync the global folder twice), so
# a skill scoped there would reach no machine. Evidence from a home-dir cwd scopes global.
_HOME_DIR_RE = re.compile(
    r"^(?:/home/[^/]+|/Users/[^/]+|/root|[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+)[\\/]*$"
)
_CANDIDATE_COLS = (
    "id, kind, name, direction, target_skills, status, score, evidence, "
    "proposal_body, base_body_hash, summary, salience, source_detector, proposed_patch"
)
_NO_DRAFT = "no draft yet; re-drafts on the next nightly run, or pass --body-file"


def _proposals_list(db_url: str) -> dict:
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        rows = c.execute(
            """SELECT id, kind, name, direction, target_skills, status, score,
                      grounded_sessions, judge_sessions, summary, trigger_phrasings,
                      salience, source_detector, (proposal_body IS NOT NULL) AS has_draft
                 FROM skills_lane.skill_gap_candidates
                WHERE status='proposed' AND (rejected_until IS NULL OR rejected_until < now())
                ORDER BY score DESC""",
        ).fetchall()
    return {"proposals": [dict(r) for r in rows]}


def _evidence_scope(c, evidence) -> str:
    """'project:<name>' when every evidence session ran in that one project, else 'global'.
    Unprovable (no sessions, a session with no episodes, an untagged or home-dir session)
    stays global. A label stored as a whole path (the parsers before the backslash fix)
    counts as its folder's name, the label a client sends today, so relabel_projects can
    still move the scope; labels that differ only in case are one project (the most
    common spelling names it); a label that still could not be a scope stays global."""
    sids = sorted(
        {e["session_id"] for e in evidence or [] if isinstance(e, dict) and e.get("session_id")}
    )
    if not sids:
        return "global"
    rows = c.execute(
        "SELECT DISTINCT session_id, project, metadata->>'cwd' AS cwd FROM episodes "
        "WHERE session_id = ANY(%s)",
        (sids,),
    ).fetchall()
    if {r["session_id"] for r in rows} != set(sids):
        return "global"
    projects = [label_from_path(r["project"]) for r in rows]
    if None in projects or len({p.lower() for p in projects}) != 1:
        return "global"
    if any(r["cwd"] and _HOME_DIR_RE.match(r["cwd"]) for r in rows):
        return "global"
    name = max(sorted(set(projects)), key=projects.count)
    scope = f"project:{name}"
    return scope if _SCOPE_RE.match(scope) else "global"


def _plan_apply(c, row, body: str, scope_override: str | None, *, lock: bool = False) -> dict:
    """Where accepting `row` with `body` lands in the registry, and what stands in the way.

    issues: [{code, detail, force}] — force=True issues yield to --force, the others never do.
    Scope: a retune keeps the existing skill's scope; a new skill (derive) is global unless
    all its evidence is from one project; an explicit scope_override wins either way."""
    kind = row["kind"]
    if kind == "retune":
        skill = (row["target_skills"] or [row["name"]])[0]
    else:
        fm = frontmatter_name(body)
        skill = fm if valid_skill_name(fm) else re.sub(r"[^a-z0-9-]", "", row["name"].lower())
    reg = c.execute(
        "SELECT scope, status, COALESCE(body, '') AS body FROM skills_lane.skill_registry "  # nosec B608 — a fixed FOR UPDATE suffix; the name is bound
        "WHERE name=%s" + (" FOR UPDATE" if lock else ""),
        (skill,),
    ).fetchone()
    exists = reg is not None
    registry_body = reg["body"] if exists else None
    current_hash = body_hash(registry_body)

    issues: list[dict] = []

    def issue(code: str, detail: str, force: bool = False) -> None:
        issues.append({"code": code, "detail": detail, "force": force})

    if not body.strip():
        issue("no_draft", _NO_DRAFT)
    if scope_override is not None and not _SCOPE_RE.match(scope_override):
        issue("bad_scope", f"scope must be 'global' or 'project:<name>', got {scope_override!r}")
    if kind not in ("derive", "retune"):  # a leftover 'consolidate' row: nothing applies it
        issue("bad_kind", f"{kind} candidates can't be applied; reject it")
    elif kind == "derive":
        if not valid_skill_name(skill):
            issue(
                "bad_name", f"no usable skill name (frontmatter name or candidate {row['name']!r})"
            )
        elif exists:
            issue(
                "exists",
                f"a skill named '{skill}' already exists in the registry ({reg['scope']}); "
                "accepting would overwrite it — rename it in --body-file, or pass --force",
                force=True,
            )
    elif kind == "retune":
        if not exists:
            issue(
                "no_target",
                f"retune target '{skill}' is not in the skill registry; pass --force to create it",
                force=True,
            )
        else:
            # The name may not change. A registry body with no frontmatter name (a legacy
            # row) only requires the new name, if any, to match the registry key.
            was, now_ = frontmatter_name(registry_body), frontmatter_name(body)
            if body.strip() and (now_ != was if was else now_ not in ("", skill)):
                issue(
                    "renamed",
                    f"the proposed body renames the skill ({was or skill!r} -> {now_!r}); "
                    "a retune must keep its name",
                )
            base = row.get("base_body_hash")
            if base and base != current_hash:
                issue(
                    "stale",
                    f"the registry body of '{skill}' changed since this proposal was drafted; "
                    "it re-drafts against the new body on the next nightly run, or pass --force "
                    "to apply this draft anyway",
                    force=True,
                )
    if exists and reg["status"] == "retired":
        issue(
            "retired",
            f"'{skill}' is retired; accepting would reactivate it — pass --force to do that",
            force=True,
        )

    if scope_override is not None:
        scope = scope_override
    elif exists:
        scope = reg["scope"]
    elif kind == "derive":
        scope = _evidence_scope(c, row["evidence"])
    else:
        scope = "global"
    return {
        "skill": skill,
        "scope": scope,
        "exists": exists,
        "registry_body": registry_body,
        "current_hash": current_hash,
        "issues": issues,
    }


def _proposal_detail(db_url: str, cid: int) -> dict:
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        r = c.execute(
            f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s",  # nosec B608 — _CANDIDATE_COLS is a literal column list
            (cid,),
        ).fetchone()
        if not r:
            return {"found": False}
        d = dict(r)
        if DRAFT_STATE.ready(c, c.info.dsn):  # why the nightly draft failed, if it did
            d.update(
                c.execute(
                    "SELECT draft_attempts, last_draft_error "
                    "FROM skills_lane.skill_gap_candidates WHERE id=%s",
                    (cid,),
                ).fetchone()
            )
        plan = _plan_apply(c, r, r["proposal_body"] or "", None)
        d["apply_to"] = {"skill": plan["skill"], "scope": plan["scope"], "exists": plan["exists"]}
        d["registry_body"] = plan["registry_body"]
        d["apply_issues"] = plan["issues"]
    d["found"] = True
    return d


def _apply_accept(
    c, cid: int, body: str | None, scope: str | None, force: bool, allowed: tuple[str, ...]
) -> dict:
    """One transaction: lock the candidate, re-check its status and the plan, write the registry
    the way a publish does, and mark the candidate promoted with the grounded accept recorded."""
    row = c.execute(
        f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s FOR UPDATE",  # nosec B608 — _CANDIDATE_COLS is a literal column list
        (cid,),
    ).fetchone()
    if not row:
        c.rollback()
        return {"found": False}
    if row["status"] not in allowed:
        c.rollback()
        return status_conflict(cid, row["status"], allowed)
    for target in row["target_skills"] or [row["name"]]:
        if isinstance(target, str):
            _lock_name(c, target)  # the same lock a publish takes
    text = body if body is not None else (row["proposal_body"] or "")
    plan = _plan_apply(c, row, text, scope, lock=True)
    blocking = [i for i in plan["issues"] if not (force and i["force"])]
    if blocking:
        c.rollback()
        return {
            "status": "refused",
            "id": cid,
            "name": row["name"],
            "reason": blocking[0]["code"],
            "detail": "; ".join(i["detail"] for i in blocking),
            "issues": blocking,
        }

    now = c.execute("SELECT now() AS now").fetchone()["now"]
    _upsert_registry(
        c.cursor(), plan["skill"], plan["scope"], text, skill_description(text) or None, now
    )
    ev = row["evidence"] or []
    ev.append({"session_id": None, "class": "grounded", "signal": "accept"})
    _js, gs, _jw, gw = _rollup(ev)
    c.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', proposal_body=%s, "
        "evidence=%s::jsonb, grounded_sessions=%s, grounded_weight=%s, updated_at=now() "
        "WHERE id=%s",
        (text, Json(ev), gs, gw, cid),
    )
    c.commit()
    forced = [i["code"] for i in plan["issues"] if i["force"]]
    return {
        "status": "promoted",
        "applied": True,
        "id": cid,
        "name": row["name"],
        "skill": plan["skill"],
        "scope": plan["scope"],
        "created": not plan["exists"],
        "forced": forced,
        "previous_body_hash": plan["current_hash"],
        "body_hash": body_hash(text),
        "detail": f"applied: '{plan['skill']}' ({plan['scope']}) written to the skill registry",
    }


# The statuses accept and reject act on: a decision is final, so a promoted, rejected or
# retired candidate refuses both (a late click must not restart a rejection's cooling window).
_DECIDABLE = ("observe", "proposed", "accepted")


def _proposal_act(
    db_url: str,
    cid: int,
    action: str,
    reason: str | None,
    *,
    body: str | None = None,
    scope: str | None = None,
    force: bool = False,
    expected: tuple[str, ...] | None = None,
) -> dict:
    """accept applies the proposal (see _apply_accept); reject drops it for 30 days. ``expected``
    narrows the statuses the action acts on (the dashboard decides only from 'proposed'); the
    status is checked under the candidate's row lock, so two decisions landing together cannot
    both win: the loser gets a 'conflict' result and the row is untouched."""
    allowed = expected or _DECIDABLE
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        if action == "accept":
            return _apply_accept(c, cid, body, scope, force, allowed)

        if action == "reject":
            row = c.execute(
                "SELECT status, evidence FROM skills_lane.skill_gap_candidates "
                "WHERE id=%s FOR UPDATE",
                (cid,),
            ).fetchone()
            if not row:
                return {"found": False}
            if row["status"] not in allowed:
                c.rollback()
                return status_conflict(cid, row["status"], allowed)
            ev = row["evidence"] or []
            ev.append({"session_id": None, "class": "grounded", "signal": "reject"})
            _js, gs, _jw, gw = _rollup(ev)
            c.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='rejected', reject_reason=%s, "
                "rejected_until=now() + interval '30 days', evidence=%s::jsonb, "
                "grounded_sessions=%s, grounded_weight=%s, updated_at=now() WHERE id=%s",
                (reason or "user_rejected", Json(ev), gs, gw, cid),
            )
            c.commit()
            return {"status": "rejected", "id": cid, "reason": reason or "user_rejected"}

    return {"status": "error", "detail": f"unknown action {action!r}"}


# -------------------------------------------------------------------- register


def register(
    mcp,
    db_url: str,
    machine_authorized: Callable[[Request], bool],
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> None:
    """Wire the /skills/* routes onto the FastMCP app. No-op when db_url is empty (dev/stdio).

    ``resolve_trust`` maps a request to its bearer-resolved verdict (the board's and the
    config lane's). Without it every caller is unknown and served nothing, so a wiring slip
    fails closed."""
    if not db_url:
        return

    def _access(request: Request) -> SkillAccess:
        # Called inside the threadpool work: resolution may read Postgres.
        return SkillAccess.of(resolve_trust(request) if resolve_trust else UNKNOWN_SURFACE)

    def _forbidden():
        return err("forbidden for this surface", 403)

    async def _guarded(request: Request, work):
        return await guarded_json(request, machine_authorized, work, label="skill route")

    async def _full_only(request: Request, work):
        """Retire, rescope and the review lane change what every machine is served."""

        def gated(b):
            return work(b) if _access(request).full else _forbidden()

        return await _guarded(request, gated)

    @mcp.custom_route("/skills/list", methods=["POST"])
    async def _list(request: Request) -> JSONResponse:
        def work(b):
            return _list_skills(db_url, b.get("scope", "global"), _access(request), b.get("held"))

        return await _guarded(request, work)

    @mcp.custom_route("/skills/fetch", methods=["POST"])
    async def _fetch(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _fetch_skill(db_url, b["name"], _access(request)))

    @mcp.custom_route("/skills/publish", methods=["POST"])
    async def _publish(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _publish_skill(db_url, b, _access(request)))

    @mcp.custom_route("/skills/retire", methods=["POST"])
    async def _retire(request: Request) -> JSONResponse:
        def work(b):
            return _retire_skill(db_url, b.get("name") if isinstance(b, dict) else None)

        return await _full_only(request, work)

    @mcp.custom_route("/skills/rescope", methods=["POST"])
    async def _rescope(request: Request) -> JSONResponse:
        def work(b):
            b = b if isinstance(b, dict) else {}
            return _rescope_skill(db_url, b.get("name"), b.get("scope"))

        return await _full_only(request, work)

    @mcp.custom_route("/skills/overwrite", methods=["POST"])
    async def _overwrite(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _record_overwrite(db_url, b, _access(request)))

    @mcp.custom_route("/skills/proposals", methods=["POST"])
    async def _proposals(request: Request) -> JSONResponse:
        def work(b):
            cid = b.get("id")
            return _proposal_detail(db_url, int(cid)) if cid else _proposals_list(db_url)

        return await _full_only(request, work)

    @mcp.custom_route("/skills/proposals/act", methods=["POST"])
    async def _act(request: Request) -> JSONResponse:
        def work(b):
            body, scope = b.get("body"), b.get("scope")
            if body is not None and not isinstance(body, str):
                return {"status": "error", "detail": "'body' must be a string"}
            if scope is not None and not isinstance(scope, str):
                return {"status": "error", "detail": "'scope' must be a string"}
            return _proposal_act(
                db_url,
                int(b["id"]),
                b["action"],
                b.get("reason"),
                body=body,
                scope=scope,
                force=bool(b.get("force")),
            )

        return await _full_only(request, work)
