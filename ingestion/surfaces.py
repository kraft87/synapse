"""Credential-bound surface trust + the audience derivation rule (schema 053/054).

The serving-side half of audience scoping. Every serving path (board, recall,
recall_full_turns, fetch, fetch_session, plain-HTTP /recall) resolves the calling
surface through :func:`resolve_caller` and filters what it serves accordingly; the
write path (remember → reconcile_note) resolves the audience tag through
:func:`derive_audience`.

**A surface is a CREDENTIAL, not a hostname (schema 054).** The first cut of audience
scoping keyed trust on the plugin's self-reported ``SYNAPSE_SURFACE``, accepted under
the one shared machine token — which meant the untrusted machine chose which trust row
applied to it. A device token closes that: the token IS the surface identity, so there
is nothing left to claim. Three kinds of caller resolve here:

1. **Device token** — ``sha256(token)`` matches ``surfaces.token_hash``. The row's
   trust applies, and only when ``status='approved'``. Devices get their token by
   ENROLLING, which requires an OAuth/OIDC identity this deployment's allowlist admits
   (``mcp_server/surface_routes``): the owner standing at the new machine is the
   authority for what that machine is, so the grant lands live with no second step.
2. **``oauth:<login>``** — MCP callers on the OAuth/OIDC lane (the claude.ai connector)
   present a verified identity rather than a device token; ``mcp_server/server``
   derives the id and passes it as ``legacy_surface_id``. That lane is unchanged.
3. **Legacy hostname** — a machine-token caller that still sends a ``surface`` param.
   Accepted for ONE release so the migration window works; spoofable exactly as it
   always was, which is why it is time-boxed and why the client stopped sending it.

Fail-closed is the whole point, so it is concentrated in ONE place: every failure mode
of :func:`resolve_caller` — no credential, no row, a non-approved row, missing table,
unreachable database, a malformed row — returns :data:`UNKNOWN_SURFACE`, which is
``restricted`` with an empty allowlist. There is no code path that turns an error into
``full``. Callers must not "handle" a lookup failure by skipping the filter; they get a
restricted verdict and serve the empty intersection.

``known`` distinguishes "a surfaces row said restricted" from "we could not tell". Both
restrict SERVING identically. They differ on the WRITE side only: rule 2 of the audience
precedence (a remember() from a restricted surface defaults ``work-safe``) fires on a
REGISTERED restricted surface, never on an unknown one — defaulting an unclassified note
to ``work-safe`` because a credential went unrecognised would invert the fail-closed
posture into a leak.

Lookups are uncached on purpose. A cache would serve a stale ``full`` verdict after a
surface is demoted or revoked, which is the one staleness this system cannot afford; the
read is an index hit on a table with a handful of rows.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

AUDIENCES = ("personal", "work-safe")
TRUST_LEVELS = ("full", "restricted")
STATUSES = ("approved", "revoked")

#: Only this status serves anything. Revoking clears the token hash too, so a revoked
#: credential matches no row at all.
APPROVED = "approved"
REVOKED = "revoked"

#: The default audience for an unclassified note — never leaves a trusted host.
DEFAULT_AUDIENCE = "personal"

#: What a restricted surface is allowed to read from the notes store.
RESTRICTED_AUDIENCE = "work-safe"


@dataclass(frozen=True)
class SurfaceTrust:
    """The resolved trust verdict for one calling surface.

    ``allowed_projects`` is meaningful only when :attr:`restricted` — a full-trust
    surface serves everything and never consults it.
    """

    surface_id: str | None = None
    trust: str = "restricted"
    allowed_projects: tuple[str, ...] = ()
    known: bool = False

    @property
    def restricted(self) -> bool:
        """True unless a surfaces row explicitly granted full trust."""
        return self.trust != "full"

    @property
    def project_filter(self) -> list[str] | None:
        """Project allowlist to apply to episode/timeline reads, or None for no filter.

        A restricted surface always gets a LIST — empty for an unknown surface, which
        makes ``project = ANY('{}')`` false for every row (NULL project included). That
        is the intended fail-closed serve: nothing, rather than everything.
        """
        return None if not self.restricted else list(self.allowed_projects)

    @property
    def audience_filter(self) -> str | None:
        """Notes audience to filter on, or None for no filter (full trust)."""
        return None if not self.restricted else RESTRICTED_AUDIENCE


#: The verdict for "no surface id", "no such surface", or any lookup failure.
UNKNOWN_SURFACE = SurfaceTrust()

#: Convenience for callers that already hold a full-trust decision (tests, CLI tools).
FULL_TRUST = SurfaceTrust(trust="full", known=True)


def token_hash(token: str) -> str:
    """The stored form of a device token: sha256 hex. The token itself is never stored.

    A device token is a bearer credential the server hands out once. Hashing means a
    dump of `surfaces` cannot be replayed against the API, and the unique index on the
    hash is what guarantees one credential resolves to exactly one surface.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def new_device_token() -> str:
    """A fresh device token — 32 bytes of urandom, URL-safe (fits a bearer header)."""
    return secrets.token_urlsafe(32)


def _row_to_trust(surface_id: str, row: Any) -> SurfaceTrust:
    """A surfaces row → verdict. A revoked row is UNKNOWN, not "restricted-known": it
    must not even set the write-side ``caller_restricted`` flag, because that flag
    WIDENS what a note is later served to."""
    if str(row.get("status") or "") != APPROVED:
        return UNKNOWN_SURFACE
    trust = str(row["trust"] or "restricted")
    if trust not in TRUST_LEVELS:  # pragma: no cover - CHECK constraint prevents it
        logger.warning("surface %r has unknown trust %r; treating as restricted", surface_id, trust)
        trust = "restricted"
    projects = tuple(str(p) for p in (row["allowed_projects"] or ()))
    return SurfaceTrust(surface_id=surface_id, trust=trust, allowed_projects=projects, known=True)


#: Written out in full rather than interpolated: every SQL string in this module is a
#: literal, so a reader (and a static scanner) can see there is no query construction
#: anywhere near a credential.
_SELECT_BY_TOKEN = (
    "SELECT surface_id, trust, allowed_projects, status FROM surfaces WHERE token_hash = %s"
)
_SELECT_BY_ID = (
    "SELECT surface_id, trust, allowed_projects, status FROM surfaces WHERE surface_id = %s"
)

#: Don't rewrite the row on every single request — one bump per device per window is
#: all an operator "is this laptop still alive" column needs.
_LAST_SEEN_WINDOW = "5 minutes"

_TOUCH_SQL = (
    "UPDATE surfaces SET last_seen_at = now() WHERE token_hash = %s "
    "AND (last_seen_at IS NULL OR last_seen_at < now() - %s::interval)"
)


def _touch_last_seen(conn: Any, th: str) -> None:
    """Best-effort liveness stamp on the device row. Swallows everything.

    Authentication must not depend on a WRITE succeeding: a read-only replica, a full
    disk, or a lock wait would otherwise turn a liveness nicety into a total outage.
    """
    try:
        conn.execute(_TOUCH_SQL, (th, _LAST_SEEN_WINDOW))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("last_seen_at stamp failed: %s", e)


def resolve_caller(
    db_url: str,
    token_hash_hex: str | None = None,
    legacy_surface_id: str | None = None,
) -> SurfaceTrust:
    """Resolve one caller to its trust verdict. NEVER raises, never fails open.

    Exactly one of the two credentials is consulted, in this order:

    * ``token_hash_hex`` — the sha256 of a device token (schema 054). The row it names
      is the caller's surface; nothing the caller *says* can change which row that is.
    * ``legacy_surface_id`` — a self-reported id. Two live users: ``oauth:<login>``
      (derived by the server from a VERIFIED identity, not self-reported at all) and
      the deprecated hostname ``surface`` param, kept for the migration window.

    A root-token caller that sends no ``surface`` param resolves to
    :data:`UNKNOWN_SURFACE` — the pre-054 behaviour for an unidentified machine-token
    call, deliberately unchanged.

    Missing credential, missing row, non-approved row, missing table (a deployment
    behind schema/054), or an unreachable database all yield :data:`UNKNOWN_SURFACE` —
    restricted with an empty allowlist.
    """
    if not db_url:
        return UNKNOWN_SURFACE
    th = (token_hash_hex or "").strip()
    sid = (legacy_surface_id or "").strip()
    if not th and not sid:
        return UNKNOWN_SURFACE
    try:
        conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
        try:
            if th:
                row = conn.execute(_SELECT_BY_TOKEN, (th,)).fetchone()
                if row is not None:
                    _touch_last_seen(conn, th)
            else:
                row = conn.execute(_SELECT_BY_ID, (sid,)).fetchone()
        finally:
            conn.close()
    except Exception as e:
        # Fail CLOSED and say so: an operator staring at an empty board needs the reason.
        logger.warning("surface resolve failed (%s); treating as restricted", e)
        return UNKNOWN_SURFACE
    if row is None:
        # A hostname/oauth id with no row keeps its id on the verdict (it is not secret
        # and it makes an empty board diagnosable). A token that matched nothing keeps
        # NOTHING — echoing a credential-derived id back would be an oracle.
        return SurfaceTrust(surface_id=sid or None)
    return _row_to_trust(str(row["surface_id"]), row)


def lookup_surface(db_url: str, surface_id: str | None) -> SurfaceTrust:
    """Deprecated alias for the id lane of :func:`resolve_caller`.

    Kept because the ``oauth:<login>`` lane and the legacy ``surface`` param both
    resolve by id, and several call sites read better spelled this way.
    """
    return resolve_caller(db_url, legacy_surface_id=surface_id)


# ---------------------------------------------------------------------------
# Write side: audience derivation
# ---------------------------------------------------------------------------


def restricted_project_union(db: Any) -> set[str]:
    """Every project any REGISTERED restricted surface is allowed to read.

    This is the provenance rule behind audience precedence #3: a note filed under a
    project that some work host can already read episodes from is work-safe by
    construction, so the work board keeps receiving new notes without anyone inventing
    a second classification step. Fail-closed (empty set) on any error — the caller then
    falls through to ``personal``.
    """
    try:
        return set(db.restricted_surface_projects())
    except Exception as e:
        logger.warning("restricted-project union failed (%s); deriving audience as personal", e)
        return set()


def derive_audience(
    db: Any,
    *,
    explicit: str | None,
    caller_restricted: bool,
    project: str | None,
) -> str:
    """The audience tag for a note being written, in spec precedence order.

    1. ``explicit`` — an ``audience`` argument on remember() wins outright.
    2. ``caller_restricted`` — a write from a REGISTERED restricted surface defaults
       ``work-safe``, symmetric with what that host is allowed to read (otherwise notes
       written at work vanish from the work board on the next session).
    3. ``project`` in the union of restricted surfaces' allowlists ⇒ ``work-safe``.
    4. Otherwise ``personal``.

    Note that an UNKNOWN surface must not set ``caller_restricted``: unknown restricts
    reads, but it must never widen a write.
    """
    if explicit:
        if explicit not in AUDIENCES:
            raise ValueError(f"invalid audience {explicit!r} — expected one of {AUDIENCES}")
        return explicit
    if caller_restricted:
        return RESTRICTED_AUDIENCE
    if project and project in restricted_project_union(db):
        return RESTRICTED_AUDIENCE
    return DEFAULT_AUDIENCE


# ---------------------------------------------------------------------------
# Enrollment / registration CRUD (the routes in mcp_server/surface_routes.py)
#
# One creation path (mint_surface, aliased enroll_surface) and one destruction path
# (revoke_surface). There is deliberately no "approve" and no pending state: a device
# token is only ever created by a caller who has ALREADY proved they may create one —
# an allowlisted OAuth identity, an approved full-trust device, or shell access to the
# database. A second confirmation step after an owner-authenticated request confirms
# nothing, and a state that serves nothing is a state that silently breaks a laptop.
# ---------------------------------------------------------------------------


class SchemaMissing(Exception):
    """schema/053+054 not applied on this deployment — reported as 503, never as success."""


def _surface_row(r: Any) -> dict[str, Any]:
    """One surfaces row in operator-view shape. The token is NEVER in here — only its
    presence, as ``has_token``: the hash is a credential-equivalent for offline attack
    and the plaintext exists exactly once, at mint time, in the mint response."""
    last = r.get("last_seen_at")
    return {
        "surface_id": r["surface_id"],
        "trust": r["trust"],
        "status": r["status"],
        "label": r.get("label"),
        "allowed_projects": list(r["allowed_projects"] or []),
        "has_token": bool(r.get("token_hash")),
        "last_seen_at": last.isoformat() if last else None,
        "created_at": r["created_at"].isoformat(),
        "updated_at": r["updated_at"].isoformat(),
    }


#: The operator view. Spelled out rather than interpolated, like every other query in
#: this module: nothing here builds SQL from anything at runtime.
_LIST_SQL = (
    "SELECT surface_id, trust, status, label, allowed_projects, token_hash,"
    " last_seen_at, created_at, updated_at"
    " FROM surfaces ORDER BY (status <> 'approved'), surface_id"
)


def list_surfaces(db_url: str) -> list[dict[str, Any]]:
    """Every surface, live ones first. Operator read — "who can see what"."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(_LIST_SQL).fetchall()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    return [_surface_row(r) for r in rows]


def _new_surface_id(kind: str = "dev") -> str:
    """An OPAQUE surface id. Deliberately not the hostname: keying a row on a
    self-reported name is how hostname spoofing gets reintroduced through the back door
    (enroll as 'trusted-laptop', inherit that row's grant). The label carries the
    hostname for humans; the id carries nothing."""
    return f"{kind}-{secrets.token_hex(6)}"


def _insert_surface(
    conn: Any,
    *,
    surface_id: str,
    trust: str,
    label: str | None,
    allowed_projects: list[str],
    th: str | None,
) -> dict[str, Any]:
    row = conn.execute(
        "INSERT INTO surfaces (surface_id, trust, allowed_projects, token_hash, status, label) "
        "VALUES (%s, %s, %s, %s, 'approved', %s) "
        "RETURNING surface_id, trust, status, label, allowed_projects, token_hash, "
        "          last_seen_at, created_at, updated_at",
        (surface_id, trust, allowed_projects, th, label),
    ).fetchone()
    assert row is not None, "INSERT ... RETURNING returned nothing"
    return _surface_row(row)


def _approved_restricted_projects(conn: Any) -> list[str]:
    """Every project an approved restricted surface may already read.

    A second work machine should see the same work projects as the first; making the
    operator re-list them by hand is how a scope silently drifts between devices. Same
    set audience derivation reads for precedence rule 3.
    """
    rows = conn.execute(
        "SELECT DISTINCT unnest(allowed_projects) AS project FROM surfaces "
        "WHERE trust = 'restricted' AND status = 'approved'"
    ).fetchall()
    return sorted(str(r["project"]) for r in rows if r["project"])


def mint_surface(
    db_url: str,
    label: str | None,
    trust: str | None = None,
    allowed_projects: list[str] | None = None,
) -> dict[str, Any]:
    """Create an APPROVED device surface and return its brand-new token.

    Returns ``{"surface": {...}, "token": "<plaintext, shown ONCE>"}``. Only the hash is
    stored, so nothing can hand the token back later — including this function.

    This is the one creation path, used by both callers that are allowed to create a
    credential, and the row lands approved in both cases because the CALLER is already
    the authority:

    * ``POST /surfaces/enroll`` — the machine being enrolled proved an OAuth/OIDC
      identity that this deployment's allowlist admits. That is the owner standing at
      the new machine saying what it is; there is nothing further to confirm.
    * ``POST /surfaces/mint`` — an already-trusted device (or the break-glass CLI)
      pre-creating a token for a headless box that will never run a browser flow.

    ``trust`` defaults to ``restricted`` when unstated. That default is for CLIENTS, not
    for humans: the plugin's install prompt makes the person choose (and defaults to
    personal, the single-user common case), so an enrollment that arrives with no stated
    role came from a client that did not ask — and an unasked question must not resolve
    to full access. A restricted surface with no stated projects inherits the union of
    what other approved restricted surfaces already read; empty when there are none,
    which serves nothing.
    """
    granted = (trust or "").strip().lower() or "restricted"
    if granted not in TRUST_LEVELS:
        raise ValueError(f"invalid trust {granted!r} — expected one of {TRUST_LEVELS}")
    explicit = (
        [str(p).strip() for p in allowed_projects if str(p).strip()]
        if allowed_projects is not None
        else None
    )
    tok = new_device_token()
    conn = psycopg.connect(db_url, row_factory=dict_row)
    try:
        with conn.transaction():
            projects: list[str] = []
            if granted == "restricted":
                projects = explicit if explicit is not None else _approved_restricted_projects(conn)
            surface = _insert_surface(
                conn,
                surface_id=_new_surface_id(),
                trust=granted,
                label=(label or "").strip()[:200] or None,
                allowed_projects=projects,
                th=token_hash(tok),
            )
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    logger.info(
        "surface created: %s trust=%s projects=%s label=%r",
        surface["surface_id"],
        surface["trust"],
        surface["allowed_projects"],
        surface["label"],
    )
    return {"surface": surface, "token": tok}


#: Enrollment and minting are the same database operation; they differ only in who is
#: allowed to ask, which is a routing concern. Named separately so call sites read as
#: what they mean.
enroll_surface = mint_surface


#: Surface-id namespace for a browser session issued by the dashboard login flow.
DASH_SURFACE_PREFIX = "dash:"


def issue_dash_token(db_url: str, identity: str) -> str:
    """Mint the full-trust device token the dashboard login hands to a browser.

    One row per identity (``dash:<login>``), so signing in repeatedly does not grow the
    operator's surface list. The token is REGENERATED on every login: only the hash is
    stored, so an existing row's plaintext cannot be recovered to hand out again. The
    practical consequence is that the newest login wins and older browser sessions for
    the same identity get a 401 — which is also the cheapest "log out everywhere".

    Full trust is correct here and not a shortcut: the identity already cleared the
    IdP's authentication AND this deployment's login allowlist, which is a strictly
    stronger check than any device token can make.
    """
    ident = (identity or "").strip().lower()
    if not ident:
        raise ValueError("identity required")
    tok = new_device_token()
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        conn.execute(
            "INSERT INTO surfaces (surface_id, trust, allowed_projects, token_hash, status, label) "
            "VALUES (%s, 'full', '{}', %s, 'approved', %s) "
            "ON CONFLICT (surface_id) DO UPDATE SET token_hash = EXCLUDED.token_hash, "
            "  trust = 'full', status = 'approved', updated_at = now()",
            (f"{DASH_SURFACE_PREFIX}{ident}", token_hash(tok), f"dashboard ({ident})"),
        )
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    return tok


def upsert_surface(
    db_url: str, surface_id: str, trust: str, allowed_projects: list[str]
) -> dict[str, Any]:
    """Register or re-register one CREDENTIAL-LESS surface by id — the ``oauth:<login>``
    lane, and legacy hostname rows during the migration window.

    Device surfaces never come through here: their id is server-generated and their
    trust is set by :func:`mint_surface` at creation, so a PUT that could name an
    arbitrary id must not be able to promote one. It writes ``status='approved'``
    because an operator PUT is itself the grant for an identity row.

    Deliberately not a PATCH: a partial update of a security allowlist is how a stale
    entry survives a demotion. The caller states the whole intended state every time.
    """
    if trust not in TRUST_LEVELS:
        raise ValueError(f"invalid trust {trust!r} — expected one of {TRUST_LEVELS}")
    projects = [str(p).strip() for p in allowed_projects if str(p).strip()]
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        row = conn.execute(
            "INSERT INTO surfaces (surface_id, trust, allowed_projects, status) "
            "VALUES (%s, %s, %s, 'approved') "
            "ON CONFLICT (surface_id) DO UPDATE SET trust = EXCLUDED.trust, "
            "  allowed_projects = EXCLUDED.allowed_projects, status = 'approved', "
            "  updated_at = now() "
            # A device row must not be reachable by id: that would let a PUT hand an
            # already-issued token full trust without an owner-authenticated enrollment.
            "  WHERE surfaces.token_hash IS NULL "
            "RETURNING surface_id, trust, status, label, allowed_projects, token_hash, "
            "          last_seen_at, created_at, updated_at",
            (surface_id, trust, projects),
        ).fetchone()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"{surface_id!r} is a device surface — use enroll/mint/revoke")
    return _surface_row(row)


def revoke_surface(db_url: str, surface_id: str) -> int:
    """Revoke a surface: status='revoked', token_hash=NULL, trust back to the floor.

    REVOKE, not DELETE. Clearing the hash is what kills the credential — the next
    request carrying that token matches no row and 401s. The row itself stays so the
    operator keeps an audit trail of what was once trusted, and the trust/allowlist
    reset means even a resurrection by hand starts from the fail-closed state.

    Returns the number of rows changed (0 for an unknown id — idempotent, not an error).
    """
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        cur = conn.execute(
            "UPDATE surfaces SET status = 'revoked', token_hash = NULL, "
            "  trust = 'restricted', allowed_projects = '{}', updated_at = now() "
            "WHERE surface_id = %s AND status <> 'revoked'",
            (surface_id,),
        )
        return cur.rowcount
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
        raise SchemaMissing from e
    finally:
        conn.close()


#: Pre-054 name. Revocation replaced deletion (the row is the audit trail), but the
#: DELETE /surfaces/{id} verb and its callers kept their spelling.
delete_surface = revoke_surface
