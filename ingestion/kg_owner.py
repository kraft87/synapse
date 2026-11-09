"""The install owner's own entity (schema 066, ``kg_entities.is_self``).

Recall never uses it as a hop seed: the owner's node carries thousands of edges, and spending
the hop leg on it returns the same few facts whatever is asked. Facts about the owner are
still served by the vector and BM25 legs; they just stay on the owner node.

The flag is set by uuid only, through ``synapse-admin owner``. Matching by name would flag a
namesake, and in an install whose facts still sit in two groups it would flag one of two
twins; the owner looks at the candidates (the highest-degree entities per group, which is how
the owner's node shows itself) and picks the row. At most one entity per ``(owner_id,
group_id)`` is flagged: until the groups merge, the owner has a node in each and flags both.

Every function takes a psycopg connection with dict rows, and changes nothing without
``apply=True``. Each returns ``(exit code, message)``: 0 done or nothing to do, 1 refused,
2 schema 066 not applied (:data:`ingestion.surfaces.FACT_AUDIENCE_PROBE`).
"""

from __future__ import annotations

import os
from typing import Any

import psycopg

from ingestion.surfaces import FACT_AUDIENCE_PROBE

#: The owner whose entities these are (the same axis the KG reader and writer use).
OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

NOT_APPLIED = "schema 066 not applied (kg_entities.is_self does not exist yet)"

# Live degree of one entity, as the recall degree gate counts it: live edges from web pages
# are not counted. Two index-driven counts, not an OR, which would scan the edge table.
_DEGREE_SQL = """
SELECT (SELECT count(*) FROM kg_relationships
         WHERE src_uuid = %(u)s AND t_invalid IS NULL AND web_artifact_id IS NULL)
     + (SELECT count(*) FROM kg_relationships
         WHERE tgt_uuid = %(u)s AND t_invalid IS NULL AND web_artifact_id IS NULL) AS n
"""

_ENTITY_SQL = """
SELECT uuid, owner_id, group_id, name, is_self FROM kg_entities WHERE uuid = %s
"""

_FLAGGED_SQL = """
SELECT uuid, owner_id, group_id, name FROM kg_entities WHERE is_self
ORDER BY owner_id, group_id
"""

_FLAGGED_IN_GROUP_SQL = """
SELECT uuid FROM kg_entities
WHERE owner_id = %s AND group_id = %s AND is_self AND uuid <> %s
"""

_CANDIDATES_SQL = """
WITH deg AS (
  SELECT u, count(*) AS d FROM (
    SELECT src_uuid AS u FROM kg_relationships
     WHERE owner_id = %(owner)s AND t_invalid IS NULL AND web_artifact_id IS NULL
    UNION ALL
    SELECT tgt_uuid AS u FROM kg_relationships
     WHERE owner_id = %(owner)s AND t_invalid IS NULL AND web_artifact_id IS NULL
  ) z GROUP BY u
), ranked AS (
  SELECT e.uuid, e.owner_id, e.group_id, e.name, e.is_self, deg.d AS live_degree,
         row_number() OVER (PARTITION BY e.group_id ORDER BY deg.d DESC, e.id) AS rn
    FROM kg_entities e JOIN deg ON deg.u = e.uuid
   WHERE e.owner_id = %(owner)s
)
SELECT uuid, owner_id, group_id, name, is_self, live_degree FROM ranked
WHERE rn <= %(n)s ORDER BY group_id, rn
"""

_SET_SQL = "UPDATE kg_entities SET is_self = true WHERE uuid = %s AND NOT is_self"
_CLEAR_SQL = "UPDATE kg_entities SET is_self = false WHERE uuid = %s AND is_self"


def _degree(conn: Any, uuid: str) -> int:
    return int(conn.execute(_DEGREE_SQL, {"u": uuid}).fetchone()["n"])


def ready(conn: Any, key: str = "") -> bool:
    """True when schema 066 is applied on ``conn``'s database."""
    return bool(FACT_AUDIENCE_PROBE.ready(conn, key))


def flagged(conn: Any) -> list[dict[str, Any]]:
    """The flagged entities, with their live degree: what recall will not hop from."""
    rows = conn.execute(_FLAGGED_SQL).fetchall()
    return [{**r, "live_degree": _degree(conn, r["uuid"])} for r in rows]


def candidates(conn: Any, owner_id: str = OWNER, per_group: int = 5) -> list[dict[str, Any]]:
    """The ``per_group`` highest live-degree entities of each group, flagged or not. The
    owner's own node is the one with the most edges: how to find it without a name."""
    cur = conn.execute(_CANDIDATES_SQL, {"owner": owner_id, "n": per_group})
    return list(cur.fetchall())


def set_owner(conn: Any, uuid: str, *, apply: bool, key: str = "") -> tuple[int, str]:
    """Flag entity ``uuid`` as its group's owner node. Refuses an unknown uuid, and a group
    that already has another flagged entity (clear that one first). A dry run unless
    ``apply``."""
    if not ready(conn, key):
        return 2, NOT_APPLIED
    row = conn.execute(_ENTITY_SQL, (uuid,)).fetchone()
    if row is None:
        return 1, f"no entity {uuid}"
    if row["is_self"]:
        return 0, f"{uuid} is already flagged"
    other = conn.execute(_FLAGGED_IN_GROUP_SQL, (row["owner_id"], row["group_id"], uuid)).fetchone()
    if other is not None:
        return 1, f"{other['uuid']} is already flagged in group {row['group_id']}: clear it first"
    what = f"{uuid} ({row['name']}, group {row['group_id']}, {_degree(conn, uuid)} live edges)"
    if not apply:
        return 0, f"would flag {what}; re-run with --apply"
    try:
        conn.execute(_SET_SQL, (uuid,))
    except psycopg.errors.UniqueViolation:
        return 1, f"another entity was flagged in group {row['group_id']} first: nothing changed"
    return 0, f"flagged {what}"


def clear_owner(conn: Any, uuid: str, *, apply: bool, key: str = "") -> tuple[int, str]:
    """Take the flag off entity ``uuid``. Refuses an unknown uuid. A dry run unless
    ``apply``."""
    if not ready(conn, key):
        return 2, NOT_APPLIED
    row = conn.execute(_ENTITY_SQL, (uuid,)).fetchone()
    if row is None:
        return 1, f"no entity {uuid}"
    if not row["is_self"]:
        return 0, f"{uuid} is not flagged"
    what = f"{uuid} ({row['name']}, group {row['group_id']})"
    if not apply:
        return 0, f"would clear {what}; re-run with --apply"
    conn.execute(_CLEAR_SQL, (uuid,))
    return 0, f"cleared {what}"
