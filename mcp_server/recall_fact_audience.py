"""RecallFactAudienceMixin: the schema-066 pieces of KG serving.

The fact label (``kg_relationships.audience``) joins the restricted serving rule, and the
install owner's flagged entity (``kg_entities.is_self``) is kept out of the hop seeds. Both
exist only on a database that has applied 066, so each is gated by a probe. A restricted
read needs the probe to say yes: without the label the only rule left is provenance, which
cannot tell a personal fact from a work one once every fact is in one memory, so a probe
that says no or could not tell serves that read nothing (:class:`FactAudienceUnknown`).
Full trust never asks the label question and is unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ingestion.surfaces import FACT_AUDIENCE_PROBE
from mcp_server.kg_pg import scope_predicate

if TYPE_CHECKING:
    from mcp_server.recall_settings import RecallSettings

logger = logging.getLogger(__name__)

# The flagged owner entities (schema 066) are re-read at most this often, so `synapse-admin
# owner set` takes effect on a running server within a minute.
_OWNER_SEEDS_TTL_S = 60.0

#: The columns a restricted statement names. An UndefinedColumn whose primary message names
#: one of these means 066 is gone; any other missing column is not the label's business.
FACT_AUDIENCE_COLUMNS = ("audience", "label_version", "label_model")


class FactAudienceUnknown(Exception):
    """The 066 probe did not say yes (the label is absent, or the probe could not tell), so a
    restricted read has no rule it may serve under. Raised inside a restricted leg and caught
    by its fail-closed handler, which serves nothing."""


class RecallFactAudienceMixin:
    _db_url: str

    if TYPE_CHECKING:

        def _settings(self) -> RecallSettings: ...

    # The owner's flagged entity uuids (schema 066), read at most every _OWNER_SEEDS_TTL_S.
    _owner_seeds: tuple[str, ...] = ()
    _owner_seeds_at: float | None = None

    def _fact_audience_on(self, conn: Any, restricted: bool) -> bool:
        """True when schema 066 is applied on ``conn``'s database: restricted reads then add
        the label rule and every read excludes the flagged owner entity as a hop seed.
        A ``restricted`` read raises :class:`FactAudienceUnknown` unless the probe says
        yes, whether the label is absent or the probe could not tell. Full trust gets False
        for both (nothing is excluded, today's behaviour)."""
        state = FACT_AUDIENCE_PROBE.check(conn, self._db_url)
        if restricted and state is not True:
            raise FactAudienceUnknown("schema 066 is not confirmed applied")
        return state is True

    def _owner_seed_uuids(self, conn: Any) -> list[str]:
        """The install owner's flagged entity uuids (``kg_entities.is_self``, schema 066),
        cached for :data:`_OWNER_SEEDS_TTL_S`. Call it only when 066 is applied. Fails open:
        a lookup error excludes nothing, today's behaviour, and is not cached."""
        at = self._owner_seeds_at
        if at is not None and time.monotonic() - at < _OWNER_SEEDS_TTL_S:
            return list(self._owner_seeds)
        try:
            with conn.transaction():
                rows = conn.execute(
                    "SELECT uuid FROM kg_entities WHERE owner_id = %s AND is_self",
                    (self._settings()._KG_OWNER,),
                ).fetchall()
        except Exception as e:
            logger.debug("owner entity lookup failed: %s", e)
            return []
        self._owner_seeds = tuple(r["uuid"] for r in rows)
        self._owner_seeds_at = time.monotonic()
        return list(self._owner_seeds)

    def _scope_predicates(self, conn: Any, aliases: tuple[str, ...]) -> str:
        """:func:`scope_predicate` for each of ``aliases``, ANDed: the restricted rule over
        both sides of a supersession pair (056, plus 066's label where the database has it).
        Raises :class:`FactAudienceUnknown` unless the probe says 066 is applied."""
        aud = self._fact_audience_on(conn, restricted=True)
        return " AND ".join(scope_predicate(a, fact_audience=aud) for a in aliases)
