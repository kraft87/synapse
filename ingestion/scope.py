"""The personal-scope switch (``SYNAPSE_PERSONAL_SCOPE``).

Synapse splits memory on one axis by default: a ``technical`` graph and a
``personal`` graph in the KG, plus the same ``personal``/``technical`` label on
timeline events. The split buys precision for a user who keeps both lives in one
Synapse, and costs recall for a user who keeps only work in it: a technical
entity that trips the personal regex ("book", "sleep", "interview") lands in a
graph the default ``group_id="technical"`` search never reads, so the fact is
stored and never served.

``SYNAPSE_PERSONAL_SCOPE=0`` turns the axis off. Every write routes to
``technical``, the timeline gate stops spending tokens on a label that is
discarded, and a read that asks for ``personal`` is answered from the one graph
that exists instead of an empty one. Nothing moves rows that were written while
the scope was on.

This is NOT the notes ``audience`` axis (``personal`` / ``work-safe``) or the
plugin's ``SYNAPSE_MACHINE_ROLE``. Those decide which machine is SERVED what;
this decides whether memory is SPLIT at all.

Every reader goes through this module so the switch has one definition. The env
var is read per call (not cached at import) so a process can be tested both
ways and so a module-level constant never freezes the answer.
"""

from __future__ import annotations

import os

# Values that read as "off". Anything else (including unset and empty) is on:
# the split is the default, and a typo must not silently collapse the graphs.
_OFF_VALUES = frozenset({"0", "false", "no", "off"})

#: The single group every write lands in when the personal scope is off.
DEFAULT_GROUP = "technical"

#: The other group, and the timeline domain, that the switch removes.
PERSONAL_GROUP = "personal"


def personal_scope_enabled() -> bool:
    """True when memory is split into technical and personal (the default)."""
    raw = os.environ.get("SYNAPSE_PERSONAL_SCOPE")
    if raw is None:
        return True
    return raw.strip().lower() not in _OFF_VALUES


def active_groups() -> tuple[str, ...]:
    """The KG groups a writer should iterate: both, or technical alone."""
    if personal_scope_enabled():
        return (DEFAULT_GROUP, PERSONAL_GROUP)
    return (DEFAULT_GROUP,)


def coerce_group(group_id: str | None) -> str | None:
    """Map a requested group/domain onto one that exists in this deployment.

    ``None`` stays ``None`` (callers use it to mean "no filter"). With the scope
    off, ``personal`` becomes ``technical`` so a model that asks for the retired
    scope gets answers rather than an empty graph. Any other value passes
    through untouched; validating it is the caller's job.
    """
    if group_id is None or personal_scope_enabled():
        return group_id
    return DEFAULT_GROUP if group_id.strip().lower() == PERSONAL_GROUP else group_id
