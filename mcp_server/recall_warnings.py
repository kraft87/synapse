"""Per-recall degradation notices, serve-nothing notices, and safe backend-error rendering."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any

from ingestion.surfaces import SurfaceTrust

_WS_RE = re.compile(r"\s+")
_WARN_SINK: ContextVar[list[str] | None] = ContextVar("synapse_recall_warnings", default=None)
_WARN_LOCK = threading.Lock()

_CONFIG_MARKERS = (
    "unauthorized",
    "forbidden",
    "401",
    "403",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "permission denied",
    "connect",
    "connection",
    "timed out",
    "timeout",
    "refused",
    "not known",
    "name resolution",
    "ssl",
    "certificate",
)
_SECRET_RE = re.compile(r"(?i)\b(api[-_]?key|authorization|token|bearer)\b['\"]?\s*[:=]?\s*\S+")


@contextmanager
def warn_sink(sink: list[str]) -> Iterator[None]:
    """Bind ``sink`` as this recall's warning list, including worker threads."""
    token = _WARN_SINK.set(sink)
    try:
        yield
    finally:
        _WARN_SINK.reset(token)


def warn(message: str) -> None:
    """Append a deduplicated degradation notice, or no-op outside a recall."""
    sink = _WARN_SINK.get()
    if sink is None:
        return
    with _WARN_LOCK:
        if message not in sink:
            sink.append(message)


def submit_with_context(executor: ThreadPoolExecutor, fn: Any, *args: Any) -> Future[Any]:
    """Run a recall leg with the caller's warning sink propagated into its worker."""
    return executor.submit(copy_context().run, fn, *args)


def error_brief(error: BaseException, cap: int = 120) -> str:
    """Return a one-line, credential-redacted backend-error summary."""
    message = _SECRET_RE.sub(r"\1=***", _WS_RE.sub(" ", str(error)).strip()).strip()
    return message[:cap] or type(error).__name__


def config_hint(detail: str, *, backend: str, env_prefix: str) -> str:
    """Return a configuration hint only when an error plausibly names misconfiguration."""
    if not any(marker in detail.lower() for marker in _CONFIG_MARKERS):
        return ""
    if backend == "voyage":
        return " Check VOYAGE_API_KEY."
    return f" Check {env_prefix}_PROVIDER / {env_prefix}_BASE_URL / {env_prefix}_API_KEY."


# ---------------------------------------------------------------------------
# Serve-nothing notices: why an authenticated caller is being served (almost) nothing
#
# Fail-closed serving is silent by construction: an unknown caller, or a restricted one
# with no projects, gets a perfectly healthy 200 with empty buckets. These notices name
# the cause on the response itself, because server-generated text is the one channel
# that reaches every client, including ones too old to explain it themselves. The
# transports attach them only for AUTHENTICATED callers (root or device token, or a
# verified sign-in): an open server has no credential to fix, and a rejected request
# never gets this far.
#
# The text deliberately names no ids: it lands in model context and transcripts, which
# are themselves ingested, so it carries the cause and the fix and nothing else.
# ---------------------------------------------------------------------------

#: The server-derived surface-id prefix of a verified OAuth/OIDC sign-in (the
#: claude.ai connector lane); see mcp_server.caller_trust.
_OAUTH_PREFIX = "oauth:"

NO_CREDENTIAL_NOTICE = (
    "This connection has no device credential, so it is served little or no memory. "
    "Run synapse-login on this machine to enroll it "
    "(server without sign-in: synapse-admin bootstrap)."
)
RESTRICTED_EMPTY_NOTICE = (
    "This device is restricted and has no projects granted, so it is served little or no "
    "memory. Its allowlist is fixed at enrollment: from a full-trust device, mint a new "
    "credential with --projects, then revoke this one."
)
SIGN_IN_NOTICE = (
    "This sign-in has no projects granted, so it is served little or no memory. "
    "From a full-trust device, grant it with PUT /surfaces/oauth:<login> "
    "(trust, allowed_projects)."
)


def serving_notice(trust: SurfaceTrust | None) -> str | None:
    """The notice explaining an empty serve for this verdict, or None when there is none.

    Fires for exactly two verdicts: UNKNOWN (no credential identity, no row, or a
    non-approved row), and a registered restricted surface whose allowlist is empty.
    Full trust, and restricted with any project at all, get nothing: narrower serving
    there is the design, not a fault. The caller decides whether the request was
    authenticated; this only maps a verdict to text.
    """
    if trust is None or not trust.restricted:
        return None
    if trust.known and trust.allowed_projects:
        return None
    if (trust.surface_id or "").startswith(_OAUTH_PREFIX):
        return SIGN_IN_NOTICE
    return RESTRICTED_EMPTY_NOTICE if trust.known else NO_CREDENTIAL_NOTICE


def with_notice(out: Any, notice: str | None) -> Any:
    """Put ``notice`` FIRST in ``out["warnings"]`` (it is the root cause; a degraded leg
    is secondary). Same contract as the degradation list: the key exists only when
    something is in it, entries are deduplicated, and nothing else in ``out`` changes.
    Non-dict results pass through untouched."""
    if not notice or not isinstance(out, dict):
        return out
    out["warnings"] = [notice, *(w for w in out.get("warnings") or () if w != notice)]
    return out
