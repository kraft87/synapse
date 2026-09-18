"""Per-recall degradation notices and safe backend-error rendering."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any

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
