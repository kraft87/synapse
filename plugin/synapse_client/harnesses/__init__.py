"""The Harness protocol and the registry of harnesses the client can wire.

A harness adapter lives in its own subpackage (``harnesses/<name>/``), imports only the
core, and is listed by ``registry()``. `synapse login`, `install`, `update` and `status`
iterate the registry; none of them names a harness.
"""

from __future__ import annotations

from typing import Any, Protocol


class Harness(Protocol):
    """One coding harness Synapse can be wired into.

    ``name`` is the identifier the CLI takes (``synapse install <name>``); ``label`` is
    what a person reads.
    """

    name: str
    label: str

    def detect(self) -> bool:
        """Whether the harness is on this machine. Never raises."""
        ...

    def install(self, url: str, *, dry_run: bool = False, **options: Any) -> str:
        """Wire Synapse at ``url`` into the harness and return a one-line report. With
        ``dry_run``, say what would be done and change nothing. Raises RuntimeError when
        nothing could be done."""
        ...

    def update(self) -> str | None:
        """Refresh an existing wiring to this client version; a one-line report, or None
        when the harness is not wired."""
        ...

    def wired(self) -> bool:
        """Whether Synapse is wired into the harness already. Never raises."""
        ...


def registry() -> list[Harness]:
    """Every harness this client knows, in the order the CLI offers them."""
    from .hermes.install import HermesHarness

    return [HermesHarness()]
