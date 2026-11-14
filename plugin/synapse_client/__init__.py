"""The Synapse client core: what every harness adapter shares.

Standard library only, relative imports only, and nothing is resolved at import: a
caller builds a ``settings.Settings`` and passes it down. See docs/client-harnesses.md.
"""

__version__ = "0.24.12"
