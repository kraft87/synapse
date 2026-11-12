"""The Hermes adapter: a memory provider, its message normaliser, and its installer.

``messages`` and ``install`` import only the core and load anywhere; ``provider`` also
imports Hermes's ``agent.memory_provider`` and so loads only inside Hermes. Nothing here
imports ``provider`` eagerly. See docs/client-harnesses.md.
"""
