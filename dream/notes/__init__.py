"""dream→notes curation lane — the notes store's missing outflow.

``remember()`` is the only thing that has ever written a note, and nothing has ever
merged, retired, or re-scoped one afterwards. The result compounds: paraphrase
duplicates, correction notes sitting live beside the notes they corrected, and
project-specific content typed as global feedback, all on a board that is injected
into every session and stays bounded only by silent truncation.

This lane runs inside the nightly dream pass and fixes those three shapes in place,
with no review step: see :mod:`dream.notes.nightly` for the entry point (``run_lane``),
the judge prompts, and the safety model.
"""
