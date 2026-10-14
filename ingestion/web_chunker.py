"""
Markdown-aware recursive chunker: web_artifacts.content_markdown, and every
episode recall compacts into passages (mcp_server/recall_passages.py).

Targets ~400-token chunks (1500 chars) with 20% overlap. Splits on the
strongest available natural boundary, falling back to weaker ones if the
chunk is still too big. Preserves fenced code blocks (```...```) as atomic
units — never splits inside one, and never repeats one in an overlap.

Output shape: list[Chunk] where each Chunk knows its character span in the
parent doc (overlap included) and its idx (0-based).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

CHUNK_TARGET = 1500  # ~400 tokens at voyage-4-large; fits cross-encoder 512-token cap
CHUNK_OVERLAP = 300  # 20% overlap

# Strongest first. We try each separator in order; the first one that yields
# pieces all <= CHUNK_TARGET wins. If none does, we recurse with the result
# of the first split that made progress.
SEPARATORS: tuple[str, ...] = (
    "\n## ",
    "\n# ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
)

# Fenced code block — an atomic unit we never split inside.
# Matches ```lang?\n ... \n``` or ~~~lang?\n ... \n~~~.
_CODE_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```|~~~[^\n]*\n.*?\n~~~", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    idx: int
    content: str
    char_start: int
    char_end: int


# Every step works on (start, end) spans of the original text, so a piece's size is
# its real size (a fence counts at full length), offsets are exact by construction,
# and a cut is legal anywhere except strictly inside a fence.


def _fence_at(pos: int, fences: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The fence that a cut at ``pos`` would split, if any."""
    for start, end in fences:
        if start < pos < end:
            return start, end
    return None


def _cuts(text: str, sep: str, lo: int, hi: int, fences: list[tuple[int, int]]) -> list[int]:
    """Positions in (lo, hi) where ``sep`` starts outside a fence. The separator stays
    attached to the *following* piece."""
    out: list[int] = []
    pos = text.find(sep, lo, hi)
    while pos >= 0:
        if pos > lo and _fence_at(pos, fences) is None:
            out.append(pos)
        pos = text.find(sep, pos + len(sep), hi)
    return out


def _hard_cut(lo: int, hi: int, fences: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Last resort: CHUNK_TARGET-sized pieces, stepping around any fence."""
    out: list[tuple[int, int]] = []
    a = lo
    while hi - a > CHUNK_TARGET:
        b = a + CHUNK_TARGET
        if fence := _fence_at(b, fences):
            b = fence[0] if fence[0] > a else fence[1]
        if b >= hi:
            break
        out.append((a, b))
        a = b
    out.append((a, hi))
    return out


def _split_recursive(
    text: str, lo: int, hi: int, fences: list[tuple[int, int]], sep_idx: int = 0
) -> list[tuple[int, int]]:
    """Split text[lo:hi] into spans <= CHUNK_TARGET using the separator ladder."""
    if hi - lo <= CHUNK_TARGET:
        return [(lo, hi)]
    if sep_idx >= len(SEPARATORS):
        return _hard_cut(lo, hi, fences)
    cuts = _cuts(text, SEPARATORS[sep_idx], lo, hi, fences)
    if not cuts:
        # This separator didn't help; try the next weaker one.
        return _split_recursive(text, lo, hi, fences, sep_idx + 1)
    bounds = [lo, *cuts, hi]
    out: list[tuple[int, int]] = []
    for a, b in pairwise(bounds):
        if b - a <= CHUNK_TARGET:
            out.append((a, b))
        else:
            out.extend(_split_recursive(text, a, b, fences, sep_idx + 1))
    return out


def _merge_small_pieces(pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedily combine adjacent spans up to CHUNK_TARGET.

    Spans are kept in order. Each output span is <= CHUNK_TARGET unless a
    single piece already exceeds that (a large fence), which passes through.
    """
    out: list[tuple[int, int]] = []
    for start, end in pieces:
        if out and end - out[-1][0] <= CHUNK_TARGET:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def _overlap_start(
    text: str, prev: tuple[int, int], start: int, overlap: int, fences: list[tuple[int, int]]
) -> int:
    """Where chunk N begins once the last ``overlap`` chars of chunk N-1 are prepended.

    The boundary is snapped right past the first space so the chunk doesn't start
    mid-word, and past any fence in the tail: a fence is served once, in its own chunk.
    """
    tail = max(prev[0], start - overlap)
    ws = text.find(" ", tail, start)
    if ws > tail:
        tail = ws + 1
    for f_start, f_end in fences:
        if f_start < start and f_end > tail:
            tail = max(tail, f_end)
    return tail


def chunk_markdown(content: str) -> list[Chunk]:
    """Chunk a markdown document into ~1500-char pieces with overlap.

    Documents <= CHUNK_TARGET return as a single chunk with no overlap.
    Code fences are preserved as atomic units; chunks split around them.
    ``content[c.char_start:c.char_end] == c.content`` for every chunk.
    """
    if not content:
        return []
    if len(content) <= CHUNK_TARGET:
        return [Chunk(idx=0, content=content, char_start=0, char_end=len(content))]

    fences = [m.span() for m in _CODE_FENCE_RE.finditer(content)]
    spans = _merge_small_pieces(_split_recursive(content, 0, len(content), fences))
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(spans):
        if i and CHUNK_OVERLAP > 0:
            start = _overlap_start(content, spans[i - 1], start, CHUNK_OVERLAP, fences)
        chunks.append(Chunk(idx=i, content=content[start:end], char_start=start, char_end=end))
    return chunks
