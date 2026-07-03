"""
Markdown-aware recursive chunker for web_artifacts.content_markdown.

Targets ~400-token chunks (1500 chars) with 20% overlap. Splits on the
strongest available natural boundary, falling back to weaker ones if the
chunk is still too big. Preserves fenced code blocks (```...```) as atomic
units — never splits inside one.

Output shape: list[Chunk] where each Chunk knows its byte offset in the
parent doc, idx (0-based), and whether it ends/starts inside a code block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

# Fenced code block — captured as an atomic unit so we never split inside.
# Matches ```lang?\n ... \n``` or ~~~lang?\n ... \n~~~.
_CODE_FENCE_RE = re.compile(r"```[^\n]*\n.*?\n```|~~~[^\n]*\n.*?\n~~~", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    idx: int
    content: str
    char_start: int
    char_end: int


def _protect_code_fences(text: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with sentinels so they survive splitting.

    Returns (protected_text, blocks). Caller must restore via _restore_fences.
    """
    blocks: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        blocks.append(m.group(0))
        return f"\x00FENCE{len(blocks) - 1}\x00"

    return _CODE_FENCE_RE.sub(_sub, text), blocks


def _restore_fences(text: str, blocks: list[str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        return blocks[int(m.group(1))]

    return re.sub(r"\x00FENCE(\d+)\x00", _sub, text)


def _split_on(text: str, sep: str) -> list[str]:
    """Split keeping the separator attached to the *following* piece."""
    if sep not in text:
        return [text]
    parts = text.split(sep)
    out = [parts[0]]
    for p in parts[1:]:
        out.append(sep + p)
    return [p for p in out if p]


def _split_recursive(text: str, sep_idx: int = 0) -> list[str]:
    """Split text into pieces <= CHUNK_TARGET using the separator ladder."""
    if len(text) <= CHUNK_TARGET:
        return [text]
    if sep_idx >= len(SEPARATORS):
        # Hard cut on character boundary as last resort
        return [text[i : i + CHUNK_TARGET] for i in range(0, len(text), CHUNK_TARGET)]
    pieces = _split_on(text, SEPARATORS[sep_idx])
    if len(pieces) == 1:
        # This separator didn't help; try the next weaker one.
        return _split_recursive(text, sep_idx + 1)
    out: list[str] = []
    for p in pieces:
        if len(p) <= CHUNK_TARGET:
            out.append(p)
        else:
            out.extend(_split_recursive(p, sep_idx + 1))
    return out


def _merge_small_pieces(pieces: list[str]) -> list[str]:
    """Greedily combine small adjacent pieces up to CHUNK_TARGET.

    Pieces are kept in order. Each output chunk is <= CHUNK_TARGET (unless
    a single piece already exceeds that, in which case we pass it through).
    """
    out: list[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
            continue
        if len(buf) + len(p) <= CHUNK_TARGET:
            buf += p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend last `overlap` chars of chunk N-1 to chunk N.

    Skipped for the first chunk. Boundary is snapped to the nearest whitespace
    on the left so we don't start mid-word.
    """
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        take = min(overlap, len(prev))
        tail = prev[-take:]
        # Snap to whitespace on the left to avoid starting mid-word
        ws = tail.find(" ")
        if ws > 0:
            tail = tail[ws + 1 :]
        out.append(tail + chunks[i])
    return out


def chunk_markdown(content: str) -> list[Chunk]:
    """Chunk a markdown document into ~1500-char pieces with overlap.

    Documents <= CHUNK_TARGET return as a single chunk with no overlap.
    Code fences are preserved as atomic units; chunks split around them.
    """
    if not content:
        return []
    if len(content) <= CHUNK_TARGET:
        return [Chunk(idx=0, content=content, char_start=0, char_end=len(content))]

    protected, blocks = _protect_code_fences(content)
    raw_pieces = _split_recursive(protected)
    merged = _merge_small_pieces(raw_pieces)
    with_overlap = _apply_overlap(merged, CHUNK_OVERLAP)
    restored = [_restore_fences(p, blocks) for p in with_overlap]

    # Compute character offsets in the *original* (unprotected) text.
    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(restored):
        # For chunks with prepended overlap, the cursor logic is approximate —
        # the overlap region overlaps the previous chunk by design.
        start = content.find(piece, max(0, cursor - CHUNK_OVERLAP * 2))
        if start < 0:
            start = cursor  # fallback if find fails (overlap snap shifted bytes)
        end = start + len(piece)
        chunks.append(Chunk(idx=i, content=piece, char_start=start, char_end=end))
        cursor = end - CHUNK_OVERLAP
    return chunks
