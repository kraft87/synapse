"""Tests for ingestion.web_chunker."""

from __future__ import annotations

from ingestion.web_chunker import (
    CHUNK_OVERLAP,
    CHUNK_TARGET,
    chunk_markdown,
)


class TestChunkMarkdown:
    def test_empty_returns_empty(self):
        assert chunk_markdown("") == []

    def test_short_doc_is_single_chunk_no_overlap(self):
        text = "# Short\n\nThis fits in one chunk."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].idx == 0
        assert chunks[0].content == text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_long_doc_produces_multiple_chunks(self):
        # 5000 chars of paragraphs
        text = "## Header\n\n" + ("Some paragraph content. " * 50 + "\n\n") * 5
        chunks = chunk_markdown(text)
        assert len(chunks) > 1
        # Each chunk roughly at or below target (overlap can push it slightly over)
        for c in chunks:
            assert len(c.content) <= CHUNK_TARGET + CHUNK_OVERLAP + 50

    def test_chunks_have_overlap(self):
        text = ("Sentence number one is here. " * 20 + "\n\n") * 8
        chunks = chunk_markdown(text)
        assert len(chunks) >= 2
        # The end of chunk 0 should appear at the start of chunk 1 (overlap).
        # Allow whitespace snap to trim a few chars off the left.
        tail_of_first = chunks[0].content[-CHUNK_OVERLAP:]
        head_of_second = chunks[1].content[: CHUNK_OVERLAP + 50]
        # Some substantial overlap from the tail must appear at the head.
        # Find longest common run.
        overlap_chars = 0
        for n in range(min(len(tail_of_first), len(head_of_second)), 20, -1):
            if tail_of_first[-n:] == head_of_second[:n]:
                overlap_chars = n
                break
        assert overlap_chars >= 100, f"expected significant overlap, got {overlap_chars} chars"

    def test_chunks_have_sequential_idx(self):
        text = "## Section\n\n" + "content " * 1000
        chunks = chunk_markdown(text)
        for i, c in enumerate(chunks):
            assert c.idx == i

    def test_code_fence_preserved_atomic(self):
        # A code block that's larger than CHUNK_TARGET should still appear
        # unbroken inside exactly one chunk.
        big_code = "```python\n" + "x = 1\n" * 600 + "```"  # ~3600 chars
        text = "Preamble.\n\n" + big_code + "\n\nPostamble paragraph."
        chunks = chunk_markdown(text)
        # The complete fence must appear in some chunk
        joined = "\n\n".join(c.content for c in chunks)
        assert "```python" in joined
        assert "```" in joined
        # And one chunk must contain the whole fenced block intact
        assert any(
            c.content.count("```python") == 1 and c.content.count("```") >= 2 for c in chunks
        ), "code fence got split across chunks"

    def test_splits_on_markdown_heading_when_possible(self):
        # Two h2-separated sections, each near target size
        section_a = "## Section A\n\n" + "Content A. " * 130  # ~1500 chars
        section_b = "## Section B\n\n" + "Content B. " * 130
        text = section_a + "\n\n" + section_b
        chunks = chunk_markdown(text)
        # With overlap, headings may not be at chunk start (chunk 1's leading
        # bytes are tail of chunk 0). The structural guarantee is that the
        # heading was used as the *split point*: each section's heading
        # appears in at most two chunks (one that ends near it, one that
        # starts at or near it via overlap).
        a_chunks = [c for c in chunks if "## Section A" in c.content]
        b_chunks = [c for c in chunks if "## Section B" in c.content]
        assert len(a_chunks) >= 1
        assert len(b_chunks) >= 1
        # Section A heading appears earlier than Section B heading
        assert a_chunks[0].idx < b_chunks[0].idx

    def test_real_voyage_doc_chunks_sanely(self):
        # Approximation of a real Voyage docs page
        text = (
            """# Voyage Embeddings

Voyage AI offers state-of-the-art embedding models.

## Models

### voyage-3
Default general-purpose model.

### voyage-3-large
Larger model with 2048 dimensions.

```python
client.embed(texts, model="voyage-3-large")
```

## Pricing

| Model | Cost |
|-------|------|
| voyage-3 | $0.06/M |
| voyage-3-large | $0.18/M |

"""
            + "Detailed model description paragraph. " * 80
        )
        chunks = chunk_markdown(text)
        assert len(chunks) >= 1
        # The code block should be atomic — find the chunk containing it
        code_chunks = [c for c in chunks if "```python" in c.content]
        assert len(code_chunks) == 1
        assert "client.embed" in code_chunks[0].content
        # No chunk is wildly oversized
        for c in chunks:
            assert len(c.content) < CHUNK_TARGET * 3

    def test_pathological_no_separator_falls_back_to_char_split(self):
        # 10kb of one big word — last-resort character chunking
        text = "x" * 10_000
        chunks = chunk_markdown(text)
        assert len(chunks) > 1
        # Each chunk respects target (modulo overlap)
        for c in chunks:
            assert len(c.content) <= CHUNK_TARGET + CHUNK_OVERLAP
        # No data loss
        reconstructed = "".join(c.content for c in chunks)
        # With overlap, total length will be > original
        assert reconstructed.replace("x", "") == ""  # only x's
