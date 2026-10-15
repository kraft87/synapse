"""Query normalization, echo detection, and passage-window selection for recall."""

from __future__ import annotations

import difflib
import math
import re

_WS_RE = re.compile(r"\s+")
_ECHO_SHINGLE_WORDS = 5
_ECHO_SHINGLE_MIN_CHARS = 25
_ECHO_SHINGLE_CAP = 2000
_RERANK_WINDOW_SIZE = 1024


def normalize_whitespace(text: str) -> str:
    """Return the lowercase whitespace-normal form used by echo matching."""
    return _WS_RE.sub(" ", text).strip().lower()


def echo_lcs_len(content: str, query: str) -> int:
    """Return the longest exact substring shared by normalized content and query."""
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(query)
    matcher.set_seq1(content)
    return matcher.find_longest_match(0, len(content), 0, len(query)).size


def query_shingles(query: str) -> list[str]:
    """Return bounded, sufficiently specific word shingles for echo pre-filtering."""
    words = query.split(" ")
    shingles: list[str] = []
    for start in range(len(words) - _ECHO_SHINGLE_WORDS + 1):
        shingle = " ".join(words[start : start + _ECHO_SHINGLE_WORDS])
        if len(shingle) >= _ECHO_SHINGLE_MIN_CHARS:
            shingles.append(shingle)
            if len(shingles) >= _ECHO_SHINGLE_CAP:
                break
    return shingles


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./#+-]+", text.lower())


def bm25_scores(docs: list[list[str]], query_tokens: list[str]) -> list[float]:
    """BM25 score of each tokenised doc against the query, with IDF and the average
    length taken over ``docs`` itself. An empty doc scores 0.0."""
    n_docs = len(docs)
    average_length = sum(len(doc) for doc in docs) / n_docs if n_docs else 0.0
    document_frequency: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    scores: list[float] = []
    for doc in docs:
        term_frequency: dict[str, int] = {}
        for token in doc:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        score = 0.0
        for token in set(query_tokens):
            frequency = term_frequency.get(token)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (n_docs - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            score += (
                inverse_frequency
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * len(doc) / average_length))
            )
        scores.append(score)
    return scores


def bm25_best_window(content: str, query_tokens: list[str], cap: int) -> str:
    """Return the cap-sized region with the strongest BM25 match to the query."""
    starts = list(range(0, len(content), _RERANK_WINDOW_SIZE))
    windows = [bm25_tokenize(content[start : start + _RERANK_WINDOW_SIZE]) for start in starts]
    if len(windows) <= 1:
        return content[:cap]
    scores = bm25_scores(windows, query_tokens)
    best_index = max(
        (index for index, window in enumerate(windows) if window), key=scores.__getitem__, default=0
    )
    start = max(0, starts[best_index] + _RERANK_WINDOW_SIZE // 2 - cap // 2)
    return content[start : start + cap]
