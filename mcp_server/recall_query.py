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


def bm25_best_window(content: str, query_tokens: list[str], cap: int) -> str:
    """Return the cap-sized region with the strongest BM25 match to the query."""
    starts = list(range(0, len(content), _RERANK_WINDOW_SIZE))
    windows = [bm25_tokenize(content[start : start + _RERANK_WINDOW_SIZE]) for start in starts]
    n_windows = len(windows)
    if n_windows <= 1:
        return content[:cap]
    average_length = sum(len(window) for window in windows) / n_windows
    document_frequency: dict[str, int] = {}
    for window in windows:
        for token in set(window):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    best_index, best_score = 0, -1.0
    for index, window in enumerate(windows):
        if not window:
            continue
        term_frequency: dict[str, int] = {}
        for token in window:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        score = 0.0
        for token in set(query_tokens):
            frequency = term_frequency.get(token)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (n_windows - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            score += (
                inverse_frequency
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * len(window) / average_length))
            )
        if score > best_score:
            best_score, best_index = score, index
    start = max(0, starts[best_index] + _RERANK_WINDOW_SIZE // 2 - cap // 2)
    return content[start : start + cap]
