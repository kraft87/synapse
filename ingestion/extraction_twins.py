"""Same-batch twin collapse: a fact restated within one extraction batch is written once.

Stage 6 dedups a new fact against edges already in the graph, never against the other
facts of its own batch, so a batch that states one thing twice wrote two edges and recall
served both. Nearly every such twin is fan-out: the model attaches one sentence to each
entity it names ("Alice and Bob both signed off on the release" -> one edge per
person), sometimes with the list reordered or trimmed.

Embedding similarity cannot separate these from genuinely different facts, because it
barely registers a changed number or a swapped name. Measured over every within-batch
pair in prod on 2026-10-09 (335K pairs): of the 821 non-identical pairs at cosine >= 0.95,
about 1 in 20 was a restatement. The rest differed in a PR number, a host, a dose, a
product or a person ("PR #12 was merged" / "PR #14 was merged", "installed via yay" /
"installed via paru", "Carol approved the budget" / "Dave approved the budget"). So the
test is lexical, with cosine only as a floor:

1. Identical identifier signature: the multiset of exact tokens. That is any token that
   is not a plain lowercase word: it carries a digit (number, version, date, dose, count,
   ticket or PR key), is a path, URL, handle or snake/dotted/hyphenated identifier, or
   has a capital letter (a name, acronym or CamelCase identifier). Negations count too.
2. The remaining content words differ on at most ONE side: a reordering or an added
   detail ("runs on alpha" / "runs on alpha, beta and gamma"), never a substitution
   ("contains camphor" / "contains menthol").
3. Cosine >= _TWIN_MIN_SIM.

Endpoints are not compared. All but one of the 363 historical twins this rule finds link
a different entity pair, so requiring the same endpoints would leave the duplicates in
place. The signature does that job instead: a swapped entity name changes it. The
dropped twin's graph link is lost, but its text lives on in the survivor.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ingestion.extraction_policy import _cosine_similarity
from ingestion.models import ExtractedFact

#: Cosine floor for a twin. In the 2026-10-09 measurement every pair passing rules 1-2 at
#: >= 0.92 was a restatement. The few passing below 0.89 were a short fact and a longer one
#: that merely mentions it ("the job runs nightly" / "the job was moved off the shared
#: host because it runs nightly"), which are separate claims.
_TWIN_MIN_SIM = 0.90

_URL = re.compile(r"https?://[^\s\"'<>)\]]+")
_EDGE_PUNCT = "()[]{}<>\"'`,;:!?*\u201c\u201d\u2018"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
# Function words only: dropping or reordering one never changes what a fact asserts.
# Quantifiers (all, any, both), comparatives, conditions (if, when) and ordering words
# (before, after, first) carry meaning and are deliberately absent.
_STOPWORDS = frozenset(
    "a an the and or of to in on at for by with from as via into onto is are was were be "
    "been being it its this that these those which who whom whose so such also because "
    "has have had do does did he she they we you i me him us them his her their our your "
    "my there here".split()
)
_NEGATIONS = frozenset(("not", "no", "never", "none", "nor", "without", "cannot"))
_SUFFIXES = ("ing", "es", "ed", "s", "e")


def _stem(word: str) -> str:
    """Crude suffix strip, so 'relaxes' / 'relaxing' or 'leans' / 'leaned' match."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def fact_tokens(text: str) -> tuple[Counter[str], Counter[str]]:
    """(identifier signature, stemmed content words) of one fact, both as multisets."""
    signature: Counter[str] = Counter()
    words: Counter[str] = Counter()
    for url in _URL.findall(text):
        signature[url.lower().rstrip("./")] += 1
    for raw in _URL.sub(" ", text.replace("\u2019", "'")).split():
        tok = raw.strip(_EDGE_PUNCT).rstrip(".").strip(_EDGE_PUNCT)
        tok = _THOUSANDS.sub("", tok.removesuffix("'s"))
        low = tok.lower()
        if low.endswith("n't"):
            signature["not"] += 1
            tok, low = tok[:-3], low[:-3]
        if not low or low in _STOPWORDS:
            continue
        if low in _NEGATIONS or tok != low or not low.isalpha():
            signature[low] += 1
        else:
            words[_stem(low)] += 1
    return signature, words


def lexical_twins(
    a: tuple[Counter[str], Counter[str]], b: tuple[Counter[str], Counter[str]]
) -> bool:
    """Rules 1-2 on two ``fact_tokens`` results."""
    (sig_a, words_a), (sig_b, words_b) = a, b
    if sig_a != sig_b:
        return False
    # Words missing on both sides is a substitution; on one side, a reorder or added detail.
    return not (words_a - words_b) or not (words_b - words_a)


def _absorb(survivor: ExtractedFact, twins: list[ExtractedFact]) -> ExtractedFact:
    """The survivor speaks for its twins: user-stated if any twin was, ongoing if any was.

    Provenance needs no merge. Stage 7 stamps every edge of a batch with the batch's
    episode ids, so the survivor already carries each twin's sources. mention_count stays
    1, as in reinforce_edges, which counts a restatement only when it brings new episodes.
    """
    update: dict[str, Any] = {}
    if survivor.attribution != "user" and any(t.attribution == "user" for t in twins):
        update["attribution"] = "user"
    if not survivor.ongoing and any(t.ongoing for t in twins):
        update["ongoing"] = True
    return survivor.model_copy(update=update) if update else survivor


def collapse_batch_twins(
    facts: list[ExtractedFact], embeddings: list[list[float]]
) -> tuple[list[ExtractedFact], list[list[float]], int]:
    """Drop the restatements among one batch's facts.

    Returns (facts, embeddings, n_dropped), survivors in their original order. The most
    specific fact of a set survives: most content words (the one the others are a subset
    of), then the longest text, then text and endpoints in sort order, so the outcome never
    depends on the order the model listed them in. Each fact is compared only against
    facts already kept, so "X uses Y for A" and "X uses Y for B" both survive although each
    restates "X uses Y".
    """
    if len(facts) < 2:
        return facts, embeddings, 0
    tokens = [fact_tokens(f.fact) for f in facts]

    def specificity(i: int) -> tuple[Any, ...]:
        f = facts[i]
        return (-tokens[i][1].total(), -len(f.fact), f.fact, f.source, f.target, f.relationship)

    kept: dict[int, list[int]] = {}
    for i in sorted(range(len(facts)), key=specificity):
        twin_of = next(
            (
                k
                for k in kept
                if lexical_twins(tokens[i], tokens[k])
                and _cosine_similarity(embeddings[i], embeddings[k]) >= _TWIN_MIN_SIM
            ),
            None,
        )
        if twin_of is None:
            kept[i] = []
        else:
            kept[twin_of].append(i)
    if len(kept) == len(facts):
        return facts, embeddings, 0
    order = sorted(kept)
    return (
        [_absorb(facts[k], [facts[j] for j in kept[k]]) for k in order],
        [embeddings[k] for k in order],
        len(facts) - len(order),
    )
