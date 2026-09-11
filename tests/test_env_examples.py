"""Drift guard for examples/env/*.env against .env.example.

The three sample files are complete, copyable `.env`s, one per deployment shape.
They rot the moment a variable is renamed or added in .env.example and nobody
touches examples/, so this test pins the relationship:

* every key a sample assigns must exist in .env.example (as a live line or as a
  commented `#VAR=` one, which is how .env.example ships its optional knobs);
* every key in .env.example's Required block must be assigned in every sample,
  even if only to CHANGEME or to blank ("not needed by this shape");
* local.env keeps the two values that make the local stack actually work;
* no sample carries anything that looks like a real credential.

Stdlib + pytest only: no dotenv parser, because the thing under test IS the file
format the deployment reads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"
SAMPLES_DIR = REPO_ROOT / "examples" / "env"
SAMPLE_NAMES = ("voyage.env", "local.env", "openrouter.env")

# `VAR=value` and its commented forms (`#VAR=`, `#   VAR=`), which .env.example
# uses for every optional knob.
_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
_COMMENTED_ASSIGNMENT = re.compile(r"^#\s*([A-Z][A-Z0-9_]*)=(.*)$")

# Shapes of real credentials that must never be committed: Anthropic keys,
# Voyage keys, Logfire tokens, and any long hex blob (a machine token from
# `openssl rand -hex 32` is 64 hex chars).
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-\S"),
    re.compile(r"\bpa-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"pylf_\S"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
)


def _samples() -> list[Path]:
    return sorted(SAMPLES_DIR.glob("*.env"))


def _assigned(path: Path) -> dict[str, str]:
    """Live `VAR=value` assignments, in file order."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = _ASSIGNMENT.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _known_keys(path: Path) -> set[str]:
    """Assigned keys plus commented-out ones, which count as documented."""
    keys = set(_assigned(path))
    for line in path.read_text().splitlines():
        m = _COMMENTED_ASSIGNMENT.match(line.strip())
        if m:
            keys.add(m.group(1))
    return keys


def _required_keys() -> set[str]:
    """Keys assigned inside .env.example's `# --- Required ---` block."""
    keys: set[str] = set()
    in_block = False
    for line in ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("# ---"):
            in_block = stripped.startswith("# --- Required")
            continue
        if in_block:
            m = _ASSIGNMENT.match(stripped)
            if m:
                keys.add(m.group(1))
    return keys


def test_the_three_documented_samples_exist():
    assert {p.name for p in _samples()} == set(SAMPLE_NAMES)


def test_required_block_is_the_one_we_think_it_is():
    """Guards the parser itself: a renamed header would silently empty the set."""
    assert _required_keys() == {
        "SYNAPSE_DB_PASSWORD",
        "SYNAPSE_DB_URL",
        "SYNAPSE_MACHINE_TOKEN",
        "VOYAGE_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }


@pytest.mark.parametrize("sample", _samples(), ids=lambda p: p.name)
def test_every_sample_key_is_documented_in_env_example(sample: Path):
    unknown = sorted(set(_assigned(sample)) - _known_keys(ENV_EXAMPLE))
    assert unknown == [], (
        f"{sample.name} sets variables .env.example does not document: {unknown}. "
        "Add them to .env.example (commented is fine) or drop them from the sample."
    )


@pytest.mark.parametrize("sample", _samples(), ids=lambda p: p.name)
def test_every_required_key_is_assigned_in_every_sample(sample: Path):
    missing = sorted(_required_keys() - set(_assigned(sample)))
    assert missing == [], (
        f"{sample.name} is not a complete .env: missing {missing}. Every required key "
        "must be present, even if assigned blank because this shape does not use it."
    )


def test_local_sample_keeps_the_two_values_that_make_it_work():
    local = _assigned(SAMPLES_DIR / "local.env")
    # Without the profile, `docker compose up -d` starts no inference container
    # and both retrieval legs point at nothing.
    assert local.get("COMPOSE_PROFILES") == "local-inference"
    # bge-base is 768-wide, and the width is frozen at the database's first boot.
    assert local.get("SYNAPSE_EMBED_DIMS") == "768"


@pytest.mark.parametrize("sample", _samples(), ids=lambda p: p.name)
def test_every_sample_pins_the_recall_floor_for_its_reranker(sample: Path):
    """The abstention floor is a property of the reranker, so the code default is off and
    each preset carries the value calibrated for the reranker it configures. A preset that
    forgets it silently runs without abstention; one that inherits another reranker's
    threshold blanks most of its results (39-56% on the LME A/B set, 2026-09-10)."""
    got = _assigned(sample).get("SYNAPSE_RECALL_FLOOR")
    assert got is not None, f"{sample.name} does not set SYNAPSE_RECALL_FLOOR"
    floor = float(got)
    if _assigned(sample).get("SYNAPSE_RERANK_MODEL", "").startswith("BAAI/"):
        assert floor == 0.0, "no calibrated floor exists for the bundled local reranker"
    else:
        assert 0.0 < floor < 1.0


@pytest.mark.parametrize("sample", [*_samples(), ENV_EXAMPLE], ids=lambda p: p.name)
def test_no_sample_ships_a_real_looking_secret(sample: Path):
    for key, value in _assigned(sample).items():
        for pattern in _SECRET_PATTERNS:
            assert not pattern.search(value), (
                f"{sample.name}: {key} holds something shaped like a real credential "
                f"(matched {pattern.pattern}). Samples ship CHANGEME or blank."
            )


@pytest.mark.parametrize("sample", _samples(), ids=lambda p: p.name)
def test_slashed_model_ids_only_under_the_openai_provider(sample: Path):
    """The mistake this repo already guards in .env.example: an OpenRouter-spelled
    id under claude-code is refused at startup, so a sample must not ship one."""
    assigned = _assigned(sample)
    provider = assigned.get("SYNAPSE_LLM_PROVIDER", "claude-code")
    if provider == "openai":
        return
    # Retrieval model ids are a different namespace (BAAI/bge-base-en-v1.5 is fine).
    llm_model_keys = {k for k in assigned if k.endswith("_MODEL")} - {
        "SYNAPSE_EMBED_MODEL",
        "SYNAPSE_RERANK_MODEL",
    }
    slashed = sorted(k for k in llm_model_keys if "/" in assigned[k])
    assert slashed == [], (
        f"{sample.name}: OpenRouter-spelled model id under SYNAPSE_LLM_PROVIDER={provider!r}: "
        f"{slashed}. The Claude CLI wants its own dashed name and refuses this at startup."
    )
