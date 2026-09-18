"""The ingest boundary must drop throwaway smoke-test workspaces but keep every real
project, including the ones whose names merely contain a scratch word."""

from __future__ import annotations

import pytest

from ingestion.scratch_projects import _ENV_VAR, is_scratch_project, scratch_projects


def test_flags_known_scratch_workspaces():
    # axon-test produced the robot-gardener fiction that reached the personal graph.
    assert is_scratch_project("axon-test")
    assert is_scratch_project("axon_test")
    assert is_scratch_project("tmp")
    assert is_scratch_project("test")
    assert is_scratch_project("scratch")


def test_match_is_case_and_whitespace_insensitive():
    assert is_scratch_project("Axon-Test")
    assert is_scratch_project("  TMP  ")


def test_keeps_real_projects_containing_a_scratch_word():
    # The whole reason matching is exact: a substring rule eats these.
    assert not is_scratch_project("axon")
    assert not is_scratch_project("axon-backend")
    assert not is_scratch_project("test-harness-rewrite")
    assert not is_scratch_project("pytest-tuning")
    assert not is_scratch_project("synapse-latest")
    assert not is_scratch_project("tmpfs-bench")
    assert not is_scratch_project("neuron")


def test_empty_and_none():
    assert not is_scratch_project(None)
    assert not is_scratch_project("")
    assert not is_scratch_project("   ")


def test_env_var_replaces_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_ENV_VAR, "sandbox, Demo-Box")
    assert is_scratch_project("sandbox")
    assert is_scratch_project("demo-box")
    # Replacement, not union — a deployment that uses "tmp" for real work opts out.
    assert not is_scratch_project("tmp")


def test_empty_env_var_disables_the_filter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(_ENV_VAR, "  ")
    assert scratch_projects() == frozenset()
    assert not is_scratch_project("axon-test")
