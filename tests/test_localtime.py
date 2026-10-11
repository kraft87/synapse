"""ingestion.localtime: the user's zone and the local calendar day of an instant."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from ingestion.localtime import local_date, local_iso, user_tz

EVENING_DT = datetime(2026, 10, 11, 0, 30, 44, 296000, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _toronto(monkeypatch):
    monkeypatch.setenv("SYNAPSE_TIMEZONE", "America/Toronto")


def test_default_zone_is_toronto(monkeypatch):
    monkeypatch.delenv("SYNAPSE_TIMEZONE", raising=False)
    assert str(user_tz()) == "America/Toronto"
    monkeypatch.setenv("SYNAPSE_TIMEZONE", "  ")
    assert str(user_tz()) == "America/Toronto"


def test_unknown_zone_falls_back_to_utc(monkeypatch):
    monkeypatch.setenv("SYNAPSE_TIMEZONE", "Mars/Olympus_Mons")
    assert user_tz() is UTC
    assert local_date(EVENING_DT) == "2026-10-11"


@pytest.mark.parametrize(
    "ts",
    [
        EVENING_DT,
        "2026-10-11T00:30:44.296000+00:00",  # datetime.isoformat()
        "2026-10-11 00:30:44.296+00",  # Postgres text
        "2026-10-11T00:30:44.296Z",  # transcript stamp
        "2026-10-11 00:30:44",  # naive = UTC
    ],
)
def test_local_date_of_an_evening_instant(ts):
    assert local_date(ts) == "2026-10-10"


def test_local_date_dst_boundaries():
    # EST (UTC-5) in January, EDT (UTC-4) in October.
    assert local_date("2026-01-15T04:59:00+00:00") == "2026-01-14"
    assert local_date("2026-01-15T05:00:00+00:00") == "2026-01-15"
    assert local_date("2026-10-11T03:59:00+00:00") == "2026-10-10"
    assert local_date("2026-10-11T04:00:00+00:00") == "2026-10-11"


def test_non_instants_pass_through():
    """A date-precision stamp has no time of day to shift; junk keeps the old slice."""
    assert local_date("2026-10-11") == "2026-10-11"
    assert local_date(date(2026, 10, 11)) == "2026-10-11"
    assert local_date("not a timestamp at all") == "not a time"
    assert local_iso("2026-10-11") == "2026-10-11"


def test_local_iso_is_the_same_instant_on_the_local_clock():
    s = local_iso("2026-10-11T00:30:44.296000+00:00")
    assert s == "2026-10-10T20:30:44.296000-04:00"
    assert datetime.fromisoformat(s) == EVENING_DT
