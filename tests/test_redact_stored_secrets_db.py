"""scripts/redact_stored_secrets.py: no rule is left out of the stored-row repair.

The script fetches only the rows its server-side prefilter matches. A redaction rule
with no entry in that prefilter never runs on a stored row that has nothing else in it,
and the report still says the table is clean. The prefilter is checked in Postgres
because Postgres is what evaluates it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ingestion.secret_redact import _TOKEN_RULES, redact_secrets
from tests.test_secret_redact import ALNUM, PROVIDER_TOKENS, WEBHOOK_URLS

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "redact_stored_secrets.py"
_spec = importlib.util.spec_from_file_location("redact_stored_secrets_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

_ASSIGNED = (
    "password passwd passphrase secret token apikey api_key api-key accesskey access_key "
    "access-key secret_key privatekey private_key private-key client_secret credential credentials"
).split()

#: (label, text): for each rule, the least a stored row can hold and still be changed.
SAMPLES: list[tuple[str, str]] = [
    *((f"{kind} #{i}", token) for i, (token, kind) in enumerate(PROVIDER_TOKENS)),
    ("private-key", "-----BEGIN " + "RSA PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n"),
    *((f"webhook {kept.split('/')[0]}", f"{kept}/{lost}") for kept, lost in WEBHOOK_URLS),
    ("bearer", "Bearer " + ALNUM),
    ("basic", "Basic " + ALNUM),
    ("url password", "redis://app:" + "Zq9xLm2Pw8" + "@cache/0"),
    *((f"{name}=", f"{name}=Zq9xLm2Pw8") for name in _ASSIGNED),
]


def test_every_rule_has_a_sample() -> None:
    marked: set[str] = set()
    for _label, text in SAMPLES:
        marked.update(script._MARK.findall(redact_secrets(text)))
    kinds = {kind for kind, _ in _TOKEN_RULES} | {"webhook", "bearer", "password", "secret"}
    assert marked == kinds


def test_prefilter_passes_every_text_a_rule_changes(conn) -> None:
    missed = []
    for label, text in SAMPLES:
        assert redact_secrets(text) != text, label
        row = conn.execute("SELECT %s::text ~* %s", (text, script._PREFILTER)).fetchone()
        if not row[0]:
            missed.append(label)
    assert missed == []


def test_apply_rewrites_the_rows_a_rule_changes_and_no_others(
    conn, db_url, monkeypatch, capsys
) -> None:
    kept, token = WEBHOOK_URLS[0]
    conn.execute("DROP TABLE IF EXISTS redact_probe")
    conn.execute("CREATE TABLE redact_probe (id int PRIMARY KEY, content text)")
    try:
        conn.execute(
            "INSERT INTO redact_probe VALUES (1, %s), (2, 'nothing to redact')",
            (f"posted to {kept}/{token}",),
        )
        monkeypatch.setattr(script, "TARGETS", {"redact_probe": ("content",)})

        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), db_url])
        script.main()
        stored = conn.execute("SELECT content FROM redact_probe WHERE id = 1").fetchone()[0]
        assert token in stored  # a dry run writes nothing

        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), db_url, "--apply"])
        script.main()
        rows = dict(conn.execute("SELECT id, content FROM redact_probe").fetchall())
        assert rows == {1: f"posted to {kept}/[REDACTED:webhook]", 2: "nothing to redact"}

        out = capsys.readouterr().out
        assert token not in out
        assert out.count("1 candidate row(s), 1 to redact {'webhook': 1}") == 2
    finally:
        conn.execute("DROP TABLE IF EXISTS redact_probe")
