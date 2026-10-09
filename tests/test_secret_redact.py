"""redact_secrets: credentials in a transcript never become memory.

Fake tokens are assembled at runtime so this public repo never contains a string a
secret scanner would flag.
"""

from __future__ import annotations

import time

import pytest

from ingestion.models import Episode
from ingestion.secret_redact import redact_secrets

ALNUM = "aB3dE5fG7hJ9kL2mN4pQ6rS8tU0vW1xY"  # 32 mixed chars


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        ("sk-" + "svcacct-" + ALNUM * 4, "openai"),
        ("sk-" + "proj-" + ALNUM * 3, "openai"),
        ("sk-" + "ant-api03-" + ALNUM * 3, "anthropic"),
        ("sk-" + "ant-oat01-" + ALNUM * 3, "anthropic"),
        ("sk-" + "or-v1-" + "0a1b2c3d" * 8, "api-key"),
        ("gh" + "p_" + ALNUM + "abcd", "github"),
        ("github" + "_pat_" + ALNUM + "_" + ALNUM, "github"),
        ("xo" + "xb-" + "123456789012-" + ALNUM[:24], "slack"),
        ("AK" + "IA" + "ABCDEFGH23456789", "aws-key-id"),
        ("AI" + "za" + "Sy" + ALNUM + "a", "google-api"),
        ("M" + "TIz" + "NDU2Nzg5MDEyMzQ1Njc4OTAx" + ".Gh1Jk2." + ALNUM[:30], "discord"),
        ("ey" + "JhbGciOiJIUzI1NiJ9" + ".ey" + "JzdWIiOiIxMjM0In0" + ".abcDEF123ghiJKL456", "jwt"),
        ("hf" + "_" + ALNUM, "huggingface"),
        ("pylf" + "_v1_us_" + ALNUM + "Ab12", "logfire"),
        ("syt" + "_bmV1cm9u_" + ALNUM[:20] + "_1aB2cD", "matrix"),
        ("pa" + "-" + ALNUM + "Ab3dE5fG7hJ9k", "voyage"),
        ("fc" + "-" + "0a1b2c3d" * 4, "firecrawl"),
        ("sk" + "_" + "0a1b2c3d4e" * 5, "elevenlabs"),
        ("123456789" + ":AA" + ALNUM + "a", "telegram"),
    ],
)
def test_provider_tokens_are_replaced(token: str, kind: str) -> None:
    out = redact_secrets(f"key is {token} ok")
    assert out == f"key is [REDACTED:{kind}] ok"


def test_private_key_block_including_json_escaped_newlines() -> None:
    begin, end = "-----BEGIN " + "OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"
    raw = f"x {begin}\nb3BlbnNzaC1rZXktdjEAAAAA\nQUFBQUFBQUFB==\n{end} y"
    escaped = f"x {begin}\\nb3BlbnNzaC1rZXktdjEAAAAA\\nQUFB==\\n{end} y"
    assert redact_secrets(raw) == "x [REDACTED:private-key] y"
    assert redact_secrets(escaped) == "x [REDACTED:private-key] y"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MCP_SECRET_PASSWORD=Zq9xLm2Pw8", "MCP_SECRET_PASSWORD=[REDACTED:secret]"),
        (
            "export LOGFIRE_TOKEN='pylf_v1_us_" + ALNUM + "'",
            "export LOGFIRE_TOKEN='[REDACTED:logfire]'",
        ),
        ('{"api_key": "a1b2c3d4e5f6a7b8"}', '{"api_key": "[REDACTED:secret]"}'),
        ("X-Api-Key: 0a1b2c3d4e5f60718293a4b5c6d7e8f9", "X-Api-Key: [REDACTED:secret]"),
        ("password: hunter2hunter2", "password: [REDACTED:secret]"),
        ("Authorization: Bearer " + ALNUM, "Authorization: Bearer [REDACTED:bearer]"),
        # JSON-escaped quotes, as inside a serialized tool result
        ('MATRIX_TOKEN=\\"a1b2c3d4e5f6g7h8\\"', 'MATRIX_TOKEN=\\"[REDACTED:secret]\\"'),
        ('{\\"token\\": \\"AnYx12345678abcd\\"}', '{\\"token\\": \\"[REDACTED:secret]\\"}'),
        (
            "postgresql://synapse:s3cretPass@192.168.0.20:5432/synapse",
            "postgresql://synapse:[REDACTED:password]@192.168.0.20:5432/synapse",
        ),
    ],
)
def test_assignments_keep_the_name_and_lose_the_value(text: str, expected: str) -> None:
    assert redact_secrets(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        'password = os.environ["SYNAPSE_DB_PASSWORD"]',
        "token: str | None = None",
        "SYNAPSE_MACHINE_TOKEN=<that value>",
        "api_key=self.api_key",
        "GITHUB_TOKEN=${{ secrets.GITHUB_TOKEN }}",
        "MCP_TOKEN=$SYNAPSE_TOKEN",
        "max_tokens=8192",
        "tokenizer=cl100k_base",
        "input_tokens: 123456789",
        "SYNAPSE_DB_PASSWORD=CHANGEME",
        "API_KEY=your-api-key-here1",
        "postgresql://user:${PGPASSWORD}@host/db",
        "https://example.com:8080/path?q=1",
        "Bearer $TOKEN",
        "the task-list-of-things-to-do-before-friday-afternoon-review is long",
    ],
)
def test_non_secrets_survive(text: str) -> None:
    assert redact_secrets(text) == text


def test_idempotent() -> None:
    text = (
        "OPENAI_API_KEY=sk-" + "proj-" + ALNUM * 3 + "\n"
        "curl -H 'Authorization: Bearer " + ALNUM + "' https://u:p4ssw0rd@h/x\n"
        "RESTIC_PASSWORD=Abc123Def456"
    )
    once = redact_secrets(text)
    assert once is not None and "sk-" not in once and ALNUM not in once
    assert redact_secrets(once) == once


def test_empty_and_none_pass_through() -> None:
    assert redact_secrets(None) is None
    assert redact_secrets("") == ""


def test_long_dashed_blob_is_linear() -> None:
    # A base64url blob has a word boundary at every '-': an unbounded lazy name prefix
    # would rescan the rest of the blob from each one.
    blob = ("Ab3_" * 15 + "-") * 4000
    start = time.perf_counter()
    redact_secrets(blob)
    assert time.perf_counter() - start < 2.0


def test_episode_redacts_every_text_field() -> None:
    key = "sk-" + "svcacct-" + ALNUM * 3
    ep = Episode(
        session_id="s",
        sequence=1,
        human_turn=f"here is {key}",
        assistant_turn=f"stored {key}",
        content=f"here is {key}\n\nstored {key}",
    )
    for field in (ep.human_turn, ep.assistant_turn, ep.content):
        assert field is not None and key not in field and "[REDACTED:openai]" in field
