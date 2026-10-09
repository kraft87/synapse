"""Redact credentials from conversation text before it becomes memory.

Transcripts carry whatever was on screen: a pasted .env, a curl with an
Authorization header, a key echoed by a tool. Once stored, recall serves it back
verbatim to every later session — the 2026-10 feedback review found an OpenAI
service-account key and MCP_SECRET_PASSWORD coming back in recall results.

``redact_secrets`` runs on the Episode model itself (``ingestion.models``), so every
ingest path (live /ingest, the disk sweep, the backfills, remember) stores, chunks and
extracts the redacted text, and the content-dedup probe compares like with like.
``scripts/redact_stored_secrets.py`` applies the same function to rows already stored.

Two rule shapes:

* Known token formats, recognised by their provider prefix, are replaced wholesale.
  High precision, no context needed.
* Credential assignments (``FOO_TOKEN=...``, ``"password": "..."``, ``Bearer ...``,
  ``scheme://user:pass@host``) keep the name and lose the value, and only when the
  value looks like a credential, so ``password=os.environ[...]``, ``token: str`` and
  ``<your-key>`` placeholders survive.

The ``[REDACTED:<kind>]`` marker keeps the useful memory (a credential was here, and
what kind) without the credential. Redaction is idempotent: no rule matches a marker.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import overload

_MARK = "[REDACTED:{}]"

#: Provider token formats. Order matters only where formats nest: the Anthropic
#: rule comes before the generic ``sk-`` rule so the marker names the provider.
_TOKEN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
            r"(?:[A-Za-z0-9+/=\s]|\\n)*"
            r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----)?"
        ),
    ),
    ("anthropic", re.compile(r"\bsk-ant-[a-z0-9]+-[A-Za-z0-9_-]{20,}")),
    ("openai", re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}")),
    ("api-key", re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{30,}")),
    ("stripe", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("gitlab", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}")),
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("slack", re.compile(r"\bxox[abposre]-[A-Za-z0-9-]{10,}")),
    ("aws-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google-api", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("discord", re.compile(r"\b[MNO][A-Za-z0-9_-]{23,27}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
)

#: The characters a bare credential value is made of. Excludes quotes, shell and
#: template sigils (``$``, ``{``, ``<``), brackets and the backslash, so an env
#: reference, a placeholder, a code expression or a JSON-escaped newline ends the value.
_VALUE = r"[^\s\"'`$<>{}()\[\],;\\]"

_ASSIGNMENT = re.compile(
    r"(?P<name>(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key"
    r"|secret[_-]?key|private[_-]?key|client[_-]?secret|credentials?))"
    r"(?P<sep>[\"']?\s*[:=]\s*[\"']?)"
    rf"(?P<value>{_VALUE}{{8,}})",
    re.IGNORECASE,
)
_BEARER = re.compile(rf"(?P<name>\b(?:bearer|basic))(?P<sep>\s+)(?P<value>{_VALUE}{{16,}})", re.I)
_URL_CREDS = re.compile(
    r"(?P<name>\b[a-z][a-z0-9+.-]{0,30}://[^\s:/@\"'<>]{1,256})(?P<sep>:)(?P<value>[^\s@/\"'<>$]{4,})(?=@)",
    re.IGNORECASE,
)

_PLACEHOLDER = re.compile(
    r"x{3,}|\*{3,}|\.\.\.|redacted|changeme|change_me|placeholder|example|your[-_]|^my[-_]"
    r"|^(?:none|null|true|false|undefined|password|secret|token|pass)$",
    re.IGNORECASE,
)


def _looks_secret(value: str) -> bool:
    """A credential, not a type annotation, attribute path or placeholder.

    Generated secrets mix letters and digits (a 43-char urlsafe token has no digit
    about once in 1,500); identifiers like ``self.api_key`` or ``settings.SECRET``
    have no digit at all."""
    if _PLACEHOLDER.search(value):
        return False
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def _keep_name(kind: str) -> Callable[[re.Match[str]], str]:
    """re.sub callback: keep the name and separator, mark the value if it is a secret."""

    def sub(m: re.Match[str]) -> str:
        if not _looks_secret(m["value"]):
            return m[0]
        return f"{m['name']}{m['sep']}{_MARK.format(kind)}"

    return sub


def _url_sub(m: re.Match[str]) -> str:
    # user:pass@host is unambiguous, so a password without digits still goes;
    # only placeholders and env references (excluded by the value class) survive.
    if _PLACEHOLDER.search(m["value"]):
        return m[0]
    return f"{m['name']}{m['sep']}{_MARK.format('password')}"


@overload
def redact_secrets(text: str) -> str: ...
@overload
def redact_secrets(text: None) -> None: ...
def redact_secrets(text: str | None) -> str | None:
    """Replace credentials in ``text`` with ``[REDACTED:<kind>]`` markers."""
    if not text:
        return text
    for kind, pattern in _TOKEN_RULES:
        text = pattern.sub(_MARK.format(kind), text)
    text = _BEARER.sub(_keep_name("bearer"), text)
    text = _URL_CREDS.sub(_url_sub, text)
    return _ASSIGNMENT.sub(_keep_name("secret"), text)
