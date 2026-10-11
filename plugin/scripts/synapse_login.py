#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse login — sign in and give this machine its own Synapse credential. Stdlib only.

No third-party dependencies. Two sign-in flows, both ending in a device token stored in
the plugin config (the zero-dependency ingest/recall hooks read it back):

  * DEVICE (default) — RFC 8628 device flow, ONE approval. Prints a short code; you
    approve it on ANY device (github.com/login/device or your OIDC provider). No
    same-host browser, no loopback redirect, so it is the right flow for servers,
    headless and remote boxes. That single approval is both the sign-in and the
    ENROLLMENT (schema 054): the server reads who approved, checks the allowlist, and
    mints a token belonging to THIS machine, scoped by the install prompt's
    personal/work answer. The shared root token is never fetched.

  * BROWSER (--browser) — the legacy loopback authorization-code flow: discovery ->
    dynamic client registration -> PKCE -> opens a browser on THIS host -> token
    exchange -> issue_machine_token. The OAuth token it gets cannot enroll a machine, so
    this flow is followed by the device-flow enrollment above: two approvals. Only
    works where a browser can open on the same machine.

Usage:
    python synapse_login.py               # device flow (recommended)
    python synapse_login.py --browser      # legacy same-host browser flow
    python synapse_login.py --reenroll     # replace this machine's device token
    SYNAPSE_URL=https://synapse.example.net python synapse_login.py

What a session is served depends on the device token, so a machine that never enrolls is
served nothing.

On a machine that is already enrolled this does nothing: there is nothing to gain from
a new token, and a sign-in that fails part-way must not cost the one it has.
`--reenroll` is the deliberate way to get a new device token; the old enrollment record
is dropped only once the new token has been minted.

Truly headless with no browser on any device? Mint a token from an already-trusted
machine (`/synapse-devices mint "<label>"`) or from the database host
(`scripts/surface_admin.py mint`), and paste it into /plugin > synapse > Synapse token
(SYNAPSE_INGEST_TOKEN in the plugin config) here, not into an environment variable.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MCP_URL = os.environ.get("SYNAPSE_MCP_URL") or config.MCP_URL
BASE = config.BASE_URL
UA = "synapse-login/1.0"
# Cloudflare (and similar) 403 a default urllib User-Agent; always send a real one.
_HDRS = {"User-Agent": UA}
_TIMEOUT = 30


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_HDRS)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={**_HDRS, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())


def _post_form(url: str, fields: dict) -> dict:
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={**_HDRS, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: str | None = None
    state: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CallbackHandler.code = (q.get("code") or [None])[0]
        _CallbackHandler.state = (q.get("state") or [None])[0]
        _CallbackHandler.error = (q.get("error") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        msg = (
            "Synapse login complete — you can close this tab."
            if _CallbackHandler.code
            else f"Synapse login failed: {_CallbackHandler.error or 'no code'}"
        )
        self.wfile.write(f"<html><body><h3>{msg}</h3></body></html>".encode())

    def log_message(self, *_args) -> None:  # silence the default stderr logging
        pass


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _mcp_call_token(access_token: str) -> str:
    """initialize + tools/call issue_machine_token over streamable HTTP (stateless)."""
    auth = {
        **_HDRS,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def rpc(payload: dict) -> list[dict]:
        req = urllib.request.Request(
            MCP_URL, data=json.dumps(payload).encode(), method="POST", headers=auth
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            raw = r.read().decode()
        out: list[dict] = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
        if not out and raw.strip():
            out.append(json.loads(raw))
        return out

    rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "synapse-login", "version": "1.0"},
            },
        }
    )
    res = rpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "issue_machine_token", "arguments": {}},
        }
    )
    for msg in res:
        result = msg.get("result", {})
        sc = result.get("structuredContent")
        if isinstance(sc, dict) and sc.get("token"):
            return sc["token"]
        for block in result.get("content", []) or []:
            text = block.get("text")
            if text:
                try:
                    tok = json.loads(text).get("token")
                    if tok:
                        return tok
                except Exception:
                    continue
    return ""


def _device_login(replace: bool = False) -> int:
    """One device flow: the approval signs you in AND enrolls this machine.

    The enrollment route reads the approving identity itself, so there is no separate
    sign-in to run first and no root token to fetch: the device token it mints is the
    only thing ever written to the slot. A refused, expired or abandoned approval writes
    nothing, so the machine stays exactly as it was.
    """
    import enroll

    state = enroll.enroll(replace=replace, open_browser=True, on_missing_route=_legacy_sign_in)
    if not state:
        return 1
    print("Run /reload-plugins (or restart) to connect the recall/remember MCP server.")
    return 0


def _legacy_sign_in(start: dict) -> dict:
    """A server with no /surfaces/enroll: redeem the SAME device code at /device/token.

    Such a server predates per-device tokens, so the shared root token is all it has to
    give, and it is what that server serves by. The code was never polled at the IdP
    (the 404 came from the route table), so the human's one approval still counts.
    Returns a truthy marker on success, ``{}`` on failure.
    """
    interval = max(1, int(start.get("interval") or 5))
    deadline = time.time() + int(start.get("expires_in") or 900)
    first = True
    while time.time() < deadline:
        if not first:
            time.sleep(interval)
        first = False
        try:
            resp = _post_json(BASE + "/device/token", {"device_code": start["device_code"]})
        except urllib.error.HTTPError as e:
            if e.code >= 500:  # transient upstream hiccup: keep polling
                continue
            try:
                resp = json.loads(e.read())
            except Exception:
                resp = {"error": f"http_{e.code}"}
        except Exception as e:
            print(f"poll error (continuing): {e}", file=sys.stderr)
            continue

        token = resp.get("token")
        if token:
            if not _store_root_token(token):
                return {}
            who = f" as {resp['login']}" if resp.get("login") else ""
            print(
                f"Logged in{who}. This server has no device enrollment, so the shared "
                "token was saved to plugin config."
            )
            return {"legacy_login": resp.get("login") or True}

        err = resp.get("error", "authorization_pending")
        if err == "authorization_pending":
            continue
        if err in ("slow_down", "temporarily_unavailable"):
            # Rate-limited at the IdP (429), not refused: back off and keep polling.
            interval += 5
            continue
        if err == "expired_token":
            print("login failed: code expired before approval. Run login again.", file=sys.stderr)
            return {}
        print(f"login failed: {err} {resp.get('error_description', '')}".rstrip(), file=sys.stderr)
        return {}

    print("login failed: timed out waiting for approval.", file=sys.stderr)
    return {}


def _browser_login() -> int:
    try:
        meta = _get_json(BASE + "/.well-known/oauth-authorization-server")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "This Synapse has no OAuth enabled. Paste a token into "
                "/plugin > synapse > Synapse token instead.",
                file=sys.stderr,
            )
        else:
            print(f"discovery failed: HTTP {e.code}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"discovery failed: {e}", file=sys.stderr)
        return 1

    # Bind an ephemeral loopback port and register that exact redirect URI (RFC 8252).
    server = http.server.HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/callback"

    try:
        reg = _post_json(
            meta["registration_endpoint"],
            {
                "client_name": "synapse-cli",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
    except Exception as e:
        server.server_close()
        print(f"client registration failed: {e}", file=sys.stderr)
        return 1
    client_id = reg["client_id"]

    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_urlsafe(16)
    scopes = " ".join(meta.get("scopes_supported") or ["user"])
    authorize_url = (
        meta["authorization_endpoint"]
        + "?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "scope": scopes,
            }
        )
    )

    print(f"Opening browser to sign in to Synapse ({BASE}) ...", file=sys.stderr)
    print(f"If it doesn't open, visit:\n  {authorize_url}\n", file=sys.stderr)
    try:
        webbrowser.open(authorize_url)
    except Exception:
        pass

    # Serve exactly one request (the OAuth callback), with a wall-clock guard.
    timed_out = threading.Event()
    timer = threading.Timer(300, lambda: (timed_out.set(), server.shutdown()))
    timer.daemon = True
    timer.start()
    server.handle_request()
    timer.cancel()
    server.server_close()

    if timed_out.is_set() or not _CallbackHandler.code:
        print(
            f"login failed: {_CallbackHandler.error or 'timed out waiting for browser sign-in'}",
            file=sys.stderr,
        )
        return 1
    if _CallbackHandler.state != state:
        print("login failed: state mismatch (possible CSRF) — aborting", file=sys.stderr)
        return 1

    try:
        tok_resp = _post_form(
            meta["token_endpoint"],
            {
                "grant_type": "authorization_code",
                "code": _CallbackHandler.code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
    except urllib.error.HTTPError as e:
        print(f"token exchange failed: HTTP {e.code} {e.read().decode()[:200]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"token exchange failed: {e}", file=sys.stderr)
        return 1

    access_token = tok_resp.get("access_token")
    if not access_token:
        print("token exchange returned no access_token", file=sys.stderr)
        return 1

    try:
        token = _mcp_call_token(access_token)
    except Exception as e:
        print(f"could not call issue_machine_token: {e}", file=sys.stderr)
        return 1
    if not token:
        print("login failed: no token returned (is auth enabled on the server?)", file=sys.stderr)
        return 1

    if not _store_root_token(token):
        return 1
    print("Logged in. Token saved to plugin config.")
    _enroll_device()
    print("Run /reload-plugins (or restart) to connect the recall/remember MCP server.")
    return 0


def _store_root_token(token: str) -> bool:
    """Put the root token a sign-in just fetched into the one credential slot.

    Only the browser flow and the legacy-server fallback get here; the default device
    flow never fetches the root token. Only reached on a machine that is not enrolled,
    or on `--reenroll` (main() stops an enrolled machine before any sign-in). The enrollment record is dropped FIRST: a crash
    between the two steps then leaves the old device token with no record, which simply
    re-enrolls, rather than the root token under a record claiming this machine is
    enrolled, which is served nothing and never re-enrolls.
    """
    try:
        config.DEVICE_FILE.unlink(missing_ok=True)
    except OSError as e:
        print(f"login failed: could not clear {config.DEVICE_FILE}: {e}", file=sys.stderr)
        return False
    config.write_user_config("SYNAPSE_INGEST_TOKEN", token)
    return True


def _enroll_device() -> bool:
    """Enroll THIS machine after a browser sign-in: its own device flow, its own token.

    The browser flow's OAuth token proves who you are to /mcp, but /surfaces/enroll
    anchors on a device-flow approval, so this is a second approval. The default device
    login does not come through here: its one approval already enrolls.

    Returns True when this machine now holds a device token.
    """
    try:
        import enroll
    except Exception:
        return False
    if enroll.is_enrolled():
        print("\nThis machine is already enrolled; leaving its device token alone.")
        return True
    return bool(enroll.enroll())


def main() -> int:
    import enroll

    # An enrolled machine already holds what a login would produce. Stop before asking
    # anyone to approve a code: the browser flow would put the root token (no surface:
    # served nothing) over the device token, and a fresh enrollment is what --reenroll
    # is for.
    if enroll.is_enrolled():
        state = config.read_device_state()
        surface = f"{state['surface_id']} ({state.get('trust') or 'unknown trust'})"
        if "--reenroll" not in sys.argv:
            print(
                f"This machine is already enrolled as {surface}; keeping its device token.\n"
                "To replace it with a new one, run `synapse-login --reenroll`."
            )
            return 0
        print(
            f"Re-enrolling: this replaces the device token for {surface}. "
            "Revoke the old device afterwards with /synapse-devices.",
            file=sys.stderr,
        )

    # Device flow is the default — browser-free, works on servers/headless. --browser opts
    # into the legacy same-host loopback flow for setups that prefer it.
    if "--browser" in sys.argv:
        return _browser_login()
    return _device_login(replace="--reenroll" in sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
