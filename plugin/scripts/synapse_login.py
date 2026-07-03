#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse login — fetch this machine's Synapse token, stdlib only.

No third-party dependencies. Two flows, both ending in the machine token stored in the
plugin's credentials file (the zero-dependency ingest/recall hooks read it back as the bearer):

  * DEVICE (default) — RFC 8628 device flow. Prints a short code; you approve at
    github.com/login/device on ANY device. No same-host browser, no loopback redirect.
    The right flow for servers / headless / remote boxes. Needs a server new enough to expose
    /device/code (and the GitHub OAuth App's "Enable Device Flow" turned on).

  * BROWSER (--browser) — the legacy loopback authorization-code flow: discovery -> dynamic
    client registration -> PKCE -> opens a browser on THIS host -> token exchange ->
    issue_machine_token. Only works where a browser can open on the same machine.

Usage:
    python synapse_login.py               # device flow (recommended)
    python synapse_login.py --browser      # legacy same-host browser flow
    SYNAPSE_URL=https://synapse.example.net python synapse_login.py

Truly headless with no second device either? Set SYNAPSE_INGEST_TOKEN directly instead.
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


def _device_login() -> int:
    """RFC 8628 device flow: print a code, approve on any device, poll for the machine token."""
    try:
        start = _post_json(BASE + "/device/code", {})
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "This Synapse has no device-login route (older server). Retry with --browser, "
                "or set SYNAPSE_INGEST_TOKEN directly.",
                file=sys.stderr,
            )
        else:
            print(f"device/code failed: HTTP {e.code} {e.read().decode()[:200]}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"device/code failed: {e}", file=sys.stderr)
        return 1

    if "user_code" not in start:
        err = start.get("error", "unknown")
        print(f"device login unavailable: {err}", file=sys.stderr)
        if err == "device_flow_disabled":
            print(
                "Turn on 'Enable Device Flow' in the GitHub OAuth App settings, then retry.",
                file=sys.stderr,
            )
        return 1

    device_code = start["device_code"]
    user_code = start["user_code"]
    verify = start.get("verification_uri") or "https://github.com/login/device"
    verify_complete = start.get("verification_uri_complete")
    interval = int(start.get("interval") or 5)
    expires_in = int(start.get("expires_in") or 900)

    print("\nTo sign in, open this on ANY device and enter the code:\n", file=sys.stderr)
    print(f"    {verify}", file=sys.stderr)
    print(f"    code:  {user_code}\n", file=sys.stderr)
    if verify_complete:
        print(f"(or open the direct link: {verify_complete})\n", file=sys.stderr)
    try:  # convenience only — the flow works fine without a local browser
        webbrowser.open(verify_complete or verify)
    except Exception:
        pass

    print("Waiting for approval...", file=sys.stderr)
    waited = 0
    while waited < expires_in:
        time.sleep(interval)
        waited += interval
        try:
            resp = _post_json(BASE + "/device/token", {"device_code": device_code})
        except urllib.error.HTTPError as e:
            if e.code >= 500:  # transient upstream hiccup — keep polling
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
            config.write_user_config("SYNAPSE_INGEST_TOKEN", token)
            who = f" as {resp['login']}" if resp.get("login") else ""
            print(f"Logged in{who}. Token saved to plugin config.")
            print("Run /reload-plugins (or restart) to connect the recall/remember MCP server.")
            return 0

        err = resp.get("error", "authorization_pending")
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            interval += 5
            continue
        if err == "access_denied":
            print(
                f"login failed: {resp.get('error_description', 'not authorized for this Synapse')}",
                file=sys.stderr,
            )
            return 1
        if err == "expired_token":
            print("login failed: code expired before approval. Run login again.", file=sys.stderr)
            return 1
        print(f"login failed: {err} {resp.get('error_description', '')}".rstrip(), file=sys.stderr)
        return 1

    print("login failed: timed out waiting for approval.", file=sys.stderr)
    return 1


def _browser_login() -> int:
    try:
        meta = _get_json(BASE + "/.well-known/oauth-authorization-server")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                "This Synapse has no OAuth enabled. Set SYNAPSE_INGEST_TOKEN directly.",
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

    config.write_user_config("SYNAPSE_INGEST_TOKEN", token)
    print("Logged in. Token saved to plugin config.")
    print("Run /reload-plugins (or restart) to connect the recall/remember MCP server.")
    return 0


def main() -> int:
    # Device flow is the default — browser-free, works on servers/headless. --browser opts
    # into the legacy same-host loopback flow for setups that prefer it.
    if "--browser" in sys.argv:
        return _browser_login()
    return _device_login()


if __name__ == "__main__":
    raise SystemExit(main())
