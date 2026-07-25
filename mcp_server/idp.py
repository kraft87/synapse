"""Identity-provider seam for the custom login flows (dashboard redirect + device grant).

FastMCP owns the MCP OAuth leg (GitHubProvider / OIDCProxy, picked in server.py); these
classes cover the two hand-rolled flows, which need the same primitives from any IdP:
an authorize URL, code -> access token, access token -> identity for the allowlist
gate, and the two RFC 8628 device-grant calls. GitHubIdP speaks GitHub's pre-standard
dialect (fixed endpoints, 200-with-error polls, /user for identity); OIDCIdP covers
any OIDC-compliant provider (Authelia, Keycloak, Pocket ID, ...) via the discovery
document + userinfo.

Both flows end in the SAME machine token; the IdP only answers "who is this and are
they allowed" — swapping providers never changes what clients receive.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_UA = "synapse-login/1.0"
_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


async def _post_form(
    url: str, data: dict[str, str], auth: tuple[str, str] | None = None
) -> dict[str, Any]:
    """Form POST -> parsed JSON body. Stubbable seam for tests.

    Deliberately ignores the HTTP status: OAuth device polls answer
    "authorization_pending" as a 400 (RFC 8628 / Authelia) or a 200 (GitHub), and the
    callers branch on the parsed error field either way."""
    kwargs: dict[str, Any] = {"auth": auth} if auth is not None else {}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            url, data=data, headers={"Accept": "application/json", "User-Agent": _UA}, **kwargs
        )
    body: dict[str, Any] = resp.json()
    return body


async def _get_json(url: str, token: str, accept: str = "application/json") -> dict[str, Any]:
    """Bearer GET -> parsed JSON body. Stubbable seam for tests."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": accept, "User-Agent": _UA},
        )
    body: dict[str, Any] = resp.json()
    return body


class GitHubIdP:
    """GitHub OAuth App. Identity = the account's login, lowercased."""

    label = "github"
    device_disabled_hint = 'Turn on "Enable Device Flow" in the GitHub OAuth App settings.'
    _AUTHORIZE = "https://github.com/login/oauth/authorize"
    _TOKEN = "https://github.com/login/oauth/access_token"
    _DEVICE_CODE = "https://github.com/login/device/code"
    _USER = "https://api.github.com/user"
    _SCOPE = "read:user"  # enough to read the login for the allowlist check

    def __init__(self, client_id: str, client_secret: str, allowed: set[str]) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.allowed = allowed

    async def authorize_url(self, redirect_uri: str, state: str) -> str:
        params = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": self._SCOPE,
                "state": state,
                "allow_signup": "false",
            }
        )
        return f"{self._AUTHORIZE}?{params}"

    async def exchange_code(self, code: str, redirect_uri: str) -> str:
        data = await _post_form(
            self._TOKEN,
            {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        return str(data.get("access_token") or "")

    async def fetch_identity(self, access_token: str) -> str:
        data = await _get_json(self._USER, access_token, accept="application/vnd.github+json")
        return str(data.get("login") or "").lower()

    async def device_start(self) -> dict[str, Any]:
        # Device endpoints authenticate by client_id alone (GitHub OAuth-App style).
        return await _post_form(
            self._DEVICE_CODE, {"client_id": self.client_id, "scope": self._SCOPE}
        )

    async def device_poll(self, device_code: str) -> dict[str, Any]:
        return await _post_form(
            self._TOKEN,
            {"client_id": self.client_id, "device_code": device_code, "grant_type": _DEVICE_GRANT},
        )


class OIDCIdP:
    """Any OIDC-compliant IdP, located by its discovery document.

    Discovery is fetched lazily and cached, so the MCP server still boots while the IdP
    is briefly down (self-hosted, they often share a box). Identity = the first
    non-empty of ``identity_claims`` in userinfo (then ``sub``), lowercased — write the
    allowlist to match. Userinfo works with opaque access tokens, so no claims/JWT
    requirements land on this path."""

    label = "oidc"
    device_disabled_hint = "The IdP's discovery document has no device_authorization_endpoint."

    def __init__(
        self,
        config_url: str,
        client_id: str,
        client_secret: str,
        allowed: set[str],
        scope: str = "openid profile email",
        identity_claims: tuple[str, ...] = ("preferred_username", "email"),
    ) -> None:
        self.config_url = config_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.allowed = allowed
        self.scope = scope
        self.identity_claims = identity_claims
        self._config: dict[str, Any] | None = None

    async def _endpoints(self) -> dict[str, Any]:
        if self._config is None:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(self.config_url, headers={"User-Agent": _UA})
                resp.raise_for_status()
                body: dict[str, Any] = resp.json()
                self._config = body
        return self._config

    def _auth(self) -> tuple[str, str]:
        return (self.client_id, self.client_secret)

    async def authorize_url(self, redirect_uri: str, state: str) -> str:
        cfg = await self._endpoints()
        params = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": self.scope,
                "state": state,
            }
        )
        return f"{cfg['authorization_endpoint']}?{params}"

    async def exchange_code(self, code: str, redirect_uri: str) -> str:
        cfg = await self._endpoints()
        data = await _post_form(
            cfg["token_endpoint"],
            {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
            auth=self._auth(),
        )
        return str(data.get("access_token") or "")

    async def fetch_identity(self, access_token: str) -> str:
        cfg = await self._endpoints()
        claims = await _get_json(cfg["userinfo_endpoint"], access_token)
        for key in (*self.identity_claims, "sub"):
            value = claims.get(key)
            if value:
                return str(value).lower()
        return ""

    async def device_start(self) -> dict[str, Any]:
        cfg = await self._endpoints()
        endpoint = cfg.get("device_authorization_endpoint")
        if not endpoint:
            return {"error": "device_flow_disabled"}
        return await _post_form(endpoint, {"scope": self.scope}, auth=self._auth())

    async def device_poll(self, device_code: str) -> dict[str, Any]:
        cfg = await self._endpoints()
        return await _post_form(
            cfg["token_endpoint"],
            {"grant_type": _DEVICE_GRANT, "device_code": device_code},
            auth=self._auth(),
        )
