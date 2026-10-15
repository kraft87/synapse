"""The UNAUTHENTICATED static routes: the React bundle under web/dist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, Response

from mcp_server.http_helpers import err

# The built bundle lives at <repo>/web/dist. Module-level so tests can monkeypatch it
# to a throwaway dir; the handlers read it as a global at request time (not captured
# into register()'s closure) so the patch takes effect.
_DIST_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"


# Long-cache immutable fonts (hashed filenames from the build); no-cache the rest.
_FONT_EXTS = {".woff2", ".woff", ".ttf", ".otf"}
_CONTENT_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".map": "application/json",
}


def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def _serve_static_file(name: str) -> Response:
    """Serve a top-level bundle file (index.html / app.js). 503 when the bundle is
    absent — the deployment shipped without a built web/dist (contract)."""
    path = _DIST_DIR / name
    if not path.is_file():
        return err("bundle not built", 503)
    media = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    if name == "index.html":
        media = "text/html; charset=utf-8"
    return _no_cache(FileResponse(path, media_type=media))


def register(mcp: Any) -> None:
    """Mount the bundle routes. They carry no memory and ride no gate: the paste-once login
    screen has to load before it has a token."""

    # The server has no other browser-facing root — a bare vhost hit (Caddy proxies
    # the whole port) 404'd. Send humans to the dashboard; machine paths unaffected.
    @mcp.custom_route("/", methods=["GET"])  # type: ignore[untyped-decorator]
    async def root_redirect(request: Request) -> Response:
        return RedirectResponse("/dash", status_code=302)

    @mcp.custom_route("/dash", methods=["GET"])  # type: ignore[untyped-decorator]
    async def dash_index(request: Request) -> Response:
        return _serve_static_file("index.html")

    @mcp.custom_route("/dash/app.js", methods=["GET"])  # type: ignore[untyped-decorator]
    async def dash_appjs(request: Request) -> Response:
        return _serve_static_file("app.js")

    @mcp.custom_route("/dash/assets/{name}", methods=["GET"])  # type: ignore[untyped-decorator]
    async def dash_assets(request: Request) -> Response:
        name = request.path_params["name"]
        assets_dir = _DIST_DIR / "assets"
        if not assets_dir.is_dir():
            return err("not found", 404)
        # Whitelist by exact basename against the directory listing — never join a raw
        # user path. A traversal attempt ("../x", an absolute/encoded path) simply won't
        # be a member of the listed set, so it 404s.
        allowed = {p.name for p in assets_dir.iterdir() if p.is_file()}
        if name not in allowed:
            return err("not found", 404)
        path = assets_dir / name
        media = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        headers = (
            {"Cache-Control": "public, max-age=31536000, immutable"}
            if path.suffix in _FONT_EXTS
            else {"Cache-Control": "no-cache"}
        )
        return FileResponse(path, media_type=media, headers=headers)
