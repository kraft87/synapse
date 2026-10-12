"""GET /install — the server hands out its own client.

    curl -sSL https://<your-synapse>/install | python3 -

The response is mcp_server/install_script.py with the client inside: plugin/ and
plugin-codex/ from this server's own tree, zipped and base64-encoded, plus the client
version (plugin.json) and the bundle's sha256. So the client a machine installs is always
the one matching the server it talks to, with no package index in between; `synapse
update` fetches this again.

Unauthenticated, like /health: the bundle is the same public code as the repository and
holds no credential. The server URL written into the script is the origin the request
came in on (honouring X-Forwarded-Proto/-Host from a reverse proxy), which is the URL the
person just typed; `| python3 - <url>` records a different one.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import zipfile
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

#: Where plugin/ and plugin-codex/ live: /app in the image, the checkout in development.
_CLIENT_ROOT = Path(__file__).resolve().parent.parent
_TREES = ("plugin", "plugin-codex")
_TEMPLATE = Path(__file__).resolve().with_name("install_script.py")
_ORIGIN_RE = re.compile(r"^https?://(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(?::\d{1,5})?$")
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # zip mtimes, so equal trees give equal bundles

_cache: dict[str, tuple[str, bytes, str]] = {}


def _client_files(root: Path) -> list[Path]:
    return sorted(
        p
        for tree in _TREES
        for p in (root / tree).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def bundle(root: Path | None = None) -> tuple[str, bytes, str]:
    """(version, zip bytes, sha256) of the client under ``root``, built once per process:
    the tree is part of the image. Raises FileNotFoundError when there is no client."""
    root = root or _CLIENT_ROOT
    cached = _cache.get(str(root))
    if cached:
        return cached
    manifest = root / "plugin" / ".claude-plugin" / "plugin.json"
    version = str(json.loads(manifest.read_text(encoding="utf-8"))["version"])
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+\-]*", version):  # never "." or ".."
        raise ValueError(f"unusable client version {version!r}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in _client_files(root):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), _FIXED_TIME)
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    data = buf.getvalue()
    _cache[str(root)] = (version, data, hashlib.sha256(data).hexdigest())
    return _cache[str(root)]


def request_origin(request: Request) -> str:
    """The origin this request was addressed to, as the client sees it."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    origin = f"{proto or request.url.scheme}://{host or request.headers.get('host', '')}"
    if _ORIGIN_RE.match(origin):
        return origin
    return str(request.base_url).rstrip("/")


def render(server: str, root: Path | None = None) -> str:
    """The installer script for ``server``, the client from ``root`` inside."""
    version, data, sha = bundle(root)
    payload = base64.encodebytes(data).decode("ascii")
    values = {
        "SERVER": json.dumps(server),
        "VERSION": json.dumps(version),
        "SHA256": json.dumps(sha),
        "PAYLOAD": '"""\n' + payload + '"""',
    }
    text = _TEMPLATE.read_text(encoding="utf-8")
    for name, literal in values.items():
        line = f'{name} = ""\n'
        if text.count(line) != 1:
            raise ValueError(f"installer template has no single {name} line")
        text = text.replace(line, f"{name} = {literal}\n")
    return text


def register(mcp) -> None:
    @mcp.custom_route("/install", methods=["GET"])
    async def install(request: Request) -> Response:
        try:
            script = render(request_origin(request))
        except (OSError, ValueError, KeyError) as e:
            logger.warning("install: no client to serve: %s", e)
            return JSONResponse(
                {"status": "error", "detail": "this server has no client to install"},
                status_code=503,
            )
        return Response(
            script,
            media_type="text/x-python; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )
