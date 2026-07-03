"""PG-backed FastMCP skills provider.

Serves every active ``skills_lane`` skill as ``skill://`` MCP resources, sourced from
Postgres instead of a directory. ``skill_registry.body`` is the SKILL.md; ``skill_files``
holds bundled files. The wire shape mirrors fastmcp's stock ``SkillProvider`` exactly, so
the client utilities (``fastmcp.utilities.skills.sync_skills`` / ``download_skill``)
round-trip against it:

  * ``skill://{name}/SKILL.md``  -> the main file (description carried on the resource)
  * ``skill://{name}/_manifest`` -> JSON {"skill": name, "files": [{path, size, hash}]},
                                    hash = "sha256:<hex>", INCLUDING the SKILL.md entry
  * ``skill://{name}/{path}``    -> a bundled file, served via a ResourceTemplate

Intended use is DISTRIBUTION: the server hosts skills; each machine runs ``sync_skills``
to materialize them into ``~/.claude/skills``, where they become native Claude Code skills.
PG is the single source of truth (the dream->skills lane writes here).

Note: ``sync_skills`` writes file contents but not the executable bit, so ``is_executable``
is metadata only — it does not survive the client-side materialization.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from typing import Any

import psycopg
from fastmcp.resources.base import Resource, ResourceResult
from fastmcp.resources.template import ResourceTemplate
from fastmcp.server.providers.base import Provider
from fastmcp.utilities.versions import VersionSpec
from pydantic import AnyUrl

_SCHEME = "skill://"
_MAIN = "SKILL.md"
_MANIFEST = "_manifest"


# --------------------------------------------------------------------------- #
# PG access (sync; invoked from async paths via asyncio.to_thread)
# --------------------------------------------------------------------------- #


def _fetch_active(db_url: str) -> list[tuple[str, str]]:
    with psycopg.connect(db_url) as c:
        rows = c.execute(
            "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
            "WHERE status = 'active' ORDER BY name"
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _skill_active(db_url: str, name: str) -> bool:
    with psycopg.connect(db_url) as c:
        row = c.execute(
            "SELECT 1 FROM skills_lane.skill_registry WHERE name = %s AND status = 'active'",
            (name,),
        ).fetchone()
    return row is not None


def _fetch_body(db_url: str, name: str) -> str | None:
    with psycopg.connect(db_url) as c:
        row = c.execute(
            "SELECT body FROM skills_lane.skill_registry WHERE name = %s AND status = 'active'",
            (name,),
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def _fetch_files_meta(db_url: str, name: str) -> list[tuple[str, int, str]]:
    with psycopg.connect(db_url) as c:
        rows = c.execute(
            "SELECT path, size, sha256 FROM skills_lane.skill_files "
            "WHERE skill_name = %s ORDER BY path",
            (name,),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _fetch_file(db_url: str, name: str, path: str) -> bytes | None:
    with psycopg.connect(db_url) as c:
        row = c.execute(
            "SELECT content FROM skills_lane.skill_files WHERE skill_name = %s AND path = %s",
            (name, path),
        ).fetchone()
    return bytes(row[0]) if row else None


def _manifest_json(db_url: str, name: str, body: str | None) -> str:
    """Build the manifest exactly like the stock provider: the SKILL.md is a file entry too."""
    files: list[dict[str, Any]] = []
    if body is not None:
        raw = body.encode("utf-8")
        files.append(
            {"path": _MAIN, "size": len(raw), "hash": "sha256:" + hashlib.sha256(raw).hexdigest()}
        )
    for path, size, sha in _fetch_files_meta(db_url, name):
        files.append({"path": path, "size": size, "hash": "sha256:" + sha})
    return json.dumps({"skill": name, "files": files}, indent=2)


def _as_text_or_bytes(content: bytes) -> str | bytes:
    """Serve as text when it decodes as UTF-8 (scripts, markdown), else as a blob."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content


def _parse(uri: str) -> tuple[str, str] | None:
    if not uri.startswith(_SCHEME):
        return None
    parts = uri[len(_SCHEME) :].split("/", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


# --------------------------------------------------------------------------- #
# Resource / template subclasses (PG-backed read)
# --------------------------------------------------------------------------- #


class _PgSkillResource(Resource):
    """SKILL.md or the synthetic manifest for one skill."""

    db_url: str
    skill_name: str
    is_manifest: bool = False

    async def read(self) -> str | bytes | ResourceResult:
        body = await asyncio.to_thread(_fetch_body, self.db_url, self.skill_name)
        if self.is_manifest:
            return await asyncio.to_thread(_manifest_json, self.db_url, self.skill_name, body)
        if body is None:
            raise FileNotFoundError(f"skill {self.skill_name!r} has no body")
        return body


class _PgSkillFileResource(Resource):
    """A single bundled file within a skill (used by the template's create_resource)."""

    db_url: str
    skill_name: str
    file_path: str

    async def read(self) -> str | bytes | ResourceResult:
        content = await asyncio.to_thread(_fetch_file, self.db_url, self.skill_name, self.file_path)
        if content is None:
            raise FileNotFoundError(f"{self.file_path!r} not in skill {self.skill_name!r}")
        return _as_text_or_bytes(content)


class _PgSkillFileTemplate(ResourceTemplate):
    """Serves any ``skill://{name}/{path}`` bundled file for one skill from PG."""

    db_url: str
    skill_name: str

    async def read(self, arguments: dict[str, Any]) -> str | bytes | ResourceResult:
        path = arguments.get("path", "")
        content = await asyncio.to_thread(_fetch_file, self.db_url, self.skill_name, path)
        if content is None:
            raise FileNotFoundError(f"{path!r} not in skill {self.skill_name!r}")
        return _as_text_or_bytes(content)

    async def _read(  # type: ignore[override]
        self, uri: str, params: dict[str, Any], task_meta: Any = None
    ) -> ResourceResult:
        return self.convert_result(await self.read(arguments=params))

    async def create_resource(self, uri: str, params: dict[str, Any]) -> Resource:
        path = params.get("path", "")
        return _PgSkillFileResource(
            uri=AnyUrl(uri),
            name=f"{self.skill_name}/{path}",
            description=f"File from {self.skill_name} skill",
            mime_type="application/octet-stream",
            db_url=self.db_url,
            skill_name=self.skill_name,
            file_path=path,
        )


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class PgSkillsProvider(Provider):
    """Expose all active ``skills_lane`` skills as ``skill://`` MCP resources from PG."""

    def __init__(self, db_url: str) -> None:
        super().__init__()
        self._db_url = db_url

    def _template(self, name: str) -> _PgSkillFileTemplate:
        return _PgSkillFileTemplate(
            uri_template=f"{_SCHEME}{name}/{{path*}}",
            name=f"{name}_files",
            description=f"Access files within {name}",
            mime_type="application/octet-stream",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            db_url=self._db_url,
            skill_name=name,
        )

    async def _list_resources(self) -> Sequence[Resource]:
        skills = await asyncio.to_thread(_fetch_active, self._db_url)
        out: list[Resource] = []
        for name, desc in skills:
            out.append(
                _PgSkillResource(
                    uri=AnyUrl(f"{_SCHEME}{name}/{_MAIN}"),
                    name=f"{name}/{_MAIN}",
                    description=desc,
                    mime_type="text/markdown",
                    db_url=self._db_url,
                    skill_name=name,
                    is_manifest=False,
                )
            )
            out.append(
                _PgSkillResource(
                    uri=AnyUrl(f"{_SCHEME}{name}/{_MANIFEST}"),
                    name=f"{name}/{_MANIFEST}",
                    description=f"File listing for {name}",
                    mime_type="application/json",
                    db_url=self._db_url,
                    skill_name=name,
                    is_manifest=True,
                )
            )
        return out

    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        parsed = _parse(uri)
        if parsed is None:
            return None
        name, fp = parsed
        if fp not in (_MAIN, _MANIFEST):
            return None  # bundled files resolve via the template
        if not await asyncio.to_thread(_skill_active, self._db_url, name):
            return None
        return _PgSkillResource(
            uri=AnyUrl(uri),
            name=f"{name}/{fp}",
            description=f"File listing for {name}" if fp == _MANIFEST else "",
            mime_type="application/json" if fp == _MANIFEST else "text/markdown",
            db_url=self._db_url,
            skill_name=name,
            is_manifest=fp == _MANIFEST,
        )

    async def _list_resource_templates(self) -> Sequence[ResourceTemplate]:
        skills = await asyncio.to_thread(_fetch_active, self._db_url)
        return [self._template(name) for name, _ in skills]

    async def _get_resource_template(
        self, uri: str, version: VersionSpec | None = None
    ) -> ResourceTemplate | None:
        parsed = _parse(uri)
        if parsed is None:
            return None
        name, fp = parsed
        if fp in (_MAIN, _MANIFEST):
            return None
        if not await asyncio.to_thread(_skill_active, self._db_url, name):
            return None
        return self._template(name)

    def __repr__(self) -> str:
        return "PgSkillsProvider(skills_lane)"
