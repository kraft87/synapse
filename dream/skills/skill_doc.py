# mypy: ignore-errors
"""SKILL.md helpers shared by the nightly drafter and the review accept path.

Pure functions, no DB and no LLM:
  * frontmatter reads (`name`; `description` is skill_measure.skill_description, the same
    reader the lane catalog and the client sync use),
  * body_hash: the pin that ties a drafted proposal to the registry body it was drafted
    against, so accept can refuse a draft whose skill changed underneath it,
  * the description-only splice a retune/widen draft is built from, and the deterministic
    check that nothing but the description moved,
  * sanity checks for a drafted revision (extend/fix/narrow) and a drafted new skill.
"""

from __future__ import annotations

import hashlib
import re

from .skill_measure import skill_description

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_KEY_RE = r"^(\s*){key}:\s*(.*?)\s*$"

# A drafted revision may grow or shrink the skill, but not wholesale: outside these bounds
# the model rewrote (or truncated) the skill instead of applying the change.
MIN_SHRINK = 0.4
MAX_GROWTH = 3
GROWTH_SLACK = 2000
MAX_DESCRIPTION = 1024  # Claude Code's skill description cap


def body_hash(body: str | None) -> str | None:
    """sha256 hex of a SKILL.md body (UTF-8). None for no body (no registry row)."""
    if body is None:
        return None
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def valid_skill_name(name: str | None) -> bool:
    """A name that is safe as a skills-dir folder and a registry key (kebab-case)."""
    return bool(name and _NAME_RE.match(name))


def strip_outer_fence(raw: str) -> str:
    """Drop a ```fence wrapping the WHOLE reply (never an inner ```bash block)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip()


# ------------------------------------------------------------------ frontmatter
def _bounds(lines: list[str]) -> tuple[int, int] | None:
    """(first, end) so lines[first:end] is the frontmatter block, or None without one."""
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return 1, i
    return None


def _field_span(lines: list[str], bounds: tuple[int, int], key: str) -> tuple[int, int] | None:
    """Line span [i, j) of the first `key:` in the frontmatter plus its more-indented
    continuation lines (block scalars, wrapped plain scalars). Same first-match rule as
    skill_description, so the splice edits exactly the field the readers read."""
    first, end = bounds
    pat = re.compile(_KEY_RE.format(key=re.escape(key)))
    for i in range(first, end):
        m = pat.match(lines[i].rstrip("\r\n"))
        if not m:
            continue
        base = len(m.group(1))
        j = i + 1
        while j < end:
            ln = lines[j].rstrip("\r\n")
            if ln.strip() and len(ln) - len(ln.lstrip()) <= base:
                break
            j += 1
        while j > i + 1 and not lines[j - 1].strip():
            j -= 1  # trailing blank lines belong to the next key, not this value
        return i, j
    return None


def frontmatter_name(text: str) -> str:
    """Frontmatter `name` (unquoted), or "" when absent."""
    lines = text.splitlines(keepends=True)
    b = _bounds(lines)
    if not b:
        return ""
    span = _field_span(lines, b, "name")
    if not span:
        return ""
    m = re.match(_KEY_RE.format(key="name"), lines[span[0]].rstrip("\r\n"))
    return m.group(2).strip().strip("\"'") if m else ""


def body_text(text: str) -> str:
    """Everything after the frontmatter block (the whole text when there is none)."""
    lines = text.splitlines(keepends=True)
    b = _bounds(lines)
    return "".join(lines[b[1] + 1 :]) if b else text


# ------------------------------------------------------------------ widen splice
def _render_description(old_value: str, new: str, indent: str) -> list[str] | None:
    """The new description in the SAME YAML style as the old one, or None when the text
    can't be written in that style without changing what the readers parse back."""
    if old_value[:1] in ("|", ">"):  # block scalar: keep the indicator, one content line
        return [f"{indent}description: {old_value}", f"{indent}  {new}"]
    if old_value[:1] == '"':
        return None if ('"' in new or "\\" in new) else [f'{indent}description: "{new}"']
    if old_value[:1] == "'":
        return None if "'" in new else [f"{indent}description: '{new}'"]
    # plain: nothing a reader would strip or reinterpret at either end, no YAML comment
    if new[:1] in "\"'|>&*!%@`#-?:[{" or new[-1:] in "\"'" or " #" in new:
        return None
    return [f"{indent}description: {new}"]


def replace_description(text: str, new: str) -> str | None:
    """`text` with ONLY the frontmatter description replaced by `new` (one line). None when
    there is no description field to replace or `new` can't be written in its style."""
    new = new.strip()
    if not new or "\n" in new or "\r" in new:
        return None
    lines = text.splitlines(keepends=True)
    b = _bounds(lines)
    if not b:
        return None
    span = _field_span(lines, b, "description")
    if not span:
        return None
    i, j = span
    m = re.match(_KEY_RE.format(key="description"), lines[i].rstrip("\r\n"))
    eol = "\r\n" if lines[i].endswith("\r\n") else "\n"
    rendered = _render_description(m.group(2), new, m.group(1))
    if rendered is None:
        return None
    return "".join(lines[:i] + [ln + eol for ln in rendered] + lines[j:])


def only_description_changed(old: str, new: str) -> bool:
    """Deterministic widen check: both texts have a frontmatter description, every line
    outside that field is byte-identical, and the description itself did change."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    ba, bb = _bounds(a), _bounds(b)
    if not ba or not bb:
        return False
    sa, sb = _field_span(a, ba, "description"), _field_span(b, bb, "description")
    if not sa or not sb:
        return False
    if a[: sa[0]] != b[: sb[0]] or a[sa[1] :] != b[sb[1] :]:
        return False
    return skill_description(old) != skill_description(new)


def widen_draft(old: str, new_description: str) -> str | None:
    """The widen draft: `old` with only its description replaced, verified. None when the
    splice can't be made, didn't round-trip, or touched anything else."""
    new_description = (new_description or "").strip()
    if not new_description or len(new_description) > MAX_DESCRIPTION:
        return None
    draft = replace_description(old, new_description)
    if draft is None or not only_description_changed(old, draft):
        return None
    if skill_description(draft) != new_description:
        return None
    return draft


# ------------------------------------------------------------------ sanity checks
def revision_problem(old: str, new: str) -> str | None:
    """Why a drafted extend/fix/narrow revision of `old` is not safe to offer, or None."""
    if not new.strip():
        return "empty draft"
    if new == old:
        return "draft is identical to the current body"
    if frontmatter_name(new) != frontmatter_name(old):
        return f"frontmatter name changed ({frontmatter_name(old)!r} -> {frontmatter_name(new)!r})"
    if not body_text(new).strip():
        return "draft has no body after the frontmatter"
    lo, hi = int(len(old) * MIN_SHRINK), len(old) * MAX_GROWTH + GROWTH_SLACK
    if not lo <= len(new) <= hi:
        return f"draft is {len(new)} chars against {len(old)} (sane range {lo}-{hi})"
    return None


def new_skill_problem(body: str) -> str | None:
    """Why a drafted NEW skill is not a usable SKILL.md, or None."""
    if not valid_skill_name(frontmatter_name(body)):
        return "frontmatter name missing or not kebab-case"
    if not skill_description(body).strip():
        return "frontmatter description missing"
    if not body_text(body).strip():
        return "no body after the frontmatter"
    return None
