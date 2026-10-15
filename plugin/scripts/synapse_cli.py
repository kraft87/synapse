#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse — set up and manage this machine's Synapse client. Stdlib only.

    synapse login [--url URL] [--role personal|work] [--skills-sync yes|no]
                  [--token TOKEN] [--reenroll] [--browser] [--no-install]
    synapse config [KEY [VALUE] | KEY --unset]
    synapse install [claude|codex] [--dry-run]
    synapse update
    synapse skill retire NAME
    synapse skill scope NAME global|project:LABEL
    synapse devices|private|import ...     (the synapse-devices/-private/-import tools)
    synapse version

`synapse login` is the one setup step on a machine: it signs in and enrolls (one
approval, see enroll.py), asks this machine's questions once (personal or work; sync
skills here?), saves the answers to ~/.synapse/config.json, then wires every harness it
finds: the Claude Code plugin and Codex's hooks + MCP entry (harness.py). Re-running it
wires a harness installed since, and never signs in again on an enrolled machine
(`--reenroll` does that on purpose).

`synapse config` reads and sets the saved settings; login is not a settings tool.
`synapse install` is the wiring step alone. `synapse update` fetches the client matching
the server's version from <server>/install, then updates the Claude plugin.

The same code runs from the managed client (~/.synapse/bin/synapse, put there by
`curl <server>/install | python3 -`) and from the Claude plugin's bin/ inside sessions.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import harness

synapse_home = config.synapse_home

_ROLES = ("personal", "work")
_KEY_HELP = {
    "url": "the Synapse server's base URL",
    "token": "this machine's credential (normally written by `synapse login`)",
    "role": "personal or work; declared to the server when this machine enrolls",
    "skills_sync": "two-way sync of your skills library at session start (on/off)",
    "config_sync": "mirror CLAUDE.md + rules/*.md to Synapse for the dream lane (on/off)",
    "config_paths": "extra globs under ~/.claude to mirror, comma-separated",
}


def _interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _ask(prompt: str) -> str | None:
    """The answer to ``prompt``, or None when nobody can answer (not a terminal, EOF)."""
    if not _interactive():
        return None
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def _ask_role() -> str | None:
    while True:
        answer = _ask(
            "Is this a personal or a work machine? A work machine is served only work-safe "
            "notes and allowlisted projects. [personal/work] (personal): "
        )
        if answer is None:
            return None
        answer = answer.lower() or "personal"
        if answer in _ROLES:
            return answer
        print("Please answer personal or work.")


def _ask_skills_sync() -> bool | None:
    while True:
        answer = _ask("Sync skills to this machine? [Y/n]: ")
        if answer is None:
            return None
        parsed = synapse_home.parse_bool(answer or "y")
        if parsed is not None:
            return parsed
        print("Please answer y or n.")


def _reload() -> None:
    """Re-read the config after this process changed it: the modules resolve their
    settings (URL, token, role) at import."""
    for name in ("config", "enroll", "synapse_login", "harness"):
        mod = sys.modules.get(name)
        if mod is not None:
            importlib.reload(mod)  # in place: every holder of the module sees the new values


def _configured_url() -> str:
    url = config._cfg("SYNAPSE_URL") or config._cfg("SYNAPSE_INGEST_URL")
    try:
        return harness.normalize_url(url) if url else ""
    except ValueError:
        return ""


def _origin(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url or "")
    return parts.scheme.lower(), parts.netloc.lower()


def _saved_url() -> str:
    """Where the saved config points (config.json, else the Claude plugin's own setting),
    never the environment: the server a token saved without a record was saved for."""
    for url in (
        synapse_home.load().get("url"),
        config._FILE_OPTIONS.get("SYNAPSE_URL"),
        os.environ.get("CLAUDE_PLUGIN_OPTION_SYNAPSE_URL"),
    ):
        try:
            if url:
                return harness.normalize_url(str(url))
        except ValueError:
            continue
    return ""


def _plugin_may_hold_token() -> bool:
    """Whether the Claude plugin's own config holds a token, readable or not: a pre-0.20
    install dialog kept it in settings.json, in ~/.claude/.credentials.json, or (macOS)
    in the keychain, which nothing but Claude Code can read, so there any configured
    plugin may hold one."""
    return bool(
        config._FILE_OPTIONS.get("SYNAPSE_INGEST_TOKEN")
        or os.environ.get("CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN")
        or harness._plugin_secret("SYNAPSE_INGEST_TOKEN")
        or sys.platform == "darwin"
    )


def _credential_origins() -> set:
    """The (scheme, host) origins this machine's saved credentials already belong to;
    empty when it holds none. An enrolled machine's device token goes only where the
    hooks send it (config.device_origins); a pasted token belongs to the URL it was saved
    with; a token in the Claude plugin's config, to the plugin's URL. Pointing anything
    at another origin would hand it a credential it never had: that is a move, and a
    move is a sign-in there."""
    out = set(config.device_origins())
    home = synapse_home.load()
    if not config.read_device_state().get("surface_id") and home.get("token"):
        out.add(_origin(_saved_url()))
    plugin_url = config._FILE_OPTIONS.get("SYNAPSE_URL") or os.environ.get(
        "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL"
    )
    if plugin_url and _plugin_may_hold_token():
        out.add(_origin(str(plugin_url)))
    return {o for o in out if o[1]}


def _held_for_move() -> set:
    """What a new URL must be in to not be a move: an enrolled machine goes by where the
    hooks send its device token (config.device_origins), and nothing else; a guess about
    a token Claude Code may keep for the plugin must not stand in for that, or a
    `--same-server` would never update the record the hooks go by."""
    if config.read_device_state().get("surface_id"):
        # Empty only for a record too old to say, whose token the hooks can't see from
        # here (Claude Code's secure storage): then the wider guess is all there is.
        return set(config.device_origins()) or _credential_origins()
    return _credential_origins()


def _names(origins: set) -> str:
    return " or ".join(sorted(f"{scheme}://{host}" for scheme, host in origins))


def _move_refusal(url: str, held: set) -> str:
    source = " (SYNAPSE_URL in your environment)" if os.environ.get("SYNAPSE_URL") else ""
    return (
        f"this machine's credential is for {_names(held)}, and {url}{source} must not be "
        f"sent it. Another server: `synapse login --url {url} --reenroll` signs in there "
        f"(or `--token` stores a token it minted). The same server at a new address: "
        f"`synapse config url {url} --same-server`"
    )


def _fail(message: str, code: int = 1) -> int:
    print(f"synapse: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- login


#: ``--token`` given with no value: ask for it (hidden) instead of taking it from argv,
#: where any local user can read it in the process list for the length of the login.
_ASK_TOKEN = object()


def _read_token(arg) -> str:
    if arg is _ASK_TOKEN:
        if _interactive():
            return getpass.getpass("Device token (input hidden): ").strip()
        return sys.stdin.readline().strip()
    if arg == "-":
        return sys.stdin.readline().strip()
    return str(arg).strip()


def _confirm(prompt: str) -> bool:
    answer = _ask(prompt)
    return bool(answer) and synapse_home.parse_bool(answer) is True


def cmd_login(args) -> int:
    try:
        adopted = harness.adopt_legacy()
    except OSError as e:
        return _fail(f"could not write {synapse_home.config_file()}: {e}")
    if adopted:
        print(
            f"Copied {', '.join(adopted)} from the Claude plugin's settings into "
            f"{synapse_home.config_file()}."
        )
        _reload()

    had_slot = bool(_static_slot().get("SYNAPSE_INGEST_TOKEN"))
    prior_env = os.environ.get("SYNAPSE_URL")
    url = (
        args.url
        or _configured_url()
        or _ask("Synapse server URL (e.g. https://synapse.example.net): ")
    )
    if not url:
        return _fail("no server URL: pass --url https://<your-synapse>")
    try:
        url = harness.normalize_url(url)
    except ValueError as e:
        return _fail(str(e))
    token = None
    if args.token is not None:
        token = _read_token(args.token)
        if not token:
            return _fail("no token given")

    import enroll

    home = synapse_home.load()
    # Whatever this machine already holds (an enrollment, or a pasted token) belongs to
    # one server; moving to another origin means a new sign-in there, and the new URL is
    # saved only once that server has minted a token.
    held = _held_for_move()
    previous = _saved_url()
    reenroll = args.reenroll
    switching = token is None and bool(held) and _origin(url) not in held
    if switching and not reenroll:
        question = (
            f"This machine's credential is for {_names(held)}. Sign in to {url} instead? [y/N]: "
        )
        if not (args.url and _confirm(question)):
            return _fail(_move_refusal(url, held))
        reenroll = True
    # A machine set up with --token holds a working credential and no record: signing in
    # again would ask for an approval (or, with no sign-in on the server, just fail).
    token_only = not enroll.is_enrolled() and bool(home.get("token"))
    enrolling = token is None and (reenroll or not (enroll.is_enrolled() or token_only))
    # A sign-in elsewhere saves its URL only once it succeeds, credential seen or not (one
    # in the macOS keychain can't be).
    moving = switching or (enrolling and bool(previous) and _origin(previous) != _origin(url))

    changes: dict = {}
    if home.get("url") != url and not moving:
        changes["url"] = url
    if args.role:
        changes["role"] = args.role
    elif enrolling and not config._cfg("SYNAPSE_MACHINE_ROLE"):
        role = _ask_role()
        if role:
            changes["role"] = role
    if args.skills_sync is not None:
        changes["skills_sync"] = args.skills_sync
    elif not config._cfg("SYNAPSE_SKILLS_SYNC"):
        answer = _ask_skills_sync()
        if answer is not None:
            changes["skills_sync"] = answer
    try:
        if changes:
            config.set_home_values(changes)
        if moving or enrolling:
            # The sign-in goes to `url`, whatever SYNAPSE_URL the shell exports, and for a
            # move config.json keeps the old server until the new one has minted a token.
            os.environ["SYNAPSE_URL"] = url
        _reload()

        if token is not None:
            config.clear_device_state()
            config.store_token(token, server=url)
            print(f"Saved the token to {synapse_home.config_file()}.")
        elif enrolling:
            import synapse_login

            argv = (["--reenroll"] if reenroll else []) + (["--browser"] if args.browser else [])
            rc = synapse_login.main(argv)
            if rc != 0:
                if moving:
                    print(f"{synapse_home.config_file()} still points at {previous}.")
                return rc
            if moving:
                config.set_home_values({"url": url})
        elif token_only:
            print(
                f"Using the token already in {synapse_home.config_file()}; this machine never "
                "enrolled by signing in (`synapse login --reenroll` does that)."
            )
        else:
            import synapse_login

            synapse_login.main([])  # says which surface this machine is enrolled as
    except OSError as e:
        return _fail(f"could not write {synapse_home.config_file()}: {e}")
    finally:
        if moving or enrolling:
            if prior_env is None:
                os.environ.pop("SYNAPSE_URL", None)
            else:
                os.environ["SYNAPSE_URL"] = prior_env
            _reload()

    if not synapse_home.load().get("token") and config.read_device_state().get("surface_id"):
        # Enrolled before 0.20 with the token pasted into Claude Code's install dialog,
        # which keeps it where nothing but Claude's hooks can read it (macOS keychain).
        print(
            "This machine's device token is in Claude Code's secure storage, where Codex and "
            "the MCP headers helper can't read it. Run `synapse login --reenroll` to give "
            f"{synapse_home.config_file()} its own copy."
        )
    if args.no_install:
        return 0
    rc = _wire(url, None)
    if had_slot:
        _restore_static_slot(url)
    return rc


def _static_slot() -> dict:
    return config.plugin_block_options()


def _restore_static_slot(url: str) -> None:
    """A pre-0.20 install's plaintext token copy feeds the Claude plugin's static header.
    store_token drops it while the plugin still points at the old server; once the
    wiring has pointed the plugin at ``url``, the copy follows the new token there, and
    only if ``url`` is that token's own server."""
    token = str(synapse_home.load().get("token") or "")
    if not token:
        return
    record = config.read_device_state()
    if record.get("token_sha256") == config.token_sha256(token) and record.get("server"):
        if _origin(str(record["server"])) != _origin(url):
            return
    opts = _static_slot()
    if opts.get("SYNAPSE_INGEST_TOKEN") or _origin(str(opts.get("SYNAPSE_URL") or "")) != _origin(
        url
    ):
        return
    try:
        config.write_user_config("SYNAPSE_INGEST_TOKEN", token)
    except OSError:
        pass


def _wire(url: str, which: str | None, dry_run: bool = False, vouched: bool = False) -> int:
    # A harness opens the MCP connection itself, and Claude Code falls back to the
    # plugin's static token header: never point one at a server this machine's
    # credential doesn't belong to (``vouched``: the user said it is the same server).
    held = _credential_origins()
    if held and not vouched and _origin(url) not in held:
        return _fail(_move_refusal(url, held))
    found = harness.detect()
    targets = [which] if which else [h for h in ("claude", "codex") if found[h]]
    if not targets:
        print(
            "No Claude Code or Codex found on this machine. Install one, then run "
            "`synapse install claude` or `synapse install codex`."
        )
        return 0
    failed = False
    for target in targets:
        try:
            if target == "claude":
                if dry_run:
                    print(f"Claude Code: would install the plugin with SYNAPSE_URL={url}")
                    continue
                print(harness.install_claude(url))
            else:
                print(harness.install_codex(url, dry_run=dry_run))
        except (RuntimeError, ValueError, OSError) as e:
            print(f"{'Claude Code' if target == 'claude' else 'Codex'}: not wired: {e}")
            failed = True
    return 1 if failed else 0


# -------------------------------------------------------------------------- config


def _show(key: str, value) -> str:
    if value is None or value == "":
        return "(unset)"
    if key == "token":
        return "(set; not shown)"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def _elsewhere(key: str) -> str:
    """What a key not set in config.json resolves to instead, and from where."""
    name = synapse_home.KEYS[key]
    if os.environ.get(name):
        source, value = f"env {name}", os.environ[name]
    elif config._FILE_OPTIONS.get(name) not in (None, ""):
        source, value = "the Claude plugin's settings", config._FILE_OPTIONS[name]
    else:
        return "(unset)"
    return f"(set; not shown, from {source})" if key == "token" else f"{value} (from {source})"


def _drop_record_of(token) -> None:
    """Forget the enrollment record when it describes ``token``, the config.json token
    being replaced or removed. A record pinned to a token kept elsewhere (a pre-0.20
    plugin config; a record with no hash only ever describes that one) still describes a
    live credential and stays."""
    pinned = config.read_device_state().get("token_sha256")
    if pinned and token and config.token_sha256(token) == pinned:
        config.clear_device_state()


def _parse_value(key: str, text: str):
    if key == "url":
        return harness.normalize_url(text)
    if key == "role":
        if text.strip().lower() not in _ROLES:
            raise ValueError("role must be personal or work")
        return text.strip().lower()
    if key in synapse_home.BOOL_KEYS:
        parsed = synapse_home.parse_bool(text)
        if parsed is None:
            raise ValueError(f"{key} takes on/off (or yes/no, 1/0)")
        return parsed
    if key == "config_paths":
        return [g for g in re.split(r"[,\s]+", text) if g]
    if not text.strip():
        raise ValueError(f"{key} can't be empty; use --unset")
    return text.strip()


def cmd_config(args) -> int:
    home = synapse_home.load()
    if not args.key:
        print(f"# {synapse_home.config_file()}")
        for key in synapse_home.KEYS:
            shown = _show(key, home.get(key)) if key in home else _elsewhere(key)
            print(f"{key:<13} {shown:<28} {_KEY_HELP[key]}")
        record = config.read_device_state()
        if record.get("surface_id"):
            print(f"{'(enrolled)':<13} {record['surface_id']} ({record.get('trust') or '?'})")
        return 0
    key = args.key
    if key not in synapse_home.KEYS:
        return _fail(f"unknown key {key!r}; one of {', '.join(synapse_home.KEYS)}")
    if args.unset:
        try:
            if key == "token":
                _drop_record_of(home.get("token"))
                # The legacy plaintext copy would otherwise go on being sent.
                harness.drop_options(["SYNAPSE_INGEST_TOKEN"])
            config.set_home_values({key: None})
        except OSError as e:
            return _fail(f"could not write {synapse_home.config_file()}: {e}")
        print(f"{key} unset")
        if key == "token":
            print(
                "A token entered through /plugin and kept in Claude Code's secure storage is "
                "not touched; clear it there."
            )
        elif _elsewhere(key) != "(unset)":
            print(f"Still in effect: {_elsewhere(key)}")
        return 0
    if args.value is None:
        print(_show(key, home.get(key)) if key in home else _elsewhere(key))
        return 0
    try:
        value = _parse_value(key, args.value)
    except ValueError as e:
        return _fail(str(e))
    if key == "url":
        # As login does: an old install's answers and record (with its token's hash)
        # come over first, so the decision and any stamp below are about the real record.
        try:
            adopted = harness.adopt_legacy()
        except OSError as e:
            return _fail(f"could not write {synapse_home.config_file()}: {e}")
        if adopted:
            print(
                f"Copied {', '.join(adopted)} from the Claude plugin's settings into "
                f"{synapse_home.config_file()}."
            )
            _reload()
    held = _held_for_move() if key == "url" else set()
    moving = bool(held) and _origin(value) not in held
    if moving and not args.same_server:
        return _fail(_move_refusal(value, (held | _credential_origins()) - {_origin(value)}))
    try:
        if moving:
            # The user vouches that the new address is the same server: the credential,
            # and the record of where it may go (one too old to have said), move with it.
            record = config.read_device_state()
            if record.get("surface_id"):
                config.write_device_state({**record, "server": value})
        if key == "token":
            _drop_record_of(home.get("token"))
            config.store_token(value)
        else:
            config.set_home_values({key: value})
    except OSError as e:
        return _fail(f"could not write {synapse_home.config_file()}: {e}")
    print(f"{key} = {_show(key, value)}")
    if key == "role" and config.read_device_state().get("surface_id"):
        print(
            "This machine is already enrolled; its served role changes when it re-enrolls "
            "(`synapse login --reenroll`) or with /synapse-devices."
        )
    if key == "url":
        return _rewire_url(value, vouched=moving)
    return 0


def _rewire_url(url: str, vouched: bool = False) -> int:
    """A harness opens the MCP connection itself, so its own copy of the URL follows."""
    wired = (("claude", harness.claude_plugin_id()), ("codex", harness.codex_wired()))
    rc = 0
    for target, present in wired:
        if present:
            rc |= _wire(url, target, vouched=vouched)
    return rc


# ------------------------------------------------------------------ install / update


def cmd_install(args) -> int:
    url = _configured_url()
    if not url:
        return _fail("no server URL yet: run `synapse login` (or `synapse config url <url>`)")
    return _wire(url, args.harness, dry_run=args.dry_run)


def cmd_update(_args) -> int:
    url = _configured_url()
    if not url:
        return _fail("no server URL yet: run `synapse login` first")
    request = urllib.request.Request(f"{url}/install", headers={"User-Agent": "synapse-cli"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            script = r.read()
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"could not fetch {url}/install: {e}")
    done = subprocess.run([sys.executable, "-", url], input=script, check=False)
    if done.returncode != 0:
        return _fail("the installer failed; the previous client is still in place")
    claude = harness.update_claude()
    if claude:
        print(claude)
    if harness.codex_wired():
        held = _credential_origins()
        if held and _origin(url) not in held:
            print(f"Codex: not re-wired: {_move_refusal(url, held)}")
            return 0
        try:
            print(harness.install_codex(url))
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Codex: not re-wired: {e}")
    return 0


# --------------------------------------------------------------------------- skills


def cmd_skill(args) -> int:
    if args.action == "scope":
        return _skill_scope(args)
    try:
        r = config.post_json("/skills/retire", {"name": args.name}, timeout=30)
    except urllib.error.HTTPError as e:
        return _fail(f"retire failed: HTTP {e.code}")
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"retire failed: {e}")
    status = r.get("status")
    if status == "retired":
        print(
            f"Retired {r['name']} ({r.get('scope') or 'global'}). Machines whose copy is "
            "the one Synapse delivered remove it at their next session start; a copy edited "
            "after now brings it back."
        )
        return 0
    if status == "already_retired":
        print(f"{args.name} was already retired ({r.get('retired_at')}).")
        return 0
    if status == "not_found":
        return _fail(f"no skill named {args.name!r} in Synapse")
    return _fail(r.get("detail") or f"retire failed: {r}")


def _skill_scope(args) -> int:
    """Move a skill to another scope: `global`, or `project:<label>` (a project as
    `synapse_project` labels it). Full-trust machines only; the server refuses others."""
    try:
        r = config.post_json(
            "/skills/rescope", {"name": args.name, "scope": args.scope}, timeout=30
        )
    except urllib.error.HTTPError as e:
        why = " (only a full-trust machine can move a skill)" if e.code == 403 else ""
        return _fail(f"scope change failed: HTTP {e.code}{why}")
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"scope change failed: {e}")
    status = r.get("status")
    if status == "rescoped":
        print(
            f"Moved {r['name']} from {r.get('from')} to {r['scope']}. Machines remove the copy "
            "Synapse delivered to the old scope's folder at their next session start, and the "
            "new scope's folders pull it."
        )
        return 0
    if status == "unchanged":
        print(f"{args.name} is already {r.get('scope')}.")
        return 0
    if status == "not_found":
        return _fail(f"no skill named {args.name!r} in Synapse")
    return _fail(r.get("detail") or f"scope change failed: {r}")


# ---------------------------------------------------------------------- passthrough

_PASSTHROUGH = {"devices": "device_admin", "private": "private_mode", "import": "import_history"}


def _passthrough(module: str, rest: list[str]) -> int:
    return importlib.import_module(module).main(rest) or 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synapse", description="Set up and manage Synapse on this machine."
    )
    sub = p.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", help="sign in, enroll this machine and wire its harnesses")
    login.add_argument("--url", help="the Synapse server's base URL")
    login.add_argument("--role", choices=_ROLES, help="declare this machine's role")
    login.add_argument(
        "--skills-sync", type=_onoff, metavar="yes|no", help="sync skills to this machine"
    )
    login.add_argument(
        "--token",
        nargs="?",
        const=_ASK_TOKEN,
        help="store a minted device token instead of signing in (asked for, hidden; '-' reads stdin)",
    )
    login.add_argument(
        "--reenroll", action="store_true", help="replace this machine's device token"
    )
    login.add_argument("--browser", action="store_true", help="legacy same-host browser sign-in")
    login.add_argument("--no-install", action="store_true", help="don't wire Claude Code / Codex")
    login.set_defaults(func=cmd_login)

    cfg = sub.add_parser("config", help="read or set this machine's saved settings")
    cfg.add_argument("key", nargs="?", help=", ".join(synapse_home.KEYS))
    cfg.add_argument("value", nargs="?")
    cfg.add_argument("--unset", action="store_true")
    cfg.add_argument(
        "--same-server",
        action="store_true",
        help="url only: the new address reaches the same server, so keep this machine's credential",
    )
    cfg.set_defaults(func=cmd_config)

    ins = sub.add_parser("install", help="wire Claude Code and/or Codex (no sign-in)")
    ins.add_argument("harness", nargs="?", choices=("claude", "codex"))
    ins.add_argument("--dry-run", action="store_true")
    ins.set_defaults(func=cmd_install)

    upd = sub.add_parser("update", help="install the client version the server serves")
    upd.set_defaults(func=cmd_update)

    skill = sub.add_parser("skill", help="manage skills in Synapse")
    skill_sub = skill.add_subparsers(dest="action", required=True)
    retire = skill_sub.add_parser("retire", help="retire a skill everywhere it was delivered")
    retire.add_argument("name")
    scope = skill_sub.add_parser("scope", help="move a skill to global or project:LABEL")
    scope.add_argument("name")
    scope.add_argument("scope", help="global, or project:<label>")
    skill.set_defaults(func=cmd_skill)

    ver = sub.add_parser("version", help="print the client version")
    ver.set_defaults(func=lambda _a: print(synapse_home.client_version()) or 0)

    for name, module in _PASSTHROUGH.items():
        sub.add_parser(name, help=f"same as synapse-{name}", add_help=False).set_defaults(
            module=module
        )
    return p


def _onoff(text: str) -> bool:
    parsed = synapse_home.parse_bool(text)
    if parsed is None:
        raise argparse.ArgumentTypeError("expected yes or no")
    return parsed


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in _PASSTHROUGH:
        return _passthrough(_PASSTHROUGH[argv[0]], argv[1:])
    args = _parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
