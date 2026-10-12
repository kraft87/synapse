# Synapse for Codex CLI

Wires OpenAI Codex CLI sessions into Synapse memory, at feature parity with
the Claude Code plugin (`../plugin/`) for the per-session surface:

| feature | Claude plugin | here |
|---|---|---|
| live turn ingest | `Stop` hook | `hooks/synapse_stop_hook.py` |
| catchup sweep + failed-upload retry | SessionStart `--catchup` | spawned by `hooks/session_start.py` |
| upload-failure warning | SessionStart `--catchup` | `hooks/session_start.py` |
| board + preferences into context | SessionStart hooks | `hooks/session_start.py` |
| recall/remember MCP | plugin manifest | `[mcp_servers.synapse]` + headers helper (installer) |
| per-prompt recall nudge | UserPromptSubmit | `hooks/user_prompt_submit.py` |
| self-session injection | PreToolUse | `hooks/pre_tool_use.py` |
| recall-feedback nudge | PostToolUse | `hooks/post_tool_use.py` |
| private mode | marker + server row | `scripts/private_mode.py` (+ SessionEnd cleanup) |
| skills sync (opt-in) | SessionStart `skills_sync.py` | `scripts/skills_sync.py`, run by `hooks/session_start.py` |

Skills sync shares the Claude plugin's engine and its opt-in (`skills_sync` in
`~/.synapse/config.json`; `synapse login` asks). It syncs Synapse's global skills two-way with Codex's user-skills
folder (`~/.agents/skills`) only; Codex's bundled skills under
`~/.codex/skills/.system` and plugin caches are never scanned or uploaded.
Both plugins on one machine converge on the same server copy, so an edit in
either folder reaches the other at its next session start.

A failed upload is not lost. When a POST fails (server down, a revoked token, a
proxy answering `403`), the Stop hook marks that rollout in its cursor file, and
every session start's catch-up sweep retries it, however long the outage
lasted. The sweep takes any lagging rollout modified in the last
`SYNAPSE_CODEX_CATCHUP_DAYS` (default 3) plus every rollout whose upload failed,
at any age, oldest first, `SYNAPSE_CODEX_CATCHUP_MAX` (default 20) per session
start. When uploads keep failing (3 in a row, or a day with no success), session
start shows one line, as a Codex warning and in the model's context:

```
Synapse uploads have been failing since 2026-01-31 09:12 (HTTP 403 Forbidden); transcripts will be retried automatically.
```

The rules and the wording are the Claude plugin's: the Stop hook imports
`../plugin/scripts/synapse_ingest_retry.py` from this checkout instead of keeping
a copy, the same way the skills sync shares its engine. The sweep never picks up
a rollout the hook didn't try to upload, so history from before install still
needs `python -m ingestion.codex_backfill`.

Not ported (machine-level curation lanes, run them from Claude Code; running
them from both hosts would double-sync): config_sync, git_feeder (board git
feeder), project-scoped skills, and the `/synapse:skill-review` /
`/synapse:config-review` commands.

## Install

On a machine with Codex (Claude Code optional), install the client from your server and log
in. That is the whole setup:

```
curl -sSL https://<your-synapse>/install | python3 -     # PowerShell: irm .../install | python -
synapse login                                            # or: synapse login --token  (asks for a device token)
```

`synapse login` enrolls the machine (or stores a token minted with
`docker compose exec mcp-server synapse-admin bootstrap "<label>"`), saves its settings in
`~/.synapse/config.json`, and writes six hook blocks plus the `synapse` MCP server into
`~/.codex/config.toml` (`synapse install codex` does that step alone). Then start `codex`,
run `/hooks`, and trust the new hooks — Codex refuses unreviewed hooks. Nothing about this
needs `~/.claude`.

From a git checkout, `python3 plugin-codex/install.py --synapse-url http://your-synapse:8765`
writes the same blocks pointing at the checkout; re-run it after upgrades.

**Commands that survive updates.** Codex trusts a hook by a hash of its command text, so
the commands never name a versioned directory. Under the managed client they call the fixed
launcher `~/.synapse/bin/synapse_run.py` with a client-relative path
(`python3 "/home/you/.synapse/bin/synapse_run.py" plugin-codex/hooks/session_start.py`),
which runs the version `~/.synapse/lib/CURRENT` names; `synapse update` changes that file,
never config.toml, so trust survives. Every command has one portable shape: a bare
interpreter token, then double-quoted, forward-slash paths. Codex runs hooks through your
shell (PowerShell by default on Windows) and the headers helper through `cmd /C` on Windows
and `sh -c` elsewhere, and that shape reads the same in all of them. On Windows the
interpreter token is the first of `py -3`, `python`, `python3` that runs Python 3.11+; the
headers helper names that interpreter by its absolute path instead, because cmd looks for a
bare `py`/`python` in the session's directory (the repository you opened) before PATH.
Re-running the installer repairs the backslash paths older versions wrote on Windows
(invalid TOML) and re-points Synapse hooks that differ; it then asks you to re-trust them.

**Auth.** MCP and hooks share one credential: `~/.synapse/config.json`, falling back to the
Claude plugin's saved options in `~/.claude/settings.json` for installs from before 0.20.
MCP uses the headers helper `plugin/scripts/mcp_headers.py` (shared with the Claude plugin)
through Codex's `http_headers_helper` setting (verified with Codex 0.155.0 and 0.162.1).
Codex runs it with an emptied environment, so it reads the file, never an exported
`SYNAPSE_INGEST_TOKEN`; it refuses to send the token to a different server origin, and since
Codex discards its stderr, its failures go to `~/.synapse/logs/mcp_headers.log`. No token is
copied into `config.toml`. `codex mcp add` is not used: it can't set a headers helper, and on
a server with OAuth it starts a browser sign-in whose token would outrank the device
credential. Once this machine is enrolled, its device token wins over an exported
`SYNAPSE_INGEST_TOKEN` in the hooks too, and session start prints one line naming the
overridden variable.

That value must be this machine's own **device** token, not the shared
`SYNAPSE_MACHINE_TOKEN`. Since schema 054 what a machine is served depends on
its own credential, and the machine token resolves to no surface, so a Codex
session using it gets an empty board and empty recalls with no error.
Background: [../docs/auth.md](../docs/auth.md).

## Why hooks in config.toml and not a Codex plugin?

Codex can install a plugin from this repository's marketplace (`codex plugin marketplace add
kraft87/synapse`, `codex plugin add synapse@synapse`), and plugin hooks do run, but that
plugin is the *Claude* one: its 13 hooks, not these six, and an MCP entry Codex can't use
(Codex expands no `${user_config.*}` and ignores `headersHelper`). So the installer writes
these hooks into `config.toml` directly. Don't install both, or every turn is ingested twice.

Codex hook I/O mirrors Claude Code's: stdin JSON payload, stdout JSON
envelope `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext"
/ "updatedInput": ...}}`. One divergence that cost debugging time: `updatedInput` is only
honored alongside `permissionDecision: "allow"`.

## Env

| var | default | |
|---|---|---|
| `SYNAPSE_URL` | `url` in `~/.synapse/config.json` | base URL |
| `SYNAPSE_INGEST_TOKEN` | `token` in `~/.synapse/config.json` | per-device bearer token; ignored once the machine is enrolled, and never seen by the MCP helper |
| `SYNAPSE_HOME` | `~/.synapse` | config.json, the installed client and its shims |
| `SYNAPSE_PRIVATE_DIR` | `~/.synapse/private` | private-mode markers |
| `SYNAPSE_CODEX_CURSORS` | `~/.synapse/codex_cursors.json` | ship cursors |
| `SYNAPSE_CODEX_CATCHUP_DAYS` | `3` | catchup sweep window for lagging rollouts; failed uploads are retried at any age |
| `SYNAPSE_CODEX_CATCHUP_MAX` | `20` | rollouts per sweep; the rest wait for the next session start |
| `SYNAPSE_CODEX_HOOK_LOG` | `/tmp/synapse-codex-hook.log` | Stop-hook log, one line per upload or failure |
| `SYNAPSE_SKILLS_SYNC` | `skills_sync` in `~/.synapse/config.json` (off) | `1` enables skills sync |
| `SYNAPSE_CODEX_SKILLS_DIR` | `~/.agents/skills` | folder the sync scans |
| `SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT` | `12` | seconds SessionStart waits for the sync |
| `SYNAPSE_BOARD` / `SYNAPSE_PREFS_BLOCK` / `SYNAPSE_RECALL_NUDGE` / `SYNAPSE_RECALL_FEEDBACK_NUDGE` / `SYNAPSE_CODEX_CATCHUP` | `1` | kill switches |
