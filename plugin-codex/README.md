# Synapse for Codex CLI

Wires OpenAI Codex CLI sessions into Synapse memory, at feature parity with
the Claude Code plugin (`../plugin/`) for the per-session surface:

| feature | Claude plugin | here |
|---|---|---|
| live turn ingest | `Stop` hook | `hooks/synapse_stop_hook.py` |
| catchup sweep | SessionStart `--catchup` | spawned by `hooks/session_start.py` |
| board + preferences into context | SessionStart hooks | `hooks/session_start.py` |
| recall/remember MCP | plugin manifest | `codex mcp add` (installer) |
| per-prompt recall nudge | UserPromptSubmit | `hooks/user_prompt_submit.py` |
| self-session injection | PreToolUse | `hooks/pre_tool_use.py` |
| recall-feedback nudge | PostToolUse | `hooks/post_tool_use.py` |
| private mode | marker + server row | `scripts/private_mode.py` (+ SessionEnd cleanup) |
| skills sync (opt-in) | SessionStart `skills_sync.py` | `scripts/skills_sync.py`, run by `hooks/session_start.py` |

Skills sync shares the Claude plugin's engine and its `SYNAPSE_SKILLS_SYNC=1`
opt-in. It syncs Synapse's global skills two-way with Codex's user-skills
folder (`~/.agents/skills`) only; Codex's bundled skills under
`~/.codex/skills/.system` and plugin caches are never scanned or uploaded.
Both plugins on one machine converge on the same server copy, so an edit in
either folder reaches the other at its next session start.

Not ported (machine-level curation lanes, run them from Claude Code; running
them from both hosts would double-sync): config_sync, git_feeder (board git
feeder), project-scoped skills, and the `/synapse:skill-review` /
`/synapse:config-review` commands.

## Install

```
python3 plugin-codex/install.py --synapse-url http://your-synapse:8765
```

Installs six hook blocks in `~/.codex/config.toml` (idempotent; re-run after
upgrades to migrate legacy matchers and MCP authentication) and registers the `synapse` MCP server. Then
start `codex`, run `/hooks`, and trust the new hooks — Codex refuses
unreviewed hooks.

Auth: MCP and hooks share the same credential source: `SYNAPSE_INGEST_TOKEN`
when explicitly set, otherwise the Claude plugin's saved options in
`~/.claude/settings.json`. MCP uses `scripts/mcp_headers.py` through Codex's
`http_headers_helper` setting (verified with Codex 0.155.0). A background Codex
daemon no longer needs to inherit a shell export. No token is copied into
`config.toml`, and the helper refuses to send it to a different server origin.
Re-run the installer to replace legacy `bearer_token_env_var` configuration,
then reload MCP or start a new session. Older Codex builds without header-helper
support must be updated first.

That value must be this machine's own **device** token, not the shared
`SYNAPSE_MACHINE_TOKEN`. Since schema 054 what a machine is served depends on
its own credential, and the machine token resolves to no surface, so a Codex
session using it gets an empty board and empty recalls with no error. Mint one
on the server host with
`docker compose exec mcp-server synapse-admin bootstrap "<label>"`, or from an
already-trusted machine with `/synapse-devices mint "<label>"`. Background:
[../docs/auth.md](../docs/auth.md).

## Why hooks in config.toml and not a plugin manifest?

Codex v0.147 validates but does not enable plugin-bundled hooks (the
`plugin_hooks` feature is off; only top-level `config.toml` hooks run), so
the installer writes hook blocks directly. When plugin hooks ship, this
directory can become a `.codex-plugin/` package and the installer goes away.

Codex hook I/O mirrors Claude Code's: stdin JSON payload, stdout JSON
envelope `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext"
/ "updatedInput": ...}}`. Two divergences that cost debugging time: the
`async` key on hook entries is rejected (docs list it; the build refuses it),
and `updatedInput` is only honored alongside `permissionDecision: "allow"`.

## Env

| var | default | |
|---|---|---|
| `SYNAPSE_URL` | `http://localhost:8765` | base URL |
| `SYNAPSE_INGEST_TOKEN` | saved plugin option | per-device bearer token |
| `SYNAPSE_PRIVATE_DIR` | `~/.synapse/private` | private-mode markers |
| `SYNAPSE_CODEX_CURSORS` | `~/.synapse/codex_cursors.json` | ship cursors |
| `SYNAPSE_CODEX_CATCHUP_DAYS` | `3` | catchup sweep window |
| `SYNAPSE_SKILLS_SYNC` | `0` (saved plugin option) | `1` enables skills sync |
| `SYNAPSE_CODEX_SKILLS_DIR` | `~/.agents/skills` | folder the sync scans |
| `SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT` | `12` | seconds SessionStart waits for the sync |
| `SYNAPSE_BOARD` / `SYNAPSE_PREFS_BLOCK` / `SYNAPSE_RECALL_NUDGE` / `SYNAPSE_RECALL_FEEDBACK_NUDGE` / `SYNAPSE_CODEX_CATCHUP` | `1` | kill switches |
