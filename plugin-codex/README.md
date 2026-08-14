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

Not ported (machine-level curation lanes, run them from Claude Code; running
them from both hosts would double-sync): skills_sync, config_sync,
git_feeder (board git feeder), and the `/synapse:skill-review` / `/synapse:config-review`
commands.

## Install

```
python3 plugin-codex/install.py --synapse-url http://your-synapse:8765
```

Appends six hook blocks to `~/.codex/config.toml` (idempotent; re-run after
upgrades to pick up new blocks) and registers the `synapse` MCP server. Then
start `codex`, run `/hooks`, and trust the new hooks — Codex refuses
unreviewed hooks.

Auth: export `SYNAPSE_INGEST_TOKEN` in the environment Codex runs under
(the MCP client hard-requires the env var). The hook scripts themselves also
fall back to the Claude plugin's saved options in `~/.claude/settings.json`,
so a machine running both plugins configures once.

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
| `SYNAPSE_INGEST_TOKEN` | – | bearer token (required for MCP) |
| `SYNAPSE_PRIVATE_DIR` | `~/.synapse/private` | private-mode markers |
| `SYNAPSE_CODEX_CURSORS` | `~/.synapse/codex_cursors.json` | ship cursors |
| `SYNAPSE_CODEX_CATCHUP_DAYS` | `3` | catchup sweep window |
| `SYNAPSE_BOARD` / `SYNAPSE_PREFS_BLOCK` / `SYNAPSE_RECALL_NUDGE` / `SYNAPSE_RECALL_FEEDBACK_NUDGE` / `SYNAPSE_CODEX_CATCHUP` | `1` | kill switches |
