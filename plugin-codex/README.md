# Synapse for Codex CLI

Wires OpenAI Codex CLI sessions into Synapse memory, mirroring the Claude Code
plugin (`../plugin/`):

- **Live ingest** — a Codex `Stop` hook ships each completed turn's rollout
  tail to Synapse `/ingest` (`format: "codex"`), parsed server-side by
  `ingestion.codex_client.CodexRolloutParser`.
- **Recall/remember** — registers the Synapse MCP server so Codex sessions can
  search and write memory.
- **Backstop sweep** — `python -m ingestion.codex_backfill` walks
  `~/.codex/sessions/` and ingests anything the hook missed. Push and sweep
  dedup on the same `codex:<item_id>` span ids, so overlap is a no-op.

## Install

```
python3 plugin-codex/install.py --synapse-url http://your-synapse:8765
```

Then start `codex`, run `/hooks`, and trust the new Stop hook (Codex refuses
unreviewed hooks). If the server is auth-gated, export `SYNAPSE_INGEST_TOKEN`
in the environment Codex runs under.

## Why not a Codex plugin manifest?

Codex v0.147 validates but does not enable plugin-bundled hooks
(`plugin_hooks` feature: removed/off; only top-level `config.toml` hooks run),
so the installer writes the hook block into `~/.codex/config.toml` directly.
When plugin hooks ship, this directory can grow a `.codex-plugin/plugin.json`
carrying the hook + MCP registration and the installer goes away.

## Pieces

- `hooks/synapse_stop_hook.py` — the Stop hook. Detached child ships a
  byte-cursor tail of the rollout file; cursor state in
  `~/.synapse/codex_cursors.json`; private sessions honored via
  `~/.synapse/private/<session_id>` markers.
- `install.py` — idempotent installer (hook block + `codex mcp add`).

## Env

| var | default | |
|---|---|---|
| `SYNAPSE_URL` | `http://localhost:8765` | base URL |
| `SYNAPSE_INGEST_TOKEN` | – | bearer token |
| `SYNAPSE_PRIVATE_DIR` | `~/.synapse/private` | private-mode markers |
| `SYNAPSE_CODEX_CURSORS` | `~/.synapse/codex_cursors.json` | cursor state |
