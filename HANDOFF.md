# Handoff log

Newest first. One entry per change that alters how a machine is set up or operated: what
changed, what to do on each machine, what was proven, and what is still open.

## 2026-10-12 — one project label everywhere (plugin 0.21.0, no schema change)

**What changed**

- **One resolver** (`plugin/scripts/synapse_project.py`, loaded by the Codex plugin too):
  a `.synapse-project` marker file (first non-comment line; searched from the working
  directory up to the git toplevel, and in a linked worktree also at the main checkout's
  root; UTF-8 with or without BOM, or UTF-16 from PowerShell 5.1), else the git toplevel's
  name (a linked worktree, Claude Code's `.claude/worktrees/<id>` included, is its main
  checkout; a bare repo's worktree is the repo's name), else the folder's name. Both
  separators, case kept. A folder that is gone (a removed worktree swept by catch-up) takes
  the label of the repository its nearest surviving parent is in. No subprocess.
- **Every client uses it**: the Claude ingest hook (Stop, catch-up, `synapse-import`) and
  the Codex Stop hook now send `"project"` with every `/ingest` body; the board hook and
  Codex session start ask `/context` for it; skills sync, skill review, config sync and the
  git feeder label with it. Claude project skills now sync into `.claude/skills` at the
  project ROOT (Claude Code reads it from any subfolder), not into each subfolder.
- **Server**: `ingestion/project_label.py` is the parsers' fallback for bodies without a
  label (both parsers used to split on "/" only, so a Windows cwd became the whole path);
  `/ingest` cuts a client-sent label to a bare name (a non-string `project` used to 500).
  A restricted allowlist matches case-insensitively (`resolve_caller` widens it to every
  stored casing via a skip scan on the 061 index, cached 60 s); skill scopes match
  case-insensitively. `ingestion/project_id.py` (dead) and `MEMORY_PROJECT` are gone.
- **`scripts/relabel_projects.py`**: maps stored labels to canonical names and rewrites
  `episodes.project` (the 056 trigger keeps KG `source_projects` in step), every other
  `project` column, `surfaces.allowed_projects`, and `project:` scopes in the skill
  registry and config lane. `--dry-run` (or `--inventory-sql` + `--dry-run --inventory`,
  which needs no DSN) writes nothing. Keep the alias file outside the repo.

**On each machine**: `synapse update` (or the plugin update) to 0.21.0. Optional: put a
`.synapse-project` file in a repository whose folder name differs between machines.

**Server**: merge → image. Then, only after Kyle approves the dry-run table:
`scripts/relabel_projects.py --apply --aliases <file>` from a checkout. Until the relabel
runs, old labels stay as they are (new sessions arrive under canonical names).

**Proven**

- Real failures reproduced first, then fixed: the server parsers labelled
  `C:\Users\…\CompromisedAccount` (and a lowercase-drive variant) with the whole path, from
  both harnesses; main's real hooks, run as subprocesses against a local server in a real
  repo, labelled a session in `MyRepo/tickets/lab` as `lab`, a worktree as `agent-x`, sent no
  `project` from either ingest hook, and synced `project:lab`. After: all five hooks send
  `MyRepo`.
- Prod (read-only, 2026-10-12): 189 label/source rows; 1,858 episodes under Windows paths;
  the work laptop's allowlist has 13 entries for its few projects. The mapping table names
  private projects and paths, so it goes to Kyle directly and is not committed.
- Tests: resolver on real git layouts (worktrees, bare, submodule, markers), hook wiring with
  real subprocesses, case-insensitive allowlists end to end on ParadeDB, relabel planner +
  apply/idempotency on ParadeDB.

**Open**

- Kyle approves the relabel mapping; then `--apply` on prod.
- Windows not run on a real host (paths exercised as strings; the resolver's filesystem
  walk is OS-neutral).

## 2026-10-12 — `synapse login` is the single setup step (#222, plugin 0.20.0, schema 060)

**What changed**

- **Harness-independent config: `~/.synapse/config.json`** (`plugin/scripts/synapse_home.py`):
  server URL, token, machine role, `skills_sync`, `config_sync`, `config_paths`, and the
  enrollment record (formerly `~/.local/share/synapse-skills/device.json`). Written 0600,
  atomically. Both `plugin/scripts/config.py` and `plugin-codex/scripts/common.py` resolve
  env → `config.json` → the Claude plugin's legacy options (`CLAUDE_PLUGIN_OPTION_*`, then
  `settings.json`) → default; the Codex Stop hook's private copy of that resolution is gone.
  `SYNAPSE_HOME` overrides `~/.synapse`.
- **Claude plugin MCP auth via a headers helper**: `plugin.json` gains
  `"headersHelper": "python3 scripts/mcp_headers.py || python … || py -3 …"` (Windows
  often has no real `python3`); Codex uses the same script
  (`plugin/scripts/mcp_headers.py`; `plugin-codex/scripts/mcp_headers.py` is now a shim
  for old `config.toml` lines). The static `Bearer ${user_config.SYNAPSE_INGEST_TOKEN}`
  header stays as the fallback. `userConfig` now asks only for `SYNAPSE_URL` and the
  optional token. Correction to the issue's premise, verified on Claude Code 2.1.296: a
  *plugin's* helper runs with no trust prompt (also in `claude -p`); only project/local
  `.mcp.json` helpers are trust-gated. So the helper is the primary path everywhere and the
  static header covers helper failure.
- **`synapse` CLI** (`plugin/scripts/synapse_cli.py`, `harness.py`): `login` (sign in /
  enroll, ask role + "Sync skills to this machine?" once, save to `config.json`, wire every
  harness found), `config`, `install claude|codex`, `update`, `skill retire`, plus
  `devices`/`private`/`import` passthroughs. `synapse-login` is now an alias of
  `synapse login`. Claude Code is wired through `claude plugin marketplace add` +
  `claude plugin install synapse@synapse --config SYNAPSE_URL=<url>` (settings-key
  bootstrap when there is no `claude` CLI); Codex through `plugin-codex/install.py`.
  First `synapse login` on an old install copies the old install-prompt answers (and a
  token Claude kept in `.credentials.json`) into `config.json`, updates a cached older
  plugin, and drops the four options the plugin no longer declares once 0.20 is
  installed. Moving to another server needs `--reenroll` (or a yes), also on a machine
  set up with a pasted token; the old server's token is never sent to the new one, and the
  new URL is saved only after it enrolls. `synapse config url`, `install` and `update`
  refuse another origin while the machine holds a credential, including one Claude Code
  keeps for the plugin (`.credentials.json`, the macOS keychain); `--same-server` is for
  a server's new address. The hooks and the headers helper send a device token only to
  the server that minted it (the record's `server`) and to wherever a plugin config that
  holds that token points (its MCP header sends it there anyway), even when `config.json`
  is edited by hand; when they withhold it, session start says so in one line. A pre-0.20
  plaintext token copy in the plugin's options follows a new token only while the plugin
  points at that token's server (a move re-points the plugin first; with `--no-install`
  the copy is removed).
- **`GET /install`** (`mcp_server/install_routes.py`, `install_script.py`): an
  unauthenticated stdlib installer with this server's own client inside. It unpacks into
  `~/.synapse/lib/<version>/`, points `lib/CURRENT` at it, writes `~/.synapse/bin/synapse`
  (+ `.cmd`) and the fixed launcher `~/.synapse/bin/synapse_run.py`, records the server URL,
  and adds `~/.synapse/bin` to the user PATH on Windows (prints the line on POSIX;
  the login-bash startup file on macOS). `synapse.cmd` is ASCII only (cmd reads batch
  files in the console code page): the runner is `%~dp0synapse_run.py`, a profile- or
  Program Files-relative interpreter is written as `%LOCALAPPDATA%\…`/`%ProgramW6432%\…`,
  and a missing one falls back to the `python` on PATH with the current directory
  excluded. The call jumps to the exit on its own line, so `synapse update` rewriting the
  running shim is harmless. All `.cmd` shims now return Python's exit code. The
  Dockerfile now copies `plugin/` and `plugin-codex/` into the image.
- **Codex installer is Windows-safe** (`plugin-codex/install.py`): every command is a bare
  interpreter token plus double-quoted forward-slash paths (no `shlex.join`, no raw
  backslashes in TOML); the whole file is validated with `tomllib` before one write; old
  broken Windows lines are repaired; Synapse hooks that point elsewhere are re-pointed in
  place. Under the managed client, commands call `~/.synapse/bin/synapse_run.py <relative
  script>`, never `lib/<version>/`, because Codex trusts a hook by a hash of its command:
  updates no longer switch the hooks off.
- **Skill retire**: schema 060 (`skill_registry.retired_at`, optional), `POST
  /skills/retire`, `/skills/list` returns retired entries under a separate `retired` key
  (old clients ignore it), and `/skills/publish` refuses to reactivate a retired skill
  unless the pushed edit is newer than the retirement (old clients can no longer resurrect
  it), nor with the retired content itself under a newer timestamp (a clone or a copy
  renews an mtime). `/skills/list` lists every retirement whatever the scope asked for. The
  client sync removes a local copy that is exactly a version Synapse delivered (ledger
  under `~/.synapse/skills/`, kept after the removal so a restored copy goes again)
  whatever its timestamp, pushes an unrecognised one edited after the retirement, and
  leaves everything else alone (symlinks and junctions included). In another scope's
  folder only the exact retired content counts: a different skill of the same name there
  is never touched.
  Accepting a retune of a retired skill now needs `--force`.

**On each machine**

1. Existing installs keep working with no action (legacy `settings.json` and `device.json`
   are still read).
2. To move over: `curl -sSL https://<server>/install | python3 -` (PowerShell:
   `irm https://<server>/install | python -`), then `synapse login`. On an enrolled machine
   it signs in again only with `--reenroll`.
3. Codex machines: after that first `synapse login`, trust the re-pointed hooks once in
   `/hooks`. The work laptop's hand-patched helper line is rewritten by it too.
4. A token pasted into Claude Code's install dialog lives in Claude's secure storage, which
   nothing else can read; `synapse login` says so and `--reenroll` fixes it.

**Server**: merge → image → apply `schema/060_skill_retired_at.sql`
(`scripts/apply_schema.sh`). Until 060 is applied the server boots and behaves as before;
only `synapse skill retire` is refused.

**Proven** (scratch homes against a local server on this branch)

- Codex-only machine (no `~/.claude`): `curl …/install | python3 -`, `synapse login
  --token …`, then a real `codex exec` (Codex 0.162.1): SessionStart/UserPromptSubmit/Stop
  hooks ran, the board text reached the model, the `mcp__synapse` tools were listed through
  the headers helper, and the turn was ingested.
- Claude Code 2.1.296: after `synapse login`, `claude mcp list` connects through the helper
  with no token in Claude's config; with the helper refusing it fails (401) — and connects
  again from the static fallback token alone.
- Retire across a Claude-side and a Codex-side machine: delivered copies removed on both;
  a copy edited after the retirement pushed back and active again.
- A legacy install (token in `settings.json`, `device.json`, no `config.json`) works
  untouched; `! synapse-login` then migrates it.

**Open**

- Windows was not run on a real host: command strings are proven by `tomllib` and by
  executing them through `sh`/`bash`, and by the documented cmd/PowerShell quoting rules;
  the `.cmd` shims (goto-based exit codes, `NoDefaultCurrentDirectoryInExePath`) by their
  text only.
- Pre-existing, not changed here: the Claude hooks' `commandWindows` in
  `plugin/hooks/hooks.json` still run `python3`, which a python.org-only Windows machine
  lacks (the MCP helper now falls back to `python`/`py -3`).
- `docs/media/setup-demo.gif` still shows the old `/plugin install` flow.
- Out of scope (separate issues): Go port of the client, SEP-2640 serving, gateway-pilot
  teardown, board cap-loop/connection-pool perf.
