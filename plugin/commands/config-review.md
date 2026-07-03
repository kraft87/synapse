---
description: Review pending dream→config proposals (behavioral rules mined from your corrections) and accept/reject them.
---

Run the dream→config review CLI to triage behavioral-rule edits the autonomous lane has proposed.

Steps:
1. List pending proposals: `python3 "$CLAUDE_PLUGIN_ROOT/scripts/config_review.py" list`
2. If the user named an id or said "show me N", run `… show <id>` and present the rule + the corrections that evidence it.
3. Act on the user's decision:
   - accept: `… accept <id>` (writes the rule to its config file on this machine — e.g. `~/.claude/rules/learned.md` — and marks it applied; add `--local` to scope the edit to this surface only)
   - reject: `… reject <id> [reason]`
4. Summarize what changed. Never accept without the user's explicit say-so — accept edits the user's live config files.

Pass any argument the user gave (an id, "accept 3", etc.) straight through. With no argument, default to `list`.
