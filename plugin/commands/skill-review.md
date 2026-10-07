---
description: Review pending dream→skills proposals (new skills, trigger retunes, merges) and accept or reject them. Accept applies the change.
---

Run the dream→skills review CLI to triage what the autonomous lane has proposed. Accepting a proposal applies it: the server writes the skill into its registry, and the two-way skills sync delivers it to whichever machine owns the skill. There is no file to move and no separate promote step.

Steps:
1. List pending proposals: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/skill_review.py" list`
2. If the user named an id or said "show me N", run `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/skill_review.py" show <id>`. Present the evidence, where accept would apply it (skill name and scope), and the change: a unified diff against the current registry body, or the full SKILL.md for a new skill. Point out any `!` warning lines.
3. Act on the user's decision:
   - accept: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/skill_review.py" accept <id>`. This applies the drafted SKILL.md. Options:
     - `--body-file PATH` applies a version the user edited instead of the draft (write it to a temp file first).
     - `--scope global|project:<name>` overrides where the skill lands.
     - `--force` applies even when the skill changed since the draft was made, or a new skill's name is taken. Only with the user's say-so, after showing them why accept refused.
   - reject: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/skill_review.py" reject <id> [reason]` (30-day cooldown).
   - If accept says there is no draft yet, tell the user it re-drafts on the next nightly run, or offer to write the SKILL.md with them and pass it via `--body-file`.
4. Summarize what changed: which skill was created or updated, its scope, and where it syncs. Never accept without the user's explicit say-so, because accept edits the live skill library.

Merge proposals (MERGE) are the exception: accept only records the decision, the merge itself is done by hand.

Pass any argument the user gave (an id, "accept 3", etc.) straight through. With no argument, default to `list`.
