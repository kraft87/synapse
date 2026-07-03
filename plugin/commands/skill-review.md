---
description: Review pending dream→skills proposals (new skills, trigger retunes, merges) and accept/reject/promote them.
---

Run the dream→skills review CLI to triage what the autonomous lane has proposed.

Steps:
1. List pending proposals: `python3 "$CLAUDE_PLUGIN_ROOT/scripts/skill_review.py" list`
2. If the user named an id or said "show me N", run `… show <id>` and present the evidence + the drafted SKILL.md.
3. Act on the user's decision:
   - accept: `… accept <id>` (records the grounded accept; prints the draft path)
   - reject: `… reject <id> [reason]` (30-day cooldown)
   - promote: `… promote <id>` (only after the user has placed the file into the skills dir — confirms it's live)
4. Summarize what changed. Never accept/promote without the user's explicit say-so — these edit the live skill library.

Pass any argument the user gave (an id, "accept 3", etc.) straight through. With no argument, default to `list`.
