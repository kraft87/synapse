-- 038: timeline domain scoping (issue #17 acceptance-run observation).
--
-- The timeline leg had no domain/group scoping: a personal-domain query word
-- pulled unrelated technical events into the recall payload (and buried the
-- genuinely relevant personal events — measured 2026-07-05: for a reworded
-- personal query, cross-domain junk sat CLOSER in embedding space (0.704) than
-- the true events (0.748+), so no distance floor can separate them; domain can).
--
-- domain is stamped at write time: the chat gate's LLM call labels each event
-- ("personal" = the user's own life; "technical" = engineering/infra work);
-- git-sourced events are technical by construction. NULL = unlabeled (legacy
-- rows pre-backfill, or a gate parse miss) and FAILS OPEN at read — the
-- personal-scope filter keeps NULLs rather than hiding unlabeled events.
ALTER TABLE timeline_events ADD COLUMN IF NOT EXISTS domain TEXT
    CHECK (domain IN ('personal', 'technical'));
