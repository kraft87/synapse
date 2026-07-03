-- 023: dream->skills ledger hardening (Gemini review of 022).
--
-- (1) Enforce THE core rule in the DB, not just app code: a candidate can only reach
--     accepted/promoted if it has a GROUNDED signal (grounded_weight > 0). The LLM judge
--     (judge_weight, discounted 0.5x into score) can NEVER advance a candidate to apply on
--     its own — even an app bug or LLM-spam burst can't promote without a real user signal.
-- (2) Decay is now time-based (skill_ledger.decay_stale uses last_seen age) instead of a
--     nightly mass-bump of runs_since_seen on every inactive row (which dead-tuple-bloated
--     rows that weren't even changing). runs_since_seen stays as a display/info column,
--     reset to 0 when a candidate is re-seen; it no longer drives retirement.
--
-- (Deferred from the review: normalizing evidence JSONB into a child table with a UNIQUE
--  idempotency constraint. The writer already dedups by (session_id, signal, class) and the
--  lane is single-user / tiny, so the MVCC/idempotency win doesn't justify the join cost yet.)

ALTER TABLE skills_lane.skill_gap_candidates
    ADD CONSTRAINT grounded_required_to_apply
    CHECK (status NOT IN ('accepted', 'promoted') OR grounded_weight > 0);
