-- 022: dream->skills lane — candidate ledger, firing log, skill registry, scan cursor.
--
-- The forward/accumulate store for the dream->skills lane. Client-side scripts
-- (skill-measure.py / skill-derive.py) write here over the host PG port. A nightly
-- run scans only NEW episodes (since skill_scan_cursor.last_scan_at), emits candidates,
-- and MERGES them into the ledger instead of re-deciding from scratch — so a frequency
-- floor works going forward (a procedure seen once/night accumulates across runs).
--
-- ISOLATED in its own schema (`skills_lane`), NOT public: this is client-side
-- bookkeeping, not Synapse memory. Synapse migrations / snapshots / dream / recall must
-- never touch it (Oracle Q6). Reaches `public.episodes` read-only via the lane scripts.
--
-- Prior art mirrored: skill_miner (carry-forward observe bucket, SEMANTIC dedup NOT
-- name-keyed, classify ladder, propose-not-apply), claude-soul/thebrain (evidence-tier
-- promotion + self-referential discount + decay), EvoSkill (failure-signal), dream-skill
-- (correction-grep heuristics). Reviewed by Oracle 2026-06-21.
--
-- THE CORE RULE: the LLM "would-have-helped"/gap-scan verdict is the system judging
-- ITSELF -> it only NOMINATES and is DISCOUNTED 0.5x; only GROUNDED user signals advance
-- a candidate toward apply. The discount is a GENERATED column so it can't drift.
-- Grounded signals (ranked): explicit request > proposal accept/reject > post-fire
-- dismissal > post-change firing > repeated correction.
--
-- WRITER CONTRACT (enforced in app, documented here):
--   * Recompute judge_weight/grounded_weight/*_sessions from the FULL evidence JSONB on
--     every merge — never incrementally (else `score` silently lies). Oracle flag.
--   * The writer may set status up to 'accepted' ONLY. 'promoted' is set solely by the
--     filesystem accept path (the mv-into-~/.claude/skills step). Oracle Q4.
--   * Dedup evidence by session_id (judge_sessions = DISTINCT sessions). Oracle Q2.
--   * Identity for 'derive' is resolved in the writer (signature_key + session-id Jaccard,
--     session-ids weighted highest as ground truth), NOT by a name unique constraint —
--     LLM-proposed names drift run-to-run and would silently fragment accumulation. Oracle Q1.

CREATE SCHEMA IF NOT EXISTS skills_lane;

-- ---------------------------------------------------------------------------
-- Candidate ledger
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills_lane.skill_gap_candidates (
    id               BIGSERIAL PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('derive', 'retune', 'consolidate')),
    name             TEXT NOT NULL,            -- display label (derive) / existing skill (retune) / merged name (consolidate)
    signature_key    TEXT,                     -- name-INDEPENDENT identity: normalized token-set of signature + tool sequence (derive)
    target_skills    TEXT[] NOT NULL DEFAULT '{}',  -- existing skill(s) affected (retune=1, consolidate=N); empty for derive
    direction        TEXT CHECK (direction IN ('widen', 'narrow')),  -- retune only: add triggers vs remove a mis-firing one
    summary          TEXT NOT NULL DEFAULT '',
    signature        TEXT,                     -- observed tool/command pattern (derive) or trigger-gap note (retune)
    trigger_phrasings JSONB NOT NULL DEFAULT '[]',
    summary_embedding halfvec(2048),           -- gap-prefilter (vs existing skill descriptions) + identity-secondary

    status           TEXT NOT NULL DEFAULT 'observe'
                       CHECK (status IN ('observe', 'proposed', 'accepted', 'promoted', 'rejected', 'retired')),
    reject_reason    TEXT,                     -- one_off | too_generic | too_specific | duplicate | stale | user_rejected | superseded

    -- evidence ledger (source of truth) + writer-recomputed roll-ups
    evidence         JSONB NOT NULL DEFAULT '[]',  -- [{session_id, ts, class:'judge'|'grounded', signal, skill?, phrasing?, why?, tools?}]
    judge_sessions    INT NOT NULL DEFAULT 0,  -- DISTINCT sessions w/ judge (soft) evidence
    grounded_sessions INT NOT NULL DEFAULT 0,  -- DISTINCT sessions w/ grounded evidence
    judge_weight      DOUBLE PRECISION NOT NULL DEFAULT 0,
    grounded_weight   DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- 0.5x self-referential discount baked in: grounded counts full, judge half.
    score            DOUBLE PRECISION GENERATED ALWAYS AS (grounded_weight + 0.5 * judge_weight) STORED,

    privacy_flags    JSONB NOT NULL DEFAULT '[]',
    proposal_path    TEXT,                     -- drafted SKILL.md / edit path once status='proposed'

    runs_since_seen  INT NOT NULL DEFAULT 0,   -- decay counter; bumped each run not re-seen; TTL -> 'retired'. CANDIDATES only — never deletes a live skill.
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen        TIMESTAMPTZ NOT NULL DEFAULT now(),
    rejected_until   TIMESTAMPTZ,              -- cooldown: don't re-surface a user-rejected candidate before this

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Stable identity only where the name is real & stable (existing skill). direction is part
-- of the key so a widen and a narrow proposal for the same skill don't collide. (Oracle Q1/Q5)
CREATE UNIQUE INDEX IF NOT EXISTS skill_gap_named_idx
    ON skills_lane.skill_gap_candidates (kind, name, COALESCE(direction, '-'))
    WHERE kind IN ('retune', 'consolidate');
-- derive identity is app-resolved; index the resolver's lookup key.
CREATE INDEX IF NOT EXISTS skill_gap_sigkey_idx
    ON skills_lane.skill_gap_candidates (signature_key)
    WHERE kind = 'derive';
-- active working set a run reconciles against
CREATE INDEX IF NOT EXISTS skill_gap_active_idx
    ON skills_lane.skill_gap_candidates (kind, score DESC)
    WHERE status IN ('observe', 'proposed');

-- ---------------------------------------------------------------------------
-- Skill firing log (the firing-history substrate)
-- ---------------------------------------------------------------------------
-- One row per skill invocation seen in transcripts. Powers: RETUNE under-fire baseline,
-- the post-fire DISMISSAL grounded signal, and the post-change "did it fire after?" metric.
-- Frequency here is NEVER on its own a reason to retire a live skill (rare != low value).
CREATE TABLE IF NOT EXISTS skills_lane.skill_usage (
    id          BIGSERIAL PRIMARY KEY,
    skill       TEXT NOT NULL,
    fired_at    TIMESTAMPTZ NOT NULL,
    session_id  TEXT,
    via         TEXT,                          -- 'tool' (Skill tool_use) | 'slash' (/command)
    dismissed   BOOLEAN,                       -- user overrode/corrected right after firing (grounded retune-negative)
    outcome     TEXT,                          -- worked | failed | partial | unknown
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (skill, session_id, fired_at)       -- idempotent across re-scans
);
CREATE INDEX IF NOT EXISTS skill_usage_skill_idx ON skills_lane.skill_usage (skill, fired_at DESC);

-- ---------------------------------------------------------------------------
-- Skill registry (per-skill metadata: overlap embedding, pin, usage rollup)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills_lane.skill_registry (
    name             TEXT PRIMARY KEY,
    description      TEXT,
    description_embedding halfvec(2048),       -- routing-overlap detection (pairwise cosine; flags consolidate candidates)
    body_hash        TEXT,                     -- detect description/body change -> re-embed
    pinned           BOOLEAN NOT NULL DEFAULT FALSE,  -- rare-but-gold protection: never proposed for retirement regardless of usage
    fire_count       INT NOT NULL DEFAULT 0,   -- rollup from skill_usage (informational)
    last_fired       TIMESTAMPTZ,              -- rollup; informational, NOT a retirement trigger
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- (<=25 skills -> brute-force cosine is instant; no HNSW index needed.)

-- ---------------------------------------------------------------------------
-- Scan watermark (singleton) — process only episodes newer than last_scan_at
-- ---------------------------------------------------------------------------
-- Scan UNIT is the whole session (re-pull any session_id with max(created_at) > last_scan_at),
-- not raw new turns — partial tails yield a different signature_key and re-fragment identity.
-- The high-water mark still advances on max(created_at). (Oracle Q2)
CREATE TABLE IF NOT EXISTS skills_lane.skill_scan_cursor (
    id               INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_scan_at     TIMESTAMPTZ,
    last_run_at      TIMESTAMPTZ,
    runs             INT NOT NULL DEFAULT 0,
    config           JSONB NOT NULL DEFAULT '{}',  -- is_substantive thresholds, excluded projects, floors — audit trail for baseline drift
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO skills_lane.skill_scan_cursor (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
