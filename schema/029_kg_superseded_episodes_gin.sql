-- 029: partial GIN on the episode-provenance array, scoped to superseded edges only.
--
-- Powers the read-side episode-validity overlay (recall._episode_supersessions): for a served
-- episode, find retired edges that CITE it (episodes @> [id]) AND carry an invalidated_by link
-- (schema 028), then surface the current superseding fact. Partial — only edges with a known
-- successor (a few hundred) — so the per-recall containment lookup is cheap and the index tiny.
CREATE INDEX IF NOT EXISTS kg_rel_superseded_episodes_gin
    ON kg_relationships USING gin (episodes jsonb_path_ops)
    WHERE invalidated_by IS NOT NULL;
