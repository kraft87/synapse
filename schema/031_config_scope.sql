-- 031: scope on config mirror rows.
--   'global'         = ~/.claude persona (CLAUDE.md + rules/*.md) — the dream lane's main target.
--   'project:<name>' = a project's .claude config (project CLAUDE.md + rules).
-- Lets auto-discovery grab BOTH the root and a project's config without file_key collisions, and
-- lets the dream lane treat global persona differently from project config. config_registry is new
-- and small, so the PK swap is safe.
ALTER TABLE config_lane.config_registry ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'global';
ALTER TABLE config_lane.config_registry DROP CONSTRAINT IF EXISTS config_registry_pkey;
ALTER TABLE config_lane.config_registry ADD PRIMARY KEY (surface_id, scope, file_key);
