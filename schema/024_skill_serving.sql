-- 024_skill_serving.sql
-- DB-hosted skill serving. skill_registry gains the SKILL.md body + scope/status;
-- skill_files holds bundled files (scripts, references) as bytes. The fastmcp
-- PgSkillsProvider (mcp_server/skills_provider.py) serves these as skill:// MCP
-- resources; clients materialize them to ~/.claude/skills via sync_skills.
-- PG stays the single source of truth (the dream->skills lane writes here).

ALTER TABLE skills_lane.skill_registry
    ADD COLUMN IF NOT EXISTS body   text,
    ADD COLUMN IF NOT EXISTS scope  text NOT NULL DEFAULT 'global',
    ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'proposed', 'retired'));

-- Only status='active' skills are served. 'proposed' = a lane draft awaiting apply;
-- 'retired' = pulled but kept for history.

CREATE TABLE IF NOT EXISTS skills_lane.skill_files (
    skill_name    text        NOT NULL REFERENCES skills_lane.skill_registry(name) ON DELETE CASCADE,
    path          text        NOT NULL,            -- POSIX-relative path within the skill
    content       bytea       NOT NULL,
    sha256        text        NOT NULL,            -- hex digest; provider serves it as 'sha256:<hex>'
    size          integer     NOT NULL,
    is_executable boolean     NOT NULL DEFAULT false,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (skill_name, path)
);
