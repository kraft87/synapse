-- 2-level entity-type taxonomy, data-driven (Oracle-reviewed design, 2026-06-19).
--   * supertype  — coarse (~32). ENTITY RESOLUTION gates on this (soft/asymmetric, later step).
--   * subtype    — finer (faceting). Carries the meaningful distinctions.
-- The mapping lives in TABLES, not code, so the taxonomy + collapse rules are INSERT/UPDATE-
-- editable (play with it live) and the whole lockdown is reversible: re-derive supertype from
-- subtype with a SQL UPDATE, no re-extraction. Raw kg_entities.entity_type is KEPT for one
-- migration cycle (don't overwrite in place) so the backfill can be revisited.

CREATE TABLE IF NOT EXISTS kg_supertypes (
    name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS kg_entity_types (
    subtype   TEXT PRIMARY KEY,
    supertype TEXT NOT NULL REFERENCES kg_supertypes(name)
);

ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS entity_supertype TEXT;
ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS entity_subtype TEXT;
CREATE INDEX IF NOT EXISTS kg_entities_supertype ON kg_entities (entity_supertype);
