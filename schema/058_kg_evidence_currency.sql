-- 058_kg_evidence_currency.sql
-- Evidence metadata on fact edges: what kind of claim an edge makes, and when the
-- user themself last supported it.
--
-- t_valid answers "when did this become true"; nothing answered "when was this last
-- confirmed". A habit extracted once ("User swims before work") stayed a live, undated
-- present-tense fact for as long as nobody contradicted it, and the reader had no
-- signal that the only evidence was a single mention long ago. Three columns fix the
-- bookkeeping without rewriting any fact text (the pipeline never invents end dates):
--
--   ongoing            TRUE when the fact asserts an activity, habit, usage, or state
--                      that can lapse without anyone saying so (a workout routine, using
--                      a tool, holding a job). FALSE for dated happenings, permanent
--                      traits, and descriptions of artifacts ("the profile lists X").
--                      NULL for every pre-058 edge: unknown, not false.
--   last_supported_at  Timestamp of the most recent USER-attributed statement that
--                      asserted this proposition (the conversation time, not the
--                      extracted valid-from date). Set on create when the extractor
--                      attributes the fact to the user; refreshed on reinforcement
--                      (a later chunk re-asserting the same edge) under the same
--                      attribution rule. Assistant restatements and re-processing of
--                      an already-cited chunk never refresh it. NULL = never
--                      user-confirmed, or pre-058.
--   last_supported_by  The source episode id behind last_supported_at (the latest
--                      episode of the supporting chunk), so the support can be read
--                      back. NULL when last_supported_at is NULL.
--
-- Recall serves ongoing + last_supported_at beside t_valid so a reader can tell
-- "stored valid-from 2025-03, never restated since" from "restated last week".
-- No backfill: attribution is not recoverable from stored edges.

ALTER TABLE kg_relationships ADD COLUMN IF NOT EXISTS ongoing BOOLEAN;
ALTER TABLE kg_relationships ADD COLUMN IF NOT EXISTS last_supported_at TIMESTAMPTZ;
ALTER TABLE kg_relationships ADD COLUMN IF NOT EXISTS last_supported_by BIGINT;
