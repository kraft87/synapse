-- 034_embedding_meta.sql
--
-- Instance metadata: record the embedding geometry (dims + model) the schema's
-- vector/halfvec columns were provisioned with, so the app can FAIL LOUDLY when
-- the configured embedding backend doesn't match (a dims/model mismatch never
-- errors at query time — it just silently breaks retrieval).
--
-- Values come from psql -v vars set by scripts/apply_schema.sh
-- (SYNAPSE_EMBED_DIMS / SYNAPSE_EMBED_MODEL, default 2048 / voyage-4-large).
-- The \if blocks default them so a direct `psql -f` run (README "Upgrading")
-- works without -v. ON CONFLICT DO NOTHING: on an existing database this
-- backfills the pre-034 production values exactly once and never overwrites.

CREATE TABLE IF NOT EXISTS synapse_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

\if :{?embed_dims}
\else
\set embed_dims 2048
\endif
\if :{?embed_model}
\else
\set embed_model voyage-4-large
\endif

INSERT INTO synapse_meta (key, value)
VALUES ('embed_dims', :'embed_dims'),
       ('embed_model', :'embed_model')
ON CONFLICT (key) DO NOTHING;
