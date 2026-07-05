#!/usr/bin/env bash
# Apply the Synapse schema, in order, to a Postgres database.
#
# THE single source of truth for schema file order. All three consumers run
# this same script so the tested path and the shipped path can't drift:
#   * CI's ephemeral test-DB provision (.github/workflows/ci.yml)
#   * first container boot (init/01_apply_schema.sh via docker-entrypoint-initdb.d)
#   * manual runs against an existing database (see README "Upgrading")
#
# Usage:
#   scripts/apply_schema.sh [DSN]
#     DSN    e.g. postgresql://synapse:pw@localhost:5432/synapse
#            Defaults to $SYNAPSE_DB_URL. If that is empty too, psql falls back
#            to its standard PG* env vars / local socket — that's how the
#            first-boot init hook connects.
#   SCHEMA_DIR=/path  overrides the schema directory (default: <repo>/schema;
#            the first-boot hook sets /schema, a read-only mount).
#
# Every file runs with ON_ERROR_STOP=1. Designed for a fresh/empty database;
# most files are idempotent (IF NOT EXISTS / IF EXISTS guards) but a full
# re-run against a large populated database can be slow (e.g. 003's ALTER
# COLUMN TYPE rewrites the episodes table) — for upgrades, prefer applying
# only the files newer than the database's current state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="${SCHEMA_DIR:-$SCRIPT_DIR/../schema}"
DSN="${1:-${SYNAPSE_DB_URL:-}}"

# Ordered list of every migration. CI's old inline provision list omitted
# 006-010 and 014; that only worked because the test suite doesn't exercise
# those paths — they are NOT superseded, the runtime needs all of them:
#   006  episodes.retrieval_count       — read+bumped by mcp_server/recall.py
#   007  drops session_summaries        — real migration (001 creates it, 004
#                                         alters it, 007 retires it)
#   008  chunks                         — written by the poller (ingestion/db.py)
#   009  synth_documents                — joined by mcp_server + dream/stage3
#                                         (generation retired, table still read)
#   010  memory_proposals               — written by dream stage 3
#   014  halfvec HNSW indexes           — instant on an empty DB; IF NOT EXISTS
#                                         no-op where already built. On a large
#                                         populated DB build CONCURRENTLY by
#                                         hand instead (see the file header).
FILES=(
  001_initial.sql
  002_span_id.sql
  003_voyage_dims.sql
  004_session_embeddings.sql
  005_paradedb.sql
  006_feedback_weights.sql
  007_drop_summaries.sql
  008_chunks.sql
  009_synth_documents.sql
  010_memory_proposals.sql
  011_web_artifacts.sql
  012_web_chunks.sql
  013_web_chunk_context.sql
  014_halfvec_hnsw_indexes.sql
  015_extraction_priority.sql
  016_extraction_claimed_at.sql
  017_kg_postgres.sql
  018_web_extraction.sql
  019_kg_mention_count.sql
  020_entity_type_taxonomy.sql
  021_recall_metrics.sql
  022_skill_gap_candidates.sql
  023_skill_ledger_hardening.sql
  024_skill_serving.sql
  025_skill_sync.sql
  026_skill_history.sql
  027_skill_proposal_body.sql
  028_kg_invalidated_by.sql
  029_kg_superseded_episodes_gin.sql
  030_config_lane.sql
  031_config_scope.sql
  032_config_scan_cursor.sql
  033_timeline.sql
  034_embedding_meta.sql
  035_preferences.sql
  036_content_md5_index.sql
  037_timeline_reported_count.sql
  038_timeline_domain.sql
  039_recall_metrics_served_ids.sql
)

# Drift guard: a numbered .sql in schema/ that isn't in the list above means
# someone added a migration without registering it here. Fail loudly.
for f in "$SCHEMA_DIR"/[0-9]*.sql; do
  b="$(basename "$f")"
  case " ${FILES[*]} " in
    *" $b "*) ;;
    *)
      echo "ERROR: $b exists in $SCHEMA_DIR but is not in apply_schema.sh's ordered list." >&2
      echo "Add it to FILES — this script is the single source of truth for schema order." >&2
      exit 1
      ;;
  esac
done

# --- Embedding geometry (pluggable embeddings) ------------------------------
# The schema declares vector/halfvec columns and HNSW index expressions at 2048
# dims (Voyage voyage-4-large, the default). A different embedding backend can
# pick a different width AT PROVISION TIME ONLY: set SYNAPSE_EMBED_DIMS (and
# SYNAPSE_EMBED_MODEL, recorded in synapse_meta) before the first boot. A
# populated database keeps the geometry it was provisioned with — the app
# validates its config against the recorded values and fails loudly on drift.
EMBED_DIMS="${SYNAPSE_EMBED_DIMS:-2048}"
EMBED_MODEL="${SYNAPSE_EMBED_MODEL:-voyage-4-large}"
if ! [[ "$EMBED_DIMS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SYNAPSE_EMBED_DIMS='$EMBED_DIMS' is not a positive integer." >&2
  exit 1
fi
if (( EMBED_DIMS > 4000 )); then
  echo "ERROR: SYNAPSE_EMBED_DIMS=$EMBED_DIMS exceeds pgvector's 4000-dim halfvec HNSW index cap." >&2
  exit 1
fi
if [[ "$EMBED_DIMS" != "2048" ]]; then
  # Substitute the dims into a temp copy of the schema; the checked-in files
  # stay byte-identical for the default path. Covers the three literal forms
  # the schema uses: vector(2048), VECTOR(2048), halfvec(2048).
  echo "==> substituting embedding dims 2048 -> $EMBED_DIMS in schema files"
  SUBST_DIR="$(mktemp -d)"
  trap 'rm -rf "$SUBST_DIR"' EXIT
  for f in "$SCHEMA_DIR"/*.sql; do
    sed -e "s/halfvec(2048)/halfvec(${EMBED_DIMS})/g" \
        -e "s/vector(2048)/vector(${EMBED_DIMS})/g" \
        -e "s/VECTOR(2048)/VECTOR(${EMBED_DIMS})/g" \
        "$f" > "$SUBST_DIR/$(basename "$f")"
  done
  SCHEMA_DIR="$SUBST_DIR"
fi

for f in "${FILES[@]}"; do
  echo "==> $f"
  if [[ -n "$DSN" ]]; then
    psql "$DSN" -v ON_ERROR_STOP=1 \
      -v embed_dims="$EMBED_DIMS" -v embed_model="$EMBED_MODEL" \
      -f "$SCHEMA_DIR/$f"
  else
    psql -v ON_ERROR_STOP=1 \
      -v embed_dims="$EMBED_DIMS" -v embed_model="$EMBED_MODEL" \
      -f "$SCHEMA_DIR/$f"
  fi
done

# Stamp the applied schema version (synapse_meta is created by 034) so
# services can verify at boot that the database matches the code — see
# ingestion/schema_check.py. Value = numeric prefix of the newest migration.
SCHEMA_VERSION="${FILES[-1]%%_*}"
STAMP_SQL="INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '${SCHEMA_VERSION}')
  ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;"
if [[ -n "$DSN" ]]; then
  psql "$DSN" -v ON_ERROR_STOP=1 -c "$STAMP_SQL"
else
  psql -v ON_ERROR_STOP=1 -c "$STAMP_SQL"
fi

echo "Schema apply complete (${#FILES[@]} files, embed dims ${EMBED_DIMS}, stamped schema_version=${SCHEMA_VERSION})."
