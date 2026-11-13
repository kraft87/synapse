#!/usr/bin/env bash
# Bring a Postgres database to the current Synapse schema: apply, in order, every file in
# schema/ that the database has not had yet, and record each one.
#
# THE single source of truth for schema file order. All three consumers run
# this same script so the tested path and the shipped path can't drift:
#   * CI's ephemeral test-DB provision (.github/workflows/ci.yml)
#   * first container boot (init/01_apply_schema.sh via docker-entrypoint-initdb.d)
#   * upgrades of an existing database (docs/install.md > Upgrading)
#
# Usage:
#   scripts/apply_schema.sh [DSN]             apply what is pending
#   scripts/apply_schema.sh --status [DSN]    print what is applied and what is pending;
#                                             changes nothing
#   scripts/apply_schema.sh --only NNN [DSN]  apply that one file (how a file marked manual
#                                             is run on a database that holds episodes)
#     DSN    e.g. postgresql://synapse:pw@localhost:5432/synapse
#            Defaults to $SYNAPSE_DB_URL. If that is empty too, psql falls back
#            to its standard PG* env vars / local socket — that's how the
#            first-boot init hook connects.
#   SCHEMA_DIR=/path  overrides the schema directory (default: <repo>/schema;
#            the first-boot hook sets /schema, a read-only mount).
#   Exit status: 0 done, 2 stopped before a manual file (below), anything else an error.
#
# THE LEDGER. public.schema_migrations holds one row per applied file, keyed by the file's
# number. This script creates the table, skips every file that has a row, and writes a
# file's row in the psql session that ran it. So one command serves an empty database,
# where every file runs, and an upgrade, where only the new ones do: no file is applied
# twice (001, 004 and 023 cannot be).
#   * A database from before the ledger carries only synapse_meta.schema_version, one
#     high-water mark. Its first run records the files up to that number without running
#     them (applied_at NULL: taken on trust from the stamp) and applies the rest.
#   * The stamp is still written after every file: the highest number recorded, never
#     lowered. Builds from before the ledger read it at boot, and
#     ingestion/schema_check.py still does on a database that has no ledger.
#   * To run a file again, delete its row. To record a file applied some other way, insert
#     its row (version, filename).
#
# MARKERS, each a comment line of its own in a schema file:
#   -- schema-check: optional   the services start without the file
#                               (ingestion/schema_check.py).
#   -- schema-apply: manual     on a database that holds episodes the file needs an
#                               operator first (services stopped, a dump taken: its header
#                               says what). An upgrade does not run it: an optional one is
#                               left alone and named, a required one stops the run there
#                               (exit 2). Run it with --only NNN. A database with no
#                               episodes (a first boot, also one that is resumed after a
#                               failure, or a scratch database) runs it like any other file.
#
# Every file runs under psql autocommit with ON_ERROR_STOP=1 and -X (a ~/.psqlrc that
# turns AUTOCOMMIT off would roll a file back, silently, when psql exits). Never wrap a
# file in a transaction: some run their own (067).
set -euo pipefail

# Associative arrays, and empty arrays under `set -u` (macOS ships bash 3.2).
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  echo "ERROR: apply_schema.sh needs bash 4.4 or newer; this is bash $BASH_VERSION." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="${SCHEMA_DIR:-$SCRIPT_DIR/../schema}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

MODE=apply
ONLY=""
DSN="${SYNAPSE_DB_URL:-}"
while (($#)); do
  case "$1" in
    --status) MODE=status ;;
    --only)
      [[ $# -ge 2 && "$2" =~ ^[0-9]{3}$ ]] || die "--only takes a schema file's number, e.g. --only 069"
      ONLY="$2"
      shift
      ;;
    -*) die "unknown option $1 (usage: apply_schema.sh [--status | --only NNN] [DSN])" ;;
    *) if [[ -n "$1" ]]; then DSN="$1"; fi ;;
  esac
  shift
done

# Ordered list of every migration. CI's old inline provision list omitted
# 006-010 and 014; that only worked because the test suite doesn't exercise
# those paths — they are NOT superseded, the runtime needs all of them:
#   006  episodes.retrieval_count       — read+bumped by mcp_server/recall.py
#   007  drops session_summaries        — real migration (001 creates it, 004
#                                         alters it, 007 retires it)
#   008  chunks                         — written by the poller (ingestion/db.py)
#   009  synth_documents                — joined by mcp_server
#                                         (generation retired, table still read)
#   014  halfvec HNSW indexes           — instant on an empty DB; IF NOT EXISTS
#                                         no-op where already built. On a large
#                                         populated DB build CONCURRENTLY by
#                                         hand instead (see the file header).
#   067  one memory                     — runs two explicit transactions of its own (the
#                                         merge, then the tail), so like every file here it
#                                         is applied under psql autocommit: never wrap it in
#                                         BEGIN or pass -1/--single-transaction. On a
#                                         populated database pipe it ALONE with the pollers
#                                         stopped (see the file header); here, on an empty
#                                         one, it only builds the indexes and the tail.
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
  040_dedup_gate_shadow.sql
  041_notes.sql
  042_dashboard.sql
  043_dash_notify.sql
  044_dream_runs.sql
  045_skills_v2_foundation.sql
  046_recall_feedback.sql
  047_drop_memory_proposals.sql
  048_feedback_found_via.sql
  049_recall_metrics_notes.sql
  050_private_sessions.sql
  051_notes_curation.sql
  052_remember_intents.sql
  053_audience_scoping.sql
  054_credential_bound_trust.sql
  055_skill_proposal_base_hash.sql
  056_kg_source_projects.sql
  057_own_surface_episodes.sql
  058_kg_evidence_currency.sql
  059_drop_dedup_gate_shadow.sql
  060_skill_retired_at.sql
  061_episodes_project_recency_index.sql
  062_skill_review_queue.sql
  063_web_audience.sql
  064_skills_autopilot.sql
  065_config_lane_identity.sql
  066_kg_fact_audience.sql
  067_kg_one_memory.sql
  068_span_id_prefix_is_platform.sql
  069_one_memory_cleanup.sql
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

# The ledger is keyed by a file's number, a database from before the ledger is read by
# "every file up to its stamp", and a name is written into SQL below. So: plain names, one
# file per number, numbers rising down the list.
declare -A FILE_OF=()
prev=""
for f in "${FILES[@]}"; do
  [[ "$f" =~ ^([0-9]{3})_[A-Za-z0-9_]+\.sql$ ]] || die "$f is not named NNN_name.sql."
  v="${BASH_REMATCH[1]}"
  [[ -z "${FILE_OF[$v]:-}" ]] || die "${FILE_OF[$v]} and $f share the number $v: renumber the newer one."
  [[ -z "$prev" || "$v" > "$prev" ]] || die "$f is listed after a file numbered $prev: FILES must rise."
  FILE_OF[$v]="$f"
  prev="$v"
done
NEWEST="$prev"
[[ -z "$ONLY" || -n "${FILE_OF[$ONLY]:-}" ]] || die "--only $ONLY: no schema file has that number."

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

PSQL=(psql -X -v ON_ERROR_STOP=1)
if [[ -n "$DSN" ]]; then PSQL+=(-d "$DSN"); fi

# One value, or rows as a|b|c, from the target database.
ask() { "${PSQL[@]}" -qAt -c "$1"; }

# marked FILE WORDS: the file carries "-- WORDS" on a line of its own.
marked() { grep -qE "^[[:space:]]*-- $2[[:space:]]*\$" "$SCHEMA_DIR/$1"; }
optional() { marked "$1" "schema-check: optional"; }
manual() { marked "$1" "schema-apply: manual"; }

# --- What the database says it has had --------------------------------------------------
state="$(ask "SELECT to_regclass('public.schema_migrations') IS NOT NULL,
                     to_regclass('public.synapse_meta') IS NOT NULL,
                     to_regclass('public.episodes') IS NOT NULL")"
IFS='|' read -r HAS_LEDGER HAS_META HAS_SCHEMA <<<"$state"

# The schema_version stamp, or nothing (synapse_meta arrives with 034).
stamp() {
  if [[ "$(ask "SELECT to_regclass('public.synapse_meta') IS NOT NULL")" == t ]]; then
    ask "SELECT value FROM public.synapse_meta WHERE key = 'schema_version'"
  fi
}
STAMP=""
if [[ "$HAS_META" == t ]]; then STAMP="$(stamp)"; fi

declare -A DONE=() WHEN=()
load_ledger() {
  DONE=()
  WHEN=()
  local rows v f at
  rows="$(ask "SELECT version, filename, COALESCE(to_char(applied_at, 'YYYY-MM-DD'), '')
                 FROM public.schema_migrations ORDER BY version")"
  while IFS='|' read -r v f at; do
    [[ -n "$v" ]] || continue
    DONE[$v]="$f"
    WHEN[$v]="$at"
  done <<<"$rows"
}
if [[ "$HAS_LEDGER" == t ]]; then load_ledger; fi

# Data to be careful with: what the manual marker is about.
HOLDS_DATA=f
if [[ "$HAS_SCHEMA" == t ]]; then HOLDS_DATA="$(ask "SELECT EXISTS (SELECT 1 FROM public.episodes)")"; fi

# A database from before the ledger: no rows, only the stamp. Every file up to the stamp
# is taken as applied.
TRUST_UPTO=""
if ((${#DONE[@]} == 0)) && [[ "$STAMP" =~ ^[0-9]{3}$ ]]; then TRUST_UPTO="$STAMP"; fi

# The same number under another name: two branches each took it, and the database has had
# only one of the two files.
CLASH=()
for f in "${FILES[@]}"; do
  v="${f%%_*}"
  if [[ -n "${DONE[$v]:-}" && "${DONE[$v]}" != "$f" ]]; then
    CLASH+=("$v is recorded as ${DONE[$v]}, and this checkout's $v is $f")
  fi
done

if [[ "$MODE" == status ]]; then
  applied=0
  trusted=0
  pending=()
  for f in "${FILES[@]}"; do
    v="${f%%_*}"
    if [[ -n "${DONE[$v]:-}" ]]; then
      applied=$((applied + 1))
      if [[ -z "${WHEN[$v]}" ]]; then trusted=$((trusted + 1)); fi
    elif [[ -n "$TRUST_UPTO" && ! "$v" > "$TRUST_UPTO" ]]; then
      applied=$((applied + 1))
      trusted=$((trusted + 1))
    else
      note=""
      if optional "$f"; then note+=", optional"; fi
      if manual "$f"; then note+=", manual: --only $v"; fi
      pending+=("$f${note:+ (${note#, })}")
    fi
  done
  echo "schema_version stamp: ${STAMP:-none}"
  if ((${#DONE[@]})); then
    echo "ledger: ${#DONE[@]} rows"
  elif [[ -n "$TRUST_UPTO" ]]; then
    echo "ledger: none yet (the first apply records the files up to $TRUST_UPTO from the stamp)"
  else
    echo "ledger: none yet"
  fi
  echo "applied: $applied of ${#FILES[@]} files ($trusted recorded without a run)"
  echo "pending: ${#pending[@]}"
  for p in "${pending[@]}"; do echo "  - $p"; done
  for v in "${!DONE[@]}"; do
    if [[ -z "${FILE_OF[$v]:-}" ]]; then echo "recorded, not in this checkout: $v ${DONE[$v]}"; fi
  done | sort
  for c in "${CLASH[@]}"; do echo "CLASH: $c"; done
  exit 0
fi

# --- Apply ------------------------------------------------------------------------------
for c in "${CLASH[@]}"; do echo "ERROR: $c" >&2; done
((${#CLASH[@]} == 0)) || die "renumber the file the database has not had, then run this again."

if ((${#DONE[@]} == 0)) && [[ "$HAS_SCHEMA" == t ]]; then
  # A schema, and nothing to say which files made it.
  [[ -n "$STAMP" ]] || die "this database has a schema but no ledger and no schema_version stamp, so there is no telling which files it has had. An install whose first boot was interrupted holds nothing: remove its data volume and start it again. Otherwise work out the last file it has had and stamp that number first (docs/install.md > Upgrading)."
  [[ -n "$TRUST_UPTO" ]] || die "the schema_version stamp is '$STAMP', not a schema file's number. Set it to the last file this database has had (docs/install.md > Upgrading)."
  [[ ! "$TRUST_UPTO" > "$NEWEST" ]] || die "this database is at schema $TRUST_UPTO, newer than this checkout (its newest file is $NEWEST). Run the script from a checkout at least as new, so the ledger starts complete."
fi

if [[ "$HAS_LEDGER" != t || -n "$TRUST_UPTO" ]]; then
  sql="
    CREATE TABLE IF NOT EXISTS public.schema_migrations (
        version    TEXT PRIMARY KEY,
        filename   TEXT NOT NULL,
        applied_at TIMESTAMPTZ
    );
    COMMENT ON TABLE public.schema_migrations IS
        'One row per applied schema file. Kept by scripts/apply_schema.sh.';
    COMMENT ON COLUMN public.schema_migrations.applied_at IS
        'When apply_schema.sh ran the file. NULL: recorded without a run (taken from the schema_version stamp when the ledger began, or inserted by hand).';"
  if [[ -n "$TRUST_UPTO" ]]; then
    echo "==> no ledger yet: recording the files up to $TRUST_UPTO from the schema_version stamp"
    for f in "${FILES[@]}"; do
      v="${f%%_*}"
      [[ ! "$v" > "$TRUST_UPTO" ]] || break
      sql+="
    INSERT INTO public.schema_migrations (version, filename) VALUES ('$v', '$f');"
    done
  fi
  "${PSQL[@]}" -q -1 -c "$sql"
  load_ledger
fi

if [[ -n "$ONLY" ]]; then
  if [[ -n "${DONE[$ONLY]:-}" ]]; then
    echo "${FILE_OF[$ONLY]} is already recorded as applied: nothing to do. (To run it again, delete its row from schema_migrations.)"
    exit 0
  fi
  for f in "${FILES[@]}"; do
    v="${f%%_*}"
    [[ "$v" != "$ONLY" ]] || break
    # In order: the one thing --only may step over is a file an upgrade leaves alone.
    if [[ -z "${DONE[$v]:-}" ]] && ! { optional "$f" && manual "$f"; }; then
      die "--only $ONLY: $f comes before it and has not been applied. Run the script without --only first; it applies the files in order."
    fi
  done
fi

# Written in the session that ran the file, right after it: the file's row, then the stamp.
# The stamp follows the ledger and never moves down; synapse_meta arrives with 034.
STAMP_SQL='DO $stamp$
BEGIN
    IF to_regclass($$public.synapse_meta$$) IS NOT NULL THEN
        INSERT INTO public.synapse_meta AS meta (key, value)
        SELECT $$schema_version$$, max(version) FROM public.schema_migrations
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
         WHERE meta.value < EXCLUDED.value;
    END IF;
END
$stamp$'

applied=0
left=()
stopped=""
for f in "${FILES[@]}"; do
  v="${f%%_*}"
  [[ -z "${DONE[$v]:-}" ]] || continue
  if [[ -n "$ONLY" ]]; then
    [[ "$v" == "$ONLY" ]] || continue
  elif [[ "$HOLDS_DATA" == t ]] && manual "$f"; then
    if optional "$f"; then
      echo "==> $f: left alone (optional and manual; apply it with --only $v)"
      left+=("$v")
      continue
    fi
    stopped="$f"
    break
  fi
  echo "==> $f"
  "${PSQL[@]}" -v embed_dims="$EMBED_DIMS" -v embed_model="$EMBED_MODEL" \
    -f "$SCHEMA_DIR/$f" \
    -c "INSERT INTO public.schema_migrations (version, filename, applied_at) VALUES ('$v', '$f', now())" \
    -c "$STAMP_SQL"
  applied=$((applied + 1))
done

if [[ -n "$stopped" ]]; then
  cat >&2 <<EOF
STOPPED before $stopped ($applied file(s) applied before it).
It is marked manual, and this database holds episodes. Its header says what it needs
first (docs/install.md > Upgrading has the steps); then apply it on its own:
    scripts/apply_schema.sh --only ${stopped%%_*} [DSN]
and run this once more for the files after it.
EOF
  exit 2
fi

STAMP="$(stamp)"
echo "Schema apply complete ($applied applied now, embed dims ${EMBED_DIMS}, schema_version=${STAMP:-none})."
if ((${#left[@]})); then echo "Left for --only: ${left[*]} (optional and manual)."; fi
