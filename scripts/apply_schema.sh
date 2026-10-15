#!/usr/bin/env bash
# Apply the Synapse schema, in order, to a Postgres database.
#
# THE single source of truth for schema file order: every numbered
# schema/NNN_*.sql, ascending. All consumers run this same script so the tested
# path and the shipped path can't drift:
#   * CI's ephemeral test-DB provision (.github/workflows/ci.yml)
#   * first container boot (init/01_apply_schema.sh via docker-entrypoint-initdb.d)
#   * manual runs against an existing database (docs/install.md "Upgrading", `make schema`)
#
# Usage:
#   scripts/apply_schema.sh [--all | --from NNN] [DSN]
#     DSN    e.g. postgresql://synapse:pw@localhost:5432/synapse
#            Defaults to $SYNAPSE_DB_URL. If that is empty too, psql falls back
#            to its standard PG* env vars / local socket — that's how the
#            first-boot init hook connects.
#     (none) applies only the files numbered ABOVE synapse_meta.schema_version, the
#            stamp this script writes (a fresh database gets every file). An
#            upgrade is therefore one command that doesn't repeat what was applied.
#     --all  applies every file regardless of the stamp. Every file is safe to
#            re-run, but it repeats their backfills too (056's), so expect it to
#            take a while on a large database.
#     --from NNN  applies the files numbered NNN and above. Needed once for a
#            database that has tables but no stamp (it predates migration 034),
#            and to catch up a migration numbered below the stamp.
#   SCHEMA_DIR=/path  overrides the schema directory (default: <repo>/schema;
#            the first-boot hook sets /schema, a read-only mount).
#
# Every file runs with ON_ERROR_STOP=1. Migrations are applied by number, so a
# new one must be numbered above every migration already shipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="${SCHEMA_DIR:-$SCRIPT_DIR/../schema}"

ALL=0
FROM=""
POSITIONAL=()
while (( $# > 0 )); do
  case "$1" in
    --all) ALL=1; shift ;;
    --from)
      FROM="${2:-}"
      if ! [[ "$FROM" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --from needs a migration number, e.g. --from 062." >&2
        exit 1
      fi
      shift 2
      ;;
    -*) echo "ERROR: unknown option $1 (see the usage in $0)." >&2; exit 1 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if (( ALL )) && [[ -n "$FROM" ]]; then
  echo "ERROR: --all and --from are mutually exclusive." >&2
  exit 1
fi
DSN="${POSITIONAL[0]:-${SYNAPSE_DB_URL:-}}"

run_psql() {
  if [[ -n "$DSN" ]]; then
    psql "$DSN" -v ON_ERROR_STOP=1 "$@"
  else
    psql -v ON_ERROR_STOP=1 "$@"
  fi
}

# Every migration, in order. Zero-padded NNN_ prefixes make the glob's order the
# numeric order; a .sql that doesn't follow the pattern would silently never run.
shopt -s nullglob
ALL_FILES=()
for f in "$SCHEMA_DIR"/*.sql; do
  b="$(basename "$f")"
  if ! [[ "$b" =~ ^[0-9]{3}_.+\.sql$ ]]; then
    echo "ERROR: $b in $SCHEMA_DIR is not named NNN_name.sql, so it would never be applied." >&2
    exit 1
  fi
  ALL_FILES+=("$b")
done
if (( ${#ALL_FILES[@]} == 0 )); then
  echo "ERROR: no NNN_*.sql files in $SCHEMA_DIR." >&2
  exit 1
fi

# First file number to apply: 0 for everything, else one past the database's stamp.
if (( ALL )); then
  FIRST=0
elif [[ -n "$FROM" ]]; then
  FIRST=$((10#$FROM))
else
  read -r HAS_TABLES HAS_META < <(
    run_psql -qAtX -F ' ' \
      -c "SELECT to_regclass('episodes') IS NOT NULL, to_regclass('synapse_meta') IS NOT NULL"
  )
  if [[ "$HAS_TABLES" == "f" ]]; then
    FIRST=0
  else
    STAMP=""
    if [[ "$HAS_META" == "t" ]]; then
      STAMP="$(run_psql -qAtX -c "SELECT value FROM synapse_meta WHERE key = 'schema_version'")"
    fi
    if ! [[ "$STAMP" =~ ^[0-9]+$ ]]; then
      echo "ERROR: this database has tables but no schema_version stamp (it predates the stamp," >&2
      echo "or init was interrupted). Say where to start: --from NNN, or --all for a full re-run." >&2
      exit 1
    fi
    FIRST=$((10#$STAMP + 1))
    echo "==> database is at schema $STAMP; applying the newer files"
  fi
fi

FILES=()
for f in "${ALL_FILES[@]}"; do
  if (( 10#${f%%_*} >= FIRST )); then
    FILES+=("$f")
  fi
done
if (( ${#FILES[@]} == 0 )); then
  echo "Schema is current (newest file ${ALL_FILES[-1]%%_*}); nothing to apply."
  exit 0
fi

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
  run_psql -v embed_dims="$EMBED_DIMS" -v embed_model="$EMBED_MODEL" -f "$SCHEMA_DIR/$f"
done

# Stamp the applied schema version (synapse_meta is created by 034) so
# services can verify at boot that the database matches the code — see
# ingestion/schema_check.py. Value = numeric prefix of the newest migration.
SCHEMA_VERSION="${FILES[-1]%%_*}"
run_psql -c "INSERT INTO synapse_meta (key, value) VALUES ('schema_version', '${SCHEMA_VERSION}')
  ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;"

echo "Schema apply complete (${#FILES[@]} files, embed dims ${EMBED_DIMS}, stamped schema_version=${SCHEMA_VERSION})."
