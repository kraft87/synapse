#!/bin/bash
# First-boot schema init. The postgres image's docker-entrypoint-initdb.d runs
# this ONLY when the data volume is empty (fresh install); existing deployments
# with a populated volume never re-run it.
#
# Delegates to scripts/apply_schema.sh — the single source of truth for schema
# order — which compose mounts at /apply_schema.sh (with ./schema at /schema).
# Connects over the local socket as the bootstrap superuser; no password needed
# during the initdb phase.
set -euo pipefail

echo "synapse init: applying schema from /schema (first boot only)"
SCHEMA_DIR=/schema PGUSER="$POSTGRES_USER" PGDATABASE="$POSTGRES_DB" bash /apply_schema.sh
