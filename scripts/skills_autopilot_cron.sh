#!/usr/bin/env bash
# Cron entry for scripts/skills_autopilot.py that keeps the database password out of the
# crontab: the DSN and the verifier's key are read from env files at run time.
#
#   skills_autopilot_cron.sh <db-env-file> <llm-env-file> [skills_autopilot.py args...]
#
#   <db-env-file>   KEY=VALUE file; only SYNAPSE_DB_URL is read from it
#   <llm-env-file>  KEY=VALUE file; only SYNAPSE_LLM_API_KEY, else OPENROUTER_API_KEY, is read
#
# Nothing else in those files reaches the run. A service .env can also carry a claude CLI
# token, which would replace this machine's own `claude` login for the reviewer; sourcing
# the whole file (set -a; . file) would pass it on.
#
# The verifier runs through ingestion.llm_client on an OpenAI-compatible provider:
# SYNAPSE_LLM_PROVIDER (default openai), SYNAPSE_LLM_BASE_URL (default OpenRouter) and
# SYNAPSE_SKILLS_VERIFY_MODEL (default z-ai/glm-5.2, the verifier the calibration ran). Set
# in the environment, those three win. Optional: SYNAPSE_DIR (default: the checkout holding
# this script), UV (default: uv on PATH). $HOME/.local/bin goes on PATH, since cron's PATH
# usually lacks the per-user installs of claude and uv.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <db-env-file> <llm-env-file> [skills_autopilot.py args...]" >&2
  exit 64
fi
db_env=$1
llm_env=$2
shift 2

# The value of KEY in a KEY=VALUE file: the last assignment, an optional leading 'export',
# one pair of surrounding quotes and a trailing CR stripped. Empty when absent.
read_key() {
  local line
  line=$(grep -E "^(export[[:space:]]+)?$2=" "$1" | tail -n 1) || true
  line=${line#export }
  line=${line#"$2"=}
  line=${line%$'\r'}
  case $line in
    \"*\" | \'*\') line=${line:1:${#line}-2} ;;
  esac
  printf '%s' "$line"
}

SYNAPSE_DB_URL=$(read_key "$db_env" SYNAPSE_DB_URL)
if [ -z "$SYNAPSE_DB_URL" ]; then
  echo "skills_autopilot_cron: no SYNAPSE_DB_URL in $db_env" >&2
  exit 78
fi
SYNAPSE_LLM_API_KEY=$(read_key "$llm_env" SYNAPSE_LLM_API_KEY)
if [ -z "$SYNAPSE_LLM_API_KEY" ]; then
  SYNAPSE_LLM_API_KEY=$(read_key "$llm_env" OPENROUTER_API_KEY)
fi
if [ -z "$SYNAPSE_LLM_API_KEY" ]; then
  echo "skills_autopilot_cron: no SYNAPSE_LLM_API_KEY or OPENROUTER_API_KEY in $llm_env" >&2
  exit 78
fi
export SYNAPSE_DB_URL SYNAPSE_LLM_API_KEY
export SYNAPSE_LLM_PROVIDER="${SYNAPSE_LLM_PROVIDER:-openai}"
export SYNAPSE_SKILLS_VERIFY_MODEL="${SYNAPSE_SKILLS_VERIFY_MODEL:-z-ai/glm-5.2}"
export PATH="$HOME/.local/bin:$PATH"

cd "${SYNAPSE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
echo "$(date -Is) skills autopilot ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
exec "${UV:-uv}" run python scripts/skills_autopilot.py "$@"
