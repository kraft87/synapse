# --- Base stage -------------------------------------------------------------
# Public bases only: anyone can `docker compose up -d --build` from a fresh
# clone with no registry auth. The build is fully self-contained.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Bake the Claude Code CLI into the image. The extraction pipeline
# (ingestion/llm_client.py) spawns it via the Agent SDK; auth comes from
# CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in the environment at runtime.
# The native installer drops the binary at /root/.local/share/claude/versions/<v>
# (which _find_cli_path discovers) and a launcher at /root/.local/bin/claude
# (the PATH fallback). Unpinned: an image rebuild picks up the current stable,
# same behavior the old host-mounted install had.
RUN curl -fsSL https://claude.ai/install.sh | bash

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
RUN mkdir -p /root/logs

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Pre-fetch the voyage-4-large tokenizer IN THE BASE so cold-start containers
# never block on HuggingFace Hub AND code merges never re-download it (this
# layer previously lived in the runtime stage, where every code change busted
# it — a per-merge HF download on ephemeral CI runners). The voyage SDK
# lazy-loads it on the first `count_tokens()` call; each uncached call does a
# HEAD etag check (~1s wall) before reading the local file. `tokenizers` is
# installed throwaway into system python just to fetch — the runtime uses the
# venv's copy; only the HF_HOME cache files matter here.
ENV HF_HOME=/root/.cache/huggingface
RUN uv pip install --system tokenizers && \
    python3 -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('voyageai/voyage-4-large')"

# --- Builder stage ----------------------------------------------------------
# Installs deps into a venv, then copies app source. Build-only state — caches,
# build tools, anything transient — stays in this stage and never reaches the
# runtime image.
FROM base AS builder

# Install dependencies (cached layer — only rebuilds when uv.lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy app source (.dockerignore strips tests/, scripts/, .git/, docs, caches)
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- Web bundle stage -------------------------------------------------------
# Builds the operator dashboard's single ESM bundle (issue #12). The dashboard
# ships as ONE self-contained bundle — web/dist/{index.html,app.js,assets/*} —
# served by the MCP server's /dash routes; there is no separate web server and
# no CDN, so the client must be baked into this image at build time. A Node
# stage does that here and only the built dist crosses into runtime (no
# node_modules, no toolchain). `npm run build` is tsc --noEmit + the esbuild
# script defined in web/package.json.
FROM node:22-bookworm-slim AS webbuild
WORKDIR /app/web
# Copy manifests first so `npm ci` caches on lockfile changes, not source edits.
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Runtime stage ----------------------------------------------------------
# Fresh from the base. Only the built .venv and the production source dirs
# come across. No uv cache, no tests, no scripts, no .git in the final image.
FROM base

# Without these, GHCR displays the BASE image's description (uv's) on the
# package page — OCI labels inherit from the final stage's ancestor.
LABEL org.opencontainers.image.title="synapse" \
      org.opencontainers.image.description="Persistent memory for AI agents: conversation episodes, a knowledge graph, and a timeline in Postgres, served over MCP." \
      org.opencontainers.image.source="https://github.com/kraft87/synapse" \
      org.opencontainers.image.licenses="MIT"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/ingestion /app/ingestion
COPY --from=builder /app/mcp_server /app/mcp_server
COPY --from=builder /app/dream /app/dream
COPY --from=builder /app/schema /app/schema
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
# The prebuilt dashboard bundle — the /dash routes serve it from here.
COPY --from=webbuild /app/web/dist /app/web/dist

ENV PATH="/app/.venv/bin:$PATH"

# The tokenizer cache ships in the base layer (see base stage). HF_HUB_OFFLINE=1
# tells huggingface_hub to skip the network HEAD entirely on every
# from_pretrained() call — pure local-file load from the read-only image layer.
ENV HF_HUB_OFFLINE=1

# Default: poller (override CMD for mcp-server or dream)
CMD ["python", "-m", "ingestion"]
