# Ready-made `.env` files

One complete `.env` per deployment shape. Copy the one that matches your accounts, then fill
in every `CHANGEME` and the blanks it marks as required, then follow the normal
[quick start](../../README.md#quick-start). Each header lists the accounts it needs and the
one setting you cannot change after the database's first boot (`SYNAPSE_EMBED_DIMS`).
[`.env.example`](../../.env.example) stays the full annotated reference for every variable.

- `voyage.env`: the default. Voyage AI for embeddings and rerank, a Claude subscription token or an Anthropic key for extraction.
- `local.env`: no retrieval signups. The bundled `local-inference` container does embeddings and rerank; extraction still uses the Claude token.
- `openrouter.env`: no Claude subscription. Voyage for retrieval, OpenRouter for extraction.

```bash
cp examples/env/local.env .env    # from the repo root
```
