// Offline fixture dataset for `#mock` mode — obviously synthetic, no real data
// (this repo is public). Enough to render every phase-1 screen without a server.
export const MOCK =
  typeof location !== 'undefined' &&
  (location.hash.includes('mock') || (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('synapse.mock') === '1'));
if (MOCK && typeof sessionStorage !== 'undefined') sessionStorage.setItem('synapse.mock', '1');

const now = Date.now();
const iso = (hAgo: number) => new Date(now - hAgo * 3600e3).toISOString();
const trace = 'user: wire service A to library B\n\nRead   src/service_a.py\nEdit   src/service_a.py:42\nBash   pytest tests/ — 3 passed';

const FIX: Record<string, unknown> = {
  catalog: {
    projects: [{ name: 'service-a', count: 12 }, { name: 'service-b', count: 8 }, { name: 'untagged', count: 3 }],
    sources: [{ name: 'claude-code', count: 14 }, { name: 'cursor', count: 9 }],
    group_ids: ['technical', 'personal'],
  },
  feed: {
    items: [
      { type: 'episode', id: '1', ts: iso(0.4), project: 'service-a', source: 'claude-code', gist: 'Example episode: wired service A to library B.', flagged: false, data: { session_id: 'sess-1', sequence: 3 } },
      { type: 'fact', id: 'ent-edge-1', ts: iso(0.6), group_id: 'technical', gist: 'example fact: service A depends on library B', flagged: false, data: { fact: 'service A depends on library B', src_name: 'service A', tgt_name: 'library B', t_valid: iso(24), t_invalid: null, provenance_episode_id: 1 } },
      { type: 'timeline_event', id: '10', ts: iso(0.9), project: 'service-a', gist: 'example event: service A shipped v2', flagged: true, sal: 0.9, data: { fact: 'service A shipped v2', t_valid: iso(48), source: 'git:example', episode_id: 1 } },
      { type: 'episode', id: '2', ts: iso(30), project: 'service-b', source: 'cursor', gist: 'Example episode: refactored the widget cache.', flagged: false, data: { session_id: 'sess-2', sequence: 1 } },
    ],
    next_cursor: null,
  },
  'episode/1': { id: 1, session_id: 'sess-1', sequence: 3, project: 'service-a', source: 'claude-code', platform: 'cli', model: 'example-model', created_at: iso(0.4), flagged: false, human_turn: 'Wire service A to library B.', assistant_turn: 'Done — added the dependency and a smoke test.', content: trace },
  'episode/2': { id: 2, session_id: 'sess-2', sequence: 1, project: 'service-b', source: 'cursor', platform: 'ide', model: 'example-model', created_at: iso(30), flagged: false, human_turn: 'Refactor the widget cache.', assistant_turn: 'Extracted a WidgetCache class; hit rate unchanged.', content: 'user: refactor the widget cache\n\nassistant: extracted a WidgetCache class.' },
  derived: { facts: [{ uuid: 'ent-edge-1', fact: 'service A depends on library B', group_id: 'technical', t_valid: iso(24), t_invalid: null }], timeline_events: [{ id: 10, fact: 'service A shipped v2', t_valid: iso(48), salience: 2 }] },
  session: { session_id: 'sess-1', project: 'service-a', source: 'claude-code', highlight: 1, episodes: [{ id: 0, sequence: 1, created_at: iso(1), human_turn: 'Set up the project.', assistant_turn: 'Scaffolded service A.', content: '' }, { id: 1, sequence: 3, created_at: iso(0.4), human_turn: 'Wire service A to library B.', assistant_turn: 'Done — added the dependency.', content: trace }] },
  entity: {
    entity: { uuid: 'ent-1', name: 'service A', entity_type: 'Project', summary: 'Example service used to demonstrate the dossier surface.', degree: 4, created_at: iso(200) },
    stats: { edges: 4, served: 42, facts: 3 },
    facts: [
      { uuid: 'ent-edge-1', fact: 'service A depends on library B', name: 'depends on', t_valid: iso(24), t_invalid: null, other: { uuid: 'ent-2', name: 'library B' }, provenance_episode_id: 1, flagged: false },
      { uuid: 'ent-edge-2', fact: 'service A used an in-memory store', name: 'used', t_valid: iso(400), t_invalid: iso(50), other: { uuid: 'ent-3', name: 'in-memory store' }, provenance_episode_id: 2, flagged: false },
    ],
    mentions: { items: [{ episode_id: 1, created_at: iso(0.4), gist: 'wired service A to library B' }, { episode_id: 2, created_at: iso(30), gist: 'refactored the widget cache' }], offset: 0, limit: 20, total: 2 },
  },
  flags: { flags: [{ id: 1, kind: 'timeline_event', item_id: '10', note: null, created_at: iso(2), gist: 'example event: service A shipped v2' }] },
  proposals: {
    pending_count: 2,
    proposals: [
      { id: 'skill:1', kind: 'skill', name: 'latency-triage', gist: 'Recurring recall latency debugging with no reusable playbook.', status: 'proposed', age_days: 0, created_at: iso(9) },
      { id: 'config:1', kind: 'config-edit', name: 'CLAUDE.md', gist: 'Add the raw-SQL / no-ORM rule the operator restated 5x.', status: 'proposed', age_days: 1, created_at: iso(30) },
      { id: 'skill:2', kind: 'skill', name: 'graph-inspect', gist: 'Two overlapping skills — merge into one.', status: 'accepted', age_days: 2, created_at: iso(52) },
      { id: 'config:2', kind: 'config-edit', name: 'rules/testing.md', gist: 'Proposed mandatory 90% coverage gate.', status: 'rejected', age_days: 4, created_at: iso(100) },
    ],
  },
  'proposals/skill:1': {
    id: 'skill:1', kind: 'skill', name: 'latency-triage', status: 'proposed',
    evidence: [
      { session_id: 'sess-1', class: 'grounded', signal: 'explicit_request', why: 'operator asked for a reusable recall-latency playbook' },
      { session_id: 'sess-2', class: 'grounded', signal: 'user_correction', why: 'repeated the same waterfall read-through by hand' },
    ],
    provenance_episodes: [1, 2],
    payload: { type: 'markdown', content: '# latency-triage\n\nWhen recall p95 regresses, read the waterfall leg-by-leg. Dominant leg → known remedy:\n\n- **rerank** → cap candidate pool (`RERANK_POOL_CAP`)\n- **kg** → depth-limit the neighborhood expand\n- **vector** → check `pgvector` index / cache hit rate\n- **bm25** → review tokenizer + stopwords\n\n```\nrecall --debug "<query>" | jq .legs\n```' },
    audit_log: [],
  },
  'proposals/config:1': {
    id: 'config:1', kind: 'config-edit', name: 'CLAUDE.md', status: 'proposed',
    evidence: [{ session_id: 'sess-3', signal: 'correction', why: 'operator restated "prefer raw SQL over the ORM" five times' }],
    provenance_episodes: [2],
    payload: { type: 'diff', content: '--- a/CLAUDE.md\n+++ b/CLAUDE.md\n@@ -12,3 +12,4 @@\n ## Conventions\n Keep functions small.\n+Prefer raw SQL over the ORM for hot-path reads.\n Log every migration.' },
    audit_log: [],
  },
  'proposals/skill:2': {
    id: 'skill:2', kind: 'skill', name: 'graph-inspect', status: 'accepted',
    evidence: [{ session_id: 'sess-4', class: 'grounded', signal: 'accept', why: 'operator accepted the merge' }],
    provenance_episodes: [],
    payload: { type: 'markdown', content: '# graph-inspect\n\nMerged from **graph-explore** + **kg-inspect** — one skill for reading the KG.' },
    audit_log: [{ ts: iso(48), action: 'proposal_approve', note: 'clear overlap; merge is right' }],
  },
  'proposals/config:2': {
    id: 'config:2', kind: 'config-edit', name: 'rules/testing.md', status: 'rejected',
    evidence: [{ session_id: 'sess-5', signal: 'correction', why: 'proposed a hard coverage gate' }],
    provenance_episodes: [],
    payload: { type: 'diff', content: '--- a/rules/testing.md\n+++ b/rules/testing.md\n@@ -1 +1,2 @@\n # Testing\n+Every PR must hit 90% line coverage.' },
    audit_log: [{ ts: iso(96), action: 'proposal_reject', note: 'too rigid — coverage is a lagging signal' }],
  },
};

// ---- metrics fixtures (phase 4) — obviously synthetic ----
const LEGS = ['embed', 'bm25', 'vector', 'kg', 'timeline', 'prefs', 'web', 'rerank'];
const metricsSeries = Array.from({ length: 24 }, (_, i) => {
  const legs: Record<string, number> = {};
  for (const l of LEGS) legs[l] = Math.round((4 + Math.random() * 40) * (l === 'rerank' ? 3 : l === 'kg' ? 2 : 1));
  const p50 = 140 + Math.round(Math.random() * 120);
  return { t: iso((23 - i) * 2), p50, p95: p50 + 120 + Math.round(Math.random() * 90), calls: 8 + Math.round(Math.random() * 20), tokens_p50: 1200 + Math.round(Math.random() * 900), legs_p50: legs };
});
const FIX_METRICS = {
  recall: {
    series: metricsSeries,
    slowest: [
      { query: 'everything I know about the lisbon trip planning and who is coming', ms_total: 612, created_at: iso(3) },
      { query: 'postgres connection pooling decisions across services', ms_total: 548, created_at: iso(7) },
      { query: 'what did I decide about the rerank pool cap', ms_total: 501, created_at: iso(11) },
      { query: 'embedding backend migration history and dims', ms_total: 470, created_at: iso(19) },
      { query: 'timeline of the homelab disk upgrades', ms_total: 442, created_at: iso(26) },
    ],
    score_hist: [0, 0, 1, 2, 4, 7, 12, 21, 33, 18].map((n, i) => ({ lo: i / 10, hi: (i + 1) / 10, n })),
  },
  ingestion: {
    queue_depth: 6,
    queue: { pending: 6, processing: 1, failed: 2 },
    throughput: {
      enqueued_per_hour: Array.from({ length: 48 }, (_, i) => ({ t: iso(47 - i), n: Math.round(Math.random() * 40) })),
      completed_per_hour: Array.from({ length: 48 }, (_, i) => ({ t: iso(47 - i), n: Math.round(Math.random() * 38) })),
    },
    failures: [
      { id: 91, episode_id: 227168, error: 'extractor timeout after 3 attempts', enqueued_at: iso(5), processed_at: iso(4), attempts: 3 },
      { id: 88, episode_id: 227101, error: 'malformed tool trace — JSON parse error', enqueued_at: iso(9), processed_at: iso(9), attempts: 2 },
    ],
    last_dream: {
      id: 5, started_at: iso(9), finished_at: iso(8.6), duration_s: 41.2, ok: true,
      stages: { skills: { ran: true, ok: true }, config: { ran: true, ok: true } },
      counts: { proposals_raised: 2, config_proposals: 1, config_corrections_found: 4, config_sessions_scanned: 12 },
      samples: { proposals: [{ id: 'skill:12', kind: 'skill', name: 'latency-triage' }, { id: 'config:4', kind: 'config-edit', name: 'rules/learned.md' }] },
      errors: [],
    },
  },
  corpus: {
    tables: [
      { name: 'episodes', rows: 44210, rows_estimated: true, spark_30d: Array.from({ length: 30 }, () => 4 + Math.round(Math.random() * 40)), delta_30d: 640 },
      { name: 'kg_entities', rows: 8123, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 20)), delta_30d: 210 },
      { name: 'kg_relationships', rows: 15980, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 30)), delta_30d: 402 },
      { name: 'timeline_events', rows: 1204, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 8)), delta_30d: 74 },
      { name: 'preferences', rows: 96, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 2)), delta_30d: 9 },
      { name: 'notes', rows: 41, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 2)), delta_30d: 6 },
      { name: 'chunks', rows: 10740, rows_estimated: false, spark_30d: Array.from({ length: 30 }, () => Math.round(Math.random() * 24)), delta_30d: 318 },
    ],
    by_project: [
      { name: 'synapse', n: 3403 }, { name: 'homelab', n: 1290 }, { name: 'transcribe-ai', n: 840 }, { name: 'untagged', n: 512 },
    ],
    by_source: [
      { name: 'claude-code', n: 31000 }, { name: 'cursor', n: 9200 }, { name: 'claude-ai', n: 3600 }, { name: 'transcribe-ai', n: 410 },
    ],
  },
};

// ---- timeline / preferences / dream / behavior fixtures (phase 5) — obviously synthetic ----
const day = (d: string) => new Date(d + 'T12:00:00Z').toISOString();
const FIX_P5 = {
  timeline: {
    events: [
      { id: 3, t_valid: day('2026-07-03'), fact: 'Rerank latency regression fixed — candidate pool capped at 96.', source: 'chat', project: 'synapse', salience: 2, sal: 0.9, event_type: 'work', episode_id: 88412, flagged: false },
      { id: 2, t_valid: day('2026-07-03'), fact: 'SSE Last-Event-ID resume shipped.', source: 'git:example', project: 'synapse', salience: 2, sal: 0.9, event_type: 'work', episode_id: 88412, flagged: false },
      { id: 1, t_valid: day('2026-07-01'), fact: 'Embedding cache moved into Postgres (pgvector table).', source: 'git:example', project: 'synapse', salience: 1, sal: 0.6, event_type: 'work', episode_id: 88101, flagged: false },
      { id: 0, t_valid: day('2026-06-30'), fact: 'Took a proper rest day.', source: 'chat', project: null, salience: 1, sal: 0.6, event_type: 'health', episode_id: null, flagged: true },
    ],
    next_before: null,
  },
  preferences: {
    preferences: [
      { id: 1, pref: 'Wants recall() debug output on by default in dev', polarity: 'like', first_seen: day('2026-06-18'), last_asserted: day('2026-07-10'), assert_count: 3, superseded_by: null, superseded_by_text: null, t_invalid: null, flagged: false },
      { id: 2, pref: 'Avoid CDN dependencies — everything self-hosted', polarity: 'dislike', first_seen: day('2026-04-11'), last_asserted: day('2026-07-02'), assert_count: 6, superseded_by: null, superseded_by_text: null, t_invalid: null, flagged: false },
      { id: 3, pref: 'Prefers boring, well-understood infra over clever abstractions', polarity: 'like', first_seen: day('2026-03-02'), last_asserted: day('2026-06-20'), assert_count: 7, superseded_by: null, superseded_by_text: null, t_invalid: null, flagged: false },
      { id: 4, pref: 'Dislikes ORMs for this project — raw SQL only', polarity: 'dislike', first_seen: day('2026-02-14'), last_asserted: day('2026-06-01'), assert_count: 5, superseded_by: null, superseded_by_text: null, t_invalid: null, flagged: false },
      { id: 5, pref: 'Prefers dark theme everywhere', polarity: 'rule', first_seen: day('2026-01-20'), last_asserted: day('2026-05-11'), assert_count: 9, superseded_by: null, superseded_by_text: null, t_invalid: null, flagged: false },
      { id: 6, pref: 'Liked hosted vector DBs', polarity: 'like', first_seen: day('2025-11-08'), last_asserted: day('2026-01-02'), assert_count: 2, superseded_by: 3, superseded_by_text: 'self-hosted pgvector', t_invalid: day('2026-02-15'), flagged: false },
    ],
  },
  dreamReport: {
    runs: [
      {
        id: 6, started_at: day('2026-07-03'), finished_at: day('2026-07-03'), duration_s: 2460, ok: true,
        stages: { skills: { ran: true, ok: true }, config: { ran: true, ok: true } },
        counts: { facts_extracted: 128, superseded: 14, dedup_merges: 1882, timeline_events: 9, proposals_raised: 3 },
        samples: { proposals: [{ id: 'config:4', kind: 'config-edit', name: 'rules/learned.md' }], facts_extracted: [{ text: 'SSE endpoint — supports → Last-Event-ID resume' }, { text: 'recall() — has parameter → RERANK_POOL_CAP (default 96)' }, { text: 'embedding cache — stored in → embeddings_cache (Postgres)' }] },
        errors: [],
      },
      {
        id: 5, started_at: day('2026-07-02'), finished_at: day('2026-07-02'), duration_s: 1980, ok: true,
        stages: { skills: { ran: true, ok: true }, config: { ran: true, ok: false } },
        counts: { proposals_raised: 1, config_proposals: 1 },
        samples: { proposals: [{ id: 'skill:12', kind: 'skill', name: 'latency-triage' }] },
        errors: ['config lane: proposer LLM call timed out'],
      },
    ],
  },
  behaviorFiles: {
    groups: [
      { name: 'CLAUDE.md', files: [{ file_key: 'CLAUDE.md', surface_id: 'cortex', scope: 'global', updated_at: day('2026-07-10'), size: 4120 }] },
      { name: 'rules', files: [
        { file_key: 'rules/voice.md', surface_id: 'cortex', scope: 'global', updated_at: day('2026-07-08'), size: 980 },
        { file_key: 'rules/memory-protocol.md', surface_id: 'cortex', scope: 'global', updated_at: day('2026-06-30'), size: 1450 },
      ] },
      { name: 'memory notes', files: [{ file_key: 'memory/project_briefing.md', surface_id: 'cortex', scope: 'global', updated_at: day('2026-07-05'), size: 720 }] },
      { name: 'other', files: [{ file_key: 'AGENTS.md', surface_id: 'cortex', scope: 'global', updated_at: day('2026-05-01'), size: 300 }] },
    ],
  },
  behaviorFile: {
    file_key: 'CLAUDE.md',
    content: '# CLAUDE.md\n\nYou are an example operator persona.\n\n- Prefer raw SQL over the ORM (see [[rules/voice.md]]).\n- Memory protocol lives in [[rules/memory-protocol.md]].\n- Standing crons: [[memory/project_briefing.md]].\n',
    meta: { surface_id: 'cortex', scope: 'global', abs_path: '/home/example/.claude/CLAUDE.md', content_hash: 'deadbeef', modified_at: day('2026-07-10'), updated_at: day('2026-07-10'), size: 4120 },
    links: ['rules/voice.md', 'rules/memory-protocol.md', 'memory/project_briefing.md'],
  },
  behaviorLinkgraph: {
    nodes: [
      { file_key: 'CLAUDE.md', scope: 'global', group: 'CLAUDE.md' },
      { file_key: 'rules/voice.md', scope: 'global', group: 'rules' },
      { file_key: 'rules/memory-protocol.md', scope: 'global', group: 'rules' },
      { file_key: 'memory/project_briefing.md', scope: 'global', group: 'memory notes' },
    ],
    edges: [
      { source: 'CLAUDE.md', target: 'rules/voice.md' },
      { source: 'CLAUDE.md', target: 'rules/memory-protocol.md' },
      { source: 'CLAUDE.md', target: 'memory/project_briefing.md' },
      { source: 'rules/memory-protocol.md', target: 'rules/voice.md' },
    ],
  },
};

const SEARCH: Record<string, unknown[]> = {
  episodes: [{ type: 'episodes', id: '1', snippet: 'wired service A to library B', meta: { project: 'service-a', source: 'claude-code', ts: iso(0.4), session_id: 'sess-1', episode_id: 1 } }],
  facts: [{ type: 'facts', id: 'ent-edge-1', snippet: 'service A depends on library B', meta: { project: 'service-a', source: 'claude-code', ts: iso(24), session_id: 'sess-1', episode_id: 1 } }],
  entities: [{ type: 'entities', id: 'ent-1', snippet: 'service A — example project', meta: { name: 'service A', entity_type: 'Project', degree: 4 } }],
  events: [{ type: 'events', id: '10', snippet: 'service A shipped v2', meta: { project: 'service-a', source: 'git:example', ts: iso(48), session_id: 'sess-1', episode_id: 1 } }],
};

// Recall debug console fixtures (synthetic — mirrors the real recall() + debug shape).
const RECALL_HISTORY = {
  items: [
    { id: 5, created_at: iso(0.03), query: 'postgres connection pooling decisions', source: 'dashboard', ms_total: 341, est_tokens: 1874, rerank_top_score: 0.91 },
    { id: 4, created_at: iso(1), query: 'what did I decide about rerank pool size', source: 'dashboard', ms_total: 296, est_tokens: 1560, rerank_top_score: 0.94 },
    { id: 3, created_at: iso(5), query: 'embedding cache implementation history', source: 'mcp', ms_total: 402, est_tokens: 2210, rerank_top_score: 0.87 },
    { id: 2, created_at: iso(28), query: 'widget cache refactor', source: 'http', ms_total: 164, est_tokens: 388, rerank_top_score: 0.81 },
  ],
};

export function mockRecall<T>(query: string): Promise<T> {
  return Promise.resolve({
    query,
    facts: [
      { fact: 'service A depends on library B', date: '2026-06-30', score: 0.88 },
      { fact: 'service A used an in-memory store', date: '2026-05-21', score: 0.43 },
    ],
    episodes: [
      { id: 'e:1', content: 'Wired service A to library B; added a shared asyncpg pool (min 2 / max 10) and a smoke test.', project: 'service-a', date: '2026-07-01', role: 'assistant', score: 0.91 },
      { id: 'e:2', content: 'Refactored the widget cache into a WidgetCache class; hit rate unchanged.', project: 'service-b', date: '2026-06-15', score: 0.71 },
    ],
    entities: [
      { name: 'service A', summary: 'Example project used to demonstrate the recall debug surface.', score: 0.81 },
    ],
    timeline: [
      { date: '2026-06-30', fact: 'service A shipped v2', type: 'milestone', salience: 2, score: 0.74 },
    ],
    preferences: [
      { pref: 'prefers boring, well-understood infra over clever abstractions', polarity: 'like', since: '2026-03-02', asserted: 7, score: 0.62 },
    ],
    web: [],
    debug: {
      total_ms: 341,
      legs_ms: { embed: 12, bm25: 48, vector: 96, kg: 141, web: 4, rerank: 168, timeline: 33, prefs: 9 },
      pool_sizes: { bm25: 100, vector: 100, fused: 100, kg_candidates: 12 },
      rerank: { model: 'rerank-2.5-lite', top_score: 0.91 },
      est_tokens: 1874,
    },
  } as T);
}

export async function mockApi<T>(path: string): Promise<T> {
  const p = path.replace(/^\//, '').split('?')[0];
  let out: unknown;
  if (p === 'catalog') out = FIX.catalog;
  else if (p === 'feed') out = FIX.feed;
  else if (p === 'flags') out = FIX.flags;
  else if (p === 'recall/history') out = RECALL_HISTORY;
  else if (p === 'metrics/recall') out = FIX_METRICS.recall;
  else if (p === 'metrics/ingestion') out = FIX_METRICS.ingestion;
  else if (p === 'metrics/corpus') out = FIX_METRICS.corpus;
  else if (p === 'timeline') out = FIX_P5.timeline;
  else if (p === 'preferences') out = FIX_P5.preferences;
  else if (p === 'dream/report') out = FIX_P5.dreamReport;
  else if (p === 'behavior/files') out = FIX_P5.behaviorFiles;
  else if (p === 'behavior/file') out = FIX_P5.behaviorFile;
  else if (p === 'behavior/linkgraph') out = FIX_P5.behaviorLinkgraph;
  else if (p === 'proposals') out = FIX.proposals;
  else if (/^proposals\//.test(p)) out = FIX['proposals/' + p.substring('proposals/'.length)] || {};
  else if (/^episode\/[^/]+\/derived$/.test(p)) out = FIX.derived;
  else if (/^episode\//.test(p)) out = FIX['episode/' + p.split('/')[1]] || FIX['episode/1'];
  else if (/^session\//.test(p)) out = FIX.session;
  else if (/^entity\//.test(p)) out = FIX.entity;
  else if (p === 'search') {
    const type = (path.split('type=')[1] || 'episodes') as string;
    out = { hits: SEARCH[type] || [], total_by_type: { episodes: 1, facts: 1, entities: 1, events: 1 }, offset: 0, limit: 20 };
  } else out = {};
  return out as T;
}

export const mockFlag = (_kind: string, _id: string) => ({ status: 'ok', flagged: true });
