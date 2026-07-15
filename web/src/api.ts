// API layer for the phase-1 dashboard. Wire shapes match docs/dashboard-contract.md
// (owned by the server agent). Everything is Bearer-token authenticated except the
// static shell. When location.hash contains "mock", the mock dataset is served instead
// so every screen renders offline.
import { mockApi, mockFlag, mockRecall, MOCK } from './mock';

export { MOCK };

const BASE = '/dash/api';
const TOKEN_KEY = 'synapse.token';

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (t: string) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

export class AuthError extends Error {}

// A 401 anywhere kicks the app back to the login screen.
let authFailHandler: () => void = () => {};
export const onAuthFail = (fn: () => void) => { authFailHandler = fn; };

async function reqAt<T>(fullPath: string, init?: RequestInit, tokenOverride?: string): Promise<T> {
  const res = await fetch(fullPath, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      Authorization: 'Bearer ' + (tokenOverride ?? getToken() ?? ''),
    },
  });
  if (res.status === 401) { authFailHandler(); throw new AuthError('unauthorized'); }
  if (!res.ok) throw new Error('request failed (' + res.status + '): ' + fullPath);
  return res.json() as Promise<T>;
}

// Most endpoints live under /dash/api; POST /recall is a top-level route (same bearer),
// so it goes through reqAt with an absolute path.
const req = <T>(path: string, init?: RequestInit, tokenOverride?: string): Promise<T> =>
  reqAt<T>(BASE + path, init, tokenOverride);

// ---------- wire types ----------
export interface Catalog {
  projects: { name: string; count: number }[];
  sources: { name: string; count: number }[];
  group_ids: string[];
}

export type FeedType = 'episode' | 'fact' | 'timeline_event';
export interface FeedItem {
  type: FeedType;
  id: string;
  ts: string;
  project?: string | null;
  source?: string | null;
  group_id?: string | null;
  gist: string;
  flagged: boolean;
  sal?: number;
  data: {
    session_id?: string;
    sequence?: number;
    fact?: string;
    src_name?: string;
    tgt_name?: string;
    t_valid?: string;
    t_invalid?: string | null;
    provenance_episode_id?: number | null;
    source?: string;
    episode_id?: number | null;
  };
}
export interface Feed { items: FeedItem[]; next_cursor: string | null; }

export interface Episode {
  id: number; session_id: string; sequence: number; project: string; source: string;
  platform: string; model: string; created_at: string; flagged: boolean;
  human_turn: string; assistant_turn: string; content: string;
}
export interface Derived {
  facts: { uuid: string; fact: string; group_id: string; t_valid: string; t_invalid: string | null }[];
  timeline_events: { id: number; fact: string; t_valid: string; salience: number }[];
}
export interface SessionEpisode {
  id: number; sequence: number; created_at: string;
  human_turn: string; assistant_turn: string; content: string;
}
export interface Session {
  session_id: string; project: string; source: string; highlight: number | null;
  episodes: SessionEpisode[];
}
export interface EntityFact {
  uuid: string; fact: string; name: string; t_valid: string; t_invalid: string | null;
  other: { uuid: string; name: string }; provenance_episode_id: number | null; flagged: boolean;
}
export interface Entity {
  entity: { uuid: string; name: string; entity_type: string; summary: string; degree: number; created_at: string };
  stats: { edges: number; served: number; facts: number };
  facts: EntityFact[];
  mentions: { items: { episode_id: number; created_at: string; gist: string }[]; offset: number; limit: number; total: number };
}
export type SearchTab = 'episodes' | 'facts' | 'entities' | 'events';
export interface SearchHit {
  type: SearchTab; id: string; snippet: string;
  meta: { project?: string; source?: string; ts?: string; session_id?: string; episode_id?: number | null; name?: string; entity_type?: string; degree?: number };
}
export interface SearchResult {
  hits: SearchHit[];
  total_by_type: Record<SearchTab, number>;
  offset: number; limit: number;
}
export interface FlagRow { id: number; kind: string; item_id: string; note: string | null; created_at: string; gist: string; }

// ---------- proposals (phase 2b) ----------
export type ProposalKind = 'skill' | 'config-edit';
export interface ProposalSummary {
  id: string; kind: ProposalKind; name: string; gist: string;
  status: string; age_days: number; created_at: string;
}
export interface ProposalList { proposals: ProposalSummary[]; pending_count: number; }
export interface EvidenceItem {
  session_id?: string; ts?: string; class?: string; signal?: string;
  why?: string; phrasing?: string; quote?: string; note?: string; [k: string]: unknown;
}
export interface ProposalDetail {
  id: string; kind: ProposalKind; name: string; status: string;
  evidence: string | EvidenceItem[]; provenance_episodes: number[];
  payload: { type: 'markdown' | 'diff'; content: string };
  audit_log: { ts: string | null; action: string; note: string | null }[];
}

const qs = (params: Record<string, string | number | undefined | null>): string => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '' && v !== 'all') p.set(k, String(v));
  }
  const s = p.toString();
  return s ? '?' + s : '';
};

// ---------- endpoints ----------
export interface FeedParams { cursor?: string | null; project?: string; group_id?: string; source?: string; limit?: number; }

export const fetchCatalog = (tokenOverride?: string): Promise<Catalog> =>
  MOCK ? mockApi('/catalog') : req('/catalog', undefined, tokenOverride);

export const fetchFeed = (p: FeedParams): Promise<Feed> =>
  MOCK ? mockApi('/feed') : req('/feed' + qs({ cursor: p.cursor, project: p.project, group_id: p.group_id, source: p.source, limit: p.limit ?? 30 }));

export const fetchEpisode = (id: string): Promise<Episode> =>
  MOCK ? mockApi('/episode/' + id) : req('/episode/' + encodeURIComponent(id));

export const fetchDerived = (id: string): Promise<Derived> =>
  MOCK ? mockApi('/episode/' + id + '/derived') : req('/episode/' + encodeURIComponent(id) + '/derived');

export const fetchSession = (id: string, highlight?: string | null): Promise<Session> =>
  MOCK ? mockApi('/session/' + id) : req('/session/' + encodeURIComponent(id) + qs({ highlight }));

export const fetchEntity = (uuid: string, mentionsOffset = 0): Promise<Entity> =>
  MOCK ? mockApi('/entity/' + uuid) : req('/entity/' + encodeURIComponent(uuid) + qs({ mentions_offset: mentionsOffset }));

export interface SearchParams { q: string; type: SearchTab; offset?: number; limit?: number; project?: string; group_id?: string; source?: string; }
export const fetchSearch = (p: SearchParams): Promise<SearchResult> =>
  MOCK ? mockApi('/search?type=' + p.type) : req('/search' + qs({ q: p.q, type: p.type, offset: p.offset ?? 0, limit: p.limit ?? 20, project: p.project, group_id: p.group_id, source: p.source }));

export const fetchFlags = (): Promise<{ flags: FlagRow[] }> =>
  MOCK ? mockApi('/flags') : req('/flags');

export const postFlag = (kind: string, id: string, note?: string): Promise<{ status: string; flagged: boolean }> =>
  MOCK ? Promise.resolve(mockFlag(kind, id)) : req('/flag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, id, note }),
  });

// ---------- recall debug console (phase 2) ----------
// The debug envelope surfaces the SAME per-leg timing / pool / rerank numbers the engine
// already measures for recall_metrics (docs/dashboard-contract.md §Phase 2). legs_ms omits
// a leg the engine did not time (timeline/prefs when disabled) — the waterfall renders it
// as skipped. Served items carry no per-item score in the real payload (only the mock adds
// `score` for demo); the Served column falls back to a muted meta token otherwise.
export interface RecallDebug {
  total_ms: number;
  legs_ms: Record<string, number>;
  pool_sizes: { bm25: number; vector: number; fused: number; kg_candidates: number };
  rerank: { model: string; top_score: number };
  est_tokens: number;
}
export interface RecallResult {
  query: string;
  facts?: { fact: string; date?: string; score?: number }[];
  episodes?: { id?: string; content: string; project?: string; date?: string; role?: string; superseded_by?: string[]; score?: number }[];
  entities?: { name: string; summary: string; score?: number }[];
  communities?: { name?: string; summary?: string; score?: number }[];
  timeline?: { date: string; fact: string; type?: string; salience?: number; score?: number }[];
  preferences?: { pref: string; polarity?: string; since?: string; asserted?: number; score?: number }[];
  web?: { context?: string; excerpt?: string; url?: string; title?: string; date?: string; score?: number }[];
  history?: { previously: string; now: string }[];
  debug?: RecallDebug;
}
export interface RecallParams { query: string; project?: string; group_id?: string; }
export const postRecall = (p: RecallParams): Promise<RecallResult> =>
  MOCK ? mockRecall(p.query) : reqAt('/recall', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // write_feedback intentionally omitted — the route defaults it false, and a debug
    // recall must never bump the retrieval-count feedback signal.
    body: JSON.stringify({ query: p.query, project: p.project, group_id: p.group_id, debug: true, source: 'dashboard' }),
  });

export interface RecallHistoryRow {
  id: number; created_at: string; query: string; source: string | null;
  ms_total: number | null; est_tokens: number | null; rerank_top_score: number | null;
}
export const fetchRecallHistory = (limit = 50): Promise<{ items: RecallHistoryRow[] }> =>
  MOCK ? mockApi('/recall/history') : req('/recall/history' + qs({ limit }));
export const fetchProposals = (params?: { status?: string; kind?: string }): Promise<ProposalList> =>
  MOCK ? mockApi('/proposals') : req('/proposals' + qs({ status: params?.status, kind: params?.kind }));

export const fetchProposal = (id: string): Promise<ProposalDetail> =>
  MOCK ? mockApi('/proposals/' + id) : req('/proposals/' + encodeURIComponent(id));

export const postProposalDecision = (
  id: string, action: 'approve' | 'reject', note?: string,
): Promise<Record<string, unknown>> =>
  MOCK
    ? Promise.resolve({ status: action === 'approve' ? 'accepted' : 'rejected' })
    : req('/proposals/' + encodeURIComponent(id) + '/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, note }),
      });

// ---------- metrics (phase 4) ----------
// Wire shapes pinned in docs/dashboard-contract.md §"Phase 4". Honesty flags carried through:
// corpus rows can be a pg_class estimate (`rows_estimated`), and ingestion has NO queue-depth
// history (only enqueue/hour + completed/hour throughput), so the client never plots a fabricated
// depth series.
export interface MetricsRecallPoint {
  t: string; p50: number | null; p95: number | null; calls: number;
  tokens_p50: number | null; legs_p50: Record<string, number>;
}
export interface MetricsRecall {
  series: MetricsRecallPoint[];
  slowest: { query: string; ms_total: number | null; created_at: string }[];
  score_hist: { lo: number; hi: number; n: number }[];
}
export interface DreamRun {
  id: number; started_at: string; finished_at: string | null; duration_s: number | null;
  ok: boolean | null; stages: Record<string, { ran?: boolean; ok?: boolean }>;
  counts: Record<string, number>;
  samples: { proposals?: { id: string; kind: string; name: string }[] };
  errors: string[];
}
export interface MetricsIngestion {
  queue_depth: number;
  queue: { pending: number; processing: number; failed: number };
  throughput: {
    enqueued_per_hour: { t: string; n: number }[];
    completed_per_hour: { t: string; n: number }[];
  };
  failures: { id: number; episode_id: number | null; error: string; enqueued_at: string; processed_at: string | null; attempts: number }[];
  last_dream: DreamRun | null;
}
export interface MetricsCorpus {
  tables: { name: string; rows: number; rows_estimated: boolean; spark_30d: number[]; delta_30d: number }[];
  by_project: { name: string; n: number }[];
  by_source: { name: string; n: number }[];
}

export const fetchMetricsRecall = (window = '7d'): Promise<MetricsRecall> =>
  MOCK ? mockApi('/metrics/recall') : req('/metrics/recall' + qs({ window }));
export const fetchMetricsIngestion = (window = '48h'): Promise<MetricsIngestion> =>
  MOCK ? mockApi('/metrics/ingestion') : req('/metrics/ingestion' + qs({ window }));
export const fetchMetricsCorpus = (): Promise<MetricsCorpus> =>
  MOCK ? mockApi('/metrics/corpus') : req('/metrics/corpus');
