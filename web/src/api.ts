// API layer for the phase-1 dashboard. Wire shapes match docs/dashboard-contract.md
// (owned by the server agent). Everything is Bearer-token authenticated except the
// static shell. When the page is opened at "#mock", the mock dataset is served instead
// so every screen renders offline.
import { MOCK } from './mode';
import { browserTz } from './tokens';

export { MOCK };

// The #mock fixtures load on demand (their own chunk), so a normal page never fetches them.
const mock = () => import('./mock');
const mockApi = <T,>(path: string): Promise<T> => mock().then((m) => m.mockApi<T>(path));
const mockRecall = <T,>(query: string): Promise<T> => mock().then((m) => m.mockRecall<T>(query));
const mockDeleteEpisode = <T,>(id: string): Promise<T> => mock().then((m) => m.mockDeleteEpisode<T>(id));
const mockDeleteFact = <T,>(id: string): Promise<T> => mock().then((m) => m.mockDeleteFact<T>(id));
const mockFlag = (kind: string, id: string, flagged: boolean) => mock().then((m) => m.mockFlag(kind, id, flagged));

const BASE = '/dash/api';
const TOKEN_KEY = 'synapse.token';

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (t: string) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

export class AuthError extends Error {}
// Any other non-2xx answer; `status` lets a caller treat one (a 404) as an answer, not a failure.
export class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

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
  if (!res.ok) throw new HttpError(res.status, 'request failed (' + res.status + '): ' + fullPath);
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
  human_turn: string; assistant_turn: string;
}
// episodes is at most the server's window of the session; total counts every stored turn.
export interface Session {
  session_id: string; project: string; source: string; total: number; highlight: number | null;
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
  meta: { project?: string; source?: string; ts?: string; t_valid?: string; t_invalid?: string | null; session_id?: string; episode_id?: number | null; source_ref?: string | null; name?: string; entity_type?: string; degree?: number };
}
export interface SearchResult {
  hits: SearchHit[];
  total_by_type?: Record<SearchTab, number>; // first page only; later pages keep the client's
  offset: number; limit: number;
}
// episode_id: the episode the Flags tab opens for the row (null: nothing to open).
export interface FlagRow { id: number; kind: string; item_id: string; note: string | null; created_at: string; gist: string; episode_id: number | null; }

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
  // The episodes the evidence names outright, and every session it cites (lane evidence is
  // keyed by session, so most proposals have sessions and no episodes).
  evidence: string | EvidenceItem[]; provenance_episodes: number[]; provenance_sessions: string[];
  payload: { type: 'markdown' | 'diff'; content: string };
  audit_log: { ts: string | null; action: string; note: string | null }[];
}

export const qs = (params: Record<string, string | number | undefined | null>): string => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.set(k, String(v));
  }
  const s = p.toString();
  return s ? '?' + s : '';
};
// The header filters say 'all' for "no filter"; user text (a search for "all") is never dropped.
export const filterVal = (v?: string): string | undefined => (v === 'all' ? undefined : v);
// A restored filter the catalog no longer lists (a renamed or emptied project) falls back to
// 'all': the select can only show 'all' for it, so it must not keep filtering unseen.
export const knownFilter = (value: string, known: { name: string }[]): string =>
  value === 'all' || known.some((k) => k.name === value) ? value : 'all';

// ---------- endpoints ----------
export interface FeedParams { cursor?: string | null; project?: string; group_id?: string; source?: string; limit?: number; }

// Is a browser sign-in (an identity provider) configured? Unauthenticated, like the shell; a
// failed probe reads as no, leaving the pasted-token form.
export const fetchSsoAvailable = (): Promise<boolean> =>
  fetch('/dash/oauth/available')
    .then((r) => (r.ok ? r.json() : { available: false }))
    .then((j: { available?: boolean }) => j.available === true)
    .catch(() => false);

export const fetchCatalog = (tokenOverride?: string): Promise<Catalog> =>
  MOCK ? mockApi('/catalog') : req('/catalog', undefined, tokenOverride);

export const fetchFeed = (p: FeedParams): Promise<Feed> =>
  MOCK ? mockApi('/feed') : req('/feed' + qs({ cursor: p.cursor, project: filterVal(p.project), group_id: filterVal(p.group_id), source: filterVal(p.source), limit: p.limit ?? 30 }));

export const fetchEpisode = (id: string): Promise<Episode> =>
  MOCK ? mockApi('/episode/' + id) : req('/episode/' + encodeURIComponent(id));

export const fetchDerived = (id: string): Promise<Derived> =>
  MOCK ? mockApi('/episode/' + id + '/derived') : req('/episode/' + encodeURIComponent(id) + '/derived');

// null: no turn is stored under this id (404), which the drawer shows as "not found".
export const fetchSession = (id: string, highlight?: string | null): Promise<Session | null> =>
  (MOCK ? mockApi<Session>('/session/' + id) : req<Session>('/session/' + encodeURIComponent(id) + qs({ highlight })))
    .catch((e) => { if (e instanceof HttpError && e.status === 404) return null; throw e; });

// One page of each list: `mentionsOffset` / `factsOffset` pick which (stats.facts and
// mentions.total are the full counts the lists page against).
export const fetchEntity = (uuid: string, mentionsOffset = 0, factsOffset = 0): Promise<Entity> =>
  MOCK ? mockApi('/entity/' + uuid) : req('/entity/' + encodeURIComponent(uuid) + qs({ mentions_offset: mentionsOffset, facts_offset: factsOffset }));

export interface SearchParams { q: string; type: SearchTab; offset?: number; limit?: number; project?: string; group_id?: string; source?: string; }
export const fetchSearch = (p: SearchParams): Promise<SearchResult> =>
  MOCK ? mockApi('/search?type=' + p.type) : req('/search' + qs({ q: p.q, type: p.type, offset: p.offset ?? 0, limit: p.limit ?? 20, project: filterVal(p.project), group_id: filterVal(p.group_id), source: filterVal(p.source) }));

export const fetchFlags = (): Promise<{ flags: FlagRow[] }> =>
  MOCK ? mockApi('/flags') : req('/flags');

// The same row has a ⚑ in several places at once (a fact in the Feed and in a dossier opened
// over it): each button follows a flag change made anywhere, through one DOM CustomEvent like
// the delete announcement below.
export interface FlaggedDetail { kind: string; id: string; flagged: boolean }
const FLAGGED_EVENT = 'synapse:flagged';
export const onFlagged = (fn: (d: FlaggedDetail) => void): (() => void) => {
  const h = (e: Event) => fn((e as CustomEvent<FlaggedDetail>).detail);
  window.addEventListener(FLAGGED_EVENT, h);
  return () => window.removeEventListener(FLAGGED_EVENT, h);
};

// `flagged` is the state the operator wants, so a stale button can't invert it; the
// server answers with the state it ended in.
export const postFlag = async (kind: string, id: string, flagged: boolean, note?: string): Promise<{ status: string; flagged: boolean }> => {
  const r: { status: string; flagged: boolean } = MOCK
    ? await mockFlag(kind, id, flagged)
    : await req('/flag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, id, flagged, note }),
    });
  window.dispatchEvent(new CustomEvent<FlaggedDetail>(FLAGGED_EVENT, { detail: { kind, id, flagged: r.flagged } }));
  return r;
};

// ---------- delete (phase 7) ----------
// The ONLY destructive calls in this client. Wire shapes pinned in
// docs/dashboard-contract.md §Phase 7. Deleting an episode cascades server-side into the
// content derived from it, and the summary says exactly what went — including the mixed
// chunks, which carry text of episodes the operator did NOT delete. The UI repeats that
// verbatim rather than reporting a bare "deleted".
export interface DeleteEpisodeResult {
  episode_id: number;
  chunks_deleted: number;
  chunks_mixed_deleted: number;
  facts_deleted: number;
  facts_unlinked: number;
  events_deleted: number;
  events_kept: number; // re-told by a surviving turn
  deleted_fact_ids: string[];
  deleted_event_ids: number[];
}
export interface DeleteFactResult {
  fact_id: string;
  deleted: boolean;
  supersession_links_cleared: number;
}

// A delete can fire from a search hit or from the episode overlay, and the lists showing
// the item (Feed, Timeline, Search, the entity dossier) are not the component that deleted
// it. One DOM CustomEvent keeps them in sync without threading state through the hash
// router. It names every row that went, ids as strings: an episode's cascade takes the facts
// and events derived only from it, and the server lists them in its response.
export interface DeletedDetail { episodes: string[]; facts: string[]; events: string[] }
const DELETED_EVENT = 'synapse:deleted';
export const onDeleted = (fn: (d: DeletedDetail) => void): (() => void) => {
  const h = (e: Event) => fn((e as CustomEvent<DeletedDetail>).detail);
  window.addEventListener(DELETED_EVENT, h);
  return () => window.removeEventListener(DELETED_EVENT, h);
};
const announceDeleted = (detail: DeletedDetail) =>
  window.dispatchEvent(new CustomEvent<DeletedDetail>(DELETED_EVENT, { detail }));

// Did this delete take the row a list is showing? `type` is a feed item type or a search tab.
const DELETED_BUCKET: Record<FeedType | SearchTab, keyof DeletedDetail | null> = {
  episode: 'episodes', fact: 'facts', timeline_event: 'events',
  episodes: 'episodes', facts: 'facts', events: 'events', entities: null,
};
export const wasDeleted = (d: DeletedDetail, type: FeedType | SearchTab, id: string | number): boolean => {
  const bucket = DELETED_BUCKET[type];
  return bucket !== null && d[bucket].includes(String(id));
};

export const deleteEpisode = async (id: string): Promise<DeleteEpisodeResult> => {
  const r: DeleteEpisodeResult = MOCK
    ? await mockDeleteEpisode(id)
    : await req('/episodes/' + encodeURIComponent(id), { method: 'DELETE' });
  announceDeleted({ episodes: [id], facts: r.deleted_fact_ids, events: r.deleted_event_ids.map(String) });
  return r;
};

export const deleteFact = async (id: string): Promise<DeleteFactResult> => {
  const r: DeleteFactResult = MOCK
    ? await mockDeleteFact(id)
    : await req('/facts/' + encodeURIComponent(id), { method: 'DELETE' });
  announceDeleted({ episodes: [], facts: [id], events: [] });
  return r;
};

const plural = (n: number, one: string) => n + ' ' + one + (n === 1 ? '' : 's');

// "deleted: 3 chunks, 2 facts, 1 event · 1 fact unlinked (shared provenance) · 1 chunk
// also spanned other episodes" — the honest readout, collateral included.
export function cascadeSummary(r: DeleteEpisodeResult): string {
  const gone: string[] = [];
  if (r.chunks_deleted) gone.push(plural(r.chunks_deleted, 'chunk'));
  if (r.facts_deleted) gone.push(plural(r.facts_deleted, 'fact'));
  if (r.events_deleted) gone.push(plural(r.events_deleted, 'event'));
  const parts = ['episode deleted' + (gone.length ? ': ' + gone.join(', ') : '')];
  if (r.facts_unlinked) parts.push(plural(r.facts_unlinked, 'fact') + ' unlinked (shared provenance)');
  if (r.events_kept) parts.push(plural(r.events_kept, 'event') + ' kept (also told by other turns)');
  if (r.chunks_mixed_deleted) parts.push(plural(r.chunks_mixed_deleted, 'chunk') + ' also spanned other episodes');
  return parts.join(' · ');
}

export const factDeleteSummary = (r: DeleteFactResult): string =>
  'fact deleted' + (r.supersession_links_cleared
    ? ' · ' + plural(r.supersession_links_cleared, 'supersession link') + ' cleared'
    : '');

// ---------- recall debug console (phase 2) ----------
// The debug envelope surfaces the SAME per-leg timing / pool / rerank numbers the engine
// already measures for recall_metrics (docs/dashboard-contract.md §Phase 2). legs_ms omits
// a leg the engine did not time (notes when disabled) — the waterfall renders it as skipped. Served items carry no per-item score in the real payload (only the mock adds
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
  // Curated notes (the board's searchable half): present when the notes leg served any.
  notes?: { id: string; hook: string; note: string }[];
  web?: { context?: string; excerpt?: string; url?: string; title?: string; date?: string; score?: number }[];
  superseded_facts?: { fact: string; superseded_by: string }[];
  // Present only when a retrieval leg degraded (dead embed/rerank backend, failed KG or
  // notes leg). Absent on a healthy recall, so treat it as optional and never expect [].
  warnings?: string[];
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

// ---------- live feed stream (phase 3) ----------
// GET /dash/api/stream is SSE, but EventSource cannot send the Authorization header, so we
// read it with fetch + a ReadableStream line parser instead (contract §Phase 3). The parser
// honors event:/data:/id: fields; `id:` rides only on feed events, so lastEventId tracks the
// resumable feed stream (not ready / processing_status / heartbeats). Manual reconnect with
// 1s→30s backoff resends Last-Event-ID; after >5 consecutive failures we give up so the
// caller can fall back to polling (contract §Phase 3, spec §2 reconnect states). A connection
// only counts as healthy once the server's `ready` frame arrives.
export interface ProcessingStatus { queue_depth: number; processing: number; failed: number; active: boolean; }

const STREAM_EVENTS = new Set(['new_episode', 'new_fact', 'new_timeline_event']);
const STREAM_MAX_BACKOFF_MS = 30_000;
const STREAM_GIVE_UP_AFTER = 5;

export interface StreamHandlers {
  onReady: () => void;      // connected and positioned: whatever is written from now on streams, so refetch page 1
  onItem: (item: FeedItem) => void;
  onReset: () => void;      // the server lost events it could not replay -> refetch page 1
  onStatus: (s: ProcessingStatus) => void;
  onDisconnect: () => void; // transient loss -> header amber "reconnecting", keep stale items
  onGiveUp: () => void;     // persistent failure -> caller starts the polling fallback
}
export interface StreamHandle { close: () => void; }

export function openFeedStream(handlers: StreamHandlers): StreamHandle {
  let closed = false;
  let lastEventId: string | null = null;
  let failures = 0;
  let controller: AbortController | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;

  const dispatch = (event: string, data: string) => {
    if (event === 'processing_status') {
      try { handlers.onStatus(JSON.parse(data) as ProcessingStatus); } catch { /* ignore */ }
    } else if (event === 'ready') {
      failures = 0;
      handlers.onReady();
    } else if (event === 'reset') {
      lastEventId = null; // resynced to page 1; a stale id would just force another reset
      handlers.onReset();
    } else if (STREAM_EVENTS.has(event)) {
      try { handlers.onItem(JSON.parse(data) as FeedItem); } catch { /* ignore */ }
    }
  };

  const readStream = async (body: ReadableStream<Uint8Array>) => {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let evName = '';
    let evData: string[] = [];
    let evId: string | null = null;
    const flush = () => {
      if (evData.length || evName) {
        if (evId !== null) lastEventId = evId;
        dispatch(evName || 'message', evData.join('\n'));
      }
      evName = ''; evData = []; evId = null;
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        let line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (line === '') { flush(); continue; }
        if (line.startsWith(':')) continue; // comment / heartbeat
        const ci = line.indexOf(':');
        const field = ci < 0 ? line : line.slice(0, ci);
        let val = ci < 0 ? '' : line.slice(ci + 1);
        if (val.startsWith(' ')) val = val.slice(1);
        if (field === 'event') evName = val;
        else if (field === 'data') evData.push(val);
        else if (field === 'id') evId = val;
      }
    }
  };

  const scheduleReconnect = () => {
    if (closed) return;
    handlers.onDisconnect();
    failures += 1;
    if (failures > STREAM_GIVE_UP_AFTER) { handlers.onGiveUp(); return; }
    const backoff = Math.min(1000 * 2 ** (failures - 1), STREAM_MAX_BACKOFF_MS);
    retryTimer = setTimeout(connect, backoff);
  };

  const connect = async () => {
    if (closed) return;
    controller = new AbortController();
    const headers: Record<string, string> = {
      Authorization: 'Bearer ' + (getToken() ?? ''),
      Accept: 'text/event-stream',
    };
    if (lastEventId) headers['Last-Event-ID'] = lastEventId;
    try {
      const res = await fetch(BASE + '/stream', { headers, signal: controller.signal });
      if (res.status === 401) { closed = true; authFailHandler(); return; }
      if (!res.ok || !res.body) { scheduleReconnect(); return; }
      await readStream(res.body);
      if (!closed) scheduleReconnect(); // server closed the stream -> reconnect
    } catch {
      if (!closed) scheduleReconnect(); // network error / abort
    }
  };

  connect();
  return {
    close: () => {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (controller) controller.abort();
    },
  };
}

// Mirror the server /feed per-type filtering client-side, so a live event only prepends when
// it passes the CURRENT filter cluster: a filter applies only to types that carry the column
// (episodes: project+source; facts: group_id; timeline: project + group_id-from-domain).
export function feedItemPassesFilter(
  item: FeedItem,
  f: { project: string; group: string; source: string },
): boolean {
  if (f.project !== 'all' && (item.type === 'episode' || item.type === 'timeline_event')) {
    if ((item.project || 'untagged') !== f.project) return false;
  }
  if (f.source !== 'all' && item.type === 'episode') {
    if ((item.source || 'untagged') !== f.source) return false;
  }
  if (f.group !== 'all' && (item.type === 'fact' || item.type === 'timeline_event')) {
    // An unlabelled event (no domain) fails open at read (schema 038): it passes any group.
    const unlabelled = item.type === 'timeline_event' && !item.group_id;
    if (!unlabelled && (item.group_id || '') !== f.group) return false;
  }
  return true;
}
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
  // The window's own percentiles: hourly percentiles do not average into a window percentile.
  summary: { calls: number; p50: number | null; p95: number | null; tokens_p50: number | null };
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

// ---------- timeline: events + preferences (phase 5) ----------
// Wire shapes pinned in docs/dashboard-contract.md §"Phase 5". `sal` is the 0.3/0.6/0.9
// readout the salience type ramp reads; `event_type` is sparsely populated (a chip filter
// narrows to it; untyped events show only in the unfiltered view).
export interface TimelineEvent {
  id: number; t_valid: string; fact: string; source: string; project: string | null;
  salience: number | null; sal: number; event_type: string | null;
  episode_id: number | null; flagged: boolean;
}
export interface TimelinePage { events: TimelineEvent[]; next_before: string | null; }
export interface TimelineParams { before?: string | null; limit?: number; type?: string; group_id?: string; }
// The event types the pipeline writes (ingestion/llm_schemas.py _EVENT_TYPES); the API
// answers 400 to any other `type`.
export const TIMELINE_TYPES = ['decision', 'action', 'finding', 'milestone'] as const;
// One month that has events under the active filters, bucketed in the browser's zone;
// `before` is the cursor that makes /timeline start at that month's last event.
export interface TimelineMonth { month: string; n: number; before: string; }

export const fetchTimeline = (p: TimelineParams): Promise<TimelinePage> =>
  MOCK ? mockApi('/timeline') : req('/timeline' + qs({ before: p.before, limit: p.limit ?? 50, type: p.type, group_id: filterVal(p.group_id) }));

export const fetchTimelineMonths = (p: { type?: string; group_id?: string }): Promise<{ months: TimelineMonth[] }> =>
  MOCK ? mockApi('/timeline/months') : req('/timeline/months' + qs({ type: p.type, group_id: filterVal(p.group_id), tz: browserTz() }));

export interface Preference {
  id: number; pref: string; polarity: string; first_seen: string; last_asserted: string;
  assert_count: number; superseded_by: number | null; superseded_by_text: string | null;
  t_invalid: string | null; flagged: boolean;
}
export const fetchPreferences = (sort: 'recency' | 'assert_count' = 'recency'): Promise<{ preferences: Preference[] }> =>
  MOCK ? mockApi('/preferences') : req('/preferences' + qs({ sort }));

// ---------- dream report + behavior files (phase 5) ----------
export interface DreamReport { runs: DreamRun[]; }
export const fetchDreamReport = (limit = 20): Promise<DreamReport> =>
  MOCK ? mockApi('/dream/report') : req('/dream/report' + qs({ limit }));

export interface BehaviorFileEntry { file_key: string; surface_id: string; scope: string; updated_at: string; size: number; }
export interface BehaviorGroup { name: string; files: BehaviorFileEntry[]; }
export interface BehaviorFiles { groups: BehaviorGroup[]; }
export interface BehaviorFileMeta {
  surface_id?: string; scope?: string; abs_path?: string; content_hash?: string;
  modified_at?: string | null; updated_at?: string | null; size?: number;
}
export interface BehaviorFile { file_key: string; content: string; meta: BehaviorFileMeta; links: string[]; }
export const fetchBehaviorFiles = (): Promise<BehaviorFiles> =>
  MOCK ? mockApi('/behavior/files') : req('/behavior/files');
export const fetchBehaviorFile = (key: string, scope = 'global', surface?: string): Promise<BehaviorFile> =>
  MOCK ? mockApi('/behavior/file') : req('/behavior/file' + qs({ key, scope, surface }));
// ---------- graph explorer (phase 6) ----------
// Wire shapes pinned in docs/dashboard-contract.md §"Phase 6". The neighborhood is a
// BFS from a resolved seed (uuid or name), depth ≤ 2, server-capped at 150 nodes
// (truncation keeps the highest-degree nodes). Edges carry bitemporal validity so the
// as-of slider scrubs client-side; the client re-queries with as_of only when truncated.
export interface GraphEntity { uuid: string; name: string; entity_type: string | null; supertype?: string | null; degree: number; }
export interface GraphNode {
  uuid: string; name: string; entity_type: string | null; supertype?: string | null; degree: number; summary: string | null;
}
export interface GraphEdge {
  uuid: string; src: string; tgt: string; name: string | null; fact: string | null;
  t_valid: string | null; t_invalid: string | null;
  provenance_episode_id: number | null; retrieval_count: number;
}
export interface Neighborhood {
  nodes: GraphNode[]; edges: GraphEdge[]; truncated: boolean; seed: string;
}
export interface NeighborhoodParams { entity: string; depth?: 1 | 2; as_of?: string | null; limit?: number; }

export const fetchGraphEntities = (q: string, limit = 10): Promise<GraphEntity[]> =>
  MOCK ? mockApi('/graph/entities?q=' + encodeURIComponent(q)) : req('/graph/entities' + qs({ q, limit }));

export const fetchGraphNeighborhood = (p: NeighborhoodParams): Promise<Neighborhood> =>
  MOCK
    ? mockApi('/graph/neighborhood?entity=' + encodeURIComponent(p.entity))
    : req('/graph/neighborhood' + qs({ entity: p.entity, depth: p.depth ?? 1, as_of: p.as_of, limit: p.limit }));
