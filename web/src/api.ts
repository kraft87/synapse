// API layer for the phase-1 dashboard. Wire shapes match docs/dashboard-contract.md
// (owned by the server agent). Everything is Bearer-token authenticated except the
// static shell. When location.hash contains "mock", the mock dataset is served instead
// so every screen renders offline.
import { mockApi, mockFlag, MOCK } from './mock';

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

async function req<T>(path: string, init?: RequestInit, tokenOverride?: string): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      Authorization: 'Bearer ' + (tokenOverride ?? getToken() ?? ''),
    },
  });
  if (res.status === 401) { authFailHandler(); throw new AuthError('unauthorized'); }
  if (!res.ok) throw new Error('request failed (' + res.status + '): ' + path);
  return res.json() as Promise<T>;
}

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
