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
};

const SEARCH: Record<string, unknown[]> = {
  episodes: [{ type: 'episodes', id: '1', snippet: 'wired service A to library B', meta: { project: 'service-a', source: 'claude-code', ts: iso(0.4), session_id: 'sess-1', episode_id: 1 } }],
  facts: [{ type: 'facts', id: 'ent-edge-1', snippet: 'service A depends on library B', meta: { project: 'service-a', source: 'claude-code', ts: iso(24), session_id: 'sess-1', episode_id: 1 } }],
  entities: [{ type: 'entities', id: 'ent-1', snippet: 'service A — example project', meta: { name: 'service A', entity_type: 'Project', degree: 4 } }],
  events: [{ type: 'events', id: '10', snippet: 'service A shipped v2', meta: { project: 'service-a', source: 'git:example', ts: iso(48), session_id: 'sess-1', episode_id: 1 } }],
};

export async function mockApi<T>(path: string): Promise<T> {
  const p = path.replace(/^\//, '').split('?')[0];
  let out: unknown;
  if (p === 'catalog') out = FIX.catalog;
  else if (p === 'feed') out = FIX.feed;
  else if (p === 'flags') out = FIX.flags;
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
