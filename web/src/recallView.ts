// What the Recall console shows for one recall() result: the latency waterfall rows, the
// Served buckets (exactly what an agent receives) and the effective scope of the query.
// Pure functions so the shaping is unit-tested; Recall.tsx only renders them.
import type { RecallDebug, RecallResult } from './api';
import { LEG_COLOR, LEG_ORDER } from './tokens';

// The parallel band (spec §5) — every one starts at embed-end; rerank waits for the
// slowest of them. embed leads, rerank trails; neither is "parallel".
const PARALLEL = ['bm25', 'vector', 'kg', 'notes', 'web'];

export interface WRow {
  name: string; left: number; width: number; color: string; glow: string;
  ms: string; msColor: string; flag: string; skipped: boolean;
}

// `debug` null: no timings to draw (the recall failed, or has not answered) — every leg reads
// "—", untimed, rather than "(skipped)", which means the engine chose not to run the leg.
export function waterfall(debug: RecallDebug | null): WRow[] {
  const total = debug?.total_ms || 1;
  const legs = debug?.legs_ms || {};
  const embedMs = legs.embed ?? 0;
  const maxParallel = PARALLEL.filter((n) => n in legs).reduce((m, n) => Math.max(m, legs[n]), 0);
  const rerankStart = embedMs + maxParallel;
  return LEG_ORDER.map((name) => {
    const present = name in legs;
    const ms = present ? legs[name] : 0;
    const start = name === 'embed' ? 0 : name === 'rerank' ? rerankStart : embedMs;
    const outlier = present && total > 0 && ms / total > 0.4;
    return {
      name,
      left: (start / total) * 100,
      width: present ? (ms / total) * 100 : 0,
      color: present ? LEG_COLOR[name] : 'var(--line2)',
      glow: outlier ? '0 0 8px ' + LEG_COLOR[name] : 'none',
      ms: present ? String(ms) : '—',
      msColor: outlier ? 'var(--err)' : 'var(--txt2)',
      flag: outlier ? ' ▲' : present || !debug ? '' : ' (skipped)',
      skipped: !present,
    };
  });
}

// ---- Served bucket item shaping ----
export interface Item { key: string; score?: number; meta?: string; head: string; rest: string; }

export function bucketItems(name: string, r: RecallResult): Item[] {
  switch (name) {
    case 'episodes':
      return (r.episodes || []).map((e, i) => ({
        key: 'e' + i, score: e.score, meta: e.date,
        head: [e.id, e.project].filter(Boolean).join(' · ') + (e.role ? ' · ' + e.role : '') + ' —',
        rest: e.content + (e.superseded_by?.length ? '  ⟳ superseded_by: ' + e.superseded_by.join('; ') : ''),
      }));
    case 'facts':
      return (r.facts || []).map((f, i) => ({
        key: 'f' + i, score: f.score, meta: f.date,
        head: f.fact, rest: f.date ? '· as of ' + f.date : '',
      }));
    case 'notes':
      return (r.notes || []).map((n, i) => ({
        key: 'n' + i, head: n.id + ' · ' + n.hook + ' —', rest: n.note,
      }));
    case 'web':
      return (r.web || []).map((w, i) => ({
        key: 'w' + i, score: w.score, meta: w.date,
        head: (w.title || w.url || 'web') + ' —', rest: w.context || w.excerpt || '',
      }));
    case 'superseded_facts':
      return (r.superseded_facts || []).map((h, i) => ({
        key: 'h' + i, head: 'superseded —',
        rest: h.fact + (h.superseded_by ? '   (now: ' + h.superseded_by + ')' : ''),
      }));
    default:
      return [];
  }
}

// The buckets an agent can be served, in display order. The engine omits a bucket it has
// nothing for, so notes shows when it came back or when its leg ran (an empty "0 served" is
// then real, not a disabled leg); superseded_facts only when there are some.
export function servedBuckets(r: RecallResult): string[] {
  return [
    'episodes', 'facts',
    ...(r.notes || (r.debug && 'notes' in r.debug.legs_ms) ? ['notes'] : []),
    'web',
    ...(r.superseded_facts?.length ? ['superseded_facts'] : []),
  ];
}

// ---- the scope recall() actually runs with ----
// The header filters are Feed/Search filters; recall() reads them differently. It searches
// exactly one group (technical when none is named), and its episode legs match `project`
// literally, so the catalog's 'untagged' bucket (NULL, '' and 'untagged') would name a
// project that holds almost nothing. The console sends what recall() honors and says so.
export interface RecallScope { project?: string; group_id: string; notes: string[]; }

export function recallScope(project: string, group: string): RecallScope {
  const notes: string[] = [];
  let scoped: string | undefined = project === 'all' ? undefined : project;
  if (project === 'untagged') {
    scoped = undefined;
    notes.push("'untagged' cannot scope recall — searching all projects");
  }
  let groupId = group;
  if (group === 'all') {
    groupId = 'technical';
    notes.push('recall searches one group — technical');
  }
  return { project: scoped, group_id: groupId, notes };
}
