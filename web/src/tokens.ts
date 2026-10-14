// Shared color maps + small formatters. Colors reference the CSS custom
// properties defined in styles.css (README §Design Tokens), so themes switch
// automatically.
import type { CSSProperties } from 'react';

// The data (monospace) font-family token + the shared uppercase mono section-head
// style, reused by the page tab bars / panel heads. Feed keeps a local 12px variant.
export const mono = 'var(--font-data)';
export const monoHead: CSSProperties = {
  fontFamily: mono, fontSize: '11px', color: 'var(--txt3)',
  textTransform: 'uppercase', letterSpacing: '.08em',
};

export const SRC_COLOR: Record<string, string> = {
  'claude-code': 'var(--src-cc)',
  'cursor': 'var(--src-cur)',
  'claude-ai': 'var(--src-cai)',
  'transcribe-ai': 'var(--src-tr)',
};
export const srcColor = (s?: string | null): string => (s && SRC_COLOR[s]) || 'var(--txt3)';

const ET_COLOR: Record<string, string> = {
  Person: 'var(--et-person)',
  Project: 'var(--et-project)',
  Technology: 'var(--et-tech)',
  Organization: 'var(--et-org)',
  Concept: 'var(--et-concept)',
  Event: 'var(--et-event)',
};
// entity_type casing is normalized to TitleCase before lookup.
export const etColor = (t?: string | null): string => {
  if (!t) return 'var(--txt3)';
  const key = t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
  return ET_COLOR[key] || 'var(--txt3)';
};

// The live KG carries ~32 canonical supertypes (kg_supertypes, schema 020) but the
// design has six color families — without this mapping ~80% of graph nodes fell
// through to gray (the "useless gray blob" report, 2026-07-15). Curated, not
// heuristic, so a new supertype lands gray until deliberately placed.
const SUPER_FAMILY: Record<string, string> = {
  Person: 'var(--et-person)',
  Project: 'var(--et-project)',
  Organization: 'var(--et-org)',
  Event: 'var(--et-event)', Activity: 'var(--et-event)', Issue: 'var(--et-event)', Decision: 'var(--et-event)',
  Agent: 'var(--et-tech)', Architecture: 'var(--et-tech)', Config: 'var(--et-tech)',
  Database: 'var(--et-tech)', Datastructure: 'var(--et-tech)', Feature: 'var(--et-tech)',
  File: 'var(--et-tech)', Function: 'var(--et-tech)', Hardware: 'var(--et-tech)',
  Library: 'var(--et-tech)', Model: 'var(--et-tech)', Product: 'var(--et-tech)',
  Service: 'var(--et-tech)', Technology: 'var(--et-tech)', Tool: 'var(--et-tech)', Url: 'var(--et-tech)',
  Concept: 'var(--et-concept)', Food: 'var(--et-concept)', Location: 'var(--et-concept)',
  Medical: 'var(--et-concept)', Metric: 'var(--et-concept)', Pattern: 'var(--et-concept)',
  Process: 'var(--et-concept)', Publication: 'var(--et-concept)', Technique: 'var(--et-concept)',
  Other: 'var(--et-concept)',
};
export const superColor = (t?: string | null): string => {
  if (!t) return 'var(--txt3)';
  const key = t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
  return SUPER_FAMILY[key] || 'var(--txt3)';
};

// Search-tab / feed type chip color keys (README §3 badge taxonomy).
export const typeColor: Record<string, string> = {
  episode: 'var(--txt2)', episodes: 'var(--txt2)',
  fact: 'var(--et-concept)', facts: 'var(--et-concept)',
  timeline_event: 'var(--et-org)', event: 'var(--et-org)', events: 'var(--et-org)',
  entity: 'var(--et-tech)', entities: 'var(--et-tech)',
};

export const typeLabel: Record<string, string> = {
  episode: 'episode', fact: 'fact', timeline_event: 'timeline',
};

// Recall waterfall leg colors — stable across the Recall waterfall AND the Metrics
// per-leg stack (README leg token table). CSS vars so themes switch automatically.
export const LEG_COLOR: Record<string, string> = {
  embed: 'var(--leg-embed)', bm25: 'var(--leg-bm25)', vector: 'var(--leg-vector)',
  kg: 'var(--leg-kg)', notes: 'var(--leg-notes)', web: 'var(--leg-web)', rerank: 'var(--leg-rerank)',
};

// Resolve a CSS custom-property reference to its concrete value so it can be painted onto a
// <canvas> (uPlot draws to canvas, where `var(--x)` is meaningless). Accepts `var(--x)` or a
// bare `--x`; falls back to the input if the property is unset. Re-resolve on theme change.
export function cssValue(ref: string): string {
  if (typeof document === 'undefined') return ref;
  const name = ref.startsWith('var(') ? ref.slice(4, -1).trim() : ref;
  if (!name.startsWith('--')) return ref;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || ref;
}
// Fixed render order (spec §5): embed, then the parallel band, then rerank last.
export const LEG_ORDER = ['embed', 'bm25', 'vector', 'kg', 'notes', 'web', 'rerank'] as const;
// Recall-item relevance score → color ramp (spec §5b Served bucket).
export const scoreColor = (score?: number): string =>
  score == null ? 'var(--txt3)' : score > 0.8 ? 'var(--acc)' : score > 0.6 ? 'var(--txt2)' : 'var(--txt3)';

// How long ago, in whole units rounded DOWN: 23.6 h is "23h ago" (24h would claim a day that
// has not passed), and anything under a minute reads "1m ago".
export function relTime(ts?: string, now: number = Date.now()): string {
  if (!ts) return '';
  const then = new Date(ts).getTime();
  if (Number.isNaN(then)) return '';
  const dh = (now - then) / 3600e3;
  if (dh < 0) return 'now';
  if (dh < 1) return Math.max(1, Math.floor(dh * 60)) + 'm ago';
  if (dh < 24) return Math.floor(dh) + 'h ago';
  return Math.floor(dh / 24) + 'd ago';
}

// ---- calendar days, in the viewer's zone ----
// An instant's calendar day is the viewer's, not the one its UTC string spells: said at 20:30
// EDT on the 30th, an event is stored as 00:30Z on the 1st, and slicing the string files it a
// day (a month, at month end) late. Every date the dashboard shows or buckets goes through
// these; the zone is the browser's, and the tests pass one.
export const browserTz = (): string => Intl.DateTimeFormat().resolvedOptions().timeZone;

const zoneFmt = new Map<string, Intl.DateTimeFormat>();
function zoneParts(iso: string, tz: string): Record<string, string> | null {
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return null;
  let fmt = zoneFmt.get(tz);
  if (!fmt) {
    fmt = new Intl.DateTimeFormat('en-CA', {
      timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
    });
    zoneFmt.set(tz, fmt);
  }
  return Object.fromEntries(fmt.formatToParts(t).map((p) => [p.type, p.value]));
}
// YYYY-MM-DD ('' for a missing or unparseable instant).
export function localDay(iso?: string | null, tz: string = browserTz()): string {
  const p = iso ? zoneParts(iso, tz) : null;
  return p ? `${p.year}-${p.month}-${p.day}` : '';
}
export const localMonth = (iso?: string | null, tz?: string): string => localDay(iso, tz).slice(0, 7);
// YYYY-MM-DD HH:MM, 24-hour.
export function localMinute(iso?: string | null, tz: string = browserTz()): string {
  const p = iso ? zoneParts(iso, tz) : null;
  return p ? `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}` : '';
}

// A tool-call trace is detected from stored episode content — the contract delegates display
// derivation to the client. Every client writes the trace as its own lines: `[tool:<name>] <input>`
// for a call and `[result] <output>` for what came back (ingestion/jsonl_client.py,
// codex_client.py, cursor_sqlite_client.py). Prose that merely starts with "Read" or "Task" is not one.
const TOOL_RE = /(^|\n)\[(tool:[^\]\n]+|result)\] /;
// A turn pair as one readable block ("user: …" / "assistant: …"); either side may be empty.
export const turnBody = (t: { human_turn?: string; assistant_turn?: string }): string =>
  [t.human_turn && 'user: ' + t.human_turn, t.assistant_turn && 'assistant: ' + t.assistant_turn].filter(Boolean).join('\n\n');

// "12 turns", or "seq 3–502 of 520 turns" when the drawer holds only the server's window of a longer session.
export function turnsLabel(episodes: { sequence: number }[], total: number): string {
  if (episodes.length === 0 || episodes.length >= total) return total + (total === 1 ? ' turn' : ' turns');
  const seqs = episodes.map((e) => e.sequence);
  return `seq ${Math.min(...seqs)}–${Math.max(...seqs)} of ${total} turns`;
}

export const hasToolTrace = (content?: string): boolean => !!content && TOOL_RE.test(content);

// Build a human validity string from bitemporal fields.
export function validityLine(t_valid?: string, t_invalid?: string | null): string {
  const v = t_valid ? 'valid since ' + localDay(t_valid) : 'valid';
  return t_invalid ? v + ' · superseded ' + localDay(t_invalid) : v;
}
