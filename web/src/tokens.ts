// Shared color maps + small formatters. Colors reference the CSS custom
// properties defined in styles.css (README §Design Tokens), so themes switch
// automatically.

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

export function relTime(ts?: string): string {
  if (!ts) return '';
  const then = new Date(ts).getTime();
  if (Number.isNaN(then)) return '';
  const dh = (Date.now() - then) / 3600e3;
  if (dh < 0) return 'now';
  if (dh < 1) return Math.max(1, Math.round(dh * 60)) + 'm ago';
  if (dh < 24) return Math.round(dh) + 'h ago';
  return Math.round(dh / 24) + 'd ago';
}

// A tool-call trace is detected heuristically from stored episode content — the
// contract delegates display derivation to the client.
const TOOL_RE = /(^|\n)\s*(Read|Edit|Write|MultiEdit|Bash|Grep|Glob|LS|NotebookEdit|Task|WebFetch|WebSearch|Tool)\b/;
export const hasToolTrace = (content?: string): boolean => !!content && TOOL_RE.test(content);

// Build a human validity string from bitemporal fields.
export function validityLine(t_valid?: string, t_invalid?: string | null): string {
  const v = t_valid ? 'valid since ' + t_valid.slice(0, 10) : 'valid';
  return t_invalid ? v + ' · superseded ' + t_invalid.slice(0, 10) : v;
}
