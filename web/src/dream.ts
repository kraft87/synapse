// Dream runs as the Review and Metrics pages present them. A run row is inserted when the
// run starts and finished when it ends. A run whose process died stays unfinished: the next
// run marks it failed (ok false) and leaves finished_at empty, since nobody knows when it
// died. Until then it is indistinguishable from a live one, bar its age. Either way it must
// not pin the pages, and must not read as still running.

export interface RunTimes { started_at: string; finished_at: string | null; ok?: boolean | null; }

// A nightly run takes minutes; past this an unfinished row nobody has marked is a dead one.
export const ABANDONED_AFTER_H = 6;

// Seconds below two minutes, minutes above: a 25 s run is not "0 min".
export const fmtDuration = (s: number): string =>
  s < 120 ? Math.round(s) + ' s' : Math.round(s / 60) + ' min';

export type RunState = 'finished' | 'in flight' | 'interrupted' | 'abandoned';
export function runState(r: RunTimes, now: number = Date.now()): RunState {
  if (r.finished_at) return 'finished';
  if (r.ok === false) return 'interrupted';
  const started = new Date(r.started_at).getTime();
  return now - started > ABANDONED_AFTER_H * 3600e3 ? 'abandoned' : 'in flight';
}

// The duration of a finished run, else what became of it.
export const runDuration = (r: RunTimes & { duration_s: number | null }, now?: number): string =>
  r.duration_s != null && r.finished_at ? fmtDuration(r.duration_s) : runState(r, now);

// The run a report opens on: the newest FINISHED one (runs arrive newest first), else the newest.
export const defaultRun = <T extends RunTimes>(runs: T[]): T | undefined =>
  runs.find((r) => r.finished_at) ?? runs[0];

// The counts the five report cards do not show (the lanes' own, plus the retention sweep's),
// by name. Only numbers: a count is never anything else.
const CARD_COUNTS = new Set(['facts_extracted', 'superseded', 'dedup_merges', 'timeline_events', 'proposals_raised']);
export const laneCounts = (counts: Record<string, number>): [string, number][] =>
  Object.entries(counts)
    .filter(([k, v]) => !CARD_COUNTS.has(k) && typeof v === 'number')
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));

// One "name   value" line, the name never running into the value.
export const countLine = (k: string, v: number): string => `${k.padEnd(24)} ${v}`;
