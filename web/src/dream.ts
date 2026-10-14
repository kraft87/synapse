// Dream runs as the Review and Metrics pages present them. A run row is inserted when the
// run starts and finished when it ends, so a run that crashed stays unfinished forever: it
// must not pin either page, and must not read as still running.

export interface RunTimes { started_at: string; finished_at: string | null; }

// A nightly run takes minutes; past this an unfinished row is a dead one.
export const ABANDONED_AFTER_H = 6;

// Seconds below two minutes, minutes above: a 25 s run is not "0 min".
export const fmtDuration = (s: number): string =>
  s < 120 ? Math.round(s) + ' s' : Math.round(s / 60) + ' min';

export type RunState = 'finished' | 'in flight' | 'abandoned';
export function runState(r: RunTimes, now: number = Date.now()): RunState {
  if (r.finished_at) return 'finished';
  const started = new Date(r.started_at).getTime();
  return now - started > ABANDONED_AFTER_H * 3600e3 ? 'abandoned' : 'in flight';
}

// The duration of a finished run, else what became of it.
export const runDuration = (r: RunTimes & { duration_s: number | null }, now?: number): string =>
  r.duration_s != null && r.finished_at ? fmtDuration(r.duration_s) : runState(r, now);

// The run a report opens on: the newest FINISHED one (runs arrive newest first), else the newest.
export const defaultRun = <T extends RunTimes>(runs: T[]): T | undefined =>
  runs.find((r) => r.finished_at) ?? runs[0];
