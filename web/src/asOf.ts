// The Graph's as-of slider: 0..24 steps back from today, one month each (24 = today).
import { localDay } from './tokens';

export const AS_OF_TODAY = 24;

// `idx` months before `now`, clamped to the last day of a shorter month (Mar 31 back one
// month is Feb 28, not Mar 3); null for today.
export function asOfDateFor(idx: number, now: Date = new Date()): Date | null {
  if (idx >= AS_OF_TODAY) return null;
  const d = new Date(now);
  const day = d.getDate();
  d.setDate(1);
  d.setMonth(d.getMonth() - (AS_OF_TODAY - idx));
  d.setDate(Math.min(day, new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate()));
  return d;
}

export const asOfLabel = (idx: number, now?: Date): string => {
  const d = asOfDateFor(idx, now);
  return d ? localDay(d.toISOString()) : 'today';
};
