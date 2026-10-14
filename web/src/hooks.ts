// Shared React hooks.
import { useCallback, useEffect, useState, type RefObject } from 'react';
import { fetchDerived, fetchEpisode, type Derived, type Episode } from './api';

// One-shot async fetch: loading / data / error, runs fn on mount and whenever a dep
// changes (or reload() is called). A `live` mount-guard drops stale resolutions so a
// fast dep change / unmount can't clobber the state with an out-of-order response.
// Promoted from pages/Metrics.tsx; `reload` re-runs fn via an internal refresh counter.
export interface AsyncResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncResult<T> {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({
    data: null, loading: true, error: null,
  });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn()
      .then((d) => { if (live) setState({ data: d, loading: false, error: null }); })
      .catch((e) => { if (live) setState({ data: null, loading: false, error: String(e?.message || e) }); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

// An episode and what was extracted from it. The derived panel is garnish: a failed
// derived fetch leaves it empty rather than failing the whole episode.
export function useEpisode(id: string): { ep: Episode | null; derived: Derived | null; error: string | null } {
  const { data, error } = useAsync(() => Promise.all([
    fetchEpisode(id),
    fetchDerived(id).catch(() => ({ facts: [], timeline_events: [] } as Derived)),
  ]).then(([ep, derived]) => ({ ep, derived })), [id]);
  return { ep: data?.ep ?? null, derived: data?.derived ?? null, error };
}

// A modal overlay's keyboard contract: focus moves into the panel on open (to its close
// button, which is there before the data is), Tab and Shift+Tab wrap inside it, and focus
// goes back to what held it when the overlay closes.
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
export function useModal(panel: RefObject<HTMLElement>): void {
  useEffect(() => {
    const el = panel.current;
    const opener = document.activeElement as HTMLElement | null;
    el?.querySelector<HTMLElement>('button[aria-label="close"]')?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !el) return;
      const items = [...el.querySelectorAll<HTMLElement>(FOCUSABLE)];
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const at = document.activeElement;
      if (!el.contains(at)) { e.preventDefault(); first.focus(); }
      else if (e.shiftKey && at === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && at === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      opener?.focus?.();
    };
  }, [panel]);
}
