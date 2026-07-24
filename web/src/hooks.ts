// Shared React hooks.
import { useEffect, useState } from 'react';

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
  const reload = () => setTick((t) => t + 1);
  return { ...state, reload };
}
