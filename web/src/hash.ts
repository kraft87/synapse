// Overlay routing lives in the URL hash so overlays are deep-linkable and
// additive over whatever page is showing (page selection is React state, not
// the hash). Also handles the one-time #token=... bootstrap.
import { useEffect, useState } from 'react';
import { setToken } from './api';

export type OverlayRoute =
  | { kind: 'none' }
  | { kind: 'episode'; id: string }
  | { kind: 'entity'; id: string }
  | { kind: 'session'; id: string; highlight?: string };

// Reads #token=... on first load, stores it, and strips it from the URL so it
// never lingers in history or logs. Preserves a "mock" marker if present.
export function bootstrapToken(): void {
  const h = location.hash;
  const m = h.match(/token=([^&]+)/);
  if (!m) return;
  setToken(decodeURIComponent(m[1]));
  const keepMock = h.includes('mock');
  history.replaceState(null, '', location.pathname + location.search + (keepMock ? '#mock' : ''));
}

export function parseRoute(hash: string): OverlayRoute {
  const h = hash.replace(/^#/, '');
  const [path, query] = h.split('?');
  const parts = path.split('/').filter(Boolean); // e.g. ["episode","227168"]
  if (parts[0] === 'episode' && parts[1]) return { kind: 'episode', id: decodeURIComponent(parts[1]) };
  if (parts[0] === 'entity' && parts[1]) return { kind: 'entity', id: decodeURIComponent(parts[1]) };
  if (parts[0] === 'session' && parts[1]) {
    const highlight = new URLSearchParams(query || '').get('highlight') || undefined;
    return { kind: 'session', id: decodeURIComponent(parts[1]), highlight };
  }
  return { kind: 'none' };
}

export function useOverlayRoute(): OverlayRoute {
  const [route, setRoute] = useState<OverlayRoute>(() => parseRoute(location.hash));
  useEffect(() => {
    const on = () => setRoute(parseRoute(location.hash));
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

// Navigation helpers — provenance links call these; they change the overlay
// layer only, never the underlying page.
export const openEpisode = (id: string | number) => { location.hash = '#/episode/' + encodeURIComponent(String(id)); };
export const openEntity = (id: string) => { location.hash = '#/entity/' + encodeURIComponent(id); };
export const openSession = (id: string, highlight?: string | number | null) => {
  location.hash = '#/session/' + encodeURIComponent(id) + (highlight != null ? '?highlight=' + encodeURIComponent(String(highlight)) : '');
};
export const closeOverlay = () => {
  // Drop the overlay route; keep a mock marker so offline reloads stay in mock.
  const keepMock = location.hash.includes('mock');
  history.replaceState(null, '', location.pathname + location.search + (keepMock ? '#mock' : ''));
  window.dispatchEvent(new HashChangeEvent('hashchange'));
};
