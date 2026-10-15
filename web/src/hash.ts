// Overlay routing lives in the URL hash so overlays are deep-linkable and
// additive over whatever page is showing (page selection is React state, not
// the hash). Also handles the one-time #token=... bootstrap.
import { useEffect, useState } from 'react';
import { setToken } from './api';
import { isMockHash } from './mode';

export type OverlayRoute =
  | { kind: 'none' }
  | { kind: 'episode'; id: string }
  | { kind: 'entity'; id: string }
  | { kind: 'session'; id: string; highlight?: string };

// decodeURIComponent throws on a malformed escape ("100%"); a hash is typed or pasted by hand.
export const dec = (s: string): string | null => {
  try { return decodeURIComponent(s); } catch { return null; }
};

// Reads #token=... on first load, stores it, and strips it from the URL so it
// never lingers in history or logs. Preserves a "mock" marker if present.
export function bootstrapToken(): void {
  const h = location.hash;
  const m = h.match(/token=([^&]+)/);
  if (!m) return;
  const token = dec(m[1]);
  if (token) setToken(token);
  history.replaceState(null, '', location.pathname + location.search + (isMockHash(h) ? '#mock' : ''));
}

// A route whose id does not decode is no route: the app shows its page, not a crash.
export function parseRoute(hash: string): OverlayRoute {
  const h = hash.replace(/^#/, '');
  const [path, query] = h.split('?');
  const parts = path.split('/').filter(Boolean); // e.g. ["episode","227168"]
  const id = parts[1] ? dec(parts[1]) : null;
  if (!id) return { kind: 'none' };
  if (parts[0] === 'episode') return { kind: 'episode', id };
  if (parts[0] === 'entity') return { kind: 'entity', id };
  if (parts[0] === 'session') {
    const highlight = new URLSearchParams(query || '').get('highlight') || undefined;
    return { kind: 'session', id, highlight };
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
//
// The overlay layer is one history entry. Opening from the page pushes it; opening from an
// overlay swaps the route inside it; closing steps back over it, so Back after a close leaves
// the page as it was instead of reopening an overlay. An overlay that was already in the URL
// when the page loaded (a pasted link, a reload) has no pushed entry to step over, so closing
// drops its hash in place. history.state marks the entry the app pushed.
const open = (hash: string) => {
  if (location.hash === hash) return;
  if (parseRoute(location.hash).kind === 'none') history.pushState({ overlay: true }, '', hash);
  else history.replaceState(history.state, '', hash);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
};
export const openEpisode = (id: string | number) => open('#/episode/' + encodeURIComponent(String(id)));
export const openEntity = (id: string) => open('#/entity/' + encodeURIComponent(id));
export const openSession = (id: string, highlight?: string | number | null) =>
  open('#/session/' + encodeURIComponent(id) + (highlight != null ? '?highlight=' + encodeURIComponent(String(highlight)) : ''));
export const closeOverlay = () => {
  if (history.state?.overlay) {
    // Unmark the entry first: a second close (a held Esc) before the traversal lands must
    // not step back out of the page. Back then forward shows the overlay as a loaded one.
    history.replaceState(null, '', location.href);
    history.back();
    return;
  }
  // Drop the overlay route; keep a mock marker so offline reloads stay in mock.
  history.replaceState(null, '', location.pathname + location.search + (isMockHash(location.hash) ? '#mock' : ''));
  window.dispatchEvent(new HashChangeEvent('hashchange'));
};
