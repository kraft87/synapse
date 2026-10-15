// Hash-routed overlay host. One overlay layer at a time (opening a provenance
// link replaces the current overlay — the underlying page never moves). Esc and
// scrim-click both close.
import { Component, useEffect, type ReactNode } from 'react';
import { closeOverlay, useOverlayRoute, type OverlayRoute } from '../hash';
import { EpisodeModal } from './EpisodeModal';
import { EntityDossier } from './EntityDossier';
import { SessionDrawer } from './SessionDrawer';

// An overlay that throws while rendering would unmount the whole app (the host sits outside
// PageBoundary): close it instead and leave the page as it was. Keyed by route, so the next
// overlay starts clean.
class OverlayBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    closeOverlay();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

const routeKey = (r: OverlayRoute): string => (r.kind === 'none' ? 'none' : r.kind + ':' + r.id);

export function Overlays() {
  const route = useOverlayRoute();

  useEffect(() => {
    if (route.kind === 'none') return;
    const on = (e: KeyboardEvent) => { if (e.key === 'Escape') closeOverlay(); };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [route.kind]);

  const overlay = route.kind === 'episode' ? <EpisodeModal id={route.id} />
    : route.kind === 'entity' ? <EntityDossier id={route.id} />
    : route.kind === 'session' ? <SessionDrawer id={route.id} highlight={route.highlight} />
    : null;
  return <OverlayBoundary key={routeKey(route)}>{overlay}</OverlayBoundary>;
}
