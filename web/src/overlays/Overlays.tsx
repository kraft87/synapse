// Hash-routed overlay host. One overlay layer at a time (opening a provenance
// link replaces the current overlay — the underlying page never moves). Esc and
// scrim-click both close.
import { useEffect } from 'react';
import { closeOverlay, useOverlayRoute } from '../hash';
import { EpisodeModal } from './EpisodeModal';
import { EntityDossier } from './EntityDossier';
import { SessionDrawer } from './SessionDrawer';

export function Overlays() {
  const route = useOverlayRoute();

  useEffect(() => {
    if (route.kind === 'none') return;
    const on = (e: KeyboardEvent) => { if (e.key === 'Escape') closeOverlay(); };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [route.kind]);

  switch (route.kind) {
    case 'episode': return <EpisodeModal id={route.id} />;
    case 'entity': return <EntityDossier id={route.id} />;
    case 'session': return <SessionDrawer id={route.id} highlight={route.highlight} />;
    default: return null;
  }
}
