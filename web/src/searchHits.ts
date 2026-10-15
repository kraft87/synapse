// What a Search hit says about itself, and whether it leads anywhere. Pure, so the rules are
// unit-tested; Search.tsx renders them.
import type { SearchHit } from './api';
import { localDay, relTime } from './tokens';

// A hit opens something only when it names a target: an entity (dossier), an episode, or the
// episode / session a fact or event traces to. A git-sourced event and a fact with no
// provenance name none, and render as plain text rather than a button that does nothing.
export const hasTarget = (h: SearchHit): boolean =>
  h.type === 'entities' || h.type === 'episodes' || h.meta.episode_id != null || !!h.meta.session_id;

export const isSuperseded = (h: SearchHit): boolean => h.type === 'facts' && !!h.meta.t_invalid;

const SHA = /^[0-9a-f]{40}$/;

export function metaLine(h: SearchHit): string {
  if (h.type === 'entities') return [h.meta.entity_type, h.meta.degree != null ? h.meta.degree + ' edges' : ''].filter(Boolean).join(' · ');
  // Events: show when the event happened (t_valid), not when the row was
  // ingested — backfills stamp thousands of events with one ingested_at.
  const ts = h.type === 'events' ? (h.meta.t_valid ?? h.meta.ts) : h.meta.ts;
  const parts = [h.meta.project, h.meta.source, relTime(ts)];
  // A git event's only provenance is its commit.
  if (h.type === 'events' && h.meta.episode_id == null && h.meta.source_ref) {
    parts.push('ref ' + (SHA.test(h.meta.source_ref) ? h.meta.source_ref.slice(0, 7) : h.meta.source_ref));
  }
  if (isSuperseded(h)) parts.push('superseded ' + localDay(h.meta.t_invalid));
  return parts.filter(Boolean).join(' · ');
}
