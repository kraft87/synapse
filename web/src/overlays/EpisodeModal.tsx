// Episode detail overlay (#/episode/:id). Turn cards + the derived-from block —
// both-directions provenance for verifying extraction (spec §IA).
import type React from 'react';
import { useState } from 'react';
import { cascadeSummary, deleteEpisode, fetchDerived, fetchEpisode, type Derived } from '../api';
import { closeOverlay } from '../hash';
import { srcColor } from '../tokens';
import { DerivedBlock } from '../components/FeedCard';
import { DeleteButton, Spinner } from '../components/ui';
import { useAsync } from '../hooks';

const chip = (bg: string, color: string, border?: string): React.CSSProperties => ({
  fontFamily: 'var(--font-data)', fontSize: '10.5px', padding: '2px 7px', borderRadius: '4px', background: bg, color, border: border || 'none',
});

export function EpisodeModal({ id }: { id: string }) {
  const { data, error } = useAsync(() => Promise.all([
    fetchEpisode(id),
    fetchDerived(id).catch(() => ({ facts: [], timeline_events: [] } as Derived)),
  ]).then(([e, d]) => ({ ep: e, derived: d })), [id]);
  // After a successful delete the episode no longer exists, so the overlay stops showing
  // it and shows the cascade summary instead — closing straight away would hide the one
  // line that says what else went (mixed chunks, unlinked facts). Dismiss closes.
  const [deleted, setDeleted] = useState<string | null>(null);
  const [delErr, setDelErr] = useState(false);
  const ep = data?.ep ?? null;
  const derived = data?.derived ?? null;
  const err = error != null;

  const turns = ep ? [
    ep.human_turn && { role: 'user', text: ep.human_turn, user: true },
    ep.assistant_turn && { role: 'assistant', text: ep.assistant_turn, user: false },
  ].filter(Boolean) as { role: string; text: string; user: boolean }[] : [];
  const meta = ep ? [ep.session_id && 'session ' + String(ep.session_id).slice(0, 8), ep.sequence != null && 'seq ' + ep.sequence, ep.created_at, ep.platform, ep.model].filter(Boolean).join(' · ') : '';

  return (
    <div onClick={closeOverlay} style={{ position: 'fixed', inset: 0, background: 'var(--scrim)', zIndex: 60, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px', backdropFilter: 'blur(2px)' }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '14px', maxWidth: '720px', width: '100%', maxHeight: '80vh', overflowY: 'auto', padding: '20px 22px', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)' }}>ep-{id}</span>
          {ep?.project && <span style={chip('var(--acc-bg)', 'var(--acc)')}>{ep.project}</span>}
          {ep?.source && <span style={chip('transparent', srcColor(ep.source), '1px solid ' + srcColor(ep.source))}>{ep.source}</span>}
          <span style={{ flex: 1 }} />
          {ep && !deleted && (
            <DeleteButton
              label="delete episode"
              onDelete={() => deleteEpisode(id).then((r) => { setDelErr(false); setDeleted(cascadeSummary(r)); })}
              onError={() => setDelErr(true)}
            />
          )}
          <button className="iconbtn" onClick={closeOverlay} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', width: 26, height: 26, fontSize: '14px', lineHeight: 1 }}>×</button>
        </div>

        {delErr && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', marginTop: '14px' }}>delete failed — the episode is still in memory.</div>}
        {deleted && (
          <div style={{ marginTop: '14px', border: '1px solid var(--line2)', background: 'var(--bg2)', borderRadius: '9px', padding: '14px 16px' }}>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '12px', color: 'var(--txt2)', lineHeight: 1.6 }}>{deleted}</div>
            <button className="softbtn" onClick={closeOverlay} style={{ marginTop: '12px', borderRadius: '6px', padding: '5px 12px', fontSize: '12px', fontFamily: 'var(--font-data)' }}>close</button>
          </div>
        )}

        {err && !deleted && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', marginTop: '14px' }}>couldn't load episode.</div>}
        {!ep && !err && !deleted && <Spinner label="loading episode…" />}

        {ep && !deleted && (
          <>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', margin: '8px 0 14px' }}>{meta}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {turns.map((t, i) => (
                <div key={i} style={{ border: '1px solid ' + (t.user ? 'var(--line2)' : 'var(--line)'), borderRadius: '9px', padding: '10px 13px', background: t.user ? 'var(--bg2)' : 'transparent' }}>
                  <div style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: t.user ? 'var(--acc)' : 'var(--txt3)', marginBottom: '5px' }}>{t.role}</div>
                  <div style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', whiteSpace: 'pre-wrap' }}>{t.text}</div>
                </div>
              ))}
            </div>
            {derived && <DerivedBlock derived={derived} heading="derived from this episode" bare />}
          </>
        )}
      </div>
    </div>
  );
}
