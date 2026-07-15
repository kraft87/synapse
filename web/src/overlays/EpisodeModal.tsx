// Episode detail overlay (#/episode/:id). Turn cards + the derived-from block —
// both-directions provenance for verifying extraction (spec §IA).
import type React from 'react';
import { useEffect, useState } from 'react';
import { fetchDerived, fetchEpisode, type Derived, type Episode } from '../api';
import { closeOverlay } from '../hash';
import { srcColor } from '../tokens';
import { DerivedBlock } from '../components/FeedCard';
import { Spinner } from '../components/ui';

const chip = (bg: string, color: string, border?: string): React.CSSProperties => ({
  fontFamily: 'var(--font-data)', fontSize: '10.5px', padding: '2px 7px', borderRadius: '4px', background: bg, color, border: border || 'none',
});

export function EpisodeModal({ id }: { id: string }) {
  const [ep, setEp] = useState<Episode | null>(null);
  const [derived, setDerived] = useState<Derived | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let live = true;
    setEp(null); setDerived(null); setErr(false);
    Promise.all([fetchEpisode(id), fetchDerived(id).catch(() => ({ facts: [], timeline_events: [] } as Derived))])
      .then(([e, d]) => { if (live) { setEp(e); setDerived(d); } })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, [id]);

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
          <button className="iconbtn" onClick={closeOverlay} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', width: 26, height: 26, fontSize: '14px', lineHeight: 1 }}>×</button>
        </div>

        {err && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', marginTop: '14px' }}>couldn't load episode.</div>}
        {!ep && !err && <Spinner label="loading episode…" />}

        {ep && (
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
