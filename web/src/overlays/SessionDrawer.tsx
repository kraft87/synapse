// Session drawer (#/session/:id?highlight=) — the whole session as a readable
// transcript, right-hand, target episode highlighted with a "viewing" chip.
import { useEffect, useRef, useState } from 'react';
import { fetchSession, type Session } from '../api';
import { closeOverlay } from '../hash';
import { Spinner } from '../components/ui';

export function SessionDrawer({ id, highlight }: { id: string; highlight?: string }) {
  const [sess, setSess] = useState<Session | null>(null);
  const [err, setErr] = useState(false);
  const targetRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let live = true;
    setSess(null); setErr(false);
    fetchSession(id, highlight).then((d) => { if (live) setSess(d); }).catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, [id, highlight]);

  useEffect(() => {
    if (sess && targetRef.current) targetRef.current.scrollIntoView({ block: 'center' });
  }, [sess]);

  const hl = highlight != null ? String(highlight) : (sess?.highlight != null ? String(sess.highlight) : null);

  return (
    <div onClick={closeOverlay} style={{ position: 'fixed', inset: 0, background: 'var(--scrim)', zIndex: 65, display: 'flex', justifyContent: 'flex-end', backdropFilter: 'blur(2px)' }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: 'var(--bg1)', borderLeft: '1px solid var(--line2)', width: 'min(520px, 92vw)', height: '100%', overflowY: 'auto', padding: '20px 22px', boxSizing: 'border-box', animation: 'slidein .25s ease' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', position: 'sticky', top: 0, background: 'var(--bg1)', paddingBottom: '10px', borderBottom: '1px solid var(--line)' }}>
          <span style={{ fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--txt)' }}>session · {String(id).slice(0, 10)}</span>
          <span style={{ flex: 1 }} />
          <button className="iconbtn" onClick={closeOverlay} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', width: 26, height: 26, fontSize: '14px', lineHeight: 1 }}>×</button>
        </div>

        {err && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', marginTop: '14px' }}>couldn't load session.</div>}
        {!sess && !err && <Spinner label="loading session…" />}

        {sess && (
          <>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', margin: '8px 0 14px' }}>
              {[sess.project, sess.source, sess.episodes.length + ' turns'].filter(Boolean).join(' · ')}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {sess.episodes.map((ep) => {
                const target = hl != null && String(ep.id) === hl;
                const body = [ep.human_turn && 'user: ' + ep.human_turn, ep.assistant_turn && 'assistant: ' + ep.assistant_turn].filter(Boolean).join('\n\n');
                return (
                  <div key={ep.id} ref={target ? targetRef : undefined} style={{ border: '1px solid ' + (target ? 'var(--line2)' : 'var(--line)'), borderRadius: '9px', padding: '10px 13px', background: target ? 'var(--bg2)' : 'transparent' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'baseline', marginBottom: '5px' }}>
                      <span style={{ fontFamily: 'var(--font-data)', fontSize: '10px', color: 'var(--txt3)' }}>seq {ep.sequence}</span>
                      {target && <span style={{ fontFamily: 'var(--font-data)', fontSize: '9.5px', padding: '1px 6px', borderRadius: '3px', background: 'var(--acc-bg)', color: 'var(--acc)' }}>viewing</span>}
                    </div>
                    <div style={{ fontSize: '13px', lineHeight: 1.55, color: 'var(--txt2)', whiteSpace: 'pre-wrap' }}>{body}</div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
