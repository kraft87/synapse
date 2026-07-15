// One card shell, three card types (README §Feed / §3). Episodes expand by
// fetching /episode/:id (+ /derived); fact & timeline items expand inline from
// the feed payload. All expansion is in place — provenance opens overlays.
import { useEffect, useState, type CSSProperties } from 'react';
import { fetchDerived, fetchEpisode, type Derived, type Episode, type FeedItem } from '../api';
import { hasToolTrace, relTime, validityLine } from '../tokens';
import { openEpisode, openSession } from '../hash';
import { FlagButton, ProjectChip, SalLabel, SourceChip, TypeChip } from './ui';

const provBtn: CSSProperties = {
  border: 'none', background: 'none', padding: 0, color: 'var(--acc)', cursor: 'pointer',
  fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline', textUnderlineOffset: '3px',
};
const softBtn: CSSProperties = {
  marginTop: '8px', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)',
  borderRadius: '5px', padding: '3px 9px', fontSize: '11.5px', fontFamily: 'var(--font-data)', cursor: 'pointer',
};

export function DerivedBlock({ derived, heading = 'extracted from this episode', bare = false }: { derived: Derived; heading?: string; bare?: boolean }) {
  const rows = [
    ...derived.facts.map((f) => ({ kind: 'fact', color: 'var(--et-concept)', text: f.fact })),
    ...derived.timeline_events.map((e) => ({ kind: 'timeline', color: 'var(--et-org)', text: e.fact })),
  ];
  if (rows.length === 0) return null;
  return (
    <div style={bare
      ? { marginTop: '14px', borderTop: '1px solid var(--line)', paddingTop: '12px' }
      : { marginTop: '10px', border: '1px solid var(--line)', borderRadius: '8px', padding: '9px 11px', background: 'var(--bg2)' }}>
      <div style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: '6px' }}>
        {heading} · {rows.length} items
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
        {rows.map((d, i) => (
          <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'baseline', fontSize: '12.5px', lineHeight: 1.45, color: 'var(--txt2)' }}>
            <span style={{ fontFamily: 'var(--font-data)', fontSize: '10px', color: d.color, minWidth: '52px' }}>{d.kind}</span>
            <span>{d.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EpisodeExpansion({ item }: { item: FeedItem }) {
  const [ep, setEp] = useState<Episode | null>(null);
  const [derived, setDerived] = useState<Derived | null>(null);
  const [err, setErr] = useState(false);
  const [traceOpen, setTraceOpen] = useState(false);

  useEffect(() => {
    let live = true;
    Promise.all([fetchEpisode(item.id), fetchDerived(item.id).catch(() => ({ facts: [], timeline_events: [] } as Derived))])
      .then(([e, d]) => { if (live) { setEp(e); setDerived(d); } })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, [item.id]);

  if (err) return <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--line)', color: 'var(--err)', fontSize: '12.5px', fontFamily: 'var(--font-data)' }}>couldn't load episode</div>;
  if (!ep) return <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--line)', color: 'var(--txt3)', fontSize: '12.5px', fontFamily: 'var(--font-data)' }}>loading…</div>;

  const body = [ep.human_turn && 'user: ' + ep.human_turn, ep.assistant_turn && 'assistant: ' + ep.assistant_turn].filter(Boolean).join('\n\n');
  const showTrace = hasToolTrace(ep.content);
  const sessionId = item.data.session_id;

  return (
    <>
      <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--line)', fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', whiteSpace: 'pre-wrap' }}>{body}</div>
      {showTrace && (
        <>
          <button className="softbtn" onClick={() => setTraceOpen((v) => !v)} style={softBtn}>{traceOpen ? 'hide tool-call trace' : 'show tool-call trace'}</button>
          {traceOpen && (
            <pre style={{ margin: '8px 0 0', background: 'var(--bg0)', border: '1px solid var(--line)', borderRadius: '6px', padding: '10px 12px', fontFamily: 'var(--font-data)', fontSize: '11.5px', lineHeight: 1.55, color: 'var(--txt2)', overflowX: 'auto' }}>{ep.content}</pre>
          )}
        </>
      )}
      {derived && <DerivedBlock derived={derived} />}
      {sessionId && (
        <div>
          <button className="softbtn" style={softBtn} onClick={() => openSession(sessionId, item.id)}>open session ↗</button>
        </div>
      )}
    </>
  );
}

function InlineExpansion({ item }: { item: FeedItem }) {
  const d = item.data;
  const isFact = item.type === 'fact';
  const detail = isFact
    ? (d.src_name && d.tgt_name ? `${d.src_name} — ${d.fact} → ${d.tgt_name}` : d.fact || '')
    : d.fact || '';
  const provEp = isFact ? d.provenance_episode_id : d.episode_id;
  const validity = isFact ? validityLine(d.t_valid, d.t_invalid) : (d.t_valid ? 'dated ' + d.t_valid.slice(0, 10) : '');
  return (
    <>
      {detail && detail !== item.gist && (
        <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--line)', fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', whiteSpace: 'pre-wrap' }}>{detail}</div>
      )}
      <div style={{ marginTop: detail && detail !== item.gist ? '8px' : '10px', paddingTop: detail && detail !== item.gist ? 0 : '10px', borderTop: detail && detail !== item.gist ? 'none' : '1px solid var(--line)', display: 'flex', gap: '14px', flexWrap: 'wrap', fontFamily: 'var(--font-data)', fontSize: '11.5px', color: 'var(--txt3)' }}>
        {provEp != null && (
          <button className="linkbtn" style={provBtn} onClick={() => openEpisode(provEp)}>source: ep-{provEp}</button>
        )}
        {validity && <span>{validity}</span>}
      </div>
    </>
  );
}

export function FeedCard({ item, fresh }: { item: FeedItem; fresh?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const sal = item.type === 'timeline_event' ? item.sal : undefined;
  // Feed-specific salience ramp (README §Feed): timeline sal>0.7 -> 15px/600.
  const big = item.type === 'timeline_event' && sal != null && sal > 0.7;
  const gistSize = big ? '15px' : '13.5px';
  const gistWeight = big ? 600 : item.type === 'fact' ? 500 : 400;

  return (
    <article style={{
      background: 'var(--bg1)', border: '1px solid ' + (fresh ? 'var(--line2)' : 'var(--line)'),
      borderRadius: '10px', padding: '12px 14px', animation: fresh ? 'slidein .5s ease' : 'none',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <TypeChip type={item.type} />
        <ProjectChip project={item.project} />
        <SourceChip source={item.source} />
        {item.type === 'timeline_event' && <SalLabel sal={sal} />}
        <span style={{ flex: 1 }} />
        <FlagButton kind={item.type} itemId={item.id} initial={item.flagged} />
        <span style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)' }}>{relTime(item.ts)}</span>
      </div>

      <div onClick={() => setExpanded((v) => !v)}
        style={{ cursor: 'pointer', marginTop: '8px', fontSize: gistSize, fontWeight: gistWeight, lineHeight: 1.5, color: 'var(--txt)' }}>
        {item.gist}
      </div>

      {expanded && (item.type === 'episode' ? <EpisodeExpansion item={item} /> : <InlineExpansion item={item} />)}
    </article>
  );
}
