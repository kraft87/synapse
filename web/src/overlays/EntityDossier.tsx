// Entity dossier overlay (#/entity/:id) — the subject-centric reading surface.
// All facts (valid + superseded struck), paginated mentions, provenance that
// opens the episode overlay. "view in graph" is disabled until the Graph phase.
import type React from 'react';
import { useEffect, useState } from 'react';
import { fetchEntity, type Entity } from '../api';
import { closeOverlay, openEpisode } from '../hash';
import { etColor, validityLine } from '../tokens';
import { Dot, FlagButton, Spinner } from '../components/ui';
import { useStore } from '../state';

const provBtn: React.CSSProperties = { border: 'none', background: 'none', padding: 0, color: 'var(--acc)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline', textUnderlineOffset: '2px' };

export function EntityDossier({ id }: { id: string }) {
  const store = useStore();
  const [data, setData] = useState<Entity | null>(null);
  const [mentions, setMentions] = useState<Entity['mentions']['items']>([]);
  const [offset, setOffset] = useState(0);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let live = true;
    setData(null); setMentions([]); setOffset(0); setErr(false);
    fetchEntity(id, 0).then((d) => { if (live) { setData(d); setMentions(d.mentions.items); setOffset(d.mentions.items.length); } })
      .catch(() => { if (live) setErr(true); });
    return () => { live = false; };
  }, [id]);

  const loadMoreMentions = () => {
    fetchEntity(id, offset).then((d) => { setMentions((prev) => prev.concat(d.mentions.items)); setOffset((o) => o + d.mentions.items.length); }).catch(() => {});
  };

  const e = data?.entity;
  const color = etColor(e?.entity_type);
  const total = data?.mentions.total ?? 0;

  return (
    <div onClick={closeOverlay} style={{ position: 'fixed', inset: 0, background: 'var(--scrim)', zIndex: 60, display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: '24px', backdropFilter: 'blur(2px)', overflowY: 'auto' }}>
      <div onClick={(ev) => ev.stopPropagation()} style={{ background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '14px', maxWidth: '760px', width: '100%', padding: '22px 24px', boxSizing: 'border-box', margin: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <Dot color={color} size={12} />
          <span style={{ fontFamily: 'var(--font-data)', fontSize: '18px', fontWeight: 500 }}>{e?.name || '…'}</span>
          {e?.entity_type && <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', padding: '2px 8px', borderRadius: '4px', border: '1px solid ' + color, color }}>{e.entity_type}</span>}
          <span style={{ flex: 1 }} />
          <button
            className="chipbtn"
            title="seed the graph explorer with this entity"
            disabled={!e}
            onClick={() => {
              if (!e) return;
              store.seedGraph(e.uuid, e.name);
              store.setPage('graph');
              closeOverlay();
            }}
            style={{ border: '1px solid var(--acc)', background: 'var(--acc-bg)', color: 'var(--acc)', borderRadius: '6px', padding: '5px 12px', fontSize: '12px', fontFamily: 'var(--font-data)', cursor: e ? 'pointer' : 'not-allowed', opacity: e ? 1 : 0.55 }}
          >view in graph</button>
          <button className="iconbtn" onClick={closeOverlay} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', width: 26, height: 26, fontSize: '14px', lineHeight: 1 }}>×</button>
        </div>

        {err && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', marginTop: '14px' }}>couldn't load entity.</div>}
        {!data && !err && <Spinner label="loading dossier…" />}

        {data && e && (
          <>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', margin: '8px 0 12px' }}>
              {data.stats.edges} edges · served {data.stats.served}× · {data.stats.facts} facts
            </div>
            {e.summary && <div style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', marginBottom: '16px' }}>{e.summary}</div>}

            <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: '8px' }}>facts · valid + superseded</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '16px' }}>
              {data.facts.map((f) => {
                const dead = f.t_invalid != null;
                return (
                  <div key={f.uuid} style={{ border: '1px solid var(--line)', borderRadius: '8px', padding: '8px 11px', opacity: dead ? 0.55 : 1 }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                      <div style={{ flex: 1, fontSize: '12.5px', lineHeight: 1.5, color: 'var(--txt)', textDecoration: dead ? 'line-through' : 'none' }}>{f.fact}</div>
                      <FlagButton kind="fact" itemId={f.uuid} initial={f.flagged} size={11} />
                    </div>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '5px', fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--txt3)', flexWrap: 'wrap' }}>
                      <span>{validityLine(f.t_valid, f.t_invalid)}</span>
                      {f.provenance_episode_id != null && <button className="linkbtn" style={provBtn} onClick={() => openEpisode(f.provenance_episode_id!)}>→ ep-{f.provenance_episode_id}</button>}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: '8px' }}>mentions · {total} episodes</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {mentions.map((m, i) => (
                <button key={i} className="result" onClick={() => openEpisode(m.episode_id)} style={{ textAlign: 'left', border: '1px solid var(--line)', background: 'var(--bg2)', borderRadius: '7px', padding: '7px 11px', cursor: 'pointer', fontSize: '12.5px', lineHeight: 1.4, color: 'var(--txt2)' }}>
                  <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--acc)' }}>ep-{m.episode_id}</span> · {m.gist}
                </button>
              ))}
            </div>
            {mentions.length < total && (
              <div style={{ textAlign: 'center', marginTop: '10px' }}>
                <button className="softbtn" onClick={loadMoreMentions} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', padding: '5px 12px', fontSize: '11.5px', fontFamily: 'var(--font-data)', cursor: 'pointer' }}>load more mentions</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
