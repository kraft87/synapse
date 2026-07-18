// Search — the omnibox target. Lexical BM25 over episodes/facts/entities/events
// (README §Search) — deliberately "not recall()". Type tabs with counts, result
// cards that deep-link (entity → dossier, others → episode overlay).
import { useEffect, useState } from 'react';
import { fetchSearch, type SearchHit, type SearchResult, type SearchTab } from '../api';
import { useStore } from '../state';
import { openEntity, openEpisode, openSession } from '../hash';
import { relTime, typeColor } from '../tokens';

const TABS: { key: SearchTab; label: string }[] = [
  { key: 'episodes', label: 'Episodes' },
  { key: 'facts', label: 'Facts' },
  { key: 'entities', label: 'Entities' },
  { key: 'events', label: 'Events' },
];
const singular: Record<SearchTab, string> = { episodes: 'episode', facts: 'fact', entities: 'entity', events: 'event' };
const LIMIT = 20;

function metaLine(h: SearchHit): string {
  if (h.type === 'entities') return [h.meta.entity_type, h.meta.degree != null ? h.meta.degree + ' edges' : ''].filter(Boolean).join(' · ');
  // Events: show when the event happened (t_valid), not when the row was
  // ingested — backfills stamp thousands of events with one ingested_at.
  const ts = h.type === 'events' ? (h.meta.t_valid ?? h.meta.ts) : h.meta.ts;
  return [h.meta.project, h.meta.source, relTime(ts)].filter(Boolean).join(' · ');
}
function openHit(h: SearchHit) {
  if (h.type === 'entities') return openEntity(h.id);
  if (h.type === 'episodes') return openEpisode(h.id);
  if (h.meta.episode_id != null) return openEpisode(h.meta.episode_id);
  if (h.meta.session_id) return openSession(h.meta.session_id);
}

export function Search() {
  const s = useStore();
  const { searchQuery, project, group, source } = s;
  const [tab, setTab] = useState<SearchTab>('episodes');
  const [res, setRes] = useState<SearchResult | null>(null);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const run = (offset: number) => {
    if (offset === 0) setLoading(true);
    setError(false);
    fetchSearch({ q: searchQuery, type: tab, offset, limit: LIMIT, project, group_id: group, source })
      .then((r) => { setRes(r); setHits((prev) => (offset === 0 ? r.hits : prev.concat(r.hits))); })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  };

  useEffect(() => { setHits([]); run(0); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [searchQuery, tab, project, group, source]);

  const counts = res?.total_by_type;
  const total = counts?.[tab] ?? 0;
  const canPage = hits.length < total;

  return (
    <main style={{ flex: 1, maxWidth: '900px', width: '100%', margin: '0 auto', padding: '20px 16px 80px', boxSizing: 'border-box' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '4px', flexWrap: 'wrap' }}>
        <div style={{ fontSize: '12px', fontFamily: 'var(--font-data)', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em' }}>lexical search</div>
        <div style={{ fontFamily: 'var(--font-data)', fontSize: '12px', color: 'var(--txt2)' }}>"{searchQuery}"</div>
      </div>
      <div style={{ fontFamily: 'var(--font-data)', fontSize: '11px', color: 'var(--txt3)', marginBottom: '14px' }}>
        BM25 over episodes · facts · entities · events — filters applied · not recall()
      </div>

      <div className="search-tabs" style={{ display: 'flex', gap: '2px', borderBottom: '1px solid var(--line)', marginBottom: '14px', flexWrap: 'wrap' }}>
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button key={t.key} onClick={() => setTab(t.key)}
              style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '8px 13px', fontSize: '13px', fontWeight: 500, color: active ? 'var(--txt)' : 'var(--txt2)', borderBottom: '2px solid ' + (active ? 'var(--acc)' : 'transparent'), marginBottom: '-1px', display: 'flex', gap: '6px', alignItems: 'center' }}>
              {t.label}
              <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--txt3)' }}>{counts ? counts[t.key] : '·'}</span>
            </button>
          );
        })}
      </div>

      {error && <div style={{ color: 'var(--err)', fontSize: '13px', fontFamily: 'var(--font-data)', padding: '8px 0' }}>search request failed.</div>}
      {loading && hits.length === 0 && !error && <div style={{ color: 'var(--txt3)', fontSize: '13px', fontFamily: 'var(--font-data)', padding: '8px 0' }}>searching…</div>}
      {!loading && hits.length === 0 && !error && <div style={{ color: 'var(--txt3)', fontSize: '13px', padding: '8px 0' }}>no {singular[tab]} matches for "{searchQuery}".</div>}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
        {hits.map((h) => (
          <button key={h.type + ':' + h.id} className="result" onClick={() => openHit(h)}
            style={{ textAlign: 'left', border: '1px solid var(--line)', background: 'var(--bg1)', borderRadius: '9px', padding: '10px 13px', cursor: 'pointer' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '5px' }}>
              <span style={{ fontFamily: 'var(--font-data)', fontSize: '10px', padding: '2px 7px', borderRadius: '4px', border: '1px solid var(--line2)', color: typeColor[h.type] || 'var(--txt2)' }}>{singular[h.type]}</span>
              {metaLine(h) && <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--txt3)' }}>{metaLine(h)}</span>}
              <span style={{ flex: 1 }} />
              <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: 'var(--txt3)' }}>↗</span>
            </div>
            <div style={{ fontSize: '13.5px', lineHeight: 1.5, color: 'var(--txt)' }}>{h.snippet}</div>
          </button>
        ))}
      </div>

      {hits.length > 0 && (
        <div style={{ textAlign: 'center', marginTop: '16px' }}>
          {canPage ? (
            <button className="softbtn" onClick={() => run(hits.length)}
              style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '7px', padding: '7px 16px', fontSize: '12.5px', fontFamily: 'var(--font-data)', cursor: 'pointer' }}>load more</button>
          ) : (
            <span style={{ fontFamily: 'var(--font-data)', fontSize: '11.5px', color: 'var(--txt3)' }}>{hits.length} of {total} · end of results</span>
          )}
        </div>
      )}
    </main>
  );
}
