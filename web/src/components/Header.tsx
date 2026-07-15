// Persistent header (README §Global structure): wordmark, 6-button nav, omnibox,
// filter cluster (dimmed where a filter doesn't apply per spec §1), live status,
// theme toggle.
import { useState, type CSSProperties } from 'react';
import { useStore, type Page } from '../state';

const NAV: { key: Page; label: string }[] = [
  { key: 'feed', label: 'Feed' },
  { key: 'recall', label: 'Recall' },
  { key: 'graph', label: 'Graph' },
  { key: 'timeline', label: 'Timeline' },
  { key: 'metrics', label: 'Metrics' },
  { key: 'review', label: 'Review' },
];

// Which filters apply on which page (spec §1 table). true = applies.
const SCOPE: Record<Page, { group: boolean; project: boolean; source: boolean }> = {
  feed: { group: true, project: true, source: true },
  search: { group: true, project: true, source: true },
  recall: { group: true, project: true, source: false },
  timeline: { group: true, project: false, source: false },
  graph: { group: false, project: false, source: false },
  metrics: { group: false, project: false, source: false },
  review: { group: false, project: false, source: false },
};

const selStyle: CSSProperties = {
  background: 'var(--bg2)', border: '1px solid var(--line2)', borderRadius: '6px',
  padding: '5px 8px', fontSize: '12.5px', fontFamily: 'var(--font-data)', cursor: 'pointer',
  // Native selects size to their longest <option>; a live catalog carries 70+
  // project names, so an unclamped select eats the whole filter row and wraps
  // the source select off the header.
  maxWidth: '160px', textOverflow: 'ellipsis',
};

// Review pending-proposal count — 0 in phase 1 (no /proposals endpoint yet), so
// the badge slot stays hidden.
const REVIEW_BADGE = 0;

export function Header() {
  const s = useStore();
  const scope = SCOPE[s.page];
  const dim = (applies: boolean): CSSProperties =>
    applies ? {} : { opacity: 0.4 };
  const dimTitle = (applies: boolean) => (applies ? undefined : 'not scoped on this page');

  const [omni, setOmni] = useState('');
  const runSearch = () => {
    const q = omni.trim();
    if (q) { s.setSearchQuery(q); s.setPage('search'); }
  };

  const grpBtn = (g: 'technical' | 'personal'): CSSProperties => ({
    border: 'none', cursor: 'pointer', padding: '5px 10px', fontSize: '12.5px', fontFamily: 'var(--font-data)',
    background: s.group === g ? 'var(--acc-bg)' : 'var(--bg2)', color: s.group === g ? 'var(--acc)' : 'var(--txt2)',
  });

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 40, display: 'flex', alignItems: 'center', gap: '16px',
      padding: '6px 16px', minHeight: '52px', boxSizing: 'border-box', rowGap: '6px',
      background: 'var(--bg1)', borderBottom: '1px solid var(--line)', flexWrap: 'wrap',
    }}>
      {/* wordmark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{ width: 14, height: 14, border: '1.5px solid var(--acc)', transform: 'rotate(45deg)', borderRadius: '2px' }} />
        <div style={{ fontFamily: 'var(--font-data)', fontWeight: 500, fontSize: '14px', letterSpacing: '.04em' }}>synapse</div>
      </div>

      {/* nav */}
      <nav style={{ display: 'flex', gap: '2px' }}>
        {NAV.map((n) => {
          const active = s.page === n.key;
          return (
            <button key={n.key} className="navbtn" onClick={() => s.setPage(n.key)}
              style={{
                position: 'relative', border: 'none', background: active ? 'var(--bg3)' : 'none',
                color: active ? 'var(--txt)' : 'var(--txt2)', padding: '6px 12px', borderRadius: '6px',
                fontSize: '13.5px', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px',
              }}>
              {n.label}
              {n.key === 'review' && REVIEW_BADGE > 0 && (
                <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', background: 'var(--acc)', color: '#0d1116', borderRadius: '9px', padding: '0 6px', lineHeight: '16px', fontWeight: 600 }}>{REVIEW_BADGE}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* omnibox */}
      <div style={{ flex: 1, minWidth: '180px', maxWidth: '340px', position: 'relative' }}>
        <span style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--txt3)', fontSize: '12px', pointerEvents: 'none' }}>⌕</span>
        <input className="field" value={omni} spellCheck={false}
          onChange={(e) => setOmni(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') runSearch(); }}
          placeholder="search episodes, facts, entities, events…"
          style={{ width: '100%', boxSizing: 'border-box', background: 'var(--bg2)', border: '1px solid var(--line2)', borderRadius: '6px', padding: '6px 10px 6px 26px', fontFamily: 'var(--font-data)', fontSize: '12px', color: 'var(--txt)' }} />
      </div>

      {/* filter cluster */}
      <div className="hdr-filters" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div title={dimTitle(scope.group)} style={{ display: 'flex', border: '1px solid var(--line2)', borderRadius: '6px', overflow: 'hidden', ...dim(scope.group) }}>
          <button onClick={() => s.setGroup(s.group === 'technical' ? 'all' : 'technical')} style={grpBtn('technical')}>technical</button>
          <button onClick={() => s.setGroup(s.group === 'personal' ? 'all' : 'personal')} style={{ ...grpBtn('personal'), borderLeft: '1px solid var(--line2)' }}>personal</button>
        </div>
        <select title={dimTitle(scope.project)} value={s.project} onChange={(e) => s.setProject(e.target.value)} style={{ ...selStyle, ...dim(scope.project) }}>
          <option value="all">all projects</option>
          {s.catalog?.projects.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
        <select title={dimTitle(scope.source)} value={s.source} onChange={(e) => s.setSource(e.target.value)} style={{ ...selStyle, ...dim(scope.source) }}>
          <option value="all">all sources</option>
          {s.catalog?.sources.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
      </div>

      {/* live status + theme */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div title={s.online ? 'polling — last feed poll ok' : 'reconnecting'} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-data)', fontSize: '11.5px', color: 'var(--txt2)' }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: s.online ? 'var(--ok)' : 'var(--warn)', animation: s.online ? 'pulse 1.6s ease-in-out infinite' : 'none' }} />
          <span>{s.online ? 'queue —' : 'reconnecting'}</span>
        </div>
        <button className="iconbtn" onClick={s.toggleTheme} title="Toggle theme"
          style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', borderRadius: '6px', width: 28, height: 28, color: 'var(--txt2)', fontSize: '13px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {s.theme === 'dark' ? '◐' : '◑'}
        </button>
      </div>
    </header>
  );
}
