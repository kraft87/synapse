// Timeline — the life/work event log (README §4, screenshots 04/05). Two tabs:
//   Events — type-chip + jump-to-date filtered, month-bucketed cards on a left rail; each
//            event's text is sized/weighted by the salience ramp (spec §6); provenance
//            "→ ep-N" opens the episode overlay.
//   Preferences — the standing like/dislike/rule log, sortable by recency or assert count;
//                 superseded rows render struck with their superseding note.
// The header group filter scopes Events (via the domain column); Preferences are
// cross-domain (spec §1), so they ignore it.
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  fetchTimeline, fetchPreferences,
  type TimelineEvent, type Preference,
} from '../api';
import { useStore } from '../state';
import { openEpisode } from '../hash';
import { FlagButton } from '../components/ui';

const mono = 'var(--font-data)';
const wrap: CSSProperties = { flex: 1, maxWidth: '900px', width: '100%', margin: '0 auto', padding: '18px 16px 80px', boxSizing: 'border-box' };
const monoHead: CSSProperties = { fontSize: '11px', fontFamily: mono, color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em' };

const TABS = [
  { key: 'events', label: 'Events' },
  { key: 'preferences', label: 'Preferences' },
] as const;
type Tab = (typeof TABS)[number]['key'];

// The five type-filter chips (README §4). Colored border+text when active.
const TL_TYPES = ['work', 'life', 'idea', 'health', 'milestone'] as const;
const TL_TYPE_COLOR: Record<string, string> = {
  work: 'var(--acc)', life: 'var(--et-person)', idea: 'var(--et-tech)',
  health: 'var(--ok)', milestone: 'var(--warn)',
};
const typeColor = (t?: string | null): string => (t && TL_TYPE_COLOR[t]) || 'var(--txt2)';

// Salience → type ramp, EXACTLY per spec §6 (only the readout goes amber; size+weight carry it).
const salRamp = (sal: number): { fontSize: string; fontWeight: number } =>
  sal > 0.75 ? { fontSize: '16px', fontWeight: 600 }
    : sal > 0.55 ? { fontSize: '14px', fontWeight: 500 }
      : { fontSize: '13px', fontWeight: 400 };
// Left-rail dot color by salience tier (high=accent, med=amber, low=muted).
const dotColor = (sal: number): string => (sal > 0.75 ? 'var(--acc)' : sal > 0.55 ? 'var(--warn)' : 'var(--txt3)');

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const monthKey = (iso: string): string => iso.slice(0, 7); // YYYY-MM
const monthLabel = (key: string): string => {
  const [y, m] = key.split('-');
  return (MONTHS[Number(m) - 1] || m) + ' ' + y;
};
// First instant of the month AFTER `key` — the `before=` boundary that loads that month down.
const monthAfter = (key: string): string => {
  const [y, m] = key.split('-').map(Number);
  const ny = m === 12 ? y + 1 : y;
  const nm = m === 12 ? 1 : m + 1;
  return `${ny}-${String(nm).padStart(2, '0')}-01T00:00:00+00:00`;
};

const errBox = (msg: string, retry?: () => void) => (
  <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', display: 'flex', alignItems: 'center', gap: '12px' }}>
    <span style={{ color: 'var(--err)', fontSize: '13px', flex: 1 }}>{msg}</span>
    {retry && <button className="softbtn" style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '5px', padding: '4px 10px', fontSize: '11.5px', fontFamily: mono, cursor: 'pointer' }} onClick={retry}>retry</button>}
  </div>
);

// =====================================================================================
// Events tab
// =====================================================================================
function EventCard({ ev }: { ev: TimelineEvent }) {
  const ramp = salRamp(ev.sal);
  return (
    <div style={{ display: 'flex', gap: '10px' }}>
      {/* left rail + dot */}
      <div style={{ position: 'relative', width: '12px', flexShrink: 0 }}>
        <div style={{ position: 'absolute', left: '5px', top: 0, bottom: '-8px', width: '2px', background: 'var(--line2)' }} />
        <div style={{ position: 'absolute', left: '0px', top: '15px', width: '11px', height: '11px', borderRadius: '50%', background: dotColor(ev.sal), border: '2px solid var(--bg0)', boxSizing: 'border-box' }} />
      </div>
      <article style={{ flex: 1, minWidth: 0, background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px', padding: '12px 14px', marginBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)' }}>{ev.t_valid.slice(5, 10)}</span>
          {ev.event_type && <span style={{ fontFamily: mono, fontSize: '11px', color: typeColor(ev.event_type) }}>{ev.event_type}</span>}
          <span style={{ flex: 1 }} />
          <FlagButton kind="timeline_event" itemId={String(ev.id)} initial={ev.flagged} />
        </div>
        <div style={{ marginTop: '5px', color: 'var(--txt)', lineHeight: 1.45, ...ramp }}>{ev.fact}</div>
        {ev.episode_id != null && (
          <button className="linkbtn" onClick={() => openEpisode(ev.episode_id as number)}
            style={{ marginTop: '8px', border: 'none', background: 'none', padding: 0, color: 'var(--acc)', cursor: 'pointer', fontFamily: mono, fontSize: '11.5px' }}>
            → ep-{ev.episode_id}
          </button>
        )}
      </article>
    </div>
  );
}

function Events() {
  const { group } = useStore();
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [nextBefore, setNextBefore] = useState<string | null>(null);
  const [activeType, setActiveType] = useState<string | null>(null);
  const [jump, setJump] = useState('');
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(false);
  const seenRef = useRef<Set<number>>(new Set());

  const groupId = group === 'all' ? undefined : group;

  const load = (before?: string) => {
    setLoading(true); setError(false);
    seenRef.current = new Set();
    fetchTimeline({ before, type: activeType || undefined, group_id: groupId })
      .then((r) => { seenRef.current = new Set(r.events.map((e) => e.id)); setEvents(r.events); setNextBefore(r.next_before); })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  };

  // (re)load whenever the type chip or the header group filter changes.
  useEffect(() => { setJump(''); load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [activeType, group]);

  const loadMore = () => {
    if (!nextBefore || loadingMore) return;
    setLoadingMore(true);
    fetchTimeline({ before: nextBefore, type: activeType || undefined, group_id: groupId })
      .then((r) => {
        const fresh = r.events.filter((e) => !seenRef.current.has(e.id));
        fresh.forEach((e) => seenRef.current.add(e.id));
        setEvents((prev) => prev.concat(fresh));
        setNextBefore(r.next_before);
      })
      .catch(() => setError(true))
      .finally(() => setLoadingMore(false));
  };

  const buckets = useMemo(() => {
    const map = new Map<string, TimelineEvent[]>();
    for (const e of events) {
      const k = monthKey(e.t_valid);
      let arr = map.get(k);
      if (!arr) { arr = []; map.set(k, arr); }
      arr.push(e);
    }
    return [...map.entries()]; // events already t_valid DESC, so keys land newest-first
  }, [events]);

  const monthsPresent = useMemo(() => buckets.map(([k]) => k), [buckets]);

  const onJump = (key: string) => {
    setJump(key);
    if (key) load(monthAfter(key));
  };

  return (
    <div>
      {/* filter row: type chips + jump-to-date */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '18px' }}>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {TL_TYPES.map((t) => {
            const on = activeType === t;
            const c = TL_TYPE_COLOR[t];
            return (
              <button key={t} className="chipbtn" onClick={() => setActiveType(on ? null : t)}
                style={{ fontFamily: mono, fontSize: '11px', borderRadius: '11px', padding: '3px 12px', cursor: 'pointer',
                  border: '1px solid ' + (on ? c : 'var(--line2)'), color: on ? c : 'var(--txt3)', background: 'transparent' }}>
                {t}
              </button>
            );
          })}
        </div>
        <span style={{ flex: 1 }} />
        <select value={jump} onChange={(e) => onJump(e.target.value)}
          style={{ background: 'var(--bg2)', border: '1px solid var(--line2)', borderRadius: '6px', padding: '5px 8px', fontFamily: mono, fontSize: '12px', color: monthsPresent.length ? 'var(--txt)' : 'var(--txt3)', cursor: 'pointer' }}>
          <option value="">jump to date…</option>
          {monthsPresent.map((k) => <option key={k} value={k}>{monthLabel(k)}</option>)}
        </select>
      </div>

      {error && <div style={{ marginBottom: '12px' }}>{errBox('Timeline request failed.', () => load())}</div>}
      {loading && events.length === 0 && <div style={{ color: 'var(--txt3)', fontFamily: mono, fontSize: '12px', padding: '20px 0' }}>loading events…</div>}
      {!loading && events.length === 0 && !error && (
        <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
          <div style={{ fontFamily: mono, fontSize: '13px', marginBottom: '6px' }}>no events{activeType ? ` of type "${activeType}"` : ''}</div>
          <div style={{ fontSize: '13px', color: 'var(--txt3)' }}>the timeline records dated happenings — shipped, decided, committed.</div>
        </div>
      )}

      {buckets.map(([key, evs]) => (
        <section key={key} style={{ marginBottom: '26px' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '12px' }}>
            <span style={{ fontFamily: mono, fontSize: '12.5px', color: 'var(--acc)' }}>{monthLabel(key)}</span>
            <div style={{ flex: 1, height: '1px', background: 'var(--line)' }} />
            <span style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)' }}>{evs.length} event{evs.length === 1 ? '' : 's'}</span>
          </div>
          {evs.map((ev) => <EventCard key={ev.id} ev={ev} />)}
        </section>
      ))}

      {nextBefore && (
        <div style={{ textAlign: 'center', marginTop: '10px' }}>
          <button className="softbtn" onClick={loadMore} disabled={loadingMore}
            style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '7px', padding: '7px 16px', fontSize: '12.5px', fontFamily: mono, cursor: 'pointer' }}>
            {loadingMore ? 'loading…' : 'load more'}
          </button>
        </div>
      )}
    </div>
  );
}

// =====================================================================================
// Preferences tab
// =====================================================================================
const POLARITY_COLOR: Record<string, string> = { like: 'var(--ok)', dislike: 'var(--err)', rule: 'var(--txt2)' };

function PolarityChip({ polarity }: { polarity: string }) {
  const c = POLARITY_COLOR[polarity] || 'var(--txt2)';
  return <span style={{ fontFamily: mono, fontSize: '10.5px', border: '1px solid ' + c, color: c, borderRadius: '4px', padding: '2px 8px', justifySelf: 'start' }}>{polarity}</span>;
}

const PREF_GRID = 'minmax(0,1fr) 74px 96px 74px';

function PrefRow({ p }: { p: Preference }) {
  const superseded = p.t_invalid != null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: PREF_GRID, alignItems: 'center', gap: '10px', padding: '11px 14px', borderTop: '1px solid var(--line)' }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: '13.5px', color: superseded ? 'var(--txt3)' : 'var(--txt)', textDecoration: superseded ? 'line-through' : 'none' }}>{p.pref}</div>
        {superseded && (
          <div style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)', marginTop: '3px' }}>
            ↳ superseded by: {p.superseded_by_text || `#${p.superseded_by}`}{p.t_invalid ? ` (${p.t_invalid.slice(0, 7)})` : ''}
          </div>
        )}
      </div>
      <PolarityChip polarity={p.polarity} />
      <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)', textAlign: 'right' }}>{(p.first_seen || '').slice(0, 10)}</span>
      <span style={{ fontFamily: mono, fontSize: '12px', color: superseded ? 'var(--txt3)' : 'var(--txt2)', textAlign: 'right' }}>{p.assert_count}×</span>
    </div>
  );
}

function Preferences() {
  const [sort, setSort] = useState<'recency' | 'assert_count'>('recency');
  const [prefs, setPrefs] = useState<Preference[] | null>(null);
  const [error, setError] = useState(false);

  const load = () => {
    setError(false);
    fetchPreferences(sort).then((r) => setPrefs(r.preferences)).catch(() => setError(true));
  };
  useEffect(() => { load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [sort]);

  const segBtn = (key: 'recency' | 'assert_count', label: string) => (
    <button onClick={() => setSort(key)}
      style={{ border: 'none', cursor: 'pointer', padding: '5px 12px', fontFamily: mono, fontSize: '12px',
        background: sort === key ? 'var(--acc-bg)' : 'var(--bg2)', color: sort === key ? 'var(--acc)' : 'var(--txt2)' }}>
      {label}
    </button>
  );

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px' }}>
        <span style={{ ...monoHead }}>sort</span>
        <div style={{ display: 'flex', border: '1px solid var(--line2)', borderRadius: '6px', overflow: 'hidden' }}>
          {segBtn('recency', 'recency')}
          <div style={{ width: '1px', background: 'var(--line2)' }} />
          {segBtn('assert_count', 'assert count')}
        </div>
      </div>

      {error && errBox('Preferences request failed.', load)}
      {!error && prefs === null && <div style={{ color: 'var(--txt3)', fontFamily: mono, fontSize: '12px', padding: '20px 0' }}>loading preferences…</div>}
      {!error && prefs !== null && prefs.length === 0 && (
        <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
          <div style={{ fontFamily: mono, fontSize: '13px' }}>no preferences recorded yet</div>
        </div>
      )}
      {!error && prefs !== null && prefs.length > 0 && (
        <div style={{ border: '1px solid var(--line)', borderRadius: '10px', background: 'var(--bg1)', overflow: 'hidden' }}>
          <div style={{ display: 'grid', gridTemplateColumns: PREF_GRID, gap: '10px', padding: '9px 14px', background: 'var(--bg2)' }}>
            <span style={monoHead}>preference</span>
            <span style={{ ...monoHead, justifySelf: 'start' }}>polarity</span>
            <span style={{ ...monoHead, textAlign: 'right' }}>since</span>
            <span style={{ ...monoHead, textAlign: 'right' }}>asserts</span>
          </div>
          {prefs.map((p) => <PrefRow key={p.id} p={p} />)}
        </div>
      )}
    </div>
  );
}

// =====================================================================================
export function Timeline() {
  const [tab, setTab] = useState<Tab>(() =>
    (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('synapse.tlTab') === 'preferences') ? 'preferences' : 'events');
  useEffect(() => {
    // A jump from Review → Flags may request the Preferences tab (one-shot).
    if (typeof sessionStorage !== 'undefined') sessionStorage.removeItem('synapse.tlTab');
  }, []);

  return (
    <main style={wrap}>
      <div className="search-tabs" style={{ display: 'flex', gap: '2px', borderBottom: '1px solid var(--line)', marginBottom: '18px', flexWrap: 'wrap' }}>
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button key={t.key} className="recall-tab" onClick={() => setTab(t.key)}
              style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '8px 13px', fontSize: '13px', fontWeight: 500, color: active ? 'var(--txt)' : 'var(--txt2)', borderBottom: '2px solid ' + (active ? 'var(--acc)' : 'transparent'), marginBottom: '-1px' }}>
              {t.label}
            </button>
          );
        })}
      </div>
      {tab === 'events' ? <Events /> : <Preferences />}
    </main>
  );
}
