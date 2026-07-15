// Feed — "watch it remember". Keyset pagination via next_cursor; a live SSE stream
// (phase 3) prepends new writes with the slidein animation, with 30s polling kept as the
// automatic fallback if the stream fails persistently. Skeleton / empty / error states
// (companion spec §2). The live dot + queue badge reflect the stream's processing_status.
import type React from 'react';
import { useEffect, useRef, useState } from 'react';
import { fetchFeed, openFeedStream, feedItemPassesFilter, MOCK, type FeedItem } from '../api';
import { useStore } from '../state';
import { FeedCard } from '../components/FeedCard';

const keyOf = (it: FeedItem) => it.type + ':' + it.id;
const POLL_MS = 30_000;

const wrap: React.CSSProperties = { flex: 1, maxWidth: '860px', width: '100%', margin: '0 auto', padding: '20px 16px 80px', boxSizing: 'border-box' };
const monoHead: React.CSSProperties = { fontSize: '12px', fontFamily: 'var(--font-data)', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em' };

function Skeletons() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} style={{ background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px', padding: '12px 14px' }}>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
            <div style={{ width: 58, height: 15, background: 'var(--bg2)', borderRadius: '4px', animation: 'skeleton 1.4s ease-in-out infinite' }} />
            <div style={{ width: 52, height: 15, background: 'var(--bg2)', borderRadius: '4px', animation: 'skeleton 1.4s ease-in-out infinite' }} />
          </div>
          <div style={{ width: '72%', height: 13, background: 'var(--bg2)', borderRadius: '4px', animation: 'skeleton 1.4s ease-in-out infinite' }} />
        </div>
      ))}
    </div>
  );
}

export function Feed() {
  const s = useStore();
  const { project, group, source, setOnline, setQueueDepth } = s;
  const [items, setItems] = useState<FeedItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(false);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  const itemsRef = useRef<FeedItem[]>([]);
  itemsRef.current = items;

  const filters = { project, group_id: group, source };

  const load = () => {
    setLoading(true); setError(false);
    fetchFeed({ ...filters })
      .then((r) => { setItems(r.items); setCursor(r.next_cursor); setFresh(new Set()); setOnline(true); })
      .catch(() => { setError(true); setOnline(false); })
      .finally(() => setLoading(false));
  };

  // (re)load page 1 whenever the filters change, then subscribe to the live SSE stream.
  // The stream prepends new writes that pass the current filter; a persistent stream
  // failure (>5 reconnect attempts) trips the 30s polling fallback. MOCK has no server,
  // so it stays on polling.
  useEffect(() => {
    load();

    // One incremental poll: fetch page 1 and prepend anything unseen (the fallback path
    // and the only path in MOCK). Captures the current filters at effect creation.
    const pollOnce = () => {
      fetchFeed({ ...filters }).then((r) => {
        const seen = new Set(itemsRef.current.map(keyOf));
        const incoming = r.items.filter((it) => !seen.has(keyOf(it)));
        setOnline(true);
        if (incoming.length) {
          setFresh(new Set(incoming.map(keyOf)));
          setItems((prev) => incoming.concat(prev));
        }
      }).catch(() => setOnline(false));
    };

    if (MOCK) {
      const t = setInterval(pollOnce, POLL_MS);
      return () => clearInterval(t);
    }

    let pollTimer: ReturnType<typeof setInterval> | null = null;
    const startPolling = () => {
      if (pollTimer != null) return;
      pollTimer = setInterval(pollOnce, POLL_MS);
    };

    const prepend = (item: FeedItem) => {
      if (!feedItemPassesFilter(item, { project, group, source })) return; // filtered out -> drop
      const k = keyOf(item);
      setItems((prev) => (prev.some((it) => keyOf(it) === k) ? prev : [item, ...prev]));
      setFresh(new Set([k]));
    };

    const stream = openFeedStream({
      onOpen: () => setOnline(true),
      onStatus: (st) => { setOnline(true); setQueueDepth(st.queue_depth); },
      onItem: (item) => { setOnline(true); prepend(item); },
      onReset: () => load(),                 // buffer aged out -> refetch page 1
      onDisconnect: () => setOnline(false),  // transient loss -> amber, keep stale items
      onGiveUp: () => { setOnline(false); startPolling(); }, // persistent -> 30s polling
    });

    return () => {
      stream.close();
      if (pollTimer != null) clearInterval(pollTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, group, source]);

  const loadMore = () => {
    if (!cursor || loadingMore) return;
    setLoadingMore(true);
    fetchFeed({ ...filters, cursor })
      .then((r) => {
        const seen = new Set(itemsRef.current.map(keyOf));
        setItems((prev) => prev.concat(r.items.filter((it) => !seen.has(keyOf(it)))));
        setCursor(r.next_cursor);
        setOnline(true);
      })
      .catch(() => setError(true))
      .finally(() => setLoadingMore(false));
  };

  const filtersActive = group !== 'all' || project !== 'all' || source !== 'all';
  const empty = !loading && items.length === 0;

  return (
    <main style={wrap}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={monoHead}>live memory stream</div>
        <div style={{ fontSize: '12px', fontFamily: 'var(--font-data)', color: 'var(--txt3)' }}>
          {items.length} items · newest first
        </div>
      </div>

      {error && (
        <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ color: 'var(--err)', fontSize: '13px', flex: 1 }}>Feed request failed — showing the last loaded items.</span>
          <button className="softbtn" style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '5px', padding: '4px 10px', fontSize: '11.5px', fontFamily: 'var(--font-data)', cursor: 'pointer' }} onClick={() => load()}>retry</button>
        </div>
      )}

      {loading && items.length === 0 && <Skeletons />}

      {empty && (
        filtersActive ? (
          <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '13px', marginBottom: '6px' }}>no items match the current filters</div>
            <button className="linkbtn" style={{ border: 'none', background: 'none', color: 'var(--acc)', cursor: 'pointer', fontSize: '13px', textDecoration: 'underline', textUnderlineOffset: '3px' }}
              onClick={() => { s.setGroup('all'); s.setProject('all'); s.setSource('all'); }}>reset filters</button>
          </div>
        ) : (
          <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
            <div style={{ fontFamily: 'var(--font-data)', fontSize: '13px', marginBottom: '6px' }}>no memories yet — connect an ingestion source</div>
            <div style={{ fontSize: '13px', color: 'var(--txt3)' }}>claude-code · cursor · claude-ai · transcribe-ai</div>
          </div>
        )
      )}

      {items.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {items.map((it) => <FeedCard key={keyOf(it)} item={it} fresh={fresh.has(keyOf(it))} />)}
        </div>
      )}

      {cursor && (
        <div style={{ textAlign: 'center', marginTop: '18px' }}>
          <button className="softbtn" onClick={loadMore} disabled={loadingMore}
            style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '7px', padding: '7px 16px', fontSize: '12.5px', fontFamily: 'var(--font-data)', cursor: 'pointer' }}>
            {loadingMore ? 'loading…' : 'load more'}
          </button>
        </div>
      )}
    </main>
  );
}
