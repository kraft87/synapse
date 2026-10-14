// Feed — "watch it remember". Keyset pagination via next_cursor; the app shell's live SSE
// stream (liveLink.ts, phase 3) puts new writes in order with the slidein animation, and the
// Feed refetches page 1 whenever the stream (re)connects or loses events. When the stream is
// down ('polling') it does the same every 30s. Skeleton / empty / error states (companion
// spec §2). The live dot + queue badge are the header's, from the same stream.
import type React from 'react';
import { useEffect, useRef, useState } from 'react';
import { fetchFeed, feedItemPassesFilter, type Feed as FeedPage, type FeedItem } from '../api';
import { insertFeedItem, keyOf, mergePage1 } from '../feedOrder';
import { useStore } from '../state';
import { FeedCard } from '../components/FeedCard';
import { ErrorBox, EmptyState } from '../components/ui';

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
  const { project, group, source, link, subscribeFeed } = s;
  const [items, setItemsState] = useState<FeedItem[]>([]);
  const [cursor, setCursorState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(false);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  // The refs are what async handlers read: always the latest list and cursor, never a closure's.
  const itemsRef = useRef<FeedItem[]>([]);
  const cursorRef = useRef<string | null>(null);
  // Bumped by every filter change: a response from an older epoch is dropped on arrival, so
  // the list and its cursor always belong to the filters in the header.
  const epoch = useRef(0);

  const setItems = (next: FeedItem[]) => { itemsRef.current = next; setItemsState(next); };
  const setCursor = (next: string | null) => { cursorRef.current = next; setCursorState(next); };

  const filters = { project, group_id: group, source };

  // Fold a fetched page 1 into the list (see mergePage1); `replace` = trust it over what's shown.
  const fold = (r: FeedPage, replace: boolean) => {
    const m = mergePage1(itemsRef.current, r.items, replace, cursorRef.current !== null);
    setItems(m.items);
    if (m.replaced) setCursor(r.next_cursor);
    if (m.fresh.length) setFresh(new Set(m.fresh));
  };

  // Start over for the current filters: nothing from the previous ones stays on screen.
  const load = () => {
    const mine = ++epoch.current;
    setItems([]); setCursor(null); setFresh(new Set());
    setLoading(true); setLoadingMore(false); setError(false);
    fetchFeed({ ...filters })
      .then((r) => { if (mine === epoch.current) fold(r, true); })
      .catch(() => { if (mine === epoch.current) setError(true); })
      .finally(() => { if (mine === epoch.current) setLoading(false); });
  };

  // Refetch page 1 and fold it in under the rows already shown: the stream (re)connected and
  // may have missed writes, or the stream is down and this is the poll.
  const resync = () => {
    const mine = epoch.current;
    fetchFeed({ ...filters })
      .then((r) => { if (mine === epoch.current) { fold(r, false); setError(false); } })
      .catch(() => { if (mine === epoch.current) setError(true); });
  };

  // (re)load page 1 whenever the filters change, and take the shell stream's rows that pass
  // the current filter (items filtered out are dropped; they reappear on the next reload).
  // Subscribing first means nothing written during the load is missed: a row committed after
  // the snapshot streams in and mergePage1 keeps it.
  useEffect(() => {
    const unsubscribe = subscribeFeed({
      onItem: (item) => {
        if (!feedItemPassesFilter(item, { project, group, source })) return;
        const r = insertFeedItem(itemsRef.current, item, cursorRef.current !== null);
        if (r.at < 0) return;
        setItems(r.items);
        if (r.at === 0) setFresh(new Set([keyOf(item)]));
      },
      onResync: resync,
    });
    load();
    return () => { epoch.current += 1; unsubscribe(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, group, source]);

  // With the stream down, page 1 is refetched every POLL_MS (and the shell retries the stream).
  useEffect(() => {
    if (link !== 'polling') return;
    const t = setInterval(resync, POLL_MS);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [link, project, group, source]);

  const loadMore = () => {
    const sent = cursor;
    if (!sent || loadingMore) return;
    const mine = epoch.current;
    setLoadingMore(true);
    fetchFeed({ ...filters, cursor: sent })
      .then((r) => {
        // Dropped if the filters changed, or a resync replaced the list under this page.
        if (mine !== epoch.current || cursorRef.current !== sent) return;
        const seen = new Set(itemsRef.current.map(keyOf));
        setItems(itemsRef.current.concat(r.items.filter((it) => !seen.has(keyOf(it)))));
        setCursor(r.next_cursor);
      })
      .catch(() => { if (mine === epoch.current) setError(true); })
      .finally(() => { if (mine === epoch.current) setLoadingMore(false); });
  };

  const filtersActive = group !== 'all' || project !== 'all' || source !== 'all';
  const empty = !loading && !error && items.length === 0;

  return (
    <main style={wrap}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={monoHead}>live memory stream</div>
        <div style={{ fontSize: '12px', fontFamily: 'var(--font-data)', color: 'var(--txt3)' }}>
          {items.length} items · newest first
        </div>
      </div>

      {error && (
        <ErrorBox onRetry={() => load()} style={{ marginBottom: '10px' }}>
          {items.length === 0 ? 'Feed request failed.' : 'Feed request failed — showing the last loaded items.'}
        </ErrorBox>
      )}

      {loading && items.length === 0 && <Skeletons />}

      {empty && (
        filtersActive ? (
          <EmptyState title="no items match the current filters">
            <button className="linkbtn" style={{ border: 'none', background: 'none', color: 'var(--acc)', cursor: 'pointer', fontSize: '13px', textDecoration: 'underline', textUnderlineOffset: '3px' }}
              onClick={() => { s.setGroup('all'); s.setProject('all'); s.setSource('all'); }}>reset filters</button>
          </EmptyState>
        ) : (
          <EmptyState title="no memories yet — connect an ingestion source" subtitle="claude-code · cursor · claude-ai · transcribe-ai" />
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
            style={{ borderRadius: '7px', padding: '7px 16px', fontSize: '12.5px', fontFamily: 'var(--font-data)' }}>
            {loadingMore ? 'loading…' : 'load more'}
          </button>
        </div>
      )}
    </main>
  );
}
