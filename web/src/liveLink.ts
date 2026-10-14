// The app shell's one live link to the server: the SSE stream (api.ts openFeedStream), held
// open on every page while signed in. It feeds the header's queue badge everywhere and the
// Feed's live rows where the Feed is open. When the stream gives up the link reports
// 'polling' (the Feed then refreshes itself on a timer) and keeps retrying the stream: every
// RETRY_MS, and at once when the browser reports the network or the tab is back.
import { openFeedStream, type FeedItem, type ProcessingStatus, type StreamHandle } from './api';

export type LinkState = 'connecting' | 'live' | 'reconnecting' | 'polling';

export interface FeedListener {
  onItem: (item: FeedItem) => void;
  onResync: () => void; // refetch page 1: the stream (re)connected or lost events
}

export interface LiveLinkHandlers extends FeedListener {
  onLink: (s: LinkState) => void;
  onQueue: (q: ProcessingStatus | null) => void;
}

export const RETRY_MS = 120_000;
// Status frames come about once a second; this much silence means the count is stale.
export const QUEUE_STALE_MS = 10_000;

export function startLiveLink(h: LiveLinkHandlers): () => void {
  let stopped = false;
  let polling = false;
  let stream: StreamHandle | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let staleTimer: ReturnType<typeof setTimeout> | null = null;

  const dropQueue = () => {
    if (staleTimer) clearTimeout(staleTimer);
    staleTimer = null;
    h.onQueue(null);
  };

  const open = () => {
    if (stopped) return;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = null;
    stream?.close();
    stream = openFeedStream({
      onReady: () => { polling = false; h.onLink('live'); h.onResync(); },
      onItem: h.onItem,
      onReset: h.onResync,
      onStatus: (q) => {
        h.onQueue(q);
        if (staleTimer) clearTimeout(staleTimer);
        staleTimer = setTimeout(dropQueue, QUEUE_STALE_MS);
      },
      // A retry that fails while polling must not flip the header back to "reconnecting".
      onDisconnect: () => { dropQueue(); if (!polling) h.onLink('reconnecting'); },
      onGiveUp: () => { polling = true; h.onLink('polling'); retryTimer = setTimeout(open, RETRY_MS); },
    });
  };

  const wake = () => { if (polling && document.visibilityState !== 'hidden') open(); };
  window.addEventListener('online', wake);
  document.addEventListener('visibilitychange', wake);

  h.onLink('connecting');
  h.onQueue(null);
  open();
  return () => {
    stopped = true;
    stream?.close();
    if (retryTimer) clearTimeout(retryTimer);
    if (staleTimer) clearTimeout(staleTimer);
    window.removeEventListener('online', wake);
    document.removeEventListener('visibilitychange', wake);
  };
}

export const sameQueue = (a: ProcessingStatus | null, b: ProcessingStatus | null): boolean =>
  a === b || (a !== null && b !== null && a.queue_depth === b.queue_depth
    && a.processing === b.processing && a.failed === b.failed);

// What the header shows for the link: the queue counts only while the stream is live (a count
// nothing is refreshing is not shown), otherwise the link's state.
export function queueBadge(link: LinkState, q: ProcessingStatus | null): {
  live: boolean; label: string; running: number; failed: number; title: string;
} {
  switch (link) {
    case 'live':
      return q
        ? { live: true, label: `queue ${q.queue_depth}`, running: q.processing, failed: q.failed,
            title: 'live stream connected — extraction queue: pending, running, failed' }
        : { live: true, label: 'queue —', running: 0, failed: 0,
            title: 'live stream connected — the queue count is unavailable' };
    case 'polling':
      return { live: false, label: 'polling', running: 0, failed: 0,
        title: 'live stream unavailable — the feed refreshes every 30 s and the stream is retried; no live queue count' };
    case 'reconnecting':
      return { live: false, label: 'reconnecting', running: 0, failed: 0,
        title: 'stream lost — reconnecting (showing last loaded items)' };
    case 'connecting':
      return { live: false, label: 'connecting', running: 0, failed: 0, title: 'connecting to the live stream' };
  }
}
