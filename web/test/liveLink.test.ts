import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { startLiveLink, queueBadge, sameQueue, RETRY_MS, QUEUE_STALE_MS, type LinkState } from '../src/liveLink';
import type { ProcessingStatus } from '../src/api';

// ---- a fake browser + server: just what api.ts and liveLink.ts touch ----
const g = globalThis as unknown as Record<string, unknown>;
g.localStorage = { getItem: () => 'token' };
const win = new EventTarget();
const doc = Object.assign(new EventTarget(), { visibilityState: 'visible' });
g.window = win;
g.document = doc;

interface Conn { push: (s: string) => void; end: () => void }
let conns: Conn[] = [];
let failNext = 0;
g.fetch = async () => {
  if (failNext > 0) { failNext -= 1; throw new Error('server down'); }
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(c) { ctrl = c; } });
  const enc = new TextEncoder();
  conns.push({ push: (s) => ctrl.enqueue(enc.encode(s)), end: () => ctrl.close() });
  return { status: 200, ok: true, body } as Response;
};
const flush = () => new Promise<void>((r) => setImmediate(r));
const frame = (event: string, data = '{}') => `event: ${event}\ndata: ${data}\n\n`;
const STATUS: ProcessingStatus = { queue_depth: 3, processing: 1, failed: 0, active: true };

function harness(serverDown = false) {
  conns = [];
  failNext = serverDown ? 99 : 0;
  mock.timers.enable({ apis: ['setTimeout'] });
  const seen = { links: [] as LinkState[], queue: null as ProcessingStatus | null, resyncs: 0, items: [] as string[] };
  const stop = startLiveLink({
    onLink: (s) => seen.links.push(s),
    onQueue: (q) => { seen.queue = q; },
    onItem: (it) => seen.items.push(it.id),
    onResync: () => { seen.resyncs += 1; },
  });
  const last = () => seen.links[seen.links.length - 1];
  return { seen, stop: () => { stop(); mock.timers.reset(); }, last };
}

test('the link is live and resyncs only once the server says ready', async () => {
  const t = harness();
  try {
    await flush();
    assert.equal(t.last(), 'connecting'); // HTTP 200 alone is not live
    conns[0].push(frame('ready') + frame('processing_status', JSON.stringify(STATUS)));
    await flush();
    assert.equal(t.last(), 'live');
    assert.equal(t.seen.resyncs, 1);
    assert.deepEqual(t.seen.queue, STATUS);
    conns[0].push(frame('new_episode', JSON.stringify({ type: 'episode', id: '7', ts: 'x', gist: '', flagged: false, data: {} })));
    await flush();
    assert.deepEqual(t.seen.items, ['7']);
    conns[0].push(frame('reset'));
    await flush();
    assert.equal(t.seen.resyncs, 2); // lost events -> refetch page 1
  } finally { t.stop(); }
});

test('a dropped stream shows reconnecting without a count, then resyncs when it is back', async () => {
  const t = harness();
  try {
    await flush();
    conns[0].push(frame('ready') + frame('processing_status', JSON.stringify(STATUS)));
    await flush();
    conns[0].end();
    await flush();
    assert.equal(t.last(), 'reconnecting');
    assert.equal(t.seen.queue, null);
    mock.timers.tick(1000); // first backoff
    await flush();
    conns[1].push(frame('ready'));
    await flush();
    assert.equal(t.last(), 'live');
    assert.equal(t.seen.resyncs, 2); // once per ready: nothing written during the gap is missed
  } finally { t.stop(); }
});

test('the queue count is withdrawn when status frames stop arriving', async () => {
  const t = harness();
  try {
    await flush();
    conns[0].push(frame('ready') + frame('processing_status', JSON.stringify(STATUS)));
    await flush();
    mock.timers.tick(QUEUE_STALE_MS - 1);
    assert.deepEqual(t.seen.queue, STATUS);
    mock.timers.tick(1);
    assert.equal(t.seen.queue, null);
    assert.equal(t.last(), 'live'); // the stream itself is fine; only the count is unknown
  } finally { t.stop(); }
});

test('after the stream gives up the link polls, and goes back to the stream when it can', async () => {
  const t = harness(true);
  try {
    await flush();
    for (const ms of [1000, 2000, 4000, 8000, 16000]) { mock.timers.tick(ms); await flush(); }
    assert.equal(t.last(), 'polling');
    // A retry that fails keeps saying polling instead of flipping to reconnecting.
    const before = t.seen.links.length;
    mock.timers.tick(RETRY_MS);
    await flush();
    assert.deepEqual(t.seen.links.slice(before), []);
    failNext = 0; // the server is back: the retry's own backoff reconnects it
    mock.timers.tick(1000);
    await flush();
    conns[conns.length - 1].push(frame('ready'));
    await flush();
    assert.equal(t.last(), 'live');
    assert.equal(t.seen.resyncs, 1);
  } finally { t.stop(); }
});

test('coming back online retries the stream at once instead of waiting out the timer', async () => {
  const t = harness(true);
  try {
    await flush();
    for (const ms of [1000, 2000, 4000, 8000, 16000]) { mock.timers.tick(ms); await flush(); }
    assert.equal(t.last(), 'polling');
    failNext = 0;
    win.dispatchEvent(new Event('online'));
    await flush();
    conns[conns.length - 1].push(frame('ready'));
    await flush();
    assert.equal(t.last(), 'live');
  } finally { t.stop(); }
});

test('sameQueue ignores the active flag a count already implies', () => {
  assert.ok(sameQueue(STATUS, { ...STATUS }));
  assert.ok(!sameQueue(STATUS, { ...STATUS, failed: 1 }));
  assert.ok(!sameQueue(STATUS, null));
  assert.ok(sameQueue(null, null));
});

test('queueBadge: counts only while live; every other state says what it is', () => {
  assert.deepEqual(queueBadge('live', STATUS), {
    live: true, label: 'queue 3', running: 1, failed: 0,
    title: 'live stream connected — extraction queue: pending, running, failed',
  });
  assert.equal(queueBadge('live', null).label, 'queue —');
  assert.equal(queueBadge('polling', STATUS).label, 'polling');
  assert.equal(queueBadge('polling', STATUS).failed, 0); // a stale count is not shown
  assert.equal(queueBadge('reconnecting', null).label, 'reconnecting');
  assert.equal(queueBadge('connecting', null).label, 'connecting');
  assert.ok(!queueBadge('polling', null).live);
});
