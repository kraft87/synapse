import test from 'node:test';
import assert from 'node:assert/strict';
import { compareFeed, insertFeedItem, mergePage1, keyOf } from '../src/feedOrder';
import type { FeedItem } from '../src/api';

const item = (id: string, ts: string, type: FeedItem['type'] = 'episode'): FeedItem =>
  ({ type, id, ts, gist: id, flagged: false, data: {} });
const ids = (xs: FeedItem[]) => xs.map((x) => x.id);

test('compareFeed: newest first, ties broken by id text descending (the server keyset order)', () => {
  const a = item('9', '2026-07-01T12:00:00+00:00');
  const b = item('10', '2026-07-01T12:00:00+00:00');
  const c = item('1', '2026-07-01T12:00:01+00:00');
  assert.deepEqual(ids([a, b, c].sort(compareFeed)), ['1', '9', '10']);
  // Offsets and fractional seconds compare as instants, not as strings.
  const early = item('x', '2026-07-01T12:00:00.5+00:00');
  const late = item('y', '2026-07-01T14:00:00+02:00'); // 12:00:00 UTC
  assert.ok(compareFeed(early, late) < 0);
});

test('insertFeedItem puts a backfilled episode at its time, not at the top', () => {
  const list = [item('3', '2026-07-03T00:00:00+00:00'), item('1', '2026-07-01T00:00:00+00:00')];
  const r = insertFeedItem(list, item('2', '2026-07-02T00:00:00+00:00'), false);
  assert.deepEqual(ids(r.items), ['3', '2', '1']);
  assert.equal(r.at, 1);
  const top = insertFeedItem(list, item('4', '2026-07-04T00:00:00+00:00'), false);
  assert.deepEqual(ids(top.items), ['4', '3', '1']);
  assert.equal(top.at, 0);
});

test('insertFeedItem skips an item older than the loaded tail while more pages exist', () => {
  const list = [item('3', '2026-07-03T00:00:00+00:00'), item('2', '2026-07-02T00:00:00+00:00')];
  const old = item('1', '2026-07-01T00:00:00+00:00');
  assert.equal(insertFeedItem(list, old, true).at, -1); // it arrives when paging
  assert.equal(insertFeedItem(list, old, true).items, list);
  assert.deepEqual(ids(insertFeedItem(list, old, false).items), ['3', '2', '1']); // nothing left to page
});

test('insertFeedItem ignores an item already on screen', () => {
  const list = [item('3', '2026-07-03T00:00:00+00:00')];
  const r = insertFeedItem(list, item('3', '2026-07-03T00:00:00+00:00'), false);
  assert.equal(r.at, -1);
  assert.equal(r.items, list);
});

test('mergePage1 adds only the unseen rows, in order, and flags the ones that landed on top', () => {
  const prev = [item('5', '2026-07-05T00:00:00+00:00'), item('4', '2026-07-04T00:00:00+00:00')];
  const page = [
    item('7', '2026-07-07T00:00:00+00:00'),
    item('6', '2026-07-06T00:00:00+00:00'),
    item('5', '2026-07-05T00:00:00+00:00'),
  ];
  const m = mergePage1(prev, page, false, true);
  assert.deepEqual(ids(m.items), ['7', '6', '5', '4']);
  assert.equal(m.replaced, false);
  assert.deepEqual(m.fresh, ['episode:7', 'episode:6']);
});

test('mergePage1 replaces the list when page 1 shares nothing with it (a burst bigger than a page)', () => {
  // The list holds 2 old rows; 3 new ones arrived and the page only reaches back to 'b'.
  const prev = [item('2', '2026-07-02T00:00:00+00:00'), item('1', '2026-07-01T00:00:00+00:00')];
  const page = [item('d', '2026-07-09T00:00:00+00:00'), item('c', '2026-07-08T00:00:00+00:00'), item('b', '2026-07-07T00:00:00+00:00')];
  const m = mergePage1(prev, page, false, true);
  assert.deepEqual(ids(m.items), ['d', 'c', 'b']); // no hole between page 1 and the old tail
  assert.equal(m.replaced, true); // the caller takes the page's cursor
  assert.deepEqual(m.fresh, []);
});

test('mergePage1 on a load keeps live rows newer than the snapshot', () => {
  // A row committed after the snapshot arrived on the stream while /feed was in flight.
  const live = item('9', '2026-07-09T00:00:00+00:00');
  const page = [item('5', '2026-07-05T00:00:00+00:00'), item('4', '2026-07-04T00:00:00+00:00')];
  const m = mergePage1([live], page, true, false);
  assert.deepEqual(ids(m.items), ['9', '5', '4']);
  assert.equal(m.replaced, true);
  // ...and drops a stale row the snapshot has already passed.
  const stale = item('1', '2026-07-01T00:00:00+00:00');
  assert.deepEqual(ids(mergePage1([stale], page, true, false).items), ['5', '4']);
});

test('mergePage1 on a load with an empty snapshot keeps the live rows', () => {
  const live = item('9', '2026-07-09T00:00:00+00:00');
  assert.deepEqual(ids(mergePage1([live], [], true, false).items), ['9']);
});

test('keyOf is type-qualified', () => {
  assert.notEqual(keyOf(item('1', 'x', 'fact')), keyOf(item('1', 'x', 'episode')));
});
