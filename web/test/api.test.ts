import test from 'node:test';
import assert from 'node:assert/strict';
import { feedItemPassesFilter, fetchSearch, fetchTimeline, fetchTimelineMonths, filterVal, qs, type FeedItem } from '../src/api';

const g = globalThis as unknown as Record<string, unknown>;
g.localStorage = { getItem: () => 'token' };
const urls: string[] = [];
g.fetch = async (url: string) => {
  urls.push(url);
  return { status: 200, ok: true, json: async () => ({ hits: [], events: [], months: [] }) } as Response;
};

test('qs sends the value "all" like any other text and drops only empty values', () => {
  assert.equal(qs({ q: 'all', type: 'episodes' }), '?q=all&type=episodes');
  assert.equal(qs({ q: '', a: undefined, b: null }), '');
});

test('filterVal turns the header\'s "all" into no filter', () => {
  assert.equal(filterVal('all'), undefined);
  assert.equal(filterVal('technical'), 'technical');
  assert.equal(filterVal(undefined), undefined);
});

test('searching for "all" sends q=all, while the "all" filters are left out', async () => {
  urls.length = 0;
  await fetchSearch({ q: 'all', type: 'episodes', project: 'all', group_id: 'all', source: 'all' });
  const u = new URL(urls[0], 'http://x');
  assert.equal(u.searchParams.get('q'), 'all');
  for (const k of ['project', 'group_id', 'source']) assert.equal(u.searchParams.has(k), false, k);
  await fetchSearch({ q: 'all', type: 'facts', project: 'synapse', group_id: 'personal' });
  const v = new URL(urls[1], 'http://x');
  assert.equal(v.searchParams.get('project'), 'synapse');
  assert.equal(v.searchParams.get('group_id'), 'personal');
});

test('the timeline requests carry the type, the group and the viewer\'s zone', async () => {
  urls.length = 0;
  await fetchTimeline({ type: 'decision', group_id: 'all' });
  await fetchTimelineMonths({ type: 'finding', group_id: 'personal' });
  const tl = new URL(urls[0], 'http://x');
  assert.equal(tl.searchParams.get('type'), 'decision');
  assert.equal(tl.searchParams.has('group_id'), false);
  const months = new URL(urls[1], 'http://x');
  assert.equal(months.pathname, '/dash/api/timeline/months');
  assert.equal(months.searchParams.get('type'), 'finding');
  assert.equal(months.searchParams.get('group_id'), 'personal');
  assert.ok(months.searchParams.get('tz'));
});

const ev = (group_id: string | null): FeedItem =>
  ({ type: 'timeline_event', id: '1', ts: '2026-07-01T00:00:00+00:00', group_id, gist: 'x', flagged: false, data: {} });
const fact = (group_id: string | null): FeedItem =>
  ({ type: 'fact', id: 'f', ts: '2026-07-01T00:00:00+00:00', group_id, gist: 'x', flagged: false, data: {} });
const f = (group: string) => ({ project: 'all', group, source: 'all' });

test('a live timeline event with no domain passes either group filter, like the server', () => {
  assert.equal(feedItemPassesFilter(ev(null), f('technical')), true);
  assert.equal(feedItemPassesFilter(ev(null), f('personal')), true);
  assert.equal(feedItemPassesFilter(ev('personal'), f('technical')), false);
  assert.equal(feedItemPassesFilter(ev('technical'), f('technical')), true);
});

test('facts keep the strict group match', () => {
  assert.equal(feedItemPassesFilter(fact(null), f('technical')), false);
  assert.equal(feedItemPassesFilter(fact('technical'), f('technical')), true);
  assert.equal(feedItemPassesFilter(fact('personal'), f('all')), true);
});
