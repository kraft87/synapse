import test from 'node:test';
import assert from 'node:assert/strict';
import { cascadeSummary, deleteEpisode, deleteFact, onDeleted, wasDeleted, type DeletedDetail, type DeleteEpisodeResult } from '../src/api';

const g = globalThis as unknown as Record<string, unknown>;
g.localStorage = { getItem: () => 'token' };
const win = new EventTarget();
g.window = win;

const cascade: DeletedDetail = { episodes: ['12'], facts: ['f-sole'], events: ['7'] };

test('a delete names the feed items and search hits it took, by kind', () => {
  assert.equal(wasDeleted(cascade, 'episode', '12'), true);
  assert.equal(wasDeleted(cascade, 'episodes', 12), true);
  assert.equal(wasDeleted(cascade, 'fact', 'f-sole'), true);
  assert.equal(wasDeleted(cascade, 'facts', 'f-sole'), true);
  assert.equal(wasDeleted(cascade, 'timeline_event', '7'), true);
  assert.equal(wasDeleted(cascade, 'events', 7), true);
});

test('ids of one kind never match another kind, and entities are never deleted', () => {
  assert.equal(wasDeleted(cascade, 'episode', '7'), false);
  assert.equal(wasDeleted(cascade, 'timeline_event', '12'), false);
  assert.equal(wasDeleted(cascade, 'fact', '12'), false);
  assert.equal(wasDeleted({ episodes: [], facts: [], events: [] }, 'entities', 'e'), false);
  assert.equal(wasDeleted({ episodes: ['x'], facts: ['x'], events: ['x'] }, 'entities', 'x'), false);
});

const result = (over: Partial<DeleteEpisodeResult> = {}): DeleteEpisodeResult => ({
  episode_id: 12, chunks_deleted: 0, chunks_mixed_deleted: 0, facts_deleted: 1, facts_unlinked: 0,
  events_deleted: 1, deleted_fact_ids: ['f-sole'], deleted_event_ids: [7], ...over,
});

test('deleting an episode announces its cascade: the facts and events the server listed', async () => {
  g.fetch = async () => ({ status: 200, ok: true, json: async () => result() }) as Response;
  const seen: DeletedDetail[] = [];
  const off = onDeleted((d) => seen.push(d));
  await deleteEpisode('12');
  off();
  assert.deepEqual(seen, [cascade]);
});

test('deleting a fact announces just that fact', async () => {
  g.fetch = async () => ({ status: 200, ok: true, json: async () => ({ fact_id: 'f1', deleted: true, supersession_links_cleared: 0 }) }) as Response;
  const seen: DeletedDetail[] = [];
  const off = onDeleted((d) => seen.push(d));
  await deleteFact('f1');
  off();
  assert.deepEqual(seen, [{ episodes: [], facts: ['f1'], events: [] }]);
});

test('the summary lists what went and admits the collateral', () => {
  assert.equal(cascadeSummary(result()), 'episode deleted: 1 fact, 1 event');
  assert.equal(
    cascadeSummary(result({ chunks_deleted: 2, chunks_mixed_deleted: 1, facts_unlinked: 3 })),
    'episode deleted: 2 chunks, 1 fact, 1 event · 3 facts unlinked (shared provenance) · 1 chunk also spanned other episodes',
  );
});
