import test from 'node:test';
import assert from 'node:assert/strict';
import { bucketItems, recallScope, servedBuckets, waterfall } from '../src/recallView';
import type { RecallDebug, RecallResult } from '../src/api';

const debug = (legs: Record<string, number>, total = 100): RecallDebug => ({
  total_ms: total, legs_ms: legs,
  pool_sizes: { bm25: 0, vector: 0, fused: 0, kg_candidates: 0 },
  rerank: { model: 'm', top_score: 0 }, est_tokens: 0,
});

test('notes are served: a bucket with the id, the hook and the note', () => {
  const r: RecallResult = { query: 'q', facts: [], notes: [{ id: 'n:12', hook: 'prefers uv', note: 'Use uv, not pip.' }] };
  assert.ok(servedBuckets(r).includes('notes'));
  assert.deepEqual(bucketItems('notes', r), [{ key: 'n0', head: 'n:12 · prefers uv —', rest: 'Use uv, not pip.' }]);
});

test('served buckets are the ones the engine emits: the retired ones never appear', () => {
  const r: RecallResult = { query: 'q', facts: [], episodes: [] };
  assert.deepEqual(servedBuckets(r), ['episodes', 'facts', 'web']);
  // a notes leg that ran and found nothing is a real "0 served"; a disabled one is absent
  assert.deepEqual(servedBuckets({ ...r, debug: debug({ notes: 12 }) }), ['episodes', 'facts', 'notes', 'web']);
  assert.deepEqual(servedBuckets({ ...r, debug: debug({ bm25: 12 }) }), ['episodes', 'facts', 'web']);
  assert.deepEqual(
    servedBuckets({ ...r, superseded_facts: [{ fact: 'old', superseded_by: 'new' }] }),
    ['episodes', 'facts', 'web', 'superseded_facts'],
  );
});

test('waterfall times the notes leg in the parallel band and leaves a disabled one skipped', () => {
  const rows = waterfall(debug({ embed: 10, bm25: 20, vector: 20, kg: 30, notes: 50, web: 5, rerank: 10 }, 100));
  const by = Object.fromEntries(rows.map((w) => [w.name, w]));
  assert.deepEqual(rows.map((w) => w.name), ['embed', 'bm25', 'vector', 'kg', 'notes', 'web', 'rerank']);
  assert.equal(by.notes.left, 10);          // starts at embed-end
  assert.equal(by.notes.width, 50);
  assert.equal(by.rerank.left, 60);          // waits for the slowest parallel leg (notes, 50)
  const off = Object.fromEntries(waterfall(debug({ embed: 10, bm25: 20, rerank: 10 })).map((w) => [w.name, w]));
  assert.equal(off.notes.skipped, true);
  assert.equal(off.notes.flag, ' (skipped)');
});

test('waterfall with no timings reads "—" for every leg, not "skipped"', () => {
  const rows = waterfall(null);
  assert.equal(rows.length, 7);
  for (const w of rows) {
    assert.equal(w.ms, '—');
    assert.equal(w.width, 0);
    assert.equal(w.flag, '');
  }
});

test('recallScope sends what recall() honors and says what it changed', () => {
  assert.deepEqual(recallScope('all', 'technical'), { project: undefined, group_id: 'technical', notes: [] });
  assert.deepEqual(recallScope('synapse', 'personal'), { project: 'synapse', group_id: 'personal', notes: [] });
  // recall() searches one group: 'all' becomes technical, visibly
  const all = recallScope('synapse', 'all');
  assert.equal(all.group_id, 'technical');
  assert.equal(all.notes.length, 1);
  // 'untagged' is the catalog's NULL bucket; recall() would match it as a literal project
  const untagged = recallScope('untagged', 'personal');
  assert.equal(untagged.project, undefined);
  assert.equal(untagged.notes.length, 1);
  assert.equal(recallScope('untagged', 'all').notes.length, 2);
});
