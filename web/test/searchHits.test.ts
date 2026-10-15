import test from 'node:test';
import assert from 'node:assert/strict';

process.env.TZ = 'America/Toronto';

import { hasTarget, isSuperseded, metaLine } from '../src/searchHits';
import type { SearchHit } from '../src/api';

const hit = (type: SearchHit['type'], meta: SearchHit['meta']): SearchHit => ({ type, id: '1', snippet: 's', meta });

test('hasTarget: entities and episodes always open; facts and events only with provenance', () => {
  assert.equal(hasTarget(hit('entities', {})), true);
  assert.equal(hasTarget(hit('episodes', {})), true);
  assert.equal(hasTarget(hit('facts', { episode_id: 12 })), true);
  assert.equal(hasTarget(hit('facts', { session_id: 's-1' })), true);
  assert.equal(hasTarget(hit('facts', { episode_id: null })), false);   // empty episodes array
  assert.equal(hasTarget(hit('events', { episode_id: null, source_ref: 'a'.repeat(40) })), false);  // git event
  assert.equal(hasTarget(hit('events', { episode_id: 7 })), true);
});

test('a superseded fact says so, with the viewer-local day', () => {
  const dead = hit('facts', { t_invalid: '2026-10-01T00:30:00+00:00' });   // 20:30 on Sep 30 in Toronto
  assert.equal(isSuperseded(dead), true);
  assert.match(metaLine(dead), /superseded 2026-09-30$/);
  assert.equal(isSuperseded(hit('facts', { t_invalid: null })), false);
  assert.doesNotMatch(metaLine(hit('facts', { t_invalid: null })), /superseded/);
  assert.equal(isSuperseded(hit('events', { t_invalid: '2026-10-01T00:30:00+00:00' })), false);
});

test('a git event shows its short SHA; an event with an episode does not', () => {
  const sha = 'abcdef0123456789abcdef0123456789abcdef01';
  assert.match(metaLine(hit('events', { source: 'git', episode_id: null, source_ref: sha })), /ref abcdef0$/);
  assert.doesNotMatch(metaLine(hit('events', { source: 'chat', episode_id: 7, source_ref: 'ep:7' })), /ref/);
});
