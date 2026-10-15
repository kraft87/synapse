import test from 'node:test';
import assert from 'node:assert/strict';
import { dec, parseRoute } from '../src/hash';
import { isMockHash } from '../src/mode';

test('parseRoute reads the three overlay routes', () => {
  assert.deepEqual(parseRoute('#/episode/227168'), { kind: 'episode', id: '227168' });
  assert.deepEqual(parseRoute('#/entity/ab%20c'), { kind: 'entity', id: 'ab c' });
  assert.deepEqual(parseRoute('#/session/s-1?highlight=42'), { kind: 'session', id: 's-1', highlight: '42' });
  assert.deepEqual(parseRoute('#/session/s-1'), { kind: 'session', id: 's-1', highlight: undefined });
  assert.deepEqual(parseRoute(''), { kind: 'none' });
  assert.deepEqual(parseRoute('#/episode'), { kind: 'none' });
  assert.deepEqual(parseRoute('#/nowhere/1'), { kind: 'none' });
});

test('a malformed percent escape is no route, not an exception', () => {
  assert.equal(dec('100%'), null);
  assert.equal(dec('a%20b'), 'a b');
  for (const h of ['#/episode/100%', '#/entity/%E0%A4%A', '#/session/%?highlight=1']) {
    assert.deepEqual(parseRoute(h), { kind: 'none' }, h);
  }
});

test('only the bare #mock marker switches to fixtures, not a hash that contains the word', () => {
  assert.equal(isMockHash('#mock'), true);
  assert.equal(isMockHash('#mock&x=1'), true);
  assert.equal(isMockHash('#/session/payments-mock-run-42'), false);
  assert.equal(isMockHash('#/episode/1?mock'), false);
  assert.equal(isMockHash('#mocking'), false);
  assert.equal(isMockHash('#token=abc'), false);
  assert.equal(isMockHash(''), false);
});
