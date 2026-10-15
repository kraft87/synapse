import test from 'node:test';
import assert from 'node:assert/strict';
import { HttpError } from '../src/api';
import { decisionProblem } from '../src/proposals';

test('a 409 says the proposal was already decided', () => {
  const p = decisionProblem(new HttpError(409, 'request failed (409): /dash/api/proposals/skill:7/decision'));
  assert.match(p, /already decided/);
  assert.doesNotMatch(p, /request failed/);
});

test('any other failure reports the error and claims no outcome', () => {
  assert.equal(decisionProblem(new HttpError(500, 'request failed (500)')), 'decision failed: request failed (500)');
  assert.equal(decisionProblem(new Error('network down')), 'decision failed: network down');
  assert.equal(decisionProblem('odd'), 'decision failed: odd');
});
