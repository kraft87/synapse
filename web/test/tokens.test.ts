import test from 'node:test';
import assert from 'node:assert/strict';
import { hasToolTrace, relTime, turnsLabel } from '../src/tokens';

const NOW = Date.parse('2026-10-14T12:00:00Z');
const ago = (ms: number) => new Date(NOW - ms).toISOString();
const MIN = 60e3;
const HOUR = 60 * MIN;

test('relTime rounds down: a unit is only claimed once it has fully passed', () => {
  assert.equal(relTime(ago(23.6 * HOUR), NOW), '23h ago');
  assert.equal(relTime(ago(36 * HOUR), NOW), '1d ago');
  assert.equal(relTime(ago(59.6 * MIN), NOW), '59m ago');
  assert.equal(relTime(ago(1.9 * HOUR), NOW), '1h ago');
  assert.equal(relTime(ago(24 * HOUR), NOW), '1d ago');
  assert.equal(relTime(ago(47.9 * HOUR), NOW), '1d ago');
  assert.equal(relTime(ago(48 * HOUR), NOW), '2d ago');
});

test('relTime: under a minute reads 1m, the future reads now, junk reads nothing', () => {
  assert.equal(relTime(ago(5e3), NOW), '1m ago');
  assert.equal(relTime(ago(-5 * MIN), NOW), 'now');
  assert.equal(relTime(undefined, NOW), '');
  assert.equal(relTime('not a date', NOW), '');
});

test('hasToolTrace: the lines the clients write for tool calls and results', () => {
  const stored = '[user] wire A to B\n\n[tool:Read] src/a.py\n[tool:Edit] src/a.py:42\n\n[result] ===== 3 passed =====\n\n[assistant] Done.';
  assert.equal(hasToolTrace(stored), true);
  assert.equal(hasToolTrace('[tool:mcp__synapse__recall] {"query": "x"}'), true);
  assert.equal(hasToolTrace('[user] hi\n\n[result] a build log long enough to keep'), true);
});

test('hasToolTrace: prose that starts with a tool-like word is not a trace', () => {
  for (const line of ['Read the docs before you start', 'Task completed', 'Edit the file', 'Write a test', 'Bash is fine', 'Tool use is off']) {
    assert.equal(hasToolTrace('[user] ' + line + '\n\n[assistant] ' + line), false, line);
    assert.equal(hasToolTrace(line), false, line);
  }
  assert.equal(hasToolTrace('see [tool:Read] in the docs'), false, 'a mention inside a line is not a trace line');
  assert.equal(hasToolTrace(''), false);
  assert.equal(hasToolTrace(undefined), false);
});

test('turnsLabel: a window of a longer session says so, a whole one does not', () => {
  const eps = (from: number, n: number) => Array.from({ length: n }, (_, i) => ({ sequence: from + i }));
  assert.equal(turnsLabel(eps(1, 12), 12), '12 turns');
  assert.equal(turnsLabel(eps(1, 1), 1), '1 turn');
  assert.equal(turnsLabel(eps(3, 500), 520), 'seq 3–502 of 520 turns');
  assert.equal(turnsLabel([], 0), '0 turns');
});
