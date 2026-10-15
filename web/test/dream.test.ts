import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultRun, fmtDuration, runDuration, runState } from '../src/dream';

const NOW = Date.parse('2026-10-14T12:00:00Z');
const run = (id: number, started_at: string, finished_at: string | null, duration_s: number | null = null) =>
  ({ id, started_at, finished_at, duration_s });

test('fmtDuration: seconds below two minutes, minutes above', () => {
  assert.equal(fmtDuration(25.4), '25 s');
  assert.equal(fmtDuration(119), '119 s');
  assert.equal(fmtDuration(120), '2 min');
  assert.equal(fmtDuration(1800), '30 min');
});

test('an unfinished run is in flight when recent and abandoned once it is long dead', () => {
  assert.equal(runState(run(1, '2026-10-14T11:00:00Z', null), NOW), 'in flight');
  assert.equal(runState(run(2, '2026-10-14T05:00:00Z', null), NOW), 'abandoned');
  assert.equal(runState(run(3, '2026-10-14T05:00:00Z', '2026-10-14T05:01:00Z'), NOW), 'finished');
});

test('runDuration: the length of a finished run, else what became of it', () => {
  assert.equal(runDuration(run(1, '2026-10-14T11:00:00Z', '2026-10-14T11:00:25Z', 25), NOW), '25 s');
  assert.equal(runDuration(run(2, '2026-10-14T11:00:00Z', null), NOW), 'in flight');
  assert.equal(runDuration(run(3, '2026-10-13T11:00:00Z', null), NOW), 'abandoned');
});

test('defaultRun skips a crashed newest run for the latest finished one', () => {
  const crashed = run(9, '2026-10-14T03:00:00Z', null);
  const done = run(8, '2026-10-13T03:00:00Z', '2026-10-13T03:40:00Z', 2400);
  assert.equal(defaultRun([crashed, done]), done);
  assert.equal(defaultRun([crashed]), crashed);       // nothing finished: the newest
  assert.equal(defaultRun([]), undefined);
});
