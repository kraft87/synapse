import test from 'node:test';
import assert from 'node:assert/strict';

// validityLine and the as-of label read the browser's zone: this run's "browser" is Toronto.
process.env.TZ = 'America/Toronto';

import { localDay, localMonth, localMinute, validityLine } from '../src/tokens';
import { asOfDateFor, asOfLabel, AS_OF_TODAY } from '../src/asOf';

const TORONTO = 'America/Toronto';

test('localDay: an evening event belongs to the day it was lived, not the UTC one', () => {
  // Said at 20:30 EDT on Sep 30, stored as 00:30Z on Oct 1.
  assert.equal(localDay('2026-10-01T00:30:00+00:00', TORONTO), '2026-09-30');
  assert.equal(localMonth('2026-10-01T00:30:00+00:00', TORONTO), '2026-09');
  assert.equal(localDay('2026-10-01T00:30:00+00:00', 'UTC'), '2026-10-01');
  // Standard time (UTC-5) and the year boundary.
  assert.equal(localDay('2027-01-01T04:30:00+00:00', TORONTO), '2026-12-31');
  assert.equal(localMonth('2027-01-01T04:30:00+00:00', TORONTO), '2026-12');
  // East of UTC the day runs ahead.
  assert.equal(localDay('2026-09-30T20:30:00+00:00', 'Asia/Tokyo'), '2026-10-01');
});

test('localDay: a date-only stamp stored at noon UTC keeps its day from UTC-11 to UTC+11', () => {
  for (const tz of ['Pacific/Pago_Pago', TORONTO, 'UTC', 'Europe/Berlin', 'Asia/Tokyo', 'Pacific/Noumea']) {
    assert.equal(localDay('2026-09-30T12:00:00+00:00', tz), '2026-09-30', tz);
  }
});

test('localDay: offsets in the stamp are instants, and nothing parseable gives an empty string', () => {
  assert.equal(localDay('2026-09-30T20:30:00-04:00', TORONTO), '2026-09-30');
  assert.equal(localDay('2026-09-30T20:30:00-04:00', 'UTC'), '2026-10-01');
  assert.equal(localDay(undefined, TORONTO), '');
  assert.equal(localDay(null, TORONTO), '');
  assert.equal(localDay('', TORONTO), '');
  assert.equal(localDay('not a date', TORONTO), '');
});

test('localMinute: 24-hour wall clock, midnight is 00', () => {
  assert.equal(localMinute('2026-10-01T00:30:00+00:00', TORONTO), '2026-09-30 20:30');
  assert.equal(localMinute('2026-10-01T04:05:00+00:00', TORONTO), '2026-10-01 00:05');
  assert.equal(localMinute(undefined, TORONTO), '');
});

test('validityLine names the viewer\'s days', () => {
  assert.equal(
    validityLine('2026-10-01T00:30:00+00:00', '2026-11-01T03:59:00+00:00'),
    'valid since 2026-09-30 · superseded 2026-10-31',
  );
  assert.equal(validityLine(undefined, null), 'valid');
});

test('asOfDateFor: a month back from the 31st lands on the last day of the shorter month', () => {
  const mar31 = new Date(2026, 2, 31, 12);
  const back = asOfDateFor(AS_OF_TODAY - 1, mar31) as Date;
  assert.deepEqual([back.getFullYear(), back.getMonth(), back.getDate()], [2026, 1, 28]);
  const leap = asOfDateFor(AS_OF_TODAY - 1, new Date(2028, 2, 31, 12)) as Date;
  assert.deepEqual([leap.getMonth(), leap.getDate()], [1, 29]);
  const nov = asOfDateFor(AS_OF_TODAY - 2, new Date(2027, 0, 31, 12)) as Date;
  assert.deepEqual([nov.getFullYear(), nov.getMonth(), nov.getDate()], [2026, 10, 30]);
  const year = asOfDateFor(AS_OF_TODAY - 12, new Date(2027, 0, 31, 12)) as Date;
  assert.deepEqual([year.getFullYear(), year.getMonth(), year.getDate()], [2026, 0, 31]);
  assert.equal(asOfDateFor(AS_OF_TODAY, mar31), null);
});

test('asOfLabel is the local day, not the UTC one, in the evening', () => {
  // 21:00 EDT on Oct 14 is already Oct 15 in UTC; a month back is Sep 14 in Toronto.
  assert.equal(asOfLabel(AS_OF_TODAY - 1, new Date(2026, 9, 14, 21, 0)), '2026-09-14');
  assert.equal(asOfLabel(AS_OF_TODAY), 'today');
});
