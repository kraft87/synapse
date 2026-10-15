import test from 'node:test';
import assert from 'node:assert/strict';
import { appendUnique } from '../src/paging';

const key = (r: { id: string }) => r.id;
const rows = (...ids: string[]) => ids.map((id) => ({ id }));

test('appendUnique adds only the rows the list does not hold, in page order', () => {
  const list = rows('a', 'b');
  assert.deepEqual(appendUnique(list, rows('b', 'c', 'd'), key), rows('a', 'b', 'c', 'd'));
});

test('appendUnique drops a row repeated inside the page itself', () => {
  assert.deepEqual(appendUnique(rows('a'), rows('b', 'b'), key), rows('a', 'b'));
});

test('appendUnique returns the same list when the page adds nothing (no re-render)', () => {
  const list = rows('a', 'b');
  assert.equal(appendUnique(list, rows('a', 'b'), key), list);
  assert.equal(appendUnique(list, [], key), list);
});
