import test from 'node:test';
import assert from 'node:assert/strict';
import { onFlagged, postFlag, type FlaggedDetail } from '../src/api';

const g = globalThis as unknown as Record<string, unknown>;
g.localStorage = { getItem: () => 'token' };
g.window = new EventTarget();
const bodies: unknown[] = [];
let answer = true;
g.fetch = async (_url: string, init: RequestInit) => {
  bodies.push(JSON.parse(String(init.body)));
  return { status: 200, ok: true, json: async () => ({ status: 'ok', flagged: answer }) } as Response;
};

test('a flag request carries the state wanted, not a bare toggle', async () => {
  bodies.length = 0;
  await postFlag('fact', 'f1', true);
  await postFlag('fact', 'f1', false, 'why');
  assert.deepEqual(bodies, [
    { kind: 'fact', id: 'f1', flagged: true },
    { kind: 'fact', id: 'f1', flagged: false, note: 'why' },
  ]);
});

test('the state the server ended in is announced to every button on that row', async () => {
  const seen: FlaggedDetail[] = [];
  const off = onFlagged((d) => seen.push(d));
  answer = false; // the server says it ended unflagged, whatever was asked
  await postFlag('preference', '3', true);
  off();
  assert.deepEqual(seen, [{ kind: 'preference', id: '3', flagged: false }]);
});
