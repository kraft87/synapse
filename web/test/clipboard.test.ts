import test from 'node:test';
import assert from 'node:assert/strict';
import { copyText } from '../src/clipboard';

const g = globalThis as unknown as Record<string, unknown>;

// A page with just enough DOM for the textarea route; `copied` records what execCommand saw.
function page(opts: { clipboard?: { writeText: (t: string) => Promise<void> }; exec?: () => boolean }) {
  const copied: string[] = [];
  const ta = { value: '', style: { cssText: '' }, setAttribute() {}, select() { copied.push(ta.value); }, remove() { removed.push(ta); } };
  const removed: unknown[] = [];
  // Node 22 defines navigator as a getter on globalThis.
  Object.defineProperty(g, 'navigator', { value: { clipboard: opts.clipboard }, configurable: true, writable: true });
  g.document = {
    activeElement: null,
    createElement: () => ta,
    body: { appendChild() {} },
    execCommand: opts.exec ?? (() => true),
  };
  return { copied, removed };
}

test('copyText uses the async clipboard when the page has one', async () => {
  const written: string[] = [];
  const p = page({ clipboard: { writeText: async (t) => { written.push(t); } } });
  assert.equal(await copyText('hello'), true);
  assert.deepEqual(written, ['hello']);
  assert.deepEqual(p.copied, []);
});

test('copyText falls back to a textarea over plain http, where navigator.clipboard is undefined', async () => {
  const p = page({});
  assert.equal(await copyText('hello'), true);
  assert.deepEqual(p.copied, ['hello']);
  assert.equal(p.removed.length, 1);
});

test('copyText tries the textarea when the clipboard refuses, and reports a copy that failed', async () => {
  const p = page({ clipboard: { writeText: async () => { throw new Error('denied'); } }, exec: () => false });
  assert.equal(await copyText('hello'), false);
  assert.deepEqual(p.copied, ['hello']);
  assert.equal(p.removed.length, 1);
});
