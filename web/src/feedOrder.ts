// The Feed's ordering and merge rules, pure so they can be unit-tested. The order is the
// server's keyset order (mcp_server/dashboard_routes.py _feed): timestamp descending, then
// the id as text descending, whatever the type. Live rows and refetched pages are folded in by
// that order, so what a live Feed shows is what a reload would show.
import type { FeedItem } from './api';

export const keyOf = (it: FeedItem) => it.type + ':' + it.id;

// Sorts newest first (negative when `a` belongs above `b`).
export function compareFeed(a: FeedItem, b: FeedItem): number {
  const d = Date.parse(b.ts) - Date.parse(a.ts);
  if (d !== 0) return d;
  return a.id === b.id ? 0 : a.id < b.id ? 1 : -1;
}

// Put one live row at its place in the list. `more` = the list has a next page: a row older
// than everything loaded is left out, it arrives when the user pages. `at` is where it landed
// (-1: left out or already there).
export function insertFeedItem(items: FeedItem[], item: FeedItem, more: boolean): { items: FeedItem[]; at: number } {
  const k = keyOf(item);
  if (items.some((it) => keyOf(it) === k)) return { items, at: -1 };
  let at = items.findIndex((it) => compareFeed(item, it) < 0);
  if (at < 0) {
    if (more) return { items, at: -1 };
    at = items.length;
  }
  return { items: [...items.slice(0, at), item, ...items.slice(at)], at };
}

// Fold a freshly fetched page 1 into the list on screen. `replaced` means the page stands in
// for the list (the caller then takes the page's cursor): a load (`replace`), or a page that
// shares no row with the list because more than a page was written meanwhile, where merging
// would leave a hole between the page and the old tail. A replaced list keeps only the live
// rows newer than the page's tail. Otherwise the unseen rows are inserted in order, and
// `fresh` names the ones that landed above everything already shown.
export function mergePage1(
  prev: FeedItem[], page: FeedItem[], replace: boolean, more: boolean,
): { items: FeedItem[]; replaced: boolean; fresh: string[] } {
  const have = new Set(prev.map(keyOf));
  const replaced = replace || !page.some((it) => have.has(keyOf(it)));
  const tail = page[page.length - 1];
  const [base, extra] = replaced
    ? [page, tail ? prev.filter((it) => compareFeed(it, tail) < 0) : prev]
    : [prev, page];
  let items = base;
  for (const it of extra) items = insertFeedItem(items, it, !replaced && more).items;
  const firstOld = items.findIndex((it) => have.has(keyOf(it)));
  const fresh = replaced || firstOld < 0 ? [] : items.slice(0, firstOld).map(keyOf);
  return { items, replaced, fresh };
}
