// Appending a "load more" page to a list that is already on screen.
// Offsets move under a live corpus (a write between two pages shifts rows into the next one),
// so a page can repeat rows the list holds: they are dropped, not rendered twice under one key.
export function appendUnique<T>(list: T[], page: T[], key: (item: T) => string): T[] {
  const seen = new Set(list.map(key));
  const fresh = page.filter((it) => {
    const k = key(it);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return fresh.length ? list.concat(fresh) : list;
}
