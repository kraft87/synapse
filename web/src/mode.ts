// `#mock` (or a sticky session flag) serves the fixtures in mock.ts instead of the API.
// Kept apart from them so the flag is a cheap synchronous import.

// Only the bare marker counts (`#mock`, or `#mock&...`): a hash that merely contains the word,
// such as a session id in `#/session/payments-mock-run-42`, is a route, not a mode switch.
export const isMockHash = (hash: string): boolean => /^#mock(&|$)/.test(hash);

export const MOCK =
  typeof location !== 'undefined' &&
  (isMockHash(location.hash) || (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('synapse.mock') === '1'));
// The sticky flag keeps a mock tab in mock across reloads; only an explicit `#mock` sets it.
if (typeof location !== 'undefined' && isMockHash(location.hash) && typeof sessionStorage !== 'undefined') sessionStorage.setItem('synapse.mock', '1');
