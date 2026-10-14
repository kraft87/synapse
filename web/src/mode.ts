// `#mock` (or a sticky session flag) serves the fixtures in mock.ts instead of the API.
// Kept apart from them so the flag is a cheap synchronous import.
export const MOCK =
  typeof location !== 'undefined' &&
  (location.hash.includes('mock') || (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('synapse.mock') === '1'));
if (MOCK && typeof sessionStorage !== 'undefined') sessionStorage.setItem('synapse.mock', '1');
