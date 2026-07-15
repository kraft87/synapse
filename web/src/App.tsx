// Root: token gate -> shell (header + active page + overlay layer). Page
// selection is React state; overlays live in the URL hash.
import { Suspense, lazy } from 'react';
import { StoreProvider, useStore } from './state';
import { MOCK } from './mock';
import { Header } from './components/Header';
import { Login } from './components/Login';
import { Feed } from './pages/Feed';
import { Search } from './pages/Search';
import { Recall } from './pages/Recall';
import { Review } from './pages/Review';
import { Metrics } from './pages/Metrics';
import { Stub } from './pages/Stub';
import { Overlays } from './overlays/Overlays';

// Graph pulls in cytoscape + cose-bilkent (~500KB) — lazy so that weight loads only on
// the first Graph visit and code-splits into its own chunk (see web/build.mjs).
const Graph = lazy(() => import('./pages/Graph').then((m) => ({ default: m.Graph })));
const GraphFallback = () => (
  <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--txt3)', fontFamily: 'var(--font-data)', fontSize: '12.5px' }}>
    loading graph explorer…
  </main>
);

function Shell() {
  const s = useStore();
  if (!MOCK && !s.token) return <Login />;
  return (
    <>
      <Header />
      {s.page === 'feed' ? <Feed />
        : s.page === 'search' ? <Search />
        : s.page === 'recall' ? <Recall />
        : s.page === 'review' ? <Review />
        : s.page === 'metrics' ? <Metrics />
        : s.page === 'graph' ? <Suspense fallback={<GraphFallback />}><Graph /></Suspense>
        : <Stub page={s.page} />}
      <Overlays />
    </>
  );
}

export function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  );
}
