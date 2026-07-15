// Root: token gate -> shell (header + active page + overlay layer). Page
// selection is React state; overlays live in the URL hash.
import { Component, Suspense, lazy, type ReactNode } from 'react';
import { StoreProvider, useStore } from './state';
import { MOCK } from './mock';
import { Header } from './components/Header';
import { Login } from './components/Login';
import { Feed } from './pages/Feed';
import { Search } from './pages/Search';
import { Recall } from './pages/Recall';
import { Review } from './pages/Review';
import { Metrics } from './pages/Metrics';
import { Timeline } from './pages/Timeline';
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

// A page that throws during render/commit must not blank the whole app (an
// unboundaried commit error unmounts the entire React tree — the Graph seed
// crash did exactly that). Contain it to the page area; header/nav stay usable.
class PageBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '40px 16px' }}>
          <div style={{ fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--err)' }}>this page crashed</div>
          <div style={{ fontFamily: 'var(--font-data)', fontSize: '11.5px', color: 'var(--txt3)', maxWidth: '560px', textAlign: 'center', overflowWrap: 'anywhere' }}>
            {String(this.state.error)}
          </div>
          <button className="chipbtn" onClick={() => this.setState({ error: null })}
            style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '7px', padding: '6px 12px', fontSize: '12px', fontFamily: 'var(--font-data)', cursor: 'pointer' }}>
            retry
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}

function Shell() {
  const s = useStore();
  if (!MOCK && !s.token) return <Login />;
  return (
    <>
      <Header />
      {/* key={s.page} resets the boundary when the user navigates away */}
      <PageBoundary key={s.page}>
        {s.page === 'feed' ? <Feed />
          : s.page === 'search' ? <Search />
          : s.page === 'recall' ? <Recall />
          : s.page === 'review' ? <Review />
          : s.page === 'metrics' ? <Metrics />
          : s.page === 'timeline' ? <Timeline />
          : s.page === 'graph' ? <Suspense fallback={<GraphFallback />}><Graph /></Suspense>
          : <Stub page={s.page} />}
      </PageBoundary>
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
