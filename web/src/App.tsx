// Root: token gate -> shell (header + active page + overlay layer). Page
// selection is React state; overlays live in the URL hash.
import { StoreProvider, useStore } from './state';
import { MOCK } from './mock';
import { Header } from './components/Header';
import { Login } from './components/Login';
import { Feed } from './pages/Feed';
import { Search } from './pages/Search';
import { Recall } from './pages/Recall';
import { Stub } from './pages/Stub';
import { Overlays } from './overlays/Overlays';

function Shell() {
  const s = useStore();
  if (!MOCK && !s.token) return <Login />;
  return (
    <>
      <Header />
      {s.page === 'feed' ? <Feed />
        : s.page === 'search' ? <Search />
        : s.page === 'recall' ? <Recall />
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
