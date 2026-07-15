// Global/header state. Theme, group, project, source, and token are persisted
// to localStorage (README §State Management). Catalog + live status live here so
// the header can render them on every page.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { fetchCatalog, fetchProposals, getToken, clearToken, onAuthFail, type Catalog } from './api';
import { MOCK } from './mock';

export type Group = 'all' | 'technical' | 'personal';
export type Page = 'feed' | 'recall' | 'graph' | 'timeline' | 'metrics' | 'review' | 'search';

const ls = {
  get: (k: string, d: string) => localStorage.getItem(k) ?? d,
  set: (k: string, v: string) => localStorage.setItem(k, v),
};

interface Store {
  token: string | null;
  setTokenValue: (t: string | null) => void;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
  group: Group;
  setGroup: (g: Group) => void;
  project: string;
  setProject: (p: string) => void;
  source: string;
  setSource: (s: string) => void;
  page: Page;
  setPage: (p: Page) => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  catalog: Catalog | null;
  online: boolean;
  setOnline: (b: boolean) => void;
  queueDepth: number;
  setQueueDepth: (n: number) => void;
  reviewPending: number;
  setReviewPending: (n: number) => void;
}

const Ctx = createContext<Store | null>(null);
export const useStore = (): Store => {
  const s = useContext(Ctx);
  if (!s) throw new Error('useStore outside provider');
  return s;
};

export function StoreProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => getToken());
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (ls.get('synapse.theme', 'dark') === 'light' ? 'light' : 'dark'));
  const [group, setGroupState] = useState<Group>(() => ls.get('synapse.group', 'all') as Group);
  const [project, setProjectState] = useState<string>(() => ls.get('synapse.project', 'all'));
  const [source, setSourceState] = useState<string>(() => ls.get('synapse.source', 'all'));
  const [page, setPage] = useState<Page>('feed');
  const [searchQuery, setSearchQuery] = useState('');
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [online, setOnline] = useState(true);
  const [queueDepth, setQueueDepth] = useState(0);
  const [reviewPending, setReviewPending] = useState(0);

  // theme -> <html data-theme> + persistence
  useEffect(() => { document.documentElement.dataset.theme = theme; ls.set('synapse.theme', theme); }, [theme]);

  // a 401 anywhere clears the token and returns to the login screen
  useEffect(() => { onAuthFail(() => { clearToken(); setToken(null); }); }, []);

  // load the catalog + the Review pending-proposal count once we have a token (or in
  // mock). The count seeds the nav badge before Review is ever opened; the Review page
  // keeps it in sync after decisions.
  useEffect(() => {
    if (!MOCK && !token) return;
    let live = true;
    fetchCatalog().then((c) => { if (live) { setCatalog(c); setOnline(true); } }).catch(() => {});
    fetchProposals().then((r) => { if (live) setReviewPending(r.pending_count || 0); }).catch(() => {});
    return () => { live = false; };
  }, [token]);

  const setTokenValue = useCallback((t: string | null) => { setToken(t); if (!t) clearToken(); }, []);
  const toggleTheme = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);
  const setGroup = useCallback((g: Group) => { setGroupState(g); ls.set('synapse.group', g); }, []);
  const setProject = useCallback((p: string) => { setProjectState(p); ls.set('synapse.project', p); }, []);
  const setSource = useCallback((s: string) => { setSourceState(s); ls.set('synapse.source', s); }, []);

  const value = useMemo<Store>(() => ({
    token, setTokenValue, theme, toggleTheme, group, setGroup, project, setProject,
    source, setSource, page, setPage, searchQuery, setSearchQuery, catalog, online, setOnline,
    queueDepth, setQueueDepth, reviewPending, setReviewPending,
  }), [token, theme, group, project, source, page, searchQuery, catalog, online, queueDepth, reviewPending, setTokenValue, toggleTheme, setGroup, setProject, setSource]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
