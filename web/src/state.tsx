// Global/header state. Theme, group, project, source, and token are persisted
// to localStorage (README §State Management). Catalog + live status live here so
// the header can render them on every page.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { fetchCatalog, fetchProposals, getToken, clearToken, knownFilter, onAuthFail, type Catalog, type ProcessingStatus } from './api';
import { startLiveLink, sameQueue, type FeedListener, type LinkState } from './liveLink';
import { MOCK } from './mode';

export type Group = 'all' | 'technical' | 'personal';
export type Page = 'feed' | 'recall' | 'graph' | 'timeline' | 'metrics' | 'review' | 'search';

const PENDING_REFRESH_MS = 60_000;
// The server caches the catalog for 5 minutes (_CATALOG_TTL_S), so asking more often gains nothing.
const CATALOG_REFRESH_MS = 300_000;

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
  // The live stream's state and its last queue snapshot (null: none to show). The stream is
  // the shell's, open on every page; the Feed subscribes for its rows and resyncs.
  link: LinkState;
  queue: ProcessingStatus | null;
  subscribeFeed: (l: FeedListener) => () => void;
  reviewPending: number;
  setReviewPending: (n: number) => void;
  // Graph seed handoff (phase 6): the entity dossier's "view in graph" sets this, then
  // navigates to the Graph page, which consumes it on mount and clears it.
  graphSeed: { uuid: string; name: string } | null;
  seedGraph: (uuid: string, name: string) => void;
  clearGraphSeed: () => void;
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
  // MOCK has no server to stream from; its Feed polls the fixtures.
  const [link, setLink] = useState<LinkState>(MOCK ? 'polling' : 'connecting');
  const [queue, setQueue] = useState<ProcessingStatus | null>(null);
  const feedListeners = useRef(new Set<FeedListener>());
  const [reviewPending, setReviewPending] = useState(0);
  const [graphSeed, setGraphSeed] = useState<{ uuid: string; name: string } | null>(null);

  // theme -> <html data-theme> + persistence
  useEffect(() => { document.documentElement.dataset.theme = theme; ls.set('synapse.theme', theme); }, [theme]);

  // a 401 anywhere clears the token and returns to the login screen
  useEffect(() => { onAuthFail(() => { clearToken(); setToken(null); }); }, []);

  // the live stream lives as long as the session, not as long as the Feed page
  useEffect(() => {
    if (MOCK || !token) return;
    return startLiveLink({
      onLink: setLink,
      onQueue: (q) => setQueue((prev) => (sameQueue(prev, q) ? prev : q)),
      onItem: (item) => feedListeners.current.forEach((l) => l.onItem(item)),
      onResync: () => feedListeners.current.forEach((l) => l.onResync()),
    });
  }, [token]);

  // load the catalog once we have a token (or in mock), and again every few minutes and when
  // the tab is looked at again, so a project created since shows up in the filter without a reload
  useEffect(() => {
    if (!MOCK && !token) return;
    let live = true;
    const load = () => { fetchCatalog().then((c) => { if (live) setCatalog(c); }).catch(() => {}); };
    const onVisible = () => { if (document.visibilityState === 'visible') load(); };
    load();
    const timer = window.setInterval(() => { if (document.visibilityState !== 'hidden') load(); }, CATALOG_REFRESH_MS);
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      live = false;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [token]);

  // The Review badge follows the server: proposals are decided from the CLI or another tab
  // and raised by the nightly dream while this page sits open, so the count is refetched
  // every minute and when the tab is looked at again. The Proposals tab also sets it from
  // its own list as soon as it loads.
  useEffect(() => {
    if (!MOCK && !token) return;
    let live = true;
    const refresh = () => {
      fetchProposals({ status: 'proposed' }).then((r) => { if (live) setReviewPending(r.pending_count || 0); }).catch(() => {});
    };
    const onVisible = () => { if (document.visibilityState === 'visible') refresh(); };
    refresh();
    const timer = window.setInterval(() => { if (document.visibilityState !== 'hidden') refresh(); }, PENDING_REFRESH_MS);
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', refresh);
    return () => {
      live = false;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', refresh);
    };
  }, [token]);

  const setTokenValue = useCallback((t: string | null) => { setToken(t); if (!t) clearToken(); }, []);
  const toggleTheme = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);
  const setGroup = useCallback((g: Group) => { setGroupState(g); ls.set('synapse.group', g); }, []);
  const setProject = useCallback((p: string) => { setProjectState(p); ls.set('synapse.project', p); }, []);
  const setSource = useCallback((s: string) => { setSourceState(s); ls.set('synapse.source', s); }, []);

  // The restored project/source may not be in the catalog (renamed, emptied): drop it rather
  // than filter by a value the header shows as "all".
  useEffect(() => {
    if (!catalog) return;
    const p = knownFilter(project, catalog.projects);
    if (p !== project) setProject(p);
    const src = knownFilter(source, catalog.sources);
    if (src !== source) setSource(src);
  }, [catalog, project, source, setProject, setSource]);
  const subscribeFeed = useCallback((l: FeedListener) => {
    feedListeners.current.add(l);
    return () => { feedListeners.current.delete(l); };
  }, []);
  const seedGraph = useCallback((uuid: string, name: string) => setGraphSeed({ uuid, name }), []);
  const clearGraphSeed = useCallback(() => setGraphSeed(null), []);

  const value = useMemo<Store>(() => ({
    token, setTokenValue, theme, toggleTheme, group, setGroup, project, setProject,
    source, setSource, page, setPage, searchQuery, setSearchQuery, catalog, link, queue, subscribeFeed,
    reviewPending, setReviewPending, graphSeed, seedGraph, clearGraphSeed,
  }), [token, theme, group, project, source, page, searchQuery, catalog, link, queue, reviewPending, graphSeed, setTokenValue, toggleTheme, setGroup, setProject, setSource, subscribeFeed, seedGraph, clearGraphSeed]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
