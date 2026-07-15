// Graph — the knowledge-graph explorer (README §3, spec §4, screenshot 03).
//
// Cytoscape + cose-bilkent render a BFS neighborhood served by
// GET /dash/api/graph/{entities,neighborhood}. The React tree owns the toolbar,
// inspector, and the accumulated {nodes,edges} model; the <div> canvas is driven
// imperatively (positions must survive expand-merges, which React reconciliation
// would clobber). Interaction contract (spec §4):
//   • first click selects (inspector populates); second click on the selected node
//     expands its depth-1 neighborhood, MERGING and keeping existing positions.
//   • hover = highlight only, never mutates.
//   • superseded edges (t_invalid ≤ as-of) render dashed at 18% opacity but stay in
//     the layout physics; the as-of slider re-filters client-side (edges carry
//     validity); nodes with zero visible edges fade to 25%.
//   • hard cap 150 rendered nodes; overflow collapses the lowest-degree leaves into a
//     synthetic "+N more" cluster chip that pages more in on click.
//   • above 100 nodes, labels show only for degree ≥ 3 / hovered / selected.
import type React from 'react';
import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import {
  fetchGraphEntities, fetchGraphNeighborhood, AuthError,
  type GraphEntity, type GraphNode, type GraphEdge,
} from '../api';
import { useStore } from '../state';
import { etColor, cssValue, validityLine } from '../tokens';
import { openEpisode } from '../hash';

const mono = 'var(--font-data)';
const NODE_CAP = 150;
const CLUSTER_PAGE = 50;
const SUPERTYPES = ['Person', 'Project', 'Technology', 'Organization', 'Concept', 'Event'];

// register the layout extension once (idempotent guard — HMR / double-mount safe)
let registered = false;
function ensureRegistered() {
  if (registered) return;
  try { cytoscape.use(coseBilkent); } catch { /* already registered */ }
  registered = true;
}

// ---- as-of slider (0..24 → months back; 24 = today/null) ----
function asOfDateFor(idx: number): Date | null {
  if (idx >= 24) return null;
  const d = new Date();
  d.setMonth(d.getMonth() - (24 - idx));
  return d;
}
const asOfLabel = (idx: number): string => (idx >= 24 ? 'today' : (asOfDateFor(idx) as Date).toISOString().slice(0, 10));

// ---- concrete-color palette (cytoscape draws to canvas, so var(--x) can't resolve) ----
function buildPalette() {
  const v = cssValue;
  return {
    txt2: v('--txt2'), txt3: v('--txt3'), line2: v('--line2'), acc: v('--acc'), bg3: v('--bg3'),
    et: (supertype?: string | null) => cssValue(etColor(supertype)),
  };
}
const nodeSize = (deg?: number) => 18 + Math.min(34, Math.sqrt(Math.max(1, deg || 1)) * 9);

type Palette = ReturnType<typeof buildPalette>;
function stylesheet(pal: Palette, sizeRef: { current: number }): cytoscape.StylesheetStyle[] {
  const labelFor = (ele: cytoscape.NodeSingular): string => {
    if (ele.data('cluster')) return ele.data('name');
    if (sizeRef.current <= 100) return ele.data('name');
    if ((ele.data('degree') || 0) >= 3 || ele.selected() || ele.hasClass('hl')) return ele.data('name');
    return '';
  };
  return [
    { selector: 'node', style: {
      'background-color': (e: cytoscape.NodeSingular) => pal.et(e.data('supertype')),
      width: (e: cytoscape.NodeSingular) => nodeSize(e.data('degree')),
      height: (e: cytoscape.NodeSingular) => nodeSize(e.data('degree')),
      label: labelFor, color: pal.txt2, 'font-family': 'IBM Plex Mono, monospace', 'font-size': 10.5,
      'text-valign': 'center', 'text-halign': 'right', 'text-margin-x': 4,
      'border-width': 0, 'transition-property': 'opacity', 'transition-duration': 0.15,
    } as unknown as cytoscape.Css.Node },
    { selector: 'node.cluster', style: {
      'background-color': pal.bg3, 'border-width': 1, 'border-color': pal.line2, shape: 'round-rectangle',
      color: pal.acc, 'text-halign': 'center', 'text-margin-x': 0, width: 'label', height: 20,
      padding: 6, 'font-size': 10.5,
    } as unknown as cytoscape.Css.Node },
    { selector: 'node:selected', style: { 'border-width': 2, 'border-color': pal.acc } },
    { selector: 'node.hl', style: { 'border-width': 2, 'border-color': pal.acc } },
    { selector: 'node.faded', style: { opacity: 0.25 } },
    { selector: 'edge', style: {
      width: (e: cytoscape.EdgeSingular) => 1 + Math.min(3, Math.log1p(e.data('rc') || 0)),
      'line-color': pal.line2, 'curve-style': 'bezier', opacity: 0.5, 'target-arrow-shape': 'none',
    } as unknown as cytoscape.Css.Edge },
    { selector: 'edge.hl', style: { 'line-color': pal.acc, opacity: 0.9 } },
    { selector: 'edge.superseded', style: { 'line-style': 'dashed', opacity: 0.18 } },
    { selector: 'edge.hidden', style: { display: 'none' } },
  ];
}

const nodeData = (n: GraphNode) => ({ id: n.uuid, name: n.name, supertype: n.entity_type, degree: n.degree });
const edgeData = (e: GraphEdge) => ({
  id: e.uuid, source: e.src, target: e.tgt, name: e.name, rc: e.retrieval_count,
  t_valid: e.t_valid, t_invalid: e.t_invalid,
});

// per-edge as-of visibility, matched to the server predicate + spec §4.
function edgeVisual(e: { t_valid: string | null; t_invalid: string | null }, asOf: Date | null) {
  if (asOf) {
    if (e.t_valid && new Date(e.t_valid) > asOf) return { hidden: true, superseded: false };
    if (e.t_invalid && new Date(e.t_invalid) <= asOf) return { hidden: false, superseded: true };
    return { hidden: false, superseded: false };
  }
  return { hidden: false, superseded: e.t_invalid != null };
}

interface GraphModel { nodes: GraphNode[]; edges: GraphEdge[]; }

export function Graph() {
  const store = useStore();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const sizeRef = useRef(0);           // node count, for the label mapper
  const selIdRef = useRef<string | null>(null);
  const pendingLayoutRef = useRef<{ mode: 'full' | 'incremental' | 'none'; added: string[] }>({ mode: 'none', added: [] });
  const expandRef = useRef<(uuid: string) => void>(() => {});
  const pageMoreRef = useRef<() => void>(() => {});
  const asOfReqRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const suggestTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [model, setModel] = useState<GraphModel>({ nodes: [], edges: [] });
  const [renderCap, setRenderCap] = useState(NODE_CAP);
  const [seedText, setSeedText] = useState('');
  const [seedUuid, setSeedUuid] = useState<string | null>(null);
  const [depth, setDepth] = useState<1 | 2>(2);
  const [asOfIdx, setAsOfIdx] = useState(24);
  const [selId, setSelId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [notFound, setNotFound] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [counts, setCounts] = useState({ nodes: 0, edges: 0 });
  const [suggest, setSuggest] = useState<GraphEntity[]>([]);
  const [suggestIdx, setSuggestIdx] = useState(-1);

  // ---- merge/replace a fetched neighborhood into the accumulated model ----
  const loadNeighborhood = useCallback((
    query: string, opts: { depth?: 1 | 2; as_of?: string | null; replace: boolean },
  ) => {
    setLoading(true); setNotFound(false);
    fetchGraphNeighborhood({ entity: query, depth: opts.depth ?? depth, as_of: opts.as_of ?? null, limit: NODE_CAP })
      .then((nb) => {
        setSeedUuid(nb.seed);
        setTruncated(nb.truncated);
        if (opts.replace) {
          pendingLayoutRef.current = { mode: 'full', added: nb.nodes.map((n) => n.uuid) };
          setModel({ nodes: nb.nodes, edges: nb.edges });
          setSelId(nb.seed);
          const s = nb.nodes.find((n) => n.uuid === nb.seed);
          if (s) setSeedText(s.name);
        } else {
          setModel((prev) => {
            const nByU = new Map(prev.nodes.map((n) => [n.uuid, n]));
            const added: string[] = [];
            for (const n of nb.nodes) if (!nByU.has(n.uuid)) { nByU.set(n.uuid, n); added.push(n.uuid); }
            const eByU = new Map(prev.edges.map((e) => [e.uuid, e]));
            for (const e of nb.edges) if (!eByU.has(e.uuid)) eByU.set(e.uuid, e);
            pendingLayoutRef.current = { mode: 'incremental', added };
            return { nodes: [...nByU.values()], edges: [...eByU.values()] };
          });
        }
      })
      .catch((err) => {
        if (err instanceof AuthError) return;
        if (String(err?.message || '').includes('404')) { if (opts.replace) { setNotFound(true); setModel({ nodes: [], edges: [] }); setSelId(null); } }
        else setToast('neighborhood query failed — showing last graph');
      })
      .finally(() => setLoading(false));
  }, [depth]);

  const runSeed = useCallback((query: string) => {
    if (!query.trim()) return;
    setRenderCap(NODE_CAP);
    setSuggest([]); setSuggestIdx(-1);
    loadNeighborhood(query.trim(), { replace: true });
  }, [loadNeighborhood]);

  const expand = useCallback((uuid: string) => { loadNeighborhood(uuid, { replace: false, depth: 1 }); }, [loadNeighborhood]);
  const pageMore = useCallback(() => { setRenderCap((c) => c + CLUSTER_PAGE); pendingLayoutRef.current = { mode: 'incremental', added: [] }; }, []);
  useEffect(() => { expandRef.current = expand; pageMoreRef.current = pageMore; }, [expand, pageMore]);

  // ---- init cytoscape once ----
  useEffect(() => {
    ensureRegistered();
    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      style: stylesheet(buildPalette(), sizeRef),
      autounselectify: true, // selection is driven by our selId, not cytoscape's tap toggle
      boxSelectionEnabled: false,
      minZoom: 0.2, maxZoom: 3,
      wheelSensitivity: 0.25,
    });
    cyRef.current = cy;
    cy.on('tap', 'node', (ev) => {
      const id = ev.target.id();
      if (id === '__cluster') { pageMoreRef.current(); return; }
      if (selIdRef.current === id) expandRef.current(id);
      else setSelId(id);
    });
    cy.on('mouseover', 'node', (ev) => {
      const n = ev.target;
      n.addClass('hl'); n.connectedEdges().addClass('hl'); n.connectedEdges().connectedNodes().addClass('hl');
      if (containerRef.current) containerRef.current.style.cursor = 'pointer';
    });
    cy.on('mouseout', 'node', (ev) => {
      ev.cy.elements().removeClass('hl');
      if (containerRef.current) containerRef.current.style.cursor = 'default';
    });
    return () => { cy.destroy(); cyRef.current = null; };
  }, []);

  // ---- consume the dossier "view in graph" handoff on mount ----
  useEffect(() => {
    if (store.graphSeed) {
      const { uuid, name } = store.graphSeed;
      setSeedText(name);
      store.clearGraphSeed();
      loadNeighborhood(uuid, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- reconcile the model into cytoscape (diff → keep positions → layout) ----
  useEffect(() => {
    const cy = cyRef.current; if (!cy) return;
    const { mode, added } = pendingLayoutRef.current;
    pendingLayoutRef.current = { mode: 'none', added: [] };

    // capped render view — keep the highest-degree nodes, always keep the seed.
    let vnodes = model.nodes;
    let overflow = 0;
    if (vnodes.length > renderCap) {
      const sorted = [...vnodes].sort((a, b) => (b.degree || 0) - (a.degree || 0));
      const keep = sorted.slice(0, renderCap - 1);
      if (seedUuid && !keep.some((n) => n.uuid === seedUuid)) {
        const s = vnodes.find((n) => n.uuid === seedUuid); if (s) keep.push(s);
      }
      overflow = vnodes.length - keep.length;
      vnodes = keep;
    }
    const keepSet = new Set(vnodes.map((n) => n.uuid));
    const vedges = model.edges.filter((e) => keepSet.has(e.src) && keepSet.has(e.tgt));
    const wantNodes = new Set(vnodes.map((n) => n.uuid));
    if (overflow > 0) wantNodes.add('__cluster');
    const wantEdges = new Set(vedges.map((e) => e.uuid));

    const seedPos = seedUuid ? cy.getElementById(seedUuid).position() : { x: 0, y: 0 };
    const addedSet = new Set(added);
    cy.batch(() => {
      cy.nodes().forEach((n) => { if (!wantNodes.has(n.id())) n.remove(); });
      cy.edges().forEach((e) => { if (!wantEdges.has(e.id())) e.remove(); });
      for (const n of vnodes) {
        const ex = cy.getElementById(n.uuid);
        if (ex.empty()) {
          const jitter = () => (Math.random() - 0.5) * 120;
          cy.add({ group: 'nodes', data: nodeData(n), position: { x: (seedPos.x || 0) + jitter(), y: (seedPos.y || 0) + jitter() } });
        } else { ex.data(nodeData(n)); }
      }
      if (overflow > 0) {
        const c = cy.getElementById('__cluster');
        if (c.empty()) cy.add({ group: 'nodes', classes: 'cluster', data: { id: '__cluster', name: `+${overflow} more`, cluster: true, degree: 0 } });
        else c.data('name', `+${overflow} more`);
      }
      for (const e of vedges) if (cy.getElementById(e.uuid).empty()) cy.add({ group: 'edges', data: edgeData(e) });
    });
    sizeRef.current = cy.nodes().length;
    setCounts({ nodes: vnodes.length, edges: vedges.length });

    if (mode === 'full') {
      cy.layout({ name: 'cose-bilkent', animate: false, randomize: true, fit: true, padding: 40, idealEdgeLength: 95, nodeRepulsion: 5000, tile: true } as cytoscape.LayoutOptions).run();
    } else if (mode === 'incremental') {
      cy.nodes().forEach((n) => { if (!addedSet.has(n.id())) n.lock(); });
      const l = cy.layout({ name: 'cose-bilkent', animate: false, randomize: false, fit: false, padding: 40, idealEdgeLength: 95, nodeRepulsion: 5000 } as cytoscape.LayoutOptions);
      l.one('layoutstop', () => cy.nodes().unlock());
      l.run();
    }
    applyAsOf(cy, asOfDateFor(asOfIdx));
    // restore selection ring after element churn
    if (selIdRef.current) { const s = cy.getElementById(selIdRef.current); if (!s.empty()) s.select(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, renderCap]);

  // ---- as-of scrub: client-side re-filter; re-query (merge) only when truncated ----
  function applyAsOf(cy: cytoscape.Core, asOf: Date | null) {
    cy.batch(() => {
      cy.edges().forEach((ed) => {
        const { hidden, superseded } = edgeVisual({ t_valid: ed.data('t_valid'), t_invalid: ed.data('t_invalid') }, asOf);
        ed.toggleClass('hidden', hidden); ed.toggleClass('superseded', superseded);
      });
      cy.nodes().forEach((n) => {
        if (n.data('cluster')) { n.removeClass('faded'); return; }
        const vis = n.connectedEdges().filter((e) => !e.hasClass('hidden')).length;
        n.toggleClass('faded', vis === 0);
      });
    });
  }
  useEffect(() => {
    const cy = cyRef.current; if (!cy) return;
    applyAsOf(cy, asOfDateFor(asOfIdx));
    if (truncated && seedUuid) {
      if (asOfReqRef.current) clearTimeout(asOfReqRef.current);
      const d = asOfDateFor(asOfIdx);
      asOfReqRef.current = setTimeout(() => loadNeighborhood(seedUuid, { replace: false, as_of: d ? d.toISOString() : null }), 400);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOfIdx]);

  // ---- selection ring + label refresh ----
  useEffect(() => {
    selIdRef.current = selId;
    const cy = cyRef.current; if (!cy) return;
    cy.nodes().unselect();
    if (selId) { const n = cy.getElementById(selId); if (!n.empty()) n.select(); }
  }, [selId]);

  // ---- theme change → rebuild the (canvas-drawn) palette ----
  useEffect(() => {
    const cy = cyRef.current; if (!cy) return;
    cy.style(stylesheet(buildPalette(), sizeRef) as cytoscape.StylesheetStyle[]);
    applyAsOf(cy, asOfDateFor(asOfIdx));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.theme]);

  const relayout = () => {
    const cy = cyRef.current; if (!cy) return;
    cy.layout({ name: 'cose-bilkent', animate: false, randomize: true, fit: true, padding: 40, idealEdgeLength: 95, nodeRepulsion: 5000, tile: true } as cytoscape.LayoutOptions).run();
  };

  // ---- toolbar typeahead ----
  const onSeedInput = (val: string) => {
    setSeedText(val); setNotFound(false);
    if (suggestTimer.current) clearTimeout(suggestTimer.current);
    if (!val.trim()) { setSuggest([]); setSuggestIdx(-1); return; }
    suggestTimer.current = setTimeout(() => {
      fetchGraphEntities(val.trim(), 8).then((r) => { setSuggest(r); setSuggestIdx(-1); }).catch(() => setSuggest([]));
    }, 180);
  };
  const chooseSuggest = (e: GraphEntity) => { setSeedText(e.name); setSuggest([]); setSuggestIdx(-1); runSeed(e.uuid); };
  const onSeedKey = (ev: React.KeyboardEvent) => {
    if (suggest.length && (ev.key === 'ArrowDown' || ev.key === 'ArrowUp')) {
      ev.preventDefault();
      setSuggestIdx((i) => { const n = suggest.length; return ev.key === 'ArrowDown' ? (i + 1) % n : (i - 1 + n) % n; });
    } else if (ev.key === 'Enter') {
      if (suggestIdx >= 0 && suggest[suggestIdx]) chooseSuggest(suggest[suggestIdx]);
      else runSeed(seedText);
    } else if (ev.key === 'Escape') { setSuggest([]); setSuggestIdx(-1); }
  };

  // ---- inspector data ----
  const asOf = asOfDateFor(asOfIdx);
  const selNode = selId ? model.nodes.find((n) => n.uuid === selId) || null : null;
  const nameByU = new Map(model.nodes.map((n) => [n.uuid, n.name]));
  const selEdges = selId ? model.edges.filter((e) => e.src === selId || e.tgt === selId) : [];
  const selVisibleDeg = selEdges.filter((e) => !edgeVisual(e, asOf).hidden).length;
  const selServed = selEdges.reduce((s, e) => s + (e.retrieval_count || 0), 0);

  // ---- render ----
  const depthBtn = (d: 1 | 2): CSSProperties => ({
    border: 'none', cursor: 'pointer', padding: '7px 11px', fontSize: '12.5px', fontFamily: mono,
    background: depth === d ? 'var(--acc-bg)' : 'var(--bg2)', color: depth === d ? 'var(--acc)' : 'var(--txt2)',
  });

  return (
    <main style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '14px 16px 16px', boxSizing: 'border-box', maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
      {/* toolbar */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '10px' }}>
        <div style={{ position: 'relative', width: '220px' }}>
          <input
            className="field" value={seedText} spellCheck={false} placeholder="seed entity…"
            onChange={(e) => onSeedInput(e.target.value)} onKeyDown={onSeedKey}
            onBlur={() => setTimeout(() => setSuggest([]), 120)}
            style={{ width: '100%', boxSizing: 'border-box', background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '7px', padding: '7px 10px', fontFamily: mono, fontSize: '12.5px', color: 'var(--txt)', outline: 'none' }}
          />
          {suggest.length > 0 && (
            <div style={{ position: 'absolute', top: '38px', left: 0, right: 0, zIndex: 30, background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '8px', overflow: 'hidden', boxShadow: '0 8px 24px rgba(0,0,0,.35)' }}>
              {suggest.map((e, i) => (
                <button
                  key={e.uuid} className="result" onMouseDown={(ev) => { ev.preventDefault(); chooseSuggest(e); }}
                  onMouseEnter={() => setSuggestIdx(i)}
                  style={{ display: 'flex', width: '100%', alignItems: 'center', gap: '8px', textAlign: 'left', border: 'none', borderBottom: '1px solid var(--line)', background: i === suggestIdx ? 'var(--bg2)' : 'transparent', padding: '6px 10px', cursor: 'pointer' }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: etColor(e.entity_type), flexShrink: 0 }} />
                  <span style={{ fontFamily: mono, fontSize: '12px', color: 'var(--txt)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{e.name}</span>
                  <span style={{ fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}>{e.degree}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="chipbtn" onClick={() => runSeed(seedText)} style={{ border: '1px solid var(--acc)', background: 'var(--acc-bg)', color: 'var(--acc)', borderRadius: '7px', padding: '7px 13px', fontSize: '12.5px', fontFamily: mono, cursor: 'pointer' }}>seed</button>
        <div style={{ display: 'flex', border: '1px solid var(--line2)', borderRadius: '7px', overflow: 'hidden' }}>
          <button onClick={() => setDepth(1)} style={depthBtn(1)}>depth 1</button>
          <button onClick={() => setDepth(2)} style={{ ...depthBtn(2), borderLeft: '1px solid var(--line2)' }}>depth 2</button>
        </div>
        <button className="gbtn" onClick={relayout} style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '7px', padding: '7px 13px', fontSize: '12.5px', fontFamily: mono, cursor: 'pointer' }}>re-run layout</button>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)' }}>as-of</span>
          <input type="range" min={0} max={24} value={asOfIdx} onChange={(e) => setAsOfIdx(Number(e.target.value))} style={{ width: '180px', accentColor: 'var(--acc)' }} />
          <span style={{ fontFamily: mono, fontSize: '12px', color: 'var(--acc)', minWidth: '86px' }}>{asOfLabel(asOfIdx)}</span>
        </div>
      </div>
      {notFound && <div style={{ fontFamily: mono, fontSize: '12px', color: 'var(--warn)', marginTop: '-4px', marginBottom: '8px' }}>no entity found for “{seedText}”.</div>}

      {/* body */}
      <div style={{ display: 'flex', gap: '12px', flex: 1, minHeight: 0, flexWrap: 'wrap' }}>
        {/* canvas */}
        <div style={{ flex: 2, minWidth: '380px', position: 'relative', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '12px', overflow: 'hidden', minHeight: '540px' }}>
          <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
          {loading && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--txt3)', fontFamily: mono, fontSize: '12px', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '8px', padding: '8px 12px' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', border: '2px solid var(--line2)', borderTopColor: 'var(--acc)', display: 'inline-block', animation: 'spin .8s linear infinite' }} />
                querying neighborhood…
              </div>
            </div>
          )}
          {model.nodes.length === 0 && !loading && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--txt3)', fontFamily: mono, fontSize: '12.5px', textAlign: 'center', padding: '0 24px', lineHeight: 1.6 }}>
              seed an entity above to explore its knowledge-graph neighborhood.
            </div>
          )}
          {/* legend */}
          <div style={{ position: 'absolute', left: '12px', bottom: '10px', display: 'flex', gap: '12px', flexWrap: 'wrap', pointerEvents: 'none' }}>
            {SUPERTYPES.map((t) => (
              <span key={t} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: etColor(t) }} />{t.toLowerCase()}
              </span>
            ))}
          </div>
          {/* counts */}
          <div style={{ position: 'absolute', right: '12px', top: '10px', fontFamily: mono, fontSize: '11px', color: 'var(--txt3)', pointerEvents: 'none' }}>
            {counts.nodes} nodes · {counts.edges} edges{truncated ? ' · capped' : ''}
          </div>
        </div>

        {/* inspector */}
        <aside style={{ flex: 1, minWidth: '280px', maxWidth: '380px', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '12px', padding: '14px 16px', overflowY: 'auto', maxHeight: '640px', boxSizing: 'border-box' }}>
          {!selNode && (
            <div style={{ color: 'var(--txt3)', fontSize: '13px', lineHeight: 1.6, paddingTop: '6px' }}>
              <div style={{ fontFamily: mono, fontSize: '11px', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '8px' }}>inspector</div>
              Click a node to inspect its facts. Click it again to expand its neighborhood. Drag the as-of slider to rewind the graph.
            </div>
          )}
          {selNode && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: etColor(selNode.entity_type), flexShrink: 0 }} />
                <span style={{ fontFamily: mono, fontSize: '14px', fontWeight: 500 }}>{selNode.name}</span>
              </div>
              <div style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)', margin: '6px 0 10px' }}>
                {selNode.entity_type || 'entity'} · {selVisibleDeg} edges · served {selServed}×
              </div>
              {selNode.summary && <div style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--txt2)', marginBottom: '12px' }}>{selNode.summary}</div>}
              <div style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '8px' }}>facts</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
                {selEdges.map((e) => {
                  const vis = edgeVisual(e, asOf);
                  if (vis.hidden) return null;
                  const other = (e.src === selId ? e.tgt : e.src);
                  const text = e.fact || `${nameByU.get(e.src) || e.src} ${e.name || '—'} ${nameByU.get(e.tgt) || e.tgt}`;
                  return (
                    <div key={e.uuid} style={{ border: '1px solid var(--line)', borderRadius: '8px', padding: '8px 10px', opacity: vis.superseded ? 0.55 : 1 }}>
                      <div style={{ fontSize: '12.5px', lineHeight: 1.5, color: 'var(--txt)', textDecoration: vis.superseded ? 'line-through' : 'none' }}>{text}</div>
                      <div style={{ display: 'flex', gap: '10px', marginTop: '5px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)', flexWrap: 'wrap', alignItems: 'center' }}>
                        {vis.superseded && <span style={{ color: 'var(--warn)' }}>superseded</span>}
                        <span>{validityLine(e.t_valid || undefined, e.t_invalid)}</span>
                        <button className="linkbtn" onMouseDown={(ev) => ev.preventDefault()} onClick={() => { if (other) { setSeedText(nameByU.get(other) || ''); setSelId(other); } }} style={{ border: 'none', background: 'none', padding: 0, color: 'var(--txt2)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 'inherit' }}>↔ {nameByU.get(other) || other}</button>
                        {e.provenance_episode_id != null && <button className="linkbtn" onClick={() => openEpisode(e.provenance_episode_id!)} style={{ border: 'none', background: 'none', padding: 0, color: 'var(--acc)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 'inherit', textDecoration: 'underline', textUnderlineOffset: '2px' }}>ep-{e.provenance_episode_id}</button>}
                      </div>
                    </div>
                  );
                })}
              </div>
              <button className="chipbtn" onClick={() => selId && expand(selId)} style={{ marginTop: '12px', width: '100%', border: '1px solid var(--acc)', background: 'var(--acc-bg)', color: 'var(--acc)', borderRadius: '7px', padding: '7px 0', fontSize: '12.5px', fontFamily: mono, cursor: 'pointer' }}>expand neighborhood</button>
            </>
          )}
        </aside>
      </div>

      {toast && (
        <div onClick={() => setToast(null)} style={{ position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)', zIndex: 70, background: 'var(--bg1)', border: '1px solid var(--err)', color: 'var(--err)', borderRadius: '8px', padding: '8px 14px', fontFamily: mono, fontSize: '12px', cursor: 'pointer', boxShadow: '0 8px 24px rgba(0,0,0,.35)' }}>
          ⚠ {toast}
        </div>
      )}
    </main>
  );
}
