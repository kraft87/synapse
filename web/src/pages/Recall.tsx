// Recall — the flagship recall() debugging console (spec §5). Runs POST /recall with
// debug + the global project/group filters (source is inert here, dimmed in the header),
// then renders a latency waterfall + Served / Raw / History tabs. The waterfall is a
// schematic Gantt: embed → the parallel band (all starting at embed-end) → rerank (at the
// max parallel end), colored by the app-wide leg tokens.
import { useEffect, useRef, useState } from 'react';
import {
  postRecall, fetchRecallHistory,
  type RecallResult, type RecallDebug, type RecallHistoryRow,
} from '../api';
import { useStore } from '../state';
import { LEG_COLOR, LEG_ORDER, scoreColor, relTime } from '../tokens';

type Tab = 'served' | 'raw' | 'history';

// The parallel band (spec §5) — every one starts at embed-end; rerank waits for the
// slowest of them. embed leads, rerank trails; neither is "parallel".
const PARALLEL = ['bm25', 'vector', 'kg', 'timeline', 'prefs', 'web'];

interface WRow {
  name: string; left: number; width: number; color: string; glow: string;
  ms: string; msColor: string; flag: string; skipped: boolean;
}

function waterfall(debug: RecallDebug): WRow[] {
  const total = debug.total_ms || 1;
  const legs = debug.legs_ms || {};
  const embedMs = legs.embed ?? 0;
  const maxParallel = PARALLEL.filter((n) => n in legs).reduce((m, n) => Math.max(m, legs[n]), 0);
  const rerankStart = embedMs + maxParallel;
  return LEG_ORDER.map((name) => {
    const present = name in legs;
    const ms = present ? legs[name] : 0;
    const start = name === 'embed' ? 0 : name === 'rerank' ? rerankStart : embedMs;
    const outlier = present && total > 0 && ms / total > 0.4;
    return {
      name,
      left: (start / total) * 100,
      width: present ? (ms / total) * 100 : 0,
      color: present ? LEG_COLOR[name] : 'var(--line2)',
      glow: outlier ? '0 0 8px ' + LEG_COLOR[name] : 'none',
      ms: present ? String(ms) : '—',
      msColor: outlier ? 'var(--err)' : 'var(--txt2)',
      flag: outlier ? ' ▲' : present ? '' : ' (skipped)',
      skipped: !present,
    };
  });
}

// ---- Served bucket item shaping ----
interface Item { key: string; score?: number; meta?: string; head: string; rest: string; }
const s = (v: unknown): string => (v == null ? '' : String(v));

function bucketItems(name: string, r: RecallResult): Item[] {
  switch (name) {
    case 'episodes':
      return (r.episodes || []).map((e, i) => ({
        key: 'e' + i, score: e.score, meta: e.date,
        head: [e.id, e.project].filter(Boolean).join(' · ') + (e.role ? ' · ' + e.role : '') + ' —',
        rest: e.content + (e.superseded_by?.length ? '  ⟳ superseded_by: ' + e.superseded_by.join('; ') : ''),
      }));
    case 'facts':
      return (r.facts || []).map((f, i) => ({
        key: 'f' + i, score: f.score, meta: f.date,
        head: f.fact, rest: f.date ? '· as of ' + f.date : '',
      }));
    case 'entities':
      return (r.entities || []).map((e, i) => ({
        key: 'n' + i, score: e.score, head: e.name + ' —', rest: e.summary,
      }));
    case 'communities':
      return (r.communities || []).map((c, i) => ({
        key: 'c' + i, score: c.score, head: s(c.name) + ' —', rest: s(c.summary),
      }));
    case 'timeline':
      return (r.timeline || []).map((t, i) => ({
        key: 't' + i, score: t.score, meta: t.salience != null ? 'sal ' + t.salience : t.type,
        head: t.date + ' —', rest: t.fact,
      }));
    case 'preferences':
      return (r.preferences || []).map((p, i) => ({
        key: 'p' + i, score: p.score, meta: p.polarity,
        head: s(p.polarity) + ' —',
        rest: p.pref + (p.since ? '  (since ' + p.since + (p.asserted ? ', ' + p.asserted + '×' : '') + ')' : ''),
      }));
    case 'web':
      return (r.web || []).map((w, i) => ({
        key: 'w' + i, score: w.score, meta: w.date,
        head: (w.title || w.url || 'web') + ' —', rest: w.context || w.excerpt || '',
      }));
    case 'history':
      return (r.history || []).map((h, i) => ({
        key: 'h' + i, head: 'now —',
        rest: h.now + (h.previously ? '   (was: ' + h.previously + ')' : ''),
      }));
    default:
      return [];
  }
}
// Canonical order (spec §2). history only shows when the payload carries it.
const BUCKETS = ['episodes', 'facts', 'entities', 'communities', 'timeline', 'preferences', 'web'];

const mono = 'var(--font-data)';

export function Recall() {
  const store = useStore();
  const [query, setQuery] = useState('');
  const [ran, setRan] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecallResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('served');
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<RecallHistoryRow[] | null>(null);
  const [histErr, setHistErr] = useState(false);
  const reqId = useRef(0);

  const run = (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setRan(true);
    setLoading(true);
    setError(null);
    setTab('served');
    const id = ++reqId.current;
    postRecall({
      query: trimmed,
      project: store.project === 'all' ? undefined : store.project,
      group_id: store.group === 'all' ? undefined : store.group,
    })
      .then((res) => { if (id === reqId.current) { setResult(res); setLoading(false); } })
      .catch((e) => { if (id === reqId.current) { setError(String(e?.message || e)); setLoading(false); } });
  };

  // History tab: load once on first open.
  useEffect(() => {
    if (tab !== 'history' || history !== null) return;
    setHistErr(false);
    fetchRecallHistory(50).then((h) => setHistory(h.items)).catch(() => setHistErr(true));
  }, [tab, history]);

  const copyRaw = () => {
    if (!result) return;
    navigator.clipboard?.writeText(JSON.stringify(result, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  };

  const rows = result?.debug ? waterfall(result.debug) : null;

  return (
    <main style={{ flex: 1, maxWidth: '980px', width: '100%', margin: '0 auto', padding: '20px 16px 80px', boxSizing: 'border-box' }}>
      {/* query bar */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <input className="field" value={query} spellCheck={false}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') run(query); }}
          placeholder="what would the agent ask? e.g. postgres connection pooling decisions"
          style={{ flex: 1, background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '8px', padding: '10px 14px', fontFamily: mono, fontSize: '13.5px', color: 'var(--txt)', outline: 'none' }} />
        <button onClick={() => run(query)}
          style={{ border: 'none', background: 'var(--acc)', color: '#0d1116', fontWeight: 600, borderRadius: '8px', padding: '0 20px', cursor: 'pointer', fontSize: '13.5px' }}>recall()</button>
      </div>

      {ran && (
        <>
          {error && (
            <div style={{ marginTop: '14px', border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 14px', color: 'var(--err)', fontFamily: mono, fontSize: '12.5px' }}>
              recall failed: {error}{result ? ' — showing the last successful result below.' : ''}
            </div>
          )}

          {/* latency waterfall */}
          <section style={{ marginTop: '18px', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px', padding: '14px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '12px' }}>
              <div style={{ fontSize: '12px', fontFamily: mono, color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em' }}>latency waterfall</div>
              <div style={{ fontFamily: mono, fontSize: '13px' }}>
                <span style={{ color: 'var(--txt2)' }}>total </span>
                <span style={{ color: 'var(--txt)', fontWeight: 500 }}>{loading ? '…' : result?.debug ? result.debug.total_ms + ' ms' : '— ms'}</span>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {loading || !rows
                ? LEG_ORDER.map((name, i) => (
                    <div key={name} style={{ display: 'grid', gridTemplateColumns: '76px 1fr 88px', gap: '10px', alignItems: 'center' }}>
                      <div style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt2)', textAlign: 'right' }}>{name}</div>
                      <div style={{ position: 'relative', height: '14px', background: 'var(--bg0)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div className="legloading" style={{ position: 'absolute', inset: 0, background: LEG_COLOR[name], borderRadius: '3px', animationDelay: i * 70 + 'ms' }} />
                      </div>
                      <div style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)' }}>· · ·</div>
                    </div>
                  ))
                : rows.map((w) => (
                    <div key={w.name} style={{ display: 'grid', gridTemplateColumns: '76px 1fr 88px', gap: '10px', alignItems: 'center' }}>
                      <div style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt2)', textAlign: 'right' }}>{w.name}</div>
                      <div style={{ position: 'relative', height: '14px', background: 'var(--bg0)', borderRadius: '3px' }}>
                        <div style={{ position: 'absolute', top: 0, bottom: 0, left: w.left + '%', width: w.width + '%', minWidth: '2px', background: w.color, borderRadius: '3px', boxShadow: w.glow }} />
                      </div>
                      <div style={{ fontFamily: mono, fontSize: '11.5px', color: w.skipped ? 'var(--txt3)' : w.msColor }}>{w.ms}{w.skipped ? '' : ' ms'}{w.flag}</div>
                    </div>
                  ))}
            </div>
          </section>

          {/* tabs */}
          <div style={{ display: 'flex', gap: '2px', marginTop: '18px', borderBottom: '1px solid var(--line)' }}>
            {(['served', 'raw', 'history'] as Tab[]).map((k) => (
              <button key={k} className="recall-tab" onClick={() => setTab(k)}
                style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '8px 14px', fontSize: '13px', fontWeight: 500, color: tab === k ? 'var(--txt)' : 'var(--txt2)', borderBottom: '2px solid ' + (tab === k ? 'var(--acc)' : 'transparent'), marginBottom: '-1px', textTransform: 'capitalize' }}>{k}</button>
            ))}
          </div>

          {/* served */}
          {tab === 'served' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '14px' }}>
              {loading && !result && <div style={{ color: 'var(--txt3)', fontFamily: mono, fontSize: '12.5px' }}>running recall…</div>}
              {result && [...BUCKETS, ...(result.history?.length ? ['history'] : [])].map((name) => {
                const items = bucketItems(name, result);
                return (
                  <section key={name}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '6px' }}>
                      <span style={{ fontFamily: mono, fontSize: '12px', color: 'var(--acc)' }}>{name}</span>
                      <span style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)' }}>{items.length} served</span>
                    </div>
                    {items.length > 0 && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        {items.map((it) => (
                          <div key={it.key} style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: '10px', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '8px', padding: '9px 12px', alignItems: 'start' }}>
                            <div style={{ fontFamily: mono, fontSize: '11.5px', color: it.score != null ? scoreColor(it.score) : 'var(--txt3)', paddingTop: '2px' }}>
                              {it.score != null ? it.score.toFixed(2) : it.meta || '·'}
                            </div>
                            <div style={{ fontSize: '13px', lineHeight: 1.55, color: 'var(--txt2)' }}>
                              <span style={{ color: 'var(--txt)' }}>{it.head}</span> {it.rest}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}

          {/* raw */}
          {tab === 'raw' && (
            <div style={{ position: 'relative', marginTop: '14px' }}>
              <button className="copybtn" onClick={copyRaw}
                style={{ position: 'absolute', top: '10px', right: '10px', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '5px', padding: '4px 10px', fontSize: '11.5px', fontFamily: mono, cursor: 'pointer' }}>{copied ? 'copied ✓' : 'copy'}</button>
              <pre style={{ margin: 0, background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px', padding: '14px 16px', fontFamily: mono, fontSize: '12px', lineHeight: 1.6, color: 'var(--txt2)', overflowX: 'auto', maxHeight: '520px', overflowY: 'auto' }}>
                {result ? JSON.stringify(result, null, 2) : loading ? 'running recall…' : ''}
              </pre>
            </div>
          )}

          {/* history */}
          {tab === 'history' && (
            <div className="scrollx" style={{ marginTop: '14px', background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px', overflow: 'hidden' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 76px 76px 70px 90px', gap: '10px', padding: '8px 14px', borderBottom: '1px solid var(--line)', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.06em' }}>
                <div>query</div><div style={{ textAlign: 'right' }}>total</div><div style={{ textAlign: 'right' }}>tokens</div><div style={{ textAlign: 'right' }}>top</div><div style={{ textAlign: 'right' }}>when</div>
              </div>
              {histErr && <div style={{ padding: '12px 14px', color: 'var(--err)', fontFamily: mono, fontSize: '12px' }}>history request failed.</div>}
              {!histErr && history === null && <div style={{ padding: '12px 14px', color: 'var(--txt3)', fontFamily: mono, fontSize: '12px' }}>loading…</div>}
              {!histErr && history?.length === 0 && <div style={{ padding: '12px 14px', color: 'var(--txt3)', fontFamily: mono, fontSize: '12px' }}>no recalls recorded yet.</div>}
              {history?.map((h) => (
                <button key={h.id} className="hist-row" onClick={() => run(h.query)}
                  style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 76px 76px 70px 90px', gap: '10px', padding: '9px 14px', border: 'none', borderBottom: '1px solid var(--line)', background: 'none', width: '100%', textAlign: 'left', cursor: 'pointer', alignItems: 'center' }}>
                  <div style={{ fontFamily: mono, fontSize: '12px', color: 'var(--txt)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.query}</div>
                  <div style={{ fontFamily: mono, fontSize: '12px', color: (h.ms_total ?? 0) > 380 ? 'var(--err)' : 'var(--txt2)', textAlign: 'right' }}>{h.ms_total != null ? Math.round(h.ms_total) + ' ms' : '—'}</div>
                  <div style={{ fontFamily: mono, fontSize: '12px', color: 'var(--txt2)', textAlign: 'right' }}>{h.est_tokens ?? '—'}</div>
                  <div style={{ fontFamily: mono, fontSize: '12px', color: 'var(--txt2)', textAlign: 'right' }}>{h.rerank_top_score != null ? h.rerank_top_score.toFixed(2) : '—'}</div>
                  <div style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)', textAlign: 'right' }}>{relTime(h.created_at)}</div>
                </button>
              ))}
            </div>
          )}
        </>
      )}

      {!ran && (
        <div style={{ marginTop: '18px', border: '1px dashed var(--line2)', borderRadius: '10px', padding: '40px 24px', textAlign: 'center', fontFamily: mono, fontSize: '12.5px', color: 'var(--txt3)', lineHeight: 1.7 }}>
          run recall() to see the latency waterfall and exactly what an agent would be served.<br />
          project + group filters apply; source is inert here.
        </div>
      )}
    </main>
  );
}
