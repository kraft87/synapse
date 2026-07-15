// Metrics — the ops page (README §5, screenshot 06). Three tabs over three aggregate
// endpoints: Recall (latency percentiles + per-leg stack + slowest + rerank histogram),
// Ingestion (queue snapshot + throughput + last dream run), Corpus (per-table counts +
// composition). Time-series charts use uPlot (the committed lib); histograms / sparklines /
// proportion bars are CSS bars per the prototype. Each tab loads from one endpoint and
// renders its own loading skeleton / "not enough data" empty / error chip; within a tab,
// each panel shows its own empty state when its slice is empty.
import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import uPlot from 'uplot';
import {
  fetchMetricsRecall, fetchMetricsIngestion, fetchMetricsCorpus,
  type MetricsRecall, type MetricsIngestion, type MetricsCorpus, type DreamRun,
} from '../api';
import { useStore } from '../state';
import { LEG_COLOR, LEG_ORDER, cssValue, srcColor, relTime } from '../tokens';
import { UPlotChart } from '../components/UPlotChart';

const mono = 'var(--font-data)';
type Tab = 'recall' | 'ingestion' | 'corpus';

// ---- tiny async hook: loading / data / error, runs fn once (and on dep change) ----
function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; loading: boolean; error: string | null } {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({
    data: null, loading: true, error: null,
  });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn()
      .then((d) => { if (live) setState({ data: d, loading: false, error: null }); })
      .catch((e) => { if (live) setState({ data: null, loading: false, error: String(e?.message || e) }); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

// ---- shared shells ----
const panel = (extra?: CSSProperties): CSSProperties => ({
  background: 'var(--bg1)', border: '1px solid var(--line)', borderRadius: '10px',
  padding: '14px 16px', ...extra,
});
const panelHead: CSSProperties = {
  fontSize: '12px', fontFamily: mono, color: 'var(--txt3)', textTransform: 'uppercase',
  letterSpacing: '.08em', marginBottom: '10px',
};

function ErrorChip({ msg }: { msg: string }) {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', color: 'var(--err)', borderRadius: '6px', padding: '4px 10px', fontFamily: mono, fontSize: '11.5px' }}>
      ⚠ {msg}
    </div>
  );
}
function Empty({ children }: { children: ReactNode }) {
  return <div style={{ color: 'var(--txt3)', fontFamily: mono, fontSize: '12px', padding: '18px 0', textAlign: 'center' }}>{children}</div>;
}
function ChartSkel({ h = 150 }: { h?: number }) {
  return <div className="chart-skel" style={{ height: h + 'px', width: '100%' }} />;
}
function StatCard({ label, value, sub, valueColor }: { label: string; value: string; sub?: string; subColor?: string; valueColor?: string }) {
  return (
    <div style={panel({ padding: '12px 14px' })}>
      <div style={{ fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
      <div style={{ fontFamily: mono, fontSize: '22px', fontWeight: 500, marginTop: '4px', color: valueColor || 'var(--txt)' }}>{value}</div>
      <div style={{ fontFamily: mono, fontSize: '11px', color: 'var(--txt3)', marginTop: '2px', minHeight: '14px' }}>{sub || ''}</div>
    </div>
  );
}
function StatGridSkel() {
  return (
    <div className="statgrid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '10px', marginBottom: '14px' }}>
      {[0, 1, 2, 3].map((i) => <div key={i} className="chart-skel" style={{ height: '78px' }} />)}
    </div>
  );
}

const bars085 = uPlot.paths.bars ? uPlot.paths.bars({ size: [0.85, 40], align: 0 }) : undefined;
const barsThin = uPlot.paths.bars ? uPlot.paths.bars({ size: [0.7, 22], align: 0 }) : undefined;
const yFrom0 = (_u: uPlot, _min: number, max: number): [number, number] => [0, max > 0 ? max * 1.05 : 1];
const baseAxes = () => {
  const ax = cssValue('--txt3');
  const grid = cssValue('--line');
  return [
    { stroke: ax, grid: { show: false }, ticks: { show: false }, font: '10px monospace', size: 26, space: 60 },
    { stroke: ax, grid: { stroke: grid, width: 1 }, ticks: { show: false }, font: '10px monospace', size: 40 },
  ] as uPlot.Axis[];
};

// =====================================================================================
// Recall tab
// =====================================================================================
function RecallTab() {
  const { theme } = useStore();
  const { data, loading, error } = useAsync<MetricsRecall>(() => fetchMetricsRecall('7d'), []);

  const cards = useMemo(() => {
    const s = data?.series || [];
    const totalCalls = s.reduce((a, p) => a + p.calls, 0);
    const wavg = (sel: (p: MetricsRecall['series'][number]) => number | null) => {
      let num = 0, den = 0;
      for (const p of s) { const v = sel(p); if (v != null) { num += v * p.calls; den += p.calls; } }
      return den ? num / den : null;
    };
    return {
      p50: wavg((p) => p.p50), p95: wavg((p) => p.p95), calls: totalCalls, tokens: wavg((p) => p.tokens_p50),
    };
  }, [data]);

  // stacked per-leg latency chart: cumulative sums, drawn largest-first so bars stack.
  const chart = useMemo(() => {
    const s = data?.series || [];
    if (s.length === 0) return null;
    const present = new Set<string>();
    for (const p of s) for (const k of Object.keys(p.legs_p50)) present.add(k);
    const stackLegs = LEG_ORDER.filter((l) => present.has(l)); // bottom→top
    if (stackLegs.length === 0) return null;
    const x = s.map((p) => Date.parse(p.t) / 1000);
    const running = new Array(s.length).fill(0);
    const cum: Record<string, number[]> = {};
    for (const leg of stackLegs) {
      for (let j = 0; j < s.length; j++) running[j] += s[j].legs_p50[leg] ?? 0;
      cum[leg] = running.slice();
    }
    const drawLegs = [...stackLegs].reverse(); // top of stack drawn first (behind)
    const opts: Omit<uPlot.Options, 'width' | 'height'> = {
      scales: { x: { time: true }, y: { range: yFrom0 } },
      legend: { show: false }, cursor: { show: false },
      axes: baseAxes(),
      series: [{}, ...drawLegs.map((leg) => ({
        label: leg, paths: bars085, points: { show: false },
        fill: cssValue(LEG_COLOR[leg]), stroke: cssValue(LEG_COLOR[leg]), width: 0,
      }))],
    };
    const chartData = [x, ...drawLegs.map((leg) => cum[leg])] as unknown as uPlot.AlignedData;
    return { opts, chartData, legLegend: stackLegs };
    // theme is a dep so colors re-resolve on toggle
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, theme]);

  if (error) return <ErrorChip msg={`recall metrics failed: ${error}`} />;
  if (loading) return (<><StatGridSkel /><div style={panel({ marginBottom: '14px' })}><ChartSkel /></div></>);

  const fmt = (v: number | null, unit = '') => (v == null ? '—' : Math.round(v) + unit);
  return (
    <>
      <div className="statgrid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '10px', marginBottom: '14px' }}>
        <StatCard label="p50 total" value={fmt(cards.p50, ' ms')} sub="median over window" />
        <StatCard label="p95 total" value={fmt(cards.p95, ' ms')} sub="tail latency" />
        <StatCard label="calls" value={cards.calls.toLocaleString()} sub="in window (7d)" />
        <StatCard label="tokens / call" value={cards.tokens == null ? '—' : Math.round(cards.tokens).toLocaleString()} sub="median served" />
      </div>

      <section style={panel({ marginBottom: '14px' })}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '10px', gap: '12px', flexWrap: 'wrap' }}>
          <div style={panelHead}>recall latency · per-leg stack, hourly</div>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {(chart?.legLegend || []).map((leg) => (
              <span key={leg} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}>
                <span style={{ width: 8, height: 8, borderRadius: '2px', background: LEG_COLOR[leg] }} />{leg}
              </span>
            ))}
          </div>
        </div>
        {chart ? <UPlotChart opts={chart.opts} data={chart.chartData} height={150} /> : <Empty>not enough data in window</Empty>}
      </section>

      <div className="duo" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        <section style={panel()}>
          <div style={panelHead}>top slowest queries</div>
          {data && data.slowest.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
              {data.slowest.map((sq, i) => (
                <div key={i} style={{ display: 'flex', gap: '10px', alignItems: 'baseline' }}>
                  <span style={{ fontFamily: mono, fontSize: '12px', color: 'var(--err)', minWidth: '64px', textAlign: 'right' }}>{sq.ms_total == null ? '—' : Math.round(sq.ms_total) + ' ms'}</span>
                  <span style={{ fontFamily: mono, fontSize: '12px', color: 'var(--txt2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sq.query || '(no query text)'}</span>
                </div>
              ))}
            </div>
          ) : <Empty>not enough data in window</Empty>}
        </section>
        <section style={panel()}>
          <div style={panelHead}>rerank top-score distribution</div>
          <ScoreHist hist={data?.score_hist || []} />
        </section>
      </div>
    </>
  );
}

function ScoreHist({ hist }: { hist: MetricsRecall['score_hist'] }) {
  const max = Math.max(1, ...hist.map((b) => b.n));
  const total = hist.reduce((a, b) => a + b.n, 0);
  if (total === 0) return <Empty>not enough data in window</Empty>;
  const color = (lo: number) => (lo < 0.4 ? 'var(--err)' : lo < 0.6 ? 'var(--warn)' : 'var(--acc)');
  return (
    <>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '110px' }}>
        {hist.map((b, i) => (
          <div key={i} title={`${b.lo.toFixed(1)}–${b.hi.toFixed(1)}: ${b.n}`}
            style={{ flex: 1, height: Math.max(2, (b.n / max) * 100) + '%', background: color(b.lo), borderRadius: '2px 2px 0 0', opacity: b.n ? 1 : 0.25 }} />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}>
        <span>0.0</span><span>0.5</span><span>1.0</span>
      </div>
    </>
  );
}

// =====================================================================================
// Ingestion tab
// =====================================================================================
function IngestionTab() {
  const { theme } = useStore();
  const { data, loading, error } = useAsync<MetricsIngestion>(() => fetchMetricsIngestion('48h'), []);

  const chart = useMemo(() => {
    if (!data) return null;
    const enq = data.throughput.enqueued_per_hour;
    const comp = data.throughput.completed_per_hour;
    if (enq.length === 0 && comp.length === 0) return null;
    const tset = new Set<number>();
    for (const r of enq) tset.add(Date.parse(r.t) / 1000);
    for (const r of comp) tset.add(Date.parse(r.t) / 1000);
    const x = [...tset].sort((a, b) => a - b);
    const em = new Map(enq.map((r) => [Date.parse(r.t) / 1000, r.n]));
    const cm = new Map(comp.map((r) => [Date.parse(r.t) / 1000, r.n]));
    const opts: Omit<uPlot.Options, 'width' | 'height'> = {
      scales: { x: { time: true }, y: { range: yFrom0 } },
      legend: { show: false }, cursor: { show: false }, axes: baseAxes(),
      series: [
        {},
        { label: 'completed', paths: barsThin, points: { show: false }, fill: cssValue('--acc'), stroke: cssValue('--acc'), width: 0 },
        { label: 'enqueued', stroke: cssValue('--txt2'), width: 1.5, points: { show: false } },
      ],
    };
    const chartData = [x, x.map((t) => cm.get(t) ?? null), x.map((t) => em.get(t) ?? null)] as unknown as uPlot.AlignedData;
    return { opts, chartData };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, theme]);

  if (error) return <ErrorChip msg={`ingestion metrics failed: ${error}`} />;
  if (loading) return (<><StatGridSkel /><div style={panel({ marginBottom: '14px' })}><ChartSkel /></div></>);
  if (!data) return <Empty>not enough data in window</Empty>;

  const compTotal = data.throughput.completed_per_hour.reduce((a, r) => a + r.n, 0);
  const peakThru = Math.max(0, ...data.throughput.completed_per_hour.map((r) => r.n), ...data.throughput.enqueued_per_hour.map((r) => r.n));
  const ld = data.last_dream;
  return (
    <>
      <div className="statgrid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '10px', marginBottom: '14px' }}>
        <StatCard label="queue depth" value={String(data.queue_depth)} sub={`${data.queue.processing} processing`} valueColor={data.queue_depth > 90 ? 'var(--warn)' : 'var(--txt)'} />
        <StatCard label="throughput" value={compTotal.toLocaleString()} sub="completed / 48h" />
        <StatCard label="failures" value={String(data.queue.failed)} sub="status=failed" valueColor={data.queue.failed > 0 ? 'var(--err)' : 'var(--txt)'} />
        <StatCard label="last run" value={ld?.duration_s != null ? Math.round(ld.duration_s) + ' s' : '—'} sub={ld ? relTime(ld.started_at) : 'no runs yet'} />
      </div>

      <section style={panel({ marginBottom: '14px' })}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '10px', gap: '12px', flexWrap: 'wrap' }}>
          <div style={panelHead}>queue throughput · 48h</div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}><span style={{ width: 8, height: 8, borderRadius: '2px', background: 'var(--acc)' }} />completed</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}><span style={{ width: 10, height: 2, background: 'var(--txt2)' }} />enqueued</span>
            {peakThru > 90 && <span style={{ fontFamily: mono, fontSize: '10.5px', color: 'var(--warn)' }}>peak {peakThru}/h</span>}
          </div>
        </div>
        {chart ? <UPlotChart opts={chart.opts} data={chart.chartData} height={140} /> : <Empty>not enough data in window</Empty>}
        <div style={{ fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)', marginTop: '6px' }}>
          throughput only — historical queue depth is not reconstructable from the queue table.
        </div>
      </section>

      <div className="duo" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        <section style={panel()}>
          <div style={panelHead}>recent failures</div>
          {data.failures.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
              {data.failures.slice(0, 8).map((f) => (
                <div key={f.id} style={{ display: 'flex', gap: '10px', alignItems: 'baseline' }}>
                  <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)', minWidth: '54px' }}>#{f.episode_id ?? f.id}</span>
                  <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--err)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.error || 'unknown error'}</span>
                </div>
              ))}
            </div>
          ) : <Empty>no failures in window</Empty>}
        </section>
        <DreamRunPanel run={ld} />
      </div>
    </>
  );
}

function DreamRunPanel({ run }: { run: DreamRun | null }) {
  const lines: string[] = [];
  if (run) {
    const stg = Object.entries(run.stages).map(([k, v]) => `${k}:${v?.ok === false ? 'ERR' : 'ok'}`).join('  ');
    lines.push(`started   ${run.started_at?.replace('T', ' ').slice(0, 19)} UTC`);
    lines.push(`duration  ${run.duration_s != null ? run.duration_s + ' s' : '(in flight)'}`);
    lines.push(`ok        ${run.ok == null ? '—' : run.ok}`);
    if (stg) lines.push(`stages    ${stg}`);
    for (const [k, v] of Object.entries(run.counts)) lines.push(`${(k + '            ').slice(0, 24)}${v}`);
    if (run.errors.length) lines.push(`errors    ${run.errors.join(' | ')}`);
  }
  return (
    <section style={panel()}>
      <div style={panelHead}>last nightly-dream run</div>
      {run ? (
        <>
          <pre style={{ margin: 0, fontFamily: mono, fontSize: '12px', lineHeight: 1.75, color: 'var(--txt2)', whiteSpace: 'pre-wrap' }}>{lines.join('\n')}</pre>
          {run.samples?.proposals?.length ? (
            <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {run.samples.proposals.map((p) => (
                <span key={p.id} style={{ fontFamily: mono, fontSize: '10.5px', padding: '2px 7px', borderRadius: '4px', background: 'var(--bg2)', color: 'var(--txt2)', border: '1px solid var(--line)' }}>{p.kind}: {p.name}</span>
              ))}
            </div>
          ) : null}
        </>
      ) : <Empty>no runs recorded yet</Empty>}
    </section>
  );
}

// =====================================================================================
// Corpus tab
// =====================================================================================
function CorpusTab() {
  const { data, loading, error } = useAsync<MetricsCorpus>(() => fetchMetricsCorpus(), []);
  if (error) return <ErrorChip msg={`corpus metrics failed: ${error}`} />;
  if (loading) return (<><div style={panel({ marginBottom: '14px' })}><ChartSkel h={220} /></div><div className="duo" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}><ChartSkel h={160} /><ChartSkel h={160} /></div></>);
  if (!data || data.tables.length === 0) return <Empty>not enough data in window</Empty>;

  const projTotal = Math.max(1, data.by_project.reduce((a, r) => a + r.n, 0));
  const srcTotal = Math.max(1, data.by_source.reduce((a, r) => a + r.n, 0));
  const grid = '160px 96px 1fr 96px';
  return (
    <>
      <section className="scrollx" style={panel({ padding: 0, overflow: 'hidden', marginBottom: '14px' })}>
        <div style={{ display: 'grid', gridTemplateColumns: grid, gap: '12px', padding: '9px 16px', borderBottom: '1px solid var(--line)', fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.06em' }}>
          <div>table</div><div style={{ textAlign: 'right' }}>rows</div><div>growth · 30d</div><div style={{ textAlign: 'right' }}>+30d</div>
        </div>
        {data.tables.map((t) => {
          const max = Math.max(1, ...t.spark_30d);
          return (
            <div key={t.name} className="corpus-row" style={{ display: 'grid', gridTemplateColumns: grid, gap: '12px', padding: '10px 16px', borderBottom: '1px solid var(--line)', alignItems: 'center' }}>
              <div style={{ fontFamily: mono, fontSize: '12.5px', color: 'var(--txt)' }}>{t.name}</div>
              <div style={{ fontFamily: mono, fontSize: '12.5px', color: 'var(--txt)', textAlign: 'right' }} title={t.rows_estimated ? 'estimated (pg_class.reltuples)' : 'exact count'}>
                {t.rows.toLocaleString()}{t.rows_estimated ? <span style={{ color: 'var(--txt3)' }}> ~</span> : ''}
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '26px' }}>
                {t.spark_30d.length ? t.spark_30d.map((v, i) => (
                  <div key={i} title={String(v)} style={{ flex: 1, height: Math.max(2, (v / max) * 100) + '%', background: 'var(--acc)', opacity: 0.55, borderRadius: '1px' }} />
                )) : <span style={{ fontFamily: mono, fontSize: '10.5px', color: 'var(--txt3)' }}>no time column</span>}
              </div>
              <div style={{ fontFamily: mono, fontSize: '12px', color: t.delta_30d > 0 ? 'var(--ok)' : 'var(--txt3)', textAlign: 'right' }}>+{t.delta_30d.toLocaleString()}</div>
            </div>
          );
        })}
      </section>

      <div className="duo" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        <section style={panel()}>
          <div style={{ ...panelHead, marginBottom: '12px' }}>episodes by project</div>
          <Proportions rows={data.by_project} total={projTotal} />
        </section>
        <section style={panel()}>
          <div style={{ ...panelHead, marginBottom: '12px' }}>episodes by source</div>
          <Proportions rows={data.by_source} total={srcTotal} colored />
        </section>
      </div>
    </>
  );
}

function Proportions({ rows, total, colored }: { rows: { name: string; n: number }[]; total: number; colored?: boolean }) {
  if (rows.length === 0) return <Empty>not enough data in window</Empty>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {rows.map((r) => {
        const c = colored ? srcColor(r.name) : 'var(--acc)';
        return (
          <div key={r.name} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 60px', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontFamily: mono, fontSize: '11.5px', color: colored ? c : 'var(--txt2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
            <div style={{ height: '10px', background: 'var(--bg0)', borderRadius: '3px' }}>
              <div style={{ height: '100%', width: Math.max(1, (r.n / total) * 100) + '%', background: c, opacity: 0.7, borderRadius: '3px' }} />
            </div>
            <span style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)', textAlign: 'right' }}>{r.n.toLocaleString()}</span>
          </div>
        );
      })}
    </div>
  );
}

// =====================================================================================
// Page shell (tabs + window label)
// =====================================================================================
export function Metrics() {
  const [tab, setTab] = useState<Tab>('recall');
  const windowLabel = tab === 'ingestion' ? 'window: 48h' : tab === 'corpus' ? 'window: 30d' : 'window: 7d';
  const tabs: Tab[] = ['recall', 'ingestion', 'corpus'];
  return (
    <main style={{ flex: 1, maxWidth: '980px', width: '100%', margin: '0 auto', padding: '20px 16px 80px', boxSizing: 'border-box' }}>
      <div style={{ display: 'flex', gap: '2px', borderBottom: '1px solid var(--line)', marginBottom: '16px', alignItems: 'center' }}>
        {tabs.map((k) => (
          <button key={k} className="metrics-tab" onClick={() => setTab(k)}
            style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '8px 14px', fontSize: '13px', fontWeight: 500, color: tab === k ? 'var(--txt)' : 'var(--txt2)', borderBottom: '2px solid ' + (tab === k ? 'var(--acc)' : 'transparent'), marginBottom: '-1px', textTransform: 'capitalize' }}>{k}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ fontFamily: mono, fontSize: '11.5px', color: 'var(--txt3)', alignSelf: 'center' }}>{windowLabel}</div>
      </div>
      {tab === 'recall' ? <RecallTab /> : tab === 'ingestion' ? <IngestionTab /> : <CorpusTab />}
    </main>
  );
}
