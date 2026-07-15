// Review — the self-improvement console (phase 2b). Tabs: Proposals / Behavior files /
// Dream report / Flags. Only Proposals is real this phase; the others render the
// phase-stub panel. Proposals is a master-detail grid over BOTH lanes (skills +
// config-edit): left = proposal cards, right = evidence + provenance + payload
// (markdown draft or unified diff) + audit trail + a decision bar on 'proposed' items.
import type React from 'react';
import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
  fetchProposal, fetchProposals, postProposalDecision,
  fetchDreamReport, fetchBehaviorFiles, fetchBehaviorFile, fetchFlags, postFlag,
  type EvidenceItem, type ProposalDetail, type ProposalSummary,
  type DreamRun, type BehaviorFileEntry, type BehaviorFile, type FlagRow,
} from '../api';
import { useStore } from '../state';
import { openEpisode } from '../hash';
import { relTime } from '../tokens';

const TABS = [
  { key: 'proposals', label: 'Proposals' },
  { key: 'behavior', label: 'Behavior files' },
  { key: 'dream', label: 'Dream report' },
  { key: 'flags', label: 'Flags' },
] as const;
type ReviewTab = (typeof TABS)[number]['key'];

// kind badge colors (README §6): skill = tech violet, config-edit = org amber.
const kindColor = (k: string): string => (k === 'skill' ? 'var(--et-tech)' : 'var(--et-org)');

const mono: CSSProperties = { fontFamily: 'var(--font-data)' };
const monoHead: CSSProperties = { ...mono, fontSize: '11px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em' };

// ---------------------------------------------------------------------------
// Tiny markdown renderer (headings, bold/code spans, lists, fenced code) — no deps.
// ---------------------------------------------------------------------------

const codeSpan: CSSProperties = { ...mono, fontSize: '12px', background: 'var(--bg2)', borderRadius: '3px', padding: '1px 4px' };
const codeBlock: CSSProperties = { ...mono, fontSize: '12px', background: 'var(--bg0)', border: '1px solid var(--line)', borderRadius: '7px', padding: '10px 12px', overflowX: 'auto', margin: '8px 0', whiteSpace: 'pre', color: 'var(--txt)' };
const headStyle = (lvl: number): CSSProperties => ({ fontFamily: 'var(--font-prose)', fontWeight: 600, fontSize: lvl === 1 ? '16px' : lvl === 2 ? '14.5px' : '13.5px', color: 'var(--txt)', margin: '12px 0 6px' });

function inline(text: string, kp: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith('`')) nodes.push(<code key={kp + i} style={codeSpan}>{tok.slice(1, -1)}</code>);
    else nodes.push(<strong key={kp + i}>{tok.slice(2, -2)}</strong>);
    last = m.index + tok.length;
    i++;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function Markdown({ src }: { src: string }) {
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];
  let list: string[] = [];
  let key = 0;
  const flush = () => {
    if (!list.length) return;
    const items = list.slice();
    blocks.push(
      <ul key={'ul' + key++} style={{ margin: '6px 0', paddingLeft: '18px' }}>
        {items.map((t, j) => <li key={j} style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--txt)', margin: '2px 0' }}>{inline(t, 'li' + key + '-' + j + '-')}</li>)}
      </ul>,
    );
    list = [];
  };
  let idx = 0;
  while (idx < lines.length) {
    const ln = lines[idx];
    if (ln.startsWith('```')) {
      flush();
      const buf: string[] = [];
      idx++;
      while (idx < lines.length && !lines[idx].startsWith('```')) { buf.push(lines[idx]); idx++; }
      idx++; // closing fence
      blocks.push(<pre key={'pre' + key++} style={codeBlock}>{buf.join('\n')}</pre>);
      continue;
    }
    const h = /^(#{1,3})\s+(.*)$/.exec(ln);
    if (h) { flush(); blocks.push(<div key={'h' + key++} style={headStyle(h[1].length)}>{inline(h[2], 'h' + key + '-')}</div>); idx++; continue; }
    const li = /^[-*]\s+(.*)$/.exec(ln);
    if (li) { list.push(li[1]); idx++; continue; }
    if (!ln.trim()) { flush(); idx++; continue; }
    flush();
    blocks.push(<p key={'p' + key++} style={{ fontSize: '13px', lineHeight: 1.6, color: 'var(--txt)', margin: '6px 0' }}>{inline(ln, 'p' + key + '-')}</p>);
    idx++;
  }
  flush();
  return <div>{blocks}</div>;
}

function DiffView({ src }: { src: string }) {
  return (
    <pre style={{ ...mono, fontSize: '12px', border: '1px solid var(--line)', borderRadius: '7px', overflowX: 'auto', margin: '8px 0', background: 'var(--bg0)', padding: '8px 0' }}>
      {src.replace(/\r\n/g, '\n').split('\n').map((ln, i) => {
        const c = ln.charAt(0);
        const color = c === '+' ? 'var(--ok)' : c === '-' ? 'var(--err)' : c === '@' ? 'var(--acc)' : 'var(--txt2)';
        const bg = c === '+' ? 'rgba(91,189,141,.08)' : c === '-' ? 'rgba(224,139,122,.08)' : 'transparent';
        return <div key={i} style={{ color, background: bg, padding: '0 10px', whiteSpace: 'pre' }}>{ln || ' '}</div>;
      })}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// Evidence + provenance
// ---------------------------------------------------------------------------

const evText = (e: EvidenceItem): string => e.why || e.quote || e.phrasing || e.note || '';

function Evidence({ detail }: { detail: ProposalDetail }) {
  const ev = detail.evidence;
  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ ...monoHead, marginBottom: '6px' }}>evidence</div>
      {typeof ev === 'string' ? (
        <p style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt)', margin: 0 }}>{ev || 'no evidence recorded.'}</p>
      ) : ev.length === 0 ? (
        <p style={{ fontSize: '13px', color: 'var(--txt3)', margin: 0 }}>no structured evidence recorded.</p>
      ) : (
        <>
          <p style={{ fontSize: '13px', color: 'var(--txt2)', margin: '0 0 8px' }}>
            {ev.length} signal{ev.length === 1 ? '' : 's'} across {new Set(ev.map((e) => e.session_id).filter(Boolean)).size || ev.length} session{ev.length === 1 ? '' : 's'}.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            {ev.map((e, i) => (
              <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'baseline' }}>
                <span style={{ ...mono, fontSize: '10.5px', color: 'var(--acc)', flexShrink: 0 }}>{e.signal || e.class || 'signal'}</span>
                <span style={{ fontSize: '12.5px', color: 'var(--txt)', lineHeight: 1.5 }}>{evText(e) || <span style={{ color: 'var(--txt3)' }}>—</span>}</span>
              </div>
            ))}
          </div>
        </>
      )}
      {detail.provenance_episodes.length > 0 && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '10px' }}>
          {detail.provenance_episodes.map((id) => (
            <button key={id} className="chipbtn" onClick={() => openEpisode(id)} title="open episode"
              style={{ ...mono, fontSize: '10.5px', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--acc)', borderRadius: '4px', padding: '2px 8px', cursor: 'pointer' }}>
              ep-{id} ↗
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Proposal detail pane
// ---------------------------------------------------------------------------

function summaryText(d: ProposalDetail): string {
  const ev = typeof d.evidence === 'string'
    ? d.evidence
    : d.evidence.map((e) => `- ${e.signal ?? 'signal'}: ${evText(e)}`.trimEnd()).join('\n');
  return [
    `Proposal ${d.id} (${d.kind}) — ${d.name}`,
    `Status: ${d.status}`,
    '',
    'Evidence:',
    ev || '(none)',
    '',
    `Payload (${d.payload.type}):`,
    d.payload.content,
  ].join('\n');
}

function Detail({ id, onDecided }: { id: string; onDecided: () => void }) {
  const [detail, setDetail] = useState<ProposalDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = () => {
    setLoading(true); setError(false);
    fetchProposal(id).then((d) => setDetail(d)).catch(() => setError(true)).finally(() => setLoading(false));
  };
  useEffect(() => { setRejecting(false); setReason(''); load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [id]);

  const decide = (action: 'approve' | 'reject', note?: string) => {
    setBusy(true);
    postProposalDecision(id, action, note)
      .then(() => { setRejecting(false); setReason(''); load(); onDecided(); })
      .catch(() => setError(true))
      .finally(() => setBusy(false));
  };

  const discuss = (d: ProposalDetail) => {
    const text = summaryText(d);
    const done = () => { setCopied(true); window.setTimeout(() => setCopied(false), 1600); };
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(done).catch(done);
    else done();
  };

  if (loading && !detail) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading proposal…</div>;
  if (error && !detail) return <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', color: 'var(--err)', fontSize: '13px' }}>failed to load this proposal.</div>;
  if (!detail) return null;

  const decided = detail.status !== 'proposed';

  return (
    <div>
      {/* header: kind + name + status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginBottom: '14px' }}>
        <span style={{ ...mono, fontSize: '10.5px', border: '1px solid ' + kindColor(detail.kind), color: kindColor(detail.kind), borderRadius: '4px', padding: '2px 7px' }}>{detail.kind}</span>
        <span style={{ ...mono, fontSize: '15px', fontWeight: 500, color: 'var(--txt)' }}>{detail.name}</span>
        <span style={{ flex: 1 }} />
        <span style={{ ...mono, fontSize: '12px', color: decided ? 'var(--txt2)' : 'var(--acc)' }}>{detail.status}</span>
      </div>

      <Evidence detail={detail} />

      {/* payload */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ ...monoHead, marginBottom: '4px' }}>{detail.payload.type === 'markdown' ? 'drafted skill.md' : 'proposed diff'}</div>
        {detail.payload.content
          ? (detail.payload.type === 'markdown' ? <Markdown src={detail.payload.content} /> : <DiffView src={detail.payload.content} />)
          : <div style={{ fontSize: '13px', color: 'var(--txt3)' }}>no payload.</div>}
      </div>

      {/* audit trail (when decided) */}
      {detail.audit_log.length > 0 && (
        <div style={{ borderLeft: '2px solid var(--line2)', paddingLeft: '12px', margin: '14px 0' }}>
          <div style={{ ...monoHead, marginBottom: '6px' }}>audit</div>
          {detail.audit_log.map((a, i) => (
            <div key={i} style={{ ...mono, fontSize: '11.5px', color: 'var(--txt2)', lineHeight: 1.7 }}>
              <span style={{ color: a.action.includes('reject') ? 'var(--err)' : a.action.includes('approve') ? 'var(--ok)' : 'var(--txt3)' }}>{a.action}</span>
              {a.ts ? ' · ' + relTime(a.ts) : ''}{a.note ? ' — ' + a.note : ''}
            </div>
          ))}
        </div>
      )}

      {/* decision bar (proposed only) */}
      {!decided && (
        <div style={{ borderTop: '1px solid var(--line)', paddingTop: '14px', marginTop: '16px' }}>
          {!rejecting ? (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
              <button disabled={busy} onClick={() => decide('approve')}
                style={{ border: 'none', background: 'var(--ok)', color: '#0d1116', borderRadius: '6px', padding: '7px 16px', fontSize: '13px', fontWeight: 600, cursor: 'pointer', opacity: busy ? 0.6 : 1 }}>approve</button>
              <button className="softbtn" disabled={busy} onClick={() => setRejecting(true)}
                style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', padding: '7px 14px', fontSize: '13px', cursor: 'pointer' }}>reject…</button>
              <span style={{ flex: 1 }} />
              <button className="softbtn" onClick={() => discuss(detail)}
                style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: copied ? 'var(--ok)' : 'var(--txt2)', borderRadius: '6px', padding: '7px 14px', fontSize: '12.5px', ...mono, cursor: 'pointer' }}>{copied ? 'copied for chat ✓' : 'discuss →'}</button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ ...monoHead }}>reason (required)</div>
              <input className="field" autoFocus value={reason} spellCheck={false}
                onChange={(e) => setReason(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && reason.trim()) decide('reject', reason.trim()); if (e.key === 'Escape') setRejecting(false); }}
                placeholder="why is this proposal wrong?"
                style={{ background: 'var(--bg2)', border: '1px solid var(--line2)', borderRadius: '6px', padding: '7px 10px', fontFamily: 'var(--font-data)', fontSize: '12.5px', color: 'var(--txt)' }} />
              <div style={{ display: 'flex', gap: '8px' }}>
                <button disabled={busy || !reason.trim()} onClick={() => decide('reject', reason.trim())}
                  style={{ border: 'none', background: 'var(--err)', color: '#0d1116', borderRadius: '6px', padding: '7px 16px', fontSize: '13px', fontWeight: 600, cursor: reason.trim() ? 'pointer' : 'not-allowed', opacity: (busy || !reason.trim()) ? 0.55 : 1 }}>confirm reject</button>
                <button className="softbtn" onClick={() => { setRejecting(false); setReason(''); }}
                  style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '6px', padding: '7px 14px', fontSize: '13px', cursor: 'pointer' }}>cancel</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Proposals master-detail
// ---------------------------------------------------------------------------

function ProposalCard({ p, selected, onClick }: { p: ProposalSummary; selected: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick}
      style={{ textAlign: 'left', width: '100%', cursor: 'pointer', borderRadius: '9px', padding: '11px 13px',
        border: '1px solid ' + (selected ? 'var(--acc)' : 'var(--line)'),
        background: selected ? 'var(--acc-bg)' : 'var(--bg1)' }}>
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px' }}>
        <span style={{ ...mono, fontSize: '10px', border: '1px solid var(--line2)', color: kindColor(p.kind), borderRadius: '4px', padding: '2px 6px' }}>{p.kind}</span>
        <span style={{ ...mono, fontSize: '10.5px', color: p.status === 'proposed' ? 'var(--acc)' : p.status === 'rejected' ? 'var(--err)' : 'var(--txt2)' }}>{p.status}</span>
        <span style={{ flex: 1 }} />
        <span style={{ ...mono, fontSize: '10.5px', color: 'var(--txt3)' }}>{relTime(p.created_at) || (p.age_days + 'd')}</span>
      </div>
      <div style={{ ...mono, fontSize: '13px', fontWeight: 500, color: 'var(--txt)', marginBottom: '3px' }}>{p.name}</div>
      <div style={{ fontSize: '12px', color: 'var(--txt2)', lineHeight: 1.45, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{p.gist}</div>
    </button>
  );
}

function Proposals() {
  const s = useStore();
  const [list, setList] = useState<ProposalSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [sel, setSel] = useState<string | null>(null);

  const load = () => {
    setError(false);
    fetchProposals().then((r) => {
      setList(r.proposals);
      s.setReviewPending(r.pending_count || 0);
      setSel((cur) => (cur && r.proposals.some((p) => p.id === cur) ? cur : (r.proposals[0]?.id ?? null)));
    }).catch(() => setError(true));
  };
  useEffect(() => { load(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  if (error && !list) return <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', color: 'var(--err)', fontSize: '13px' }}>failed to load proposals.</div>;
  if (!list) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading proposals…</div>;
  if (list.length === 0) {
    return (
      <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
        <div style={{ ...mono, fontSize: '13px', marginBottom: '6px' }}>no proposals</div>
        <div style={{ fontSize: '13px', color: 'var(--txt3)' }}>the dream pipeline hasn't raised anything to review.</div>
      </div>
    );
  }

  return (
    <div className="proposals-grid" style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '16px', alignItems: 'start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {list.map((p) => <ProposalCard key={p.id} p={p} selected={p.id === sel} onClick={() => setSel(p.id)} />)}
      </div>
      <div style={{ border: '1px solid var(--line)', borderRadius: '11px', background: 'var(--bg1)', padding: '16px 18px', minWidth: 0 }}>
        {sel ? <Detail id={sel} onDecided={load} /> : <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)' }}>select a proposal.</div>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dream report (phase 5)
// ---------------------------------------------------------------------------

const monoCard: CSSProperties = { ...mono };
// The 5 drill-in cards: count key, sample bucket key, and value color. An absent count
// renders "—" (the dream lanes propose; they do not extract facts yet — contract §Phase 5).
const DREAM_CARDS: { count: string; sample: string; label: string; color: string }[] = [
  { count: 'facts_extracted', sample: 'facts_extracted', label: 'facts extracted', color: 'var(--ok)' },
  { count: 'superseded', sample: 'superseded', label: 'facts superseded', color: 'var(--warn)' },
  { count: 'dedup_merges', sample: 'dedup_merges', label: 'dedup merges', color: 'var(--txt)' },
  { count: 'timeline_events', sample: 'timeline_events', label: 'timeline events', color: 'var(--txt)' },
  { count: 'proposals_raised', sample: 'proposals', label: 'proposals raised', color: 'var(--acc)' },
];

const runLabel = (r: DreamRun): string => (r.started_at || '').replace('T', ' ').slice(0, 16);
const sampleText = (s: unknown): string => {
  if (typeof s === 'string') return s;
  if (s && typeof s === 'object') {
    const o = s as Record<string, unknown>;
    return String(o.text ?? o.fact ?? o.name ?? o.id ?? JSON.stringify(o));
  }
  return String(s);
};

function DreamTab() {
  const [runs, setRuns] = useState<DreamRun[] | null>(null);
  const [error, setError] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const [drill, setDrill] = useState<string>('proposals_raised');

  useEffect(() => {
    fetchDreamReport().then((r) => {
      setRuns(r.runs);
      setSel(r.runs[0]?.id ?? null);
    }).catch(() => setError(true));
  }, []);

  const run = useMemo(() => runs?.find((r) => r.id === sel) || null, [runs, sel]);

  if (error && !runs) return <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', color: 'var(--err)', fontSize: '13px' }}>failed to load dream runs.</div>;
  if (!runs) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading dream runs…</div>;
  if (runs.length === 0 || !run) {
    return (
      <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
        <div style={{ ...mono, fontSize: '13px' }}>no runs recorded yet</div>
        <div style={{ fontSize: '13px', color: 'var(--txt3)', marginTop: '6px' }}>the nightly dream pipeline writes a row per run.</div>
      </div>
    );
  }

  const mins = run.duration_s != null ? Math.round(run.duration_s / 60) : null;
  const samples = (run.samples || {}) as Record<string, unknown[]>;
  const drillCard = DREAM_CARDS.find((c) => c.count === drill);
  const drillItems: unknown[] = drillCard ? (samples[drillCard.sample] || []) : [];

  return (
    <div>
      {/* run select + meta line */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px', flexWrap: 'wrap' }}>
        <span style={{ ...mono, fontSize: '11px', color: 'var(--txt3)' }}>run</span>
        <select value={sel ?? ''} onChange={(e) => setSel(Number(e.target.value))}
          style={{ background: 'var(--bg2)', border: '1px solid var(--line2)', borderRadius: '6px', padding: '5px 8px', ...mono, fontSize: '12.5px', color: 'var(--txt)', cursor: 'pointer' }}>
          {runs.map((r) => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
        </select>
        <span style={{ ...mono, fontSize: '11.5px', color: 'var(--txt3)' }}>
          {mins != null ? `${mins} min` : 'in flight'} · {run.errors.length} error{run.errors.length === 1 ? '' : 's'}
          {run.ok === false ? ' · failed' : ''}
        </span>
      </div>

      {/* 5 drill-in stat cards */}
      <div className="statgrid" style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: '10px', marginBottom: '16px' }}>
        {DREAM_CARDS.map((c) => {
          const v = run.counts[c.count];
          const active = drill === c.count;
          const has = typeof v === 'number';
          return (
            <button key={c.count} onClick={() => setDrill(c.count)}
              style={{ textAlign: 'left', cursor: 'pointer', borderRadius: '10px', padding: '12px 14px',
                border: '1px solid ' + (active ? 'var(--acc)' : 'var(--line)'), background: active ? 'var(--acc-bg)' : 'var(--bg1)' }}>
              <div style={{ ...monoCard, fontSize: '22px', fontWeight: 500, color: has ? c.color : 'var(--txt3)' }}>{has ? (v as number).toLocaleString() : '—'}</div>
              <div style={{ ...monoCard, fontSize: '10.5px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.06em', marginTop: '4px' }}>{c.label}</div>
            </button>
          );
        })}
      </div>

      {/* sample panel for the selected card */}
      <div style={{ border: '1px solid var(--line)', borderRadius: '10px', background: 'var(--bg1)', padding: '14px 16px' }}>
        <div style={{ ...monoHead, marginBottom: '10px' }}>{(drillCard?.label || drill)} · sample</div>
        {drillItems.length === 0 ? (
          <div style={{ fontSize: '13px', color: 'var(--txt3)' }}>no sample recorded for this run.</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {drillItems.map((it, i) => {
              const struck = !!(it && typeof it === 'object' && (it as Record<string, unknown>).superseded);
              return (
                <div key={i} style={{ display: 'flex', gap: '10px', alignItems: 'baseline' }}>
                  <span style={{ ...mono, fontSize: '10px', color: 'var(--et-concept)', minWidth: '44px' }}>{drill === 'proposals_raised' ? 'prop' : 'fact'}</span>
                  <span style={{ fontSize: '12.5px', color: 'var(--txt2)', lineHeight: 1.5, textDecoration: struck ? 'line-through' : 'none' }}>{sampleText(it)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* lane errors */}
      {run.errors.length > 0 && (
        <div style={{ marginTop: '14px', border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px' }}>
          <div style={{ ...monoHead, color: 'var(--err)', marginBottom: '6px' }}>errors</div>
          {run.errors.map((e, i) => <div key={i} style={{ ...mono, fontSize: '11.5px', color: 'var(--err)', lineHeight: 1.6 }}>{e}</div>)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Behavior files (phase 5)
// ---------------------------------------------------------------------------

const fileKeyNoExt = (fk: string): string => fk.replace(/\.[^./]+$/, '');

function BehaviorTab() {
  const [entries, setEntries] = useState<{ group: string; file: BehaviorFileEntry }[] | null>(null);
  const [groups, setGroups] = useState<{ name: string; files: BehaviorFileEntry[] }[]>([]);
  const [error, setError] = useState(false);
  const [sel, setSel] = useState<BehaviorFileEntry | null>(null);
  const [file, setFile] = useState<BehaviorFile | null>(null);
  const [fileErr, setFileErr] = useState(false);

  useEffect(() => {
    fetchBehaviorFiles().then((r) => {
      setGroups(r.groups);
      const flat = r.groups.flatMap((g) => g.files.map((f) => ({ group: g.name, file: f })));
      setEntries(flat);
      setSel(flat[0]?.file ?? null);
    }).catch(() => setError(true));
  }, []);

  useEffect(() => {
    if (!sel) return;
    setFile(null); setFileErr(false);
    fetchBehaviorFile(sel.file_key, sel.scope, sel.surface_id)
      .then(setFile).catch(() => setFileErr(true));
  }, [sel]);

  // Resolve a [[wikilink]] to a known file entry (exact file_key, or basename-without-ext).
  const resolveLink = (target: string): BehaviorFileEntry | null => {
    if (!entries) return null;
    const t = target.trim();
    const exact = entries.find((e) => e.file.file_key === t);
    if (exact) return exact.file;
    const byBase = entries.find((e) => fileKeyNoExt(e.file.file_key) === fileKeyNoExt(t)
      || e.file.file_key.split('/').pop() === t || fileKeyNoExt(e.file.file_key.split('/').pop() || '') === t);
    return byBase?.file ?? null;
  };

  if (error && !entries) return <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', color: 'var(--err)', fontSize: '13px' }}>failed to load behavior files.</div>;
  if (!entries) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading behavior files…</div>;
  if (entries.length === 0) {
    return (
      <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
        <div style={{ ...mono, fontSize: '13px' }}>no behavior files mirrored</div>
        <div style={{ fontSize: '13px', color: 'var(--txt3)', marginTop: '6px' }}>the config lane publishes CLAUDE.md / rules / notes here.</div>
      </div>
    );
  }

  return (
    <div className="proposals-grid" style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '16px', alignItems: 'start' }}>
      {/* left: grouped file list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {groups.map((g) => (
          <div key={g.name}>
            <div style={{ ...monoHead, marginBottom: '6px' }}>{g.name}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {g.files.map((f) => {
                const active = sel?.file_key === f.file_key && sel?.scope === f.scope && sel?.surface_id === f.surface_id;
                return (
                  <button key={f.scope + '/' + f.surface_id + '/' + f.file_key} onClick={() => setSel(f)}
                    style={{ textAlign: 'left', cursor: 'pointer', border: 'none', borderRadius: '6px', padding: '6px 9px', ...mono, fontSize: '12px',
                      background: active ? 'var(--acc-bg)' : 'transparent', color: active ? 'var(--acc)' : 'var(--txt2)' }}>
                    {f.file_key}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* right: selected file */}
      <div style={{ border: '1px solid var(--line)', borderRadius: '11px', background: 'var(--bg1)', padding: '16px 18px', minWidth: 0 }}>
        {fileErr ? (
          <div style={{ color: 'var(--err)', fontSize: '13px' }}>couldn't load this file.</div>
        ) : !file ? (
          <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)' }}>loading file…</div>
        ) : (
          <>
            <div style={{ ...mono, fontSize: '15px', fontWeight: 500, color: 'var(--txt)', marginBottom: '4px' }}>{file.file_key}</div>
            <div style={{ ...mono, fontSize: '11px', color: 'var(--txt3)', marginBottom: '14px' }}>
              {[file.meta.scope, file.meta.surface_id, file.meta.updated_at ? relTime(file.meta.updated_at) : null, file.meta.size != null ? file.meta.size + ' B' : null].filter(Boolean).join(' · ')}
            </div>
            {file.links.length > 0 && (
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                {file.links.map((lnk) => {
                  const target = resolveLink(lnk);
                  return (
                    <button key={lnk} className={target ? 'chipbtn' : undefined} onClick={() => target && setSel(target)} disabled={!target}
                      title={target ? 'open ' + target.file_key : 'no mirrored file for this link'}
                      style={{ ...mono, fontSize: '10.5px', border: '1px solid var(--line2)', background: 'var(--bg2)', color: target ? 'var(--acc)' : 'var(--txt3)', borderRadius: '4px', padding: '2px 8px', cursor: target ? 'pointer' : 'default' }}>
                      [[{lnk}]]
                    </button>
                  );
                })}
              </div>
            )}
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'var(--font-prose)', fontSize: '13px', lineHeight: 1.6, color: 'var(--txt2)' }}>{file.content}</pre>
            {/* Change history intentionally absent: config_registry stores only the current
                version — no history table exists (contract §Phase 5). */}
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Flags (phase 5)
// ---------------------------------------------------------------------------

const FLAG_KIND_COLOR: Record<string, string> = {
  episode: 'var(--txt2)', fact: 'var(--et-concept)', timeline_event: 'var(--et-org)',
  preference: 'var(--et-person)', note: 'var(--et-tech)',
};

function FlagsTab() {
  const s = useStore();
  const [flags, setFlags] = useState<FlagRow[] | null>(null);
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const load = () => {
    setError(false);
    fetchFlags().then((r) => setFlags(r.flags)).catch(() => setError(true));
  };
  useEffect(() => { load(); }, []);

  const jump = (f: FlagRow) => {
    if (f.kind === 'episode') openEpisode(f.item_id);
    else if (f.kind === 'timeline_event') s.setPage('timeline');
    else if (f.kind === 'preference') {
      if (typeof sessionStorage !== 'undefined') sessionStorage.setItem('synapse.tlTab', 'preferences');
      s.setPage('timeline');
    } else s.setPage('feed'); // fact / note surface in the live feed
  };

  const unflag = (f: FlagRow) => {
    const k = f.kind + ':' + f.item_id;
    setBusy(k);
    postFlag(f.kind, f.item_id).then(() => load()).catch(() => setError(true)).finally(() => setBusy(null));
  };

  if (error && !flags) return <div style={{ border: '1px solid var(--err)', background: 'rgba(224,139,122,.08)', borderRadius: '8px', padding: '10px 13px', color: 'var(--err)', fontSize: '13px' }}>failed to load flags.</div>;
  if (!flags) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading flags…</div>;
  if (flags.length === 0) {
    return (
      <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '48px 24px', textAlign: 'center', color: 'var(--txt2)' }}>
        <div style={{ ...mono, fontSize: '13px' }}>nothing flagged</div>
        <div style={{ fontSize: '13px', color: 'var(--txt3)', marginTop: '6px' }}>flag an item anywhere (⚑) to queue it for review here.</div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {flags.map((f) => {
        const k = f.kind + ':' + f.item_id;
        return (
          <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '10px', border: '1px solid var(--line)', borderRadius: '9px', background: 'var(--bg1)', padding: '10px 13px' }}>
            <span style={{ color: 'var(--warn)', fontSize: '13px' }}>⚑</span>
            <span style={{ ...mono, fontSize: '10.5px', border: '1px solid var(--line2)', color: FLAG_KIND_COLOR[f.kind] || 'var(--txt2)', borderRadius: '4px', padding: '2px 7px', whiteSpace: 'nowrap' }}>{f.kind}</span>
            <span style={{ flex: 1, minWidth: 0, fontSize: '13px', color: 'var(--txt)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.gist || <span style={{ color: 'var(--txt3)' }}>({f.kind} #{f.item_id})</span>}</span>
            {f.note && <span style={{ ...mono, fontSize: '11px', color: 'var(--txt3)' }}>{f.note}</span>}
            <button className="linkbtn" onClick={() => jump(f)}
              style={{ border: 'none', background: 'none', color: 'var(--acc)', cursor: 'pointer', ...mono, fontSize: '11.5px' }}>jump ↗</button>
            <button className="softbtn" disabled={busy === k} onClick={() => unflag(f)}
              style={{ border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '5px', padding: '4px 10px', ...mono, fontSize: '11px', cursor: 'pointer', opacity: busy === k ? 0.6 : 1 }}>unflag</button>
          </div>
        );
      })}
    </div>
  );
}

export function Review() {
  const [tab, setTab] = useState<ReviewTab>('proposals');
  return (
    <main style={{ flex: 1, maxWidth: '1160px', width: '100%', margin: '0 auto', padding: '18px 16px 80px', boxSizing: 'border-box' }}>
      <div className="search-tabs" style={{ display: 'flex', gap: '2px', borderBottom: '1px solid var(--line)', marginBottom: '18px', flexWrap: 'wrap' }}>
        {TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button key={t.key} onClick={() => setTab(t.key)}
              style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '8px 13px', fontSize: '13px', fontWeight: 500, color: active ? 'var(--txt)' : 'var(--txt2)', borderBottom: '2px solid ' + (active ? 'var(--acc)' : 'transparent'), marginBottom: '-1px' }}>
              {t.label}
            </button>
          );
        })}
      </div>
      {tab === 'proposals' ? <Proposals />
        : tab === 'dream' ? <DreamTab />
        : tab === 'behavior' ? <BehaviorTab />
        : <FlagsTab />}
    </main>
  );
}
