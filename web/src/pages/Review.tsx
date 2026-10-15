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
import { copyText } from '../clipboard';
import { useStore } from '../state';
import { openEpisode, openSession } from '../hash';
import { relTime, monoHead, localMinute } from '../tokens';
import { ErrorBox, EmptyState } from '../components/ui';
import { useAsync } from '../hooks';
import { defaultRun, laneCounts, runDuration, runState } from '../dream';
import { decisionProblem } from '../proposals';

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
const chipStyle: CSSProperties = { ...mono, fontSize: '10.5px', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--acc)', borderRadius: '4px', padding: '2px 8px', cursor: 'pointer' };

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
      {(detail.provenance_episodes.length > 0 || detail.provenance_sessions.length > 0) && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '10px' }}>
          {detail.provenance_episodes.map((id) => (
            <button key={'ep' + id} className="chipbtn" onClick={() => openEpisode(id)} title="open episode"
              style={chipStyle}>
              ep-{id} ↗
            </button>
          ))}
          {detail.provenance_sessions.map((sid) => (
            <button key={'s' + sid} className="chipbtn" onClick={() => openSession(sid)} title={'open session ' + sid}
              style={chipStyle}>
              session {sid.slice(0, 8)} ↗
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
  const { data: detail, loading, error, reload } = useAsync(() => fetchProposal(id), [id]);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  // What the last press of "discuss" did, shown for a moment.
  const [copied, setCopied] = useState<'ok' | 'failed' | null>(null);
  // Why a decision did not go through. A 409 (see decisionProblem) and a refusal (200, status
  // 'refused': a skills approve that can't apply the draft, e.g. no draft yet, a stale draft,
  // a taken name) both changed nothing. A failed request says nothing about the outcome, so
  // the pane is reloaded to show the status the server has.
  const [problem, setProblem] = useState<string | null>(null);
  // The pane is mounted per proposal (see Proposals), so nothing here outlives a selection.
  const locked = busy || loading;

  const decide = (action: 'approve' | 'reject', note?: string) => {
    setBusy(true);
    setProblem(null);
    postProposalDecision(id, action, note)
      .then((r) => {
        if (r.status === 'refused') {
          setProblem('not applied: ' + String(r.detail ?? 'refused by the lane'));
          reload(); onDecided();
          return;
        }
        setRejecting(false); setReason(''); reload(); onDecided();
      })
      .catch((e) => { setProblem(decisionProblem(e)); reload(); onDecided(); })
      .finally(() => setBusy(false));
  };

  const discuss = (d: ProposalDetail) => {
    const text = summaryText(d);
    void copyText(text).then((ok) => {
      setCopied(ok ? 'ok' : 'failed');
      window.setTimeout(() => setCopied(null), 1600);
    });
  };

  if (loading && !detail) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading proposal…</div>;
  if (error && !detail) return <ErrorBox>failed to load this proposal.</ErrorBox>;
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

      {/* why the last decision did not go through; a 409 leaves the row decided, so it shows without the bar */}
      {problem && <div role="alert" style={{ fontSize: '12.5px', color: 'var(--err)', lineHeight: 1.5, marginTop: '16px' }}>{problem}</div>}

      {/* decision bar (proposed only) */}
      {!decided && (
        <div style={{ borderTop: '1px solid var(--line)', paddingTop: '14px', marginTop: '16px' }}>
          {!rejecting ? (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
              <button disabled={locked} onClick={() => decide('approve')}
                style={{ border: 'none', background: 'var(--ok)', color: '#0d1116', borderRadius: '6px', padding: '7px 16px', fontSize: '13px', fontWeight: 600, cursor: 'pointer', opacity: locked ? 0.6 : 1 }}>approve</button>
              <button className="softbtn" disabled={locked} onClick={() => setRejecting(true)}
                style={{ borderRadius: '6px', padding: '7px 14px', fontSize: '13px' }}>reject…</button>
              <span style={{ flex: 1 }} />
              <button className="softbtn" onClick={() => discuss(detail)}
                style={{ color: copied === 'ok' ? 'var(--ok)' : copied === 'failed' ? 'var(--err)' : 'var(--txt2)', borderRadius: '6px', padding: '7px 14px', fontSize: '12.5px', ...mono }}>{copied === 'ok' ? 'copied for chat ✓' : copied === 'failed' ? 'copy failed' : 'discuss →'}</button>
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
                <button disabled={locked || !reason.trim()} onClick={() => decide('reject', reason.trim())}
                  style={{ border: 'none', background: 'var(--err)', color: '#0d1116', borderRadius: '6px', padding: '7px 16px', fontSize: '13px', fontWeight: 600, cursor: reason.trim() ? 'pointer' : 'not-allowed', opacity: (locked || !reason.trim()) ? 0.55 : 1 }}>confirm reject</button>
                <button className="softbtn" onClick={() => { setRejecting(false); setReason(''); }}
                  style={{ borderRadius: '6px', padding: '7px 14px', fontSize: '13px' }}>cancel</button>
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
  const { data, error, reload } = useAsync(() => fetchProposals(), []);
  const list = data?.proposals ?? null;
  const [sel, setSel] = useState<string | null>(null);

  // side effects of a (re)load: sync the nav badge + keep the selection valid.
  useEffect(() => {
    if (!data) return;
    s.setReviewPending(data.pending_count || 0);
    setSel((cur) => (cur && data.proposals.some((p) => p.id === cur) ? cur : (data.proposals[0]?.id ?? null)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  if (error && !list) return <ErrorBox>failed to load proposals.</ErrorBox>;
  if (!list) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading proposals…</div>;
  if (list.length === 0) {
    return <EmptyState title="no proposals" subtitle="the dream pipeline hasn't raised anything to review." />;
  }

  return (
    <div className="proposals-grid" style={{ display: 'grid', gridTemplateColumns: '340px minmax(0,1fr)', gap: '16px', alignItems: 'start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {list.map((p) => <ProposalCard key={p.id} p={p} selected={p.id === sel} onClick={() => setSel(p.id)} />)}
      </div>
      <div style={{ border: '1px solid var(--line)', borderRadius: '11px', background: 'var(--bg1)', padding: '16px 18px', minWidth: 0 }}>
        {sel ? <Detail key={sel} id={sel} onDecided={reload} /> : <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)' }}>select a proposal.</div>}
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

const runLabel = (r: DreamRun): string => localMinute(r.started_at) + (r.finished_at ? '' : ' · ' + runState(r));
const sampleText = (s: unknown): string => {
  if (typeof s === 'string') return s;
  if (s && typeof s === 'object') {
    const o = s as Record<string, unknown>;
    return String(o.text ?? o.fact ?? o.name ?? o.id ?? JSON.stringify(o));
  }
  return String(s);
};

function DreamTab() {
  const { data, error } = useAsync(() => fetchDreamReport(), []);
  const runs = data?.runs ?? null;
  // default-select the newest finished run without an effect (avoids an empty-state flash
  // on load); a crashed run stays in the list but does not pin the report.
  const [selRaw, setSel] = useState<number | null>(null);
  const sel = selRaw ?? (runs ? defaultRun(runs)?.id : undefined) ?? null;
  const [drill, setDrill] = useState<string>('proposals_raised');

  const run = useMemo(() => runs?.find((r) => r.id === sel) || null, [runs, sel]);

  if (error && !runs) return <ErrorBox>failed to load dream runs.</ErrorBox>;
  if (!runs) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading dream runs…</div>;
  if (runs.length === 0 || !run) {
    return <EmptyState title="no runs recorded yet" subtitle="the nightly dream pipeline writes a row per run." />;
  }

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
          {runDuration(run)} · {run.errors.length} error{run.errors.length === 1 ? '' : 's'}
          {run.ok === false && runState(run) !== 'interrupted' ? ' · failed' : ''}
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

      {/* the counts the lanes keep besides the five above */}
      {laneCounts(run.counts).length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '16px' }}>
          {laneCounts(run.counts).map(([k, v]) => (
            <span key={k} style={{ ...mono, fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: 'var(--bg2)', border: '1px solid var(--line)', color: 'var(--txt2)' }}>
              {k} <span style={{ color: 'var(--txt)' }}>{v.toLocaleString()}</span>
            </span>
          ))}
        </div>
      )}

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

  // The last click wins: a slower earlier file arriving after it is dropped.
  useEffect(() => {
    if (!sel) return;
    let live = true;
    setFile(null); setFileErr(false);
    fetchBehaviorFile(sel.file_key, sel.scope, sel.surface_id)
      .then((f) => { if (live) setFile(f); })
      .catch(() => { if (live) setFileErr(true); });
    return () => { live = false; };
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

  if (error && !entries) return <ErrorBox>failed to load behavior files.</ErrorBox>;
  if (!entries) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading behavior files…</div>;
  if (entries.length === 0) {
    return <EmptyState title="no behavior files mirrored" subtitle="the config lane publishes CLAUDE.md / rules / notes here." />;
  }

  return (
    <div className="proposals-grid" style={{ display: 'grid', gridTemplateColumns: '260px minmax(0,1fr)', gap: '16px', alignItems: 'start' }}>
      {/* left: grouped file list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {groups.map((g) => (
          <div key={g.name}>
            <div style={{ ...monoHead, marginBottom: '6px' }}>{g.name}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {g.files.map((f) => {
                const active = sel?.file_key === f.file_key && sel?.scope === f.scope && sel?.surface_id === f.surface_id;
                // The same file mirrored from two surfaces (or scopes) would read as one entry twice.
                const twin = g.files.some((o) => o !== f && o.file_key === f.file_key);
                return (
                  <button key={f.scope + '/' + f.surface_id + '/' + f.file_key} onClick={() => setSel(f)}
                    style={{ textAlign: 'left', cursor: 'pointer', border: 'none', borderRadius: '6px', padding: '6px 9px', ...mono, fontSize: '12px',
                      background: active ? 'var(--acc-bg)' : 'transparent', color: active ? 'var(--acc)' : 'var(--txt2)' }}>
                    {f.file_key}
                    {twin && <span style={{ display: 'block', fontSize: '10.5px', color: 'var(--txt3)' }}>{[f.surface_id, f.scope].filter(Boolean).join(' · ')}</span>}
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

  // Where a flag leads: the episode behind it (the item itself for an episode, a fact's
  // provenance, the turn an event or note was born from), or the Preferences tab. A row with
  // no episode to open (a git event, a note from `remember`) has no jump: its gist is shown,
  // and a note's body with it.
  const canJump = (f: FlagRow): boolean => f.kind === 'preference' || f.episode_id != null;
  const jump = (f: FlagRow) => {
    if (f.kind === 'preference') {
      if (typeof sessionStorage !== 'undefined') sessionStorage.setItem('synapse.tlTab', 'preferences');
      s.setPage('timeline');
    } else if (f.episode_id != null) openEpisode(f.episode_id);
  };

  const unflag = (f: FlagRow) => {
    const k = f.kind + ':' + f.item_id;
    setBusy(k);
    postFlag(f.kind, f.item_id, false).then(() => load()).catch(() => setError(true)).finally(() => setBusy(null));
  };

  if (error && !flags) return <ErrorBox>failed to load flags.</ErrorBox>;
  if (!flags) return <div style={{ ...mono, fontSize: '12px', color: 'var(--txt3)', padding: '10px 0' }}>loading flags…</div>;
  if (flags.length === 0) {
    return <EmptyState title="nothing flagged" subtitle="flag an episode, fact, event or preference (⚑) to queue it for review here." />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {flags.map((f) => {
        const k = f.kind + ':' + f.item_id;
        return (
          <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '10px', border: '1px solid var(--line)', borderRadius: '9px', background: 'var(--bg1)', padding: '10px 13px' }}>
            <span style={{ color: 'var(--warn)', fontSize: '13px' }}>⚑</span>
            <span style={{ ...mono, fontSize: '10.5px', border: '1px solid var(--line2)', color: FLAG_KIND_COLOR[f.kind] || 'var(--txt2)', borderRadius: '4px', padding: '2px 7px', whiteSpace: 'nowrap' }}>{f.kind}</span>
            <span style={{ flex: 1, minWidth: 0, fontSize: '13px', lineHeight: 1.45, color: 'var(--txt)', overflowWrap: 'anywhere' }}>
              {f.gist || <span style={{ color: 'var(--txt3)' }}>({f.kind} #{f.item_id})</span>}
              {f.body && <div style={{ marginTop: '5px', maxHeight: '14em', overflowY: 'auto', whiteSpace: 'pre-wrap', fontSize: '12.5px', color: 'var(--txt2)' }}>{f.body}</div>}
            </span>
            {f.note && <span style={{ ...mono, fontSize: '11px', color: 'var(--txt3)' }}>{f.note}</span>}
            {canJump(f) && <button className="linkbtn" onClick={() => jump(f)}
              style={{ border: 'none', background: 'none', color: 'var(--acc)', cursor: 'pointer', ...mono, fontSize: '11.5px' }}>jump ↗</button>}
            <button className="softbtn" disabled={busy === k} onClick={() => unflag(f)}
              style={{ borderRadius: '5px', padding: '4px 10px', ...mono, fontSize: '11px', opacity: busy === k ? 0.6 : 1 }}>unflag</button>
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
