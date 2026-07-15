// Review — the self-improvement console (phase 2b). Tabs: Proposals / Behavior files /
// Dream report / Flags. Only Proposals is real this phase; the others render the
// phase-stub panel. Proposals is a master-detail grid over BOTH lanes (skills +
// config-edit): left = proposal cards, right = evidence + provenance + payload
// (markdown draft or unified diff) + audit trail + a decision bar on 'proposed' items.
import type React from 'react';
import { useEffect, useState, type CSSProperties } from 'react';
import {
  fetchProposal, fetchProposals, postProposalDecision,
  type EvidenceItem, type ProposalDetail, type ProposalSummary,
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

const PHASE_STUB: Record<string, string> = {
  behavior: 'Behavior files — CLAUDE.md / rules / memory notes with change history and [[wikilinks]]',
  dream: 'Dream report — per-run extraction stats with drill-in sample panels',
  flags: 'Flags — the flagged-item queue with jump + unflag',
};

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

function PhaseStub({ tab }: { tab: string }) {
  return (
    <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '56px 24px', textAlign: 'center' }}>
      <div style={{ ...mono, fontSize: '13px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '8px' }}>ships in a later phase</div>
      <div style={{ ...mono, fontSize: '12.5px', color: 'var(--txt3)', lineHeight: 1.6 }}>{PHASE_STUB[tab] || tab}</div>
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
      {tab === 'proposals' ? <Proposals /> : <PhaseStub tab={tab} />}
    </main>
  );
}
