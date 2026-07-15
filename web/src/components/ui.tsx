// Small shared primitives: the badge chips (README §3 badge taxonomy) and the
// optimistic ⚑ flag toggle (spec §5c — quiet, txt3 at rest, warn when set).
import type React from 'react';
import { useState, type CSSProperties } from 'react';
import { postFlag } from '../api';
import { srcColor, typeColor, typeLabel } from '../tokens';

const chipBase: CSSProperties = {
  fontFamily: 'var(--font-data)', fontSize: '10.5px', padding: '2px 7px',
  borderRadius: '4px', whiteSpace: 'nowrap', lineHeight: 1.4,
};

export const TypeChip = ({ type }: { type: string }) => (
  <span style={{ ...chipBase, background: 'var(--bg2)', color: typeColor[type] || 'var(--txt2)', border: '1px solid var(--line)' }}>
    {typeLabel[type] || type}
  </span>
);

export const ProjectChip = ({ project }: { project?: string | null }) =>
  project ? <span style={{ ...chipBase, background: 'var(--acc-bg)', color: 'var(--acc)' }}>{project}</span> : null;

export const SourceChip = ({ source }: { source?: string | null }) =>
  source ? <span style={{ ...chipBase, border: '1px solid ' + srcColor(source), color: srcColor(source) }}>{source}</span> : null;

export const SalLabel = ({ sal }: { sal?: number }) =>
  sal != null ? (
    <span style={{ fontFamily: 'var(--font-data)', fontSize: '10.5px', color: sal > 0.7 ? 'var(--warn)' : 'var(--txt3)' }}>
      sal {sal.toFixed(2)}
    </span>
  ) : null;

// entity dot for dossiers/results
export const Dot = ({ color, size = 10 }: { color: string; size?: number }) => (
  <span style={{ width: size, height: size, borderRadius: '50%', background: color, flexShrink: 0 }} />
);

export function FlagButton({ kind, itemId, initial, size = 12 }: { kind: string; itemId: string; initial: boolean; size?: number }) {
  const [on, setOn] = useState(initial);
  const [busy, setBusy] = useState(false);
  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (busy) return;
    const next = !on;
    setOn(next); // optimistic
    setBusy(true);
    postFlag(kind, itemId).then((r) => { if (typeof r.flagged === 'boolean') setOn(r.flagged); })
      .catch(() => setOn(!next)) // revert on failure
      .finally(() => setBusy(false));
  };
  return (
    <button
      className="flagbtn"
      onClick={toggle}
      title={on ? 'flagged for review — click to unflag' : 'flag for review'}
      aria-pressed={on}
      style={{ border: 'none', background: 'none', padding: '2px 4px', cursor: 'pointer', color: on ? 'var(--warn)' : 'var(--txt3)', fontSize: size + 'px', lineHeight: 1 }}
    >
      ⚑
    </button>
  );
}

export const Spinner = ({ label }: { label?: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--txt3)', fontFamily: 'var(--font-data)', fontSize: '12px', padding: '20px 0' }}>
    <span style={{ width: 12, height: 12, borderRadius: '50%', border: '2px solid var(--line2)', borderTopColor: 'var(--acc)', display: 'inline-block', animation: 'spin .8s linear infinite' }} />
    {label}
  </div>
);
