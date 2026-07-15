// Paste-once token login. No token in localStorage -> this centered card. The
// pasted token is verified with a live GET /dash/api/catalog before it's stored.
import { useState } from 'react';
import { fetchCatalog, setToken } from '../api';
import { useStore } from '../state';

export function Login() {
  const s = useStore();
  const [value, setValue] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const t = value.trim();
    if (!t || busy) return;
    setBusy(true); setErr('');
    try {
      await fetchCatalog(t); // 401 -> throws
      setToken(t);
      s.setTokenValue(t);
    } catch {
      setErr('Token rejected. Check the machine token and try again.');
      setBusy(false);
    }
  };

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
      <div style={{ width: '100%', maxWidth: '380px', background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '14px', padding: '24px 24px 22px', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <div style={{ width: 14, height: 14, border: '1.5px solid var(--acc)', transform: 'rotate(45deg)', borderRadius: '2px' }} />
          <div style={{ fontFamily: 'var(--font-data)', fontWeight: 500, fontSize: '15px', letterSpacing: '.04em' }}>synapse</div>
        </div>
        <div style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', marginBottom: '14px' }}>
          Paste the machine token to open the operator dashboard. It's stored locally in this browser only.
        </div>
        <input className="field" type="password" autoFocus value={value} spellCheck={false}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }}
          placeholder="SYNAPSE_MACHINE_TOKEN"
          style={{ width: '100%', boxSizing: 'border-box', background: 'var(--bg0)', border: '1px solid var(--line2)', borderRadius: '8px', padding: '9px 12px', fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--txt)', marginBottom: '10px' }} />
        {err && <div style={{ color: 'var(--err)', fontSize: '12px', marginBottom: '10px', fontFamily: 'var(--font-data)' }}>{err}</div>}
        <button onClick={() => void submit()} disabled={busy}
          style={{ width: '100%', border: 'none', background: 'var(--acc)', color: '#0d1116', fontWeight: 600, borderRadius: '8px', padding: '10px 0', cursor: busy ? 'default' : 'pointer', fontSize: '13.5px', opacity: busy ? 0.7 : 1 }}>
          {busy ? 'verifying…' : 'unlock'}
        </button>
      </div>
    </div>
  );
}
