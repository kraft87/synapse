// Login card: browser sign-in (redirect flow, offered when the server has an identity
// provider) or paste-once token (fallback).
//
// "sign in" is a plain navigation to /dash/oauth/start: the server
// sends the browser to GitHub, GitHub returns to the public callback, the server
// gates the account by ALLOWED_GITHUB_USERS — the same identity gate as MCP
// auth — and bounces back here with #token=... (consumed by the existing
// fragment bootstrap before first render). Failures come back as #login_error=.
import { useEffect, useState } from 'react';
import { fetchCatalog, fetchSsoAvailable, setToken } from '../api';
import { dec } from '../hash';
import { useStore } from '../state';

// Read-and-strip a #login_error=... fragment left by the OAuth callback.
function consumeLoginError(): string {
  const m = location.hash.match(/login_error=([^&]+)/);
  if (!m) return '';
  history.replaceState(null, '', location.pathname + location.search);
  return dec(m[1]) ?? '';
}

export function Login() {
  const s = useStore();
  const [value, setValue] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  // "sign in" only where the server has an identity provider; without one /dash/oauth/start is a 404.
  const [sso, setSso] = useState(false);
  useEffect(() => { const e = consumeLoginError(); if (e) setErr(e); }, []);
  useEffect(() => {
    let live = true;
    void fetchSsoAvailable().then((ok) => { if (live) setSso(ok); });
    return () => { live = false; };
  }, []);

  const submit = async () => {
    const t = value.trim();
    if (!t || busy) return;
    setBusy(true); setErr('');
    try {
      await fetchCatalog(t); // 401 -> throws
      setToken(t);
      s.setTokenValue(t);
    } catch {
      setErr('Token rejected — the dashboard needs a full-trust device token (the root machine token is refused).');
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
          Open the operator dashboard. Credentials are stored locally in this browser only.
        </div>
        {sso && (
          <>
            <a href="/dash/oauth/start"
              style={{ display: 'block', textAlign: 'center', textDecoration: 'none', border: 'none', background: 'var(--acc)', color: '#0d1116', fontWeight: 600, borderRadius: '8px', padding: '10px 0', fontSize: '13.5px', marginBottom: '14px' }}>
              sign in
            </a>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: '0 0 12px', color: 'var(--txt3)', fontFamily: 'var(--font-data)', fontSize: '10.5px' }}>
              <span style={{ flex: 1, borderTop: '1px solid var(--line)' }} /> or <span style={{ flex: 1, borderTop: '1px solid var(--line)' }} />
            </div>
          </>
        )}
        <input className="field" type="password" value={value} spellCheck={false}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }}
          placeholder="full-trust device token"
          style={{ width: '100%', boxSizing: 'border-box', background: 'var(--bg0)', border: '1px solid var(--line2)', borderRadius: '8px', padding: '9px 12px', fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--txt)', marginBottom: '10px' }} />
        <button onClick={() => void submit()} disabled={busy}
          style={{ width: '100%', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '8px', padding: '9px 0', cursor: busy ? 'default' : 'pointer', fontSize: '13px' }}>
          {busy ? 'verifying…' : 'unlock with token'}
        </button>
        {err && <div style={{ color: 'var(--err)', fontSize: '12px', marginTop: '10px', fontFamily: 'var(--font-data)' }}>{err}</div>}
      </div>
    </div>
  );
}
