// Login card: GitHub device flow (primary) or paste-once token (fallback).
//
// The device flow reuses the server's `synapse login` routes (/device/code +
// /device/token, RFC 8628): GitHub approves on any device, the server gates the
// approving login by ALLOWED_GITHUB_USERS — the same identity gate as MCP auth —
// and hands back the machine token, which is then stored exactly like a paste.
// No redirect URIs, so it works on any host the dashboard is served from.
import { useEffect, useRef, useState } from 'react';
import { fetchCatalog, setToken } from '../api';
import { useStore } from '../state';

interface DeviceStart {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete?: string | null;
  expires_in: number;
  interval: number;
}

export function Login() {
  const s = useStore();
  const [value, setValue] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [device, setDevice] = useState<DeviceStart | null>(null);
  const [copied, setCopied] = useState(false);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stopPoll = () => { if (pollRef.current) { clearTimeout(pollRef.current); pollRef.current = null; } };
  useEffect(() => stopPoll, []);

  const finish = (t: string) => { setToken(t); s.setTokenValue(t); };

  const submit = async () => {
    const t = value.trim();
    if (!t || busy) return;
    setBusy(true); setErr('');
    try {
      await fetchCatalog(t); // 401 -> throws
      finish(t);
    } catch {
      setErr('Token rejected. Check the machine token and try again.');
      setBusy(false);
    }
  };

  const startDevice = async () => {
    setErr(''); setBusy(true);
    try {
      const r = await fetch('/device/code', { method: 'POST' });
      const d = await r.json();
      if (!r.ok || !d.device_code) throw new Error(d.error_description || d.error || 'device flow unavailable');
      setDevice(d as DeviceStart);
      poll(d as DeviceStart, (d as DeviceStart).interval || 5, Date.now() + ((d as DeviceStart).expires_in || 900) * 1000);
    } catch (e) {
      setErr(String((e as Error).message || e));
    } finally {
      setBusy(false);
    }
  };

  const poll = (d: DeviceStart, interval: number, deadline: number) => {
    pollRef.current = setTimeout(async () => {
      if (Date.now() > deadline) { setDevice(null); setErr('Login expired — try again.'); return; }
      try {
        const r = await fetch('/device/token', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ device_code: d.device_code }),
        });
        const out = await r.json();
        if (out.token) { finish(out.token); return; }
        if (out.error === 'slow_down') { poll(d, interval + 5, deadline); return; } // RFC 8628 §3.5
        if (out.error === 'authorization_pending' || out.error === 'server_error') { poll(d, interval, deadline); return; }
        // access_denied (not in allowlist) / expired_token — terminal.
        setDevice(null);
        setErr(out.error_description || out.error || 'login failed');
      } catch {
        poll(d, interval, deadline); // transient network blip — keep polling
      }
    }, interval * 1000);
  };

  const cancelDevice = () => { stopPoll(); setDevice(null); setErr(''); };
  const copyCode = () => {
    if (!device) return;
    void navigator.clipboard?.writeText(device.user_code).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
      <div style={{ width: '100%', maxWidth: '380px', background: 'var(--bg1)', border: '1px solid var(--line2)', borderRadius: '14px', padding: '24px 24px 22px', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <div style={{ width: 14, height: 14, border: '1.5px solid var(--acc)', transform: 'rotate(45deg)', borderRadius: '2px' }} />
          <div style={{ fontFamily: 'var(--font-data)', fontWeight: 500, fontSize: '15px', letterSpacing: '.04em' }}>synapse</div>
        </div>

        {device ? (
          <>
            <div style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', marginBottom: '14px' }}>
              Enter this code at{' '}
              <a href={device.verification_uri_complete || device.verification_uri} target="_blank" rel="noreferrer"
                style={{ color: 'var(--acc)' }}>
                {device.verification_uri.replace('https://', '')}
              </a>
            </div>
            <button onClick={copyCode} title="copy"
              style={{ width: '100%', border: '1px dashed var(--line2)', background: 'var(--bg0)', borderRadius: '8px', padding: '12px 0', marginBottom: '12px', cursor: 'pointer', fontFamily: 'var(--font-data)', fontSize: '22px', letterSpacing: '.12em', color: 'var(--txt)' }}>
              {copied ? 'copied ✓' : device.user_code}
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--font-data)', fontSize: '12px', color: 'var(--txt3)', marginBottom: '12px' }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--ok)', animation: 'pulse 1.6s ease-in-out infinite' }} />
              waiting for GitHub approval…
            </div>
            <button onClick={cancelDevice}
              style={{ width: '100%', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '8px', padding: '9px 0', cursor: 'pointer', fontSize: '13px' }}>
              cancel
            </button>
          </>
        ) : (
          <>
            <div style={{ fontSize: '13.5px', lineHeight: 1.6, color: 'var(--txt2)', marginBottom: '14px' }}>
              Sign in to open the operator dashboard. Credentials are stored locally in this browser only.
            </div>
            <button onClick={() => void startDevice()} disabled={busy}
              style={{ width: '100%', border: 'none', background: 'var(--acc)', color: '#0d1116', fontWeight: 600, borderRadius: '8px', padding: '10px 0', cursor: busy ? 'default' : 'pointer', fontSize: '13.5px', opacity: busy ? 0.7 : 1, marginBottom: '14px' }}>
              {busy ? 'starting…' : 'sign in with GitHub'}
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: '0 0 12px', color: 'var(--txt3)', fontFamily: 'var(--font-data)', fontSize: '10.5px' }}>
              <span style={{ flex: 1, borderTop: '1px solid var(--line)' }} /> or <span style={{ flex: 1, borderTop: '1px solid var(--line)' }} />
            </div>
            <input className="field" type="password" value={value} spellCheck={false}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void submit(); }}
              placeholder="SYNAPSE_MACHINE_TOKEN"
              style={{ width: '100%', boxSizing: 'border-box', background: 'var(--bg0)', border: '1px solid var(--line2)', borderRadius: '8px', padding: '9px 12px', fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--txt)', marginBottom: '10px' }} />
            <button onClick={() => void submit()} disabled={busy}
              style={{ width: '100%', border: '1px solid var(--line2)', background: 'var(--bg2)', color: 'var(--txt2)', borderRadius: '8px', padding: '9px 0', cursor: busy ? 'default' : 'pointer', fontSize: '13px' }}>
              unlock with token
            </button>
          </>
        )}
        {err && <div style={{ color: 'var(--err)', fontSize: '12px', marginTop: '10px', fontFamily: 'var(--font-data)' }}>{err}</div>}
      </div>
    </div>
  );
}
