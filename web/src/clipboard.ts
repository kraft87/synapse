// Copy text to the clipboard. navigator.clipboard exists only in a secure context, and the
// dashboard is often opened over plain http on a LAN address, where it is undefined: fall back
// to a hidden textarea and execCommand there. Resolves false when neither route copied, so a
// caller can say so instead of showing a success it did not get.
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch { /* permission refused: try the textarea route */ }
  }
  const prev = document.activeElement as HTMLElement | null;
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none';
  document.body.appendChild(ta);
  ta.select();
  try {
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    ta.remove();
    prev?.focus();
  }
}
