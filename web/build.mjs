// Build: bundle src/main.tsx -> dist/app.js (ES module entry) + code-split chunks
// under dist/assets/, copy the five IBM Plex latin woff2 into dist/assets/, and emit
// dist/index.html from src/index.html with src/styles.css inlined into a <style> tag.
// No CDN, no separate CSS file — the Python MCP server serves dist/ verbatim under /dash.
//
// Code-splitting (phase 6): the graph explorer pulls in cytoscape + cose-bilkent (~500KB),
// which would push a single IIFE past ~800KB and tax every page load. The Graph page is
// React.lazy-imported, so esbuild's `splitting` peels cytoscape into a hashed chunk that
// loads only on the first Graph visit. Splitting REQUIRES format:'esm', hence the
// <script type="module"> in index.html. app.js stays at /dash/app.js (its own server
// route); chunks land in dist/assets/ and ride the existing whitelisted /dash/assets/*
// route — no server change. app.js's ESM imports resolve relative to /dash/, so
// `./assets/chunk-*.js` → /dash/assets/chunk-*.js.
import { build } from 'esbuild';
import {
  mkdirSync, rmSync, copyFileSync, readFileSync, writeFileSync, existsSync, statSync,
  readdirSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, 'dist');
const assets = join(dist, 'assets');

rmSync(dist, { recursive: true, force: true });
mkdirSync(assets, { recursive: true });

// 1. bundle the app as an ES module with code-splitting (entry = dist/app.js; shared +
//    lazy chunks = dist/assets/chunk-[hash].js).
await build({
  entryPoints: [join(root, 'src/main.tsx')],
  bundle: true,
  minify: true,
  format: 'esm',
  splitting: true,
  jsx: 'automatic',
  target: 'es2020',
  define: { 'process.env.NODE_ENV': '"production"' },
  outdir: dist,
  entryNames: 'app',
  chunkNames: 'assets/chunk-[hash]',
  assetNames: 'assets/[name]-[hash]',
  logLevel: 'info',
});

// 2. self-host the fonts (sans 400/500/600, mono 400/500)
const fonts = [
  ['@fontsource/ibm-plex-sans', 'ibm-plex-sans-latin-400-normal.woff2'],
  ['@fontsource/ibm-plex-sans', 'ibm-plex-sans-latin-500-normal.woff2'],
  ['@fontsource/ibm-plex-sans', 'ibm-plex-sans-latin-600-normal.woff2'],
  ['@fontsource/ibm-plex-mono', 'ibm-plex-mono-latin-400-normal.woff2'],
  ['@fontsource/ibm-plex-mono', 'ibm-plex-mono-latin-500-normal.woff2'],
];
for (const [pkg, file] of fonts) {
  const src = join(root, 'node_modules', pkg, 'files', file);
  if (!existsSync(src)) throw new Error('missing font file: ' + src);
  copyFileSync(src, join(assets, file));
}

// 3. index.html with the hand-written CSS inlined (font urls point at /dash/assets)
const css = readFileSync(join(root, 'src/styles.css'), 'utf8');
const tmpl = readFileSync(join(root, 'src/index.html'), 'utf8');
if (!tmpl.includes('<!--STYLES-->')) throw new Error('src/index.html missing <!--STYLES--> marker');
writeFileSync(join(dist, 'index.html'), tmpl.replace('<!--STYLES-->', '<style>\n' + css + '\n</style>'));

const kb = (n) => (n / 1024).toFixed(1) + ' KB';
const sz = (p) => statSync(p).size;
const chunks = readdirSync(assets).filter((f) => f.startsWith('chunk-') && f.endsWith('.js'));
const chunkBytes = chunks.reduce((s, f) => s + sz(join(assets, f)), 0);
console.log('build complete:');
console.log('  dist/app.js       ' + kb(sz(join(dist, 'app.js'))) + ' (entry, loaded on every page)');
for (const c of chunks) console.log('  dist/assets/' + c + '  ' + kb(sz(join(assets, c))) + ' (lazy chunk)');
console.log('  js total          ' + kb(sz(join(dist, 'app.js')) + chunkBytes));
console.log('  dist/index.html   ' + kb(sz(join(dist, 'index.html'))));
console.log('  dist/assets/      ' + fonts.length + ' woff2 + ' + chunks.length + ' chunk(s)');
