// Build: bundle src/main.tsx -> dist/app.js (one minified IIFE), copy the five
// IBM Plex latin woff2 into dist/assets/, and emit dist/index.html from
// src/index.html with src/styles.css inlined into a <style> tag. No CDN, no
// separate CSS file — the Python MCP server serves dist/ verbatim under /dash.
import { build } from 'esbuild';
import {
  mkdirSync, rmSync, copyFileSync, readFileSync, writeFileSync, existsSync, statSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, 'dist');
const assets = join(dist, 'assets');

rmSync(dist, { recursive: true, force: true });
mkdirSync(assets, { recursive: true });

// 1. bundle the app into a single IIFE
await build({
  entryPoints: [join(root, 'src/main.tsx')],
  bundle: true,
  minify: true,
  format: 'iife',
  jsx: 'automatic',
  target: 'es2020',
  define: { 'process.env.NODE_ENV': '"production"' },
  outfile: join(dist, 'app.js'),
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

const kb = (p) => (statSync(p).size / 1024).toFixed(1) + ' KB';
console.log('build complete:');
console.log('  dist/app.js     ' + kb(join(dist, 'app.js')));
console.log('  dist/index.html ' + kb(join(dist, 'index.html')));
console.log('  dist/assets/    ' + fonts.length + ' woff2');
