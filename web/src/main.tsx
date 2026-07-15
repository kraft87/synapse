import { createRoot } from 'react-dom/client';
import { App } from './App';
import { bootstrapToken } from './hash';

// Consume a #token=... fragment (if any) before first render.
bootstrapToken();

// Reflect the persisted theme immediately so there's no flash.
const theme = localStorage.getItem('synapse.theme');
if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;

const el = document.getElementById('root');
if (el) createRoot(el).render(<App />);
