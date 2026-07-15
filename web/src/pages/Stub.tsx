// Later-phase pages (Recall/Graph/Timeline/Metrics/Review) render this placeholder
// in phase 1 — dashed border, mono, txt3, per the build brief.
const PHASE: Record<string, string> = {
  graph: 'Graph — the knowledge-graph explorer',
  timeline: 'Timeline — the life/work event log and preferences',
  metrics: 'Metrics — recall / ingestion / corpus ops charts',
  review: 'Review — the self-improvement proposal console',
};

export function Stub({ page }: { page: string }) {
  return (
    <main style={{ flex: 1, maxWidth: '860px', width: '100%', margin: '0 auto', padding: '20px 16px 80px', boxSizing: 'border-box' }}>
      <div style={{ border: '1px dashed var(--line2)', borderRadius: '10px', padding: '56px 24px', textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--font-data)', fontSize: '13px', color: 'var(--txt3)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '8px' }}>
          ships in a later phase
        </div>
        <div style={{ fontFamily: 'var(--font-data)', fontSize: '12.5px', color: 'var(--txt3)', lineHeight: 1.6 }}>
          {PHASE[page] || page}
        </div>
      </div>
    </main>
  );
}
