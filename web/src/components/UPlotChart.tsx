// Thin React wrapper around uPlot (the committed ~40KB time-series lib, per the build brief).
// It owns the instance lifecycle: build on mount / when opts or data change, size to the
// container via a ResizeObserver, destroy on unmount. Callers pass fully-built uPlot options
// (minus width/height) + AlignedData; theme-dependent colors must be resolved to concrete
// values by the caller (canvas can't read CSS vars) — see tokens.cssValue.
import { useEffect, useRef } from 'react';
import uPlot from 'uplot';

export function UPlotChart({
  opts,
  data,
  height = 150,
}: {
  opts: Omit<uPlot.Options, 'width' | 'height'>;
  data: uPlot.AlignedData;
  height?: number;
}) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const width = Math.max(1, Math.floor(node.clientWidth) || 600);
    const u = new uPlot({ ...opts, width, height } as uPlot.Options, data, node);
    const ro = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0].contentRect.width);
      if (w > 0) u.setSize({ width: w, height });
    });
    ro.observe(node);
    return () => {
      ro.disconnect();
      u.destroy();
    };
  }, [opts, data, height]);

  return <div ref={host} style={{ width: '100%' }} />;
}
