import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import type { ForecastResponse } from "@/lib/forecast-types";
import { categoryLabel, saffirKey } from "@/lib/forecast-types";
import { saffirColor } from "./scale";

interface Props {
  data: ForecastResponse;
  nowHour: number;
  currentStep?: number;
  onStepChange?: (step: number) => void;
}

export function IntensityChart({ data, nowHour, currentStep = 0, onStepChange }: Props) {
  const [viewMode, setViewMode] = useState<"realtime" | "audit">("realtime");
  const [smoothingMode, setSmoothingMode] = useState<"raw" | "ema">("raw");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);

  // Fallback lifecycle array if not present
  const lifecycle = useMemo(() => {
    if (data.lifecycle && data.lifecycle.length > 0) {
      return data.lifecycle;
    }
    // Synthesize from timeline
    return data.timeline.map((item, idx) => ({
      step_index: idx,
      elapsed_hours: item.t,
      observed_kt: item.observed_kt,
      actual_plus_24h: item.observed_kt,
      pred_6h: item.predicted_kt,
      pred_12h: item.predicted_kt,
      pred_24h: item.predicted_kt,
      ema_6h: item.predicted_kt,
      ema_12h: item.predicted_kt,
      ema_24h: item.predicted_kt,
    }));
  }, [data.lifecycle, data.timeline]);

  const N = lifecycle.length;
  const safeCurrentStep = Math.max(0, Math.min(N - 1, currentStep));

  // Render chart on canvas
  const renderChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    if (W === 0 || H === 0) return;

    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    if (N === 0) {
      ctx.restore();
      return;
    }

    const padLeft = 46;
    const padRight = 32;
    const padTop = 22;
    const padBottom = 28;
    const chartW = W - padLeft - padRight;
    const chartH = H - padTop - padBottom;

    const maxObserved = Math.max(...lifecycle.map((t) => t.observed_kt));
    const maxPred = Math.max(...lifecycle.map((t) => t.pred_24h));
    const maxVal = Math.max(130, maxObserved, maxPred, 140);
    const minVal = 15;

    const getX = (idx: number) => padLeft + (idx / Math.max(1, N - 1)) * chartW;
    const getY = (val: number) => padTop + chartH - ((val - minVal) / (maxVal - minVal)) * chartH;

    // 1. Clean, neutral horizontal gridlines (every 20 kt)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let v = 20; v <= maxVal; v += 20) {
      const y = getY(v);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + chartW, y);
      ctx.stroke();

      ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(`${v}`, padLeft - 6, y + 3);
    }

    // X-Axis Baseline
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + chartH);
    ctx.lineTo(padLeft + chartW, padTop + chartH);
    ctx.stroke();

    // Time ticks along X axis
    const stepInterval = Math.max(1, Math.ceil(N / 8));
    for (let i = 0; i < N; i += stepInterval) {
      const x = getX(i);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.beginPath();
      ctx.moveTo(x, padTop + chartH);
      ctx.lineTo(x, padTop + chartH + 4);
      ctx.stroke();

      ctx.fillStyle = "#64748B";
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`+${lifecycle[i]?.elapsed_hours}h`, x, padTop + chartH + 15);
    }

    const currStep = lifecycle[safeCurrentStep]!;
    const nowX = getX(safeCurrentStep);
    const nowY = getY(currStep.observed_kt);

    // Calculation for forecast horizons (+6h, +12h, +24h)
    const future24Idx = Math.min(safeCurrentStep + 8, N - 1);
    const p6Idx = Math.min(safeCurrentStep + 2, N - 1);
    const p12Idx = Math.min(safeCurrentStep + 4, N - 1);
    const p24Idx = future24Idx;

    const isEma = smoothingMode === "ema";
    const raw6 = currStep.pred_6h;
    const raw12 = currStep.pred_12h;
    const raw24 = currStep.pred_24h;

    const ema6 = currStep.ema_6h;
    const ema12 = currStep.ema_12h;
    const ema24 = currStep.ema_24h;

    const mainP6 = isEma ? ema6 : raw6;
    const mainP12 = isEma ? ema12 : raw12;
    const mainP24 = isEma ? ema24 : raw24;

    if (viewMode === "realtime") {
      // A. +24h Forecast Active Corridor Highlight
      if (future24Idx > safeCurrentStep) {
        const future24X = getX(future24Idx);

        // Corridor gradient fill
        const grad = ctx.createLinearGradient(nowX, 0, future24X, 0);
        grad.addColorStop(0, "rgba(56, 189, 248, 0.08)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0.02)");
        ctx.fillStyle = grad;
        ctx.fillRect(nowX, padTop, future24X - nowX, chartH);

        // Right boundary of corridor
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(future24X, padTop);
        ctx.lineTo(future24X, padTop + chartH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Horizon tag
        ctx.fillStyle = "rgba(56, 189, 248, 0.7)";
        ctx.font = "8.5px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("+24h WINDOW", (nowX + future24X) / 2, padTop + 12);
      }

      // 0a. Full Lifecycle Ground-Truth Envelope (Faint Dashed White)
      ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.observed_kt);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // B. Observed Past Intensity Path (Solid Crisp White)
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i <= safeCurrentStep; i++) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.observed_kt);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // C. Observed Past Points
      for (let i = 0; i <= safeCurrentStep; i += Math.max(1, Math.floor(N / 20))) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.observed_kt);
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }

      // D. Ground-Truth Actual Outcome Corridor (Next 24h - Dashed Red)
      ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = safeCurrentStep; i <= future24Idx; i++) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.observed_kt);
        if (i === safeCurrentStep) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Primary Forecast Vector (Sea Blue #4988C4)
      ctx.strokeStyle = "#4988C4";
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(nowX, nowY);
      ctx.lineTo(getX(p6Idx), getY(mainP6));
      ctx.lineTo(getX(p12Idx), getY(mainP12));
      ctx.lineTo(getX(p24Idx), getY(mainP24));
      ctx.stroke();

      // Horizon prediction dots with badges (Cold Ice #BDE8F5)
      const horizonPts = [
        { idx: p6Idx, val: mainP6, lbl: "+6h" },
        { idx: p12Idx, val: mainP12, lbl: "+12h" },
        { idx: p24Idx, val: mainP24, lbl: "+24h" },
      ];

      horizonPts.forEach((pt) => {
        const px = getX(pt.idx);
        const py = getY(pt.val);

        // Glowing dot
        ctx.fillStyle = "#BDE8F5";
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "rgba(73, 136, 196, 0.4)";
        ctx.lineWidth = 3;
        ctx.stroke();

        // Value text
        ctx.fillStyle = "#BDE8F5";
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`${pt.lbl}: ${Math.round(pt.val)}k`, px + 6, py - 4);
      });
    } else {
      // ==========================================
      // Full Lifecycle Audit View
      // ==========================================
      // 1. Observed Best Track across all timesteps (Solid White)
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.observed_kt);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // 2. Predicted +24h Trajectory across Full Lifecycle (Sea Blue #4988C4)
      ctx.strokeStyle = "#4988C4";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const val = isEma ? lifecycle[i]!.ema_24h : lifecycle[i]!.pred_24h;
        const x = getX(i);
        const y = getY(val);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // 3. Actual +24h Ground Truth Target Path (Dashed Red)
      ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const x = getX(i);
        const y = getY(lifecycle[i]!.actual_plus_24h);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // "NOW" Vertical Timeline Marker Line (Cold Ice #BDE8F5 / Sea Blue #4988C4)
    ctx.strokeStyle = "#4988C4";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(nowX, padTop);
    ctx.lineTo(nowX, padTop + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Observation Beacon at NOW
    ctx.fillStyle = saffirColor(currStep.observed_kt);
    ctx.beginPath();
    ctx.arc(nowX, nowY, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.stroke();

    // NOW Pill Label at top
    ctx.fillStyle = "#BDE8F5";
    ctx.font = "bold 9px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("NOW", nowX, padTop - 5);

    // Interactive Hover Indicator
    if (hoveredIdx !== null && hoveredIdx >= 0 && hoveredIdx < N && hoveredIdx !== safeCurrentStep) {
      const hovX = getX(hoveredIdx);
      const hovItem = lifecycle[hoveredIdx]!;
      const hovY = getY(hovItem.observed_kt);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hovX, padTop);
      ctx.lineTo(hovX, padTop + chartH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#FFFFFF";
      ctx.beginPath();
      ctx.arc(hovX, hovY, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    ctx.restore();
  }, [lifecycle, safeCurrentStep, viewMode, smoothingMode, hoveredIdx, N]);

  // Handle Resize and render
  useEffect(() => {
    renderChart();

    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new ResizeObserver(() => {
      renderChart();
    });
    observer.observe(canvas);

    return () => observer.disconnect();
  }, [renderChart]);

  // Mouse interaction for seeking and inspection
  const handlePointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || N === 0) return;
    const rect = canvas.getBoundingClientRect();
    const padLeft = 46;
    const padRight = 32;
    const chartW = rect.width - padLeft - padRight;
    const relX = e.clientX - rect.left - padLeft;
    const frac = Math.max(0, Math.min(1, relX / chartW));
    const targetIdx = Math.round(frac * (N - 1));

    setHoveredIdx(targetIdx);

    if (isDraggingRef.current && onStepChange) {
      onStepChange(targetIdx);
    }
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    handlePointer(e);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (isDraggingRef.current) {
      isDraggingRef.current = false;
      try {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // Pointer capture release safety
      }
    }
  };

  const activeHoverItem = hoveredIdx !== null ? lifecycle[hoveredIdx] : lifecycle[safeCurrentStep];

  return (
    <section
      ref={containerRef}
      className="flex h-full flex-col border border-[#1C4D8D] bg-panel shadow-md transition-all"
    >
      {/* Workstation Header and Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1C4D8D] bg-[#1C4D8D]/40 px-4 py-2.5">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-[#BDE8F5] animate-pulse" />
            <h2 className="text-xs font-semibold tracking-wider uppercase text-[#BDE8F5] font-mono">
              Intensity Evolution &amp; Operational Forecast Window
            </h2>
          </div>
        </div>

        {/* Control Toggles */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Mode Toggle: Realtime vs Audit */}
          <div className="flex overflow-hidden rounded border border-[#1C4D8D] bg-[#0A1C3C] p-0.5">
            <button
              type="button"
              onClick={() => setViewMode("realtime")}
              className={`px-2.5 py-1 text-[11px] font-mono transition-colors rounded-sm ${
                viewMode === "realtime"
                  ? "bg-[#4988C4] text-[#060D1A] font-bold shadow-xs"
                  : "text-[#8CB8E8] hover:text-[#EEF8FC]"
              }`}
            >
              Real-Time Operations
            </button>
            <button
              type="button"
              onClick={() => setViewMode("audit")}
              className={`px-2.5 py-1 text-[11px] font-mono transition-colors rounded-sm ${
                viewMode === "audit"
                  ? "bg-[#4988C4] text-[#060D1A] font-bold shadow-xs"
                  : "text-[#8CB8E8] hover:text-[#EEF8FC]"
              }`}
            >
              Full Lifecycle Audit
            </button>
          </div>

          {/* Smoothing Toggle: Raw vs EMA */}
          <div className="flex overflow-hidden rounded border border-[#1C4D8D] bg-[#0A1C3C] p-0.5">
            <button
              type="button"
              onClick={() => setSmoothingMode("raw")}
              className={`px-2.5 py-1 text-[11px] font-mono transition-colors rounded-sm ${
                smoothingMode === "raw"
                  ? "bg-[#4988C4] text-[#060D1A] font-bold shadow-xs"
                  : "text-[#8CB8E8] hover:text-[#EEF8FC]"
              }`}
            >
              Raw Model
            </button>
            <button
              type="button"
              onClick={() => setSmoothingMode("ema")}
              className={`px-2.5 py-1 text-[11px] font-mono transition-colors rounded-sm ${
                smoothingMode === "ema"
                  ? "bg-[#4988C4] text-[#060D1A] font-bold shadow-xs"
                  : "text-[#8CB8E8] hover:text-[#EEF8FC]"
              }`}
            >
              EMA Smoothed (α=0.35)
            </button>
          </div>
        </div>
      </div>

      {/* Legend Banner */}
      <div className="flex flex-wrap items-center justify-between border-b border-[#1C4D8D] bg-[#0A1C3C]/80 px-4 py-1.5 text-[11px]">
        <div className="flex flex-wrap items-center gap-4">
          <LegendSwatch color="#FFFFFF" label="Observed Past" solid />
          <LegendSwatch color="#4988C4" label="Model Forecast (+24h)" solid />
          <LegendSwatch color="#EF4444" label="Ground-Truth Verification (+24h)" dashed />
          {viewMode === "realtime" && (
            <LegendSwatch color="rgba(255, 255, 255, 0.4)" label="Truth Envelope" dashed />
          )}
          {viewMode === "realtime" && (
            <span className="flex items-center gap-1.5 text-muted-foreground font-mono text-[10px]">
              <span className="inline-block h-2 w-3 rounded-xs bg-cyan-400/20 border border-cyan-400/40" />
              +24h Active Corridor
            </span>
          )}
        </div>

        {/* Live Scrub/Hover Telemetry HUD */}
        {activeHoverItem && (() => {
          const forecast24 = smoothingMode === "ema" ? activeHoverItem.ema_24h : activeHoverItem.pred_24h;
          const observed24 = activeHoverItem.actual_plus_24h;
          const err = forecast24 - observed24;
          return (
            <div className="flex flex-wrap items-center gap-2.5 font-mono text-[11px]">
              <span className="rounded-xs bg-white/10 px-1.5 py-0.5 text-white font-semibold">
                CURRENT OBS (T+{activeHoverItem.elapsed_hours}h): {activeHoverItem.observed_kt} kt
              </span>
              <span className="rounded-xs bg-cyan-500/15 border border-cyan-500/30 px-1.5 py-0.5 text-cyan-300 font-semibold">
                +24h FORECAST: {forecast24} kt
              </span>
              <span className="rounded-xs bg-red-500/15 border border-red-500/30 px-1.5 py-0.5 text-red-300 font-semibold">
                +24h OBSERVED: {observed24} kt
              </span>
              <span className="text-muted-foreground text-[10px]">
                ERROR: {err > 0 ? `+${err}` : err} kt
              </span>
            </div>
          );
        })()}
      </div>

      {/* Canvas Area with high-DPI scaling */}
      <div className="relative min-h-[340px] flex-1 w-full select-none cursor-crosshair">
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointer}
          onPointerUp={handlePointerUp}
          onPointerLeave={() => {
            isDraggingRef.current = false;
            setHoveredIdx(null);
          }}
          className="absolute inset-0 h-full w-full"
        />
      </div>

      {/* Canvas footer tip */}
      <div className="flex items-center justify-between border-t border-hairline/60 px-4 py-1.5 text-[10px] text-muted-foreground font-mono">
        <span>Click or drag on the canvas to scrub through cyclone timesteps</span>
        <span>Hardware-Accelerated 60 FPS Canvas</span>
      </div>
    </section>
  );
}

function LegendSwatch({
  color,
  label,
  solid,
  dashed,
}: {
  color: string;
  label: string;
  solid?: boolean;
  dashed?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5 text-muted-foreground font-mono text-[10px]">
      <span
        className="inline-block h-0 w-4"
        style={{
          borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
        }}
      />
      {label}
    </span>
  );
}
