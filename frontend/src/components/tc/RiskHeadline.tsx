import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { RI_BASE_RATE, riskTier, type ForecastResponse } from "@/lib/forecast-types";
import { riskColor } from "./scale";

interface Props {
  data: ForecastResponse;
}

const TREND_ICON = {
  Intensifying: ArrowUpRight,
  Stable: Minus,
  Weakening: ArrowDownRight,
} as const;

export function RiskHeadline({ data }: Props) {
  const prob = data.ri_probability;
  const tier = riskTier(prob);
  const color = riskColor(tier);
  const TrendIcon = TREND_ICON[data.trend];
  const climatologyMultiplier = (prob / RI_BASE_RATE).toFixed(1);

  return (
    <section
      className="relative border border-hairline bg-panel shadow-xs"
      style={{
        borderTop: `3px solid ${color}`,
        background: `linear-gradient(180deg, color-mix(in oklab, ${color} 7%, var(--panel)) 0%, var(--panel) 60%)`,
      }}
    >
      <div className="flex items-baseline justify-between border-b border-hairline bg-panel-raised px-6 py-2.5">
        <h2 className="text-xs font-semibold tracking-wider uppercase font-mono text-foreground">
          24-Hour Rapid Intensification Guidance
        </h2>
        <span className="readout text-[11px] text-muted-foreground font-mono">
          Threshold: ≥ 30 kt / 24 h
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-6 px-6 py-4">
        <div className="flex flex-wrap items-center gap-8">
          <div>
            <p className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider font-semibold">
              Macro Intensity Trend
            </p>
            <div className="mt-1 flex items-center gap-2">
              <TrendIcon className="h-5 w-5" style={{ color }} />
              <span
                className="text-2xl font-bold tracking-tight font-mono"
                style={{ color }}
              >
                {data.trend}
              </span>
            </div>
          </div>

          <div className="hidden sm:block h-14 w-px bg-hairline" />

          <div>
            <p className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider font-semibold">
              Predicted RI Probability (≥30 kt / 24h)
            </p>
            <div className="mt-1 flex items-baseline gap-3">
              <span
                className="readout text-6xl sm:text-7xl leading-none font-bold font-mono tracking-tight"
                style={{ color }}
              >
                {(prob * 100).toFixed(1)}%
              </span>
              <div className="flex flex-col text-xs text-muted-foreground font-mono leading-tight">
                <span>(p = {prob.toFixed(3)})</span>
                <span className="font-semibold text-foreground/80 mt-0.5">
                  {climatologyMultiplier}× climatology
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Clean Risk Progress Bar */}
        <div className="min-w-[200px] flex-1 max-w-xs">
          <div className="h-2.5 w-full bg-panel-raised rounded-full overflow-hidden border border-hairline/50">
            <div
              className="h-full transition-[width] duration-500 rounded-full"
              style={{ width: `${Math.min(100, prob * 100)}%`, background: color }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
