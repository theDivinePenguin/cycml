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
  const multiplier = prob / RI_BASE_RATE;
  const tier = riskTier(multiplier);
  const color = riskColor(tier);
  const TrendIcon = TREND_ICON[data.trend];

  return (
    <section
      className="relative border border-hairline bg-panel"
      style={{
        borderTop: `3px solid ${color}`,
        background: `linear-gradient(180deg, color-mix(in oklab, ${color} 12%, var(--panel)) 0%, var(--panel) 60%)`,
      }}
    >
      <div className="flex items-baseline justify-between border-b border-hairline px-6 py-2.5">
        <h2 className="text-sm font-semibold tracking-tight">
          24-hour rapid intensification guidance
        </h2>
        <span className="readout text-[11px] text-muted-foreground">
          ≥30 kt / 24 h · valid from analysis time
        </span>
      </div>

      <div className="grid gap-6 px-6 py-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div>
          <p className="text-xs text-muted-foreground">Macro intensity trend</p>
          <div className="mt-1 flex items-center gap-2">
            <TrendIcon className="h-7 w-7" style={{ color }} />
            <span
              className="text-4xl leading-none font-semibold tracking-tight"
              style={{ color }}
            >
              {data.trend}
            </span>
          </div>

          <div className="mt-6 flex items-end gap-4">
            <div>
              <p className="text-xs text-muted-foreground">RI probability</p>
              <div className="flex items-end gap-1">
                <span
                  className="readout text-[5.5rem] leading-[0.85] font-semibold"
                  style={{ color }}
                >
                  {(prob * 100).toFixed(1)}
                </span>
                <span className="readout mb-2 text-2xl text-muted-foreground">%</span>
              </div>
            </div>
          </div>

          <div className="mt-4 h-2 w-full max-w-md bg-panel-raised">
            <div
              className="h-full transition-[width] duration-500"
              style={{ width: `${Math.min(100, prob * 100)}%`, background: color }}
            />
          </div>
          <p className="readout mt-2 text-[11px] text-muted-foreground">
            raw model output · not calibrated away · p = {prob.toFixed(3)}
          </p>
        </div>

        <div className="flex flex-col justify-center gap-4 border-hairline lg:border-l lg:pl-6">
          <div className="grid grid-cols-2 gap-3 font-mono">
            <div className="rounded border border-hairline bg-panel-raised/50 p-3">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                CLIMATOLOGICAL BASE RATE
              </p>
              <p className="mt-1 text-2xl font-semibold text-foreground/90 font-mono">
                {(RI_BASE_RATE * 100).toFixed(1)}%
              </p>
              <p className="mt-0.5 text-[10px] text-muted-foreground">SHIPS/HURDAT2</p>
            </div>

            <div className="rounded border border-hairline bg-panel-raised/50 p-3">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                RELATIVE TO BASE RATE
              </p>
              <p className="mt-1 text-2xl font-semibold font-mono" style={{ color }}>
                {multiplier.toFixed(2)}×
              </p>
              <p className="mt-0.5 text-[10px] text-muted-foreground">Ratio p / base</p>
            </div>
          </div>

          <div
            className="inline-flex w-fit items-center gap-2 rounded-xs px-3 py-1 text-xs font-semibold font-mono"
            style={{
              background: `color-mix(in oklab, ${color} 18%, transparent)`,
              color,
              border: `1px solid color-mix(in oklab, ${color} 55%, transparent)`,
            }}
          >
            {tier}
          </div>

          <p className="max-w-sm text-[12px] leading-relaxed text-muted-foreground">
            Probability and multiplier are reported together: the relative departure compares the model's 24-hour RI probability against the {(RI_BASE_RATE * 100).toFixed(1)}% historical climatological base rate.
          </p>
        </div>
      </div>
    </section>
  );
}
