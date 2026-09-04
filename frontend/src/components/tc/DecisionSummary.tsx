import { Minus, Zap, TrendingDown, TrendingUp } from "lucide-react";
import type { ForecastResponse } from "@/lib/forecast-types";
import { saffirColor } from "./scale";

interface Props {
  data: ForecastResponse;
}

export function DecisionSummary({ data }: Props) {
  const currentKt = data.current_wind_kt;
  const forecast24Kt = data.forecast["+24h"];
  const delta24 = forecast24Kt - currentKt;
  const probPercent = (data.ri_probability * 100).toFixed(1);

  // Decision classification
  const isRI = data.ri_probability >= 0.35 || delta24 >= 30;

  let headline = "Stable intensity expected";
  let badgeColor = "var(--risk-low)";
  let TrendIcon = Minus;

  if (isRI) {
    headline = "Rapid Intensification expected";
    badgeColor = "var(--risk-critical)";
    TrendIcon = Zap;
  } else if (delta24 >= 10 || data.trend === "Intensifying") {
    headline = "Intensification expected";
    badgeColor = "var(--risk-elevated)";
    TrendIcon = TrendingUp;
  } else if (delta24 <= -10 || data.trend === "Weakening") {
    headline = "Weakening expected";
    badgeColor = "var(--risk-low)";
    TrendIcon = TrendingDown;
  }

  return (
    <section
      className="relative overflow-hidden border border-hairline bg-panel"
      style={{
        borderLeft: `3px solid ${badgeColor}`,
        background: `linear-gradient(135deg, color-mix(in oklab, ${badgeColor} 6%, var(--panel)) 0%, var(--panel) 100%)`,
      }}
    >
      <div className="flex items-center justify-between border-b border-hairline bg-panel-raised/60 px-4 py-2">
        <h2 className="flex items-center gap-1.5 font-mono text-xs font-semibold tracking-wider text-foreground uppercase">
          <TrendIcon className="h-3.5 w-3.5" style={{ color: badgeColor }} />
          24-HOUR OUTLOOK
        </h2>
        <span
          className="rounded-xs px-2 py-0.5 font-mono text-[10px] font-semibold"
          style={{
            background: `color-mix(in oklab, ${badgeColor} 12%, transparent)`,
            color: badgeColor,
            border: `1px solid color-mix(in oklab, ${badgeColor} 30%, transparent)`,
          }}
        >
          {data.trend.toUpperCase()}
        </span>
      </div>

      <div className="space-y-3.5 p-4 font-mono">
        <div>
          <p className="text-base font-semibold tracking-tight" style={{ color: badgeColor }}>
            {headline}
          </p>
          <div className="mt-1.5 flex flex-wrap items-baseline gap-2">
            <span className="text-2xl font-semibold" style={{ color: saffirColor(currentKt) }}>
              {currentKt}
            </span>
            <span className="text-sm text-muted-foreground">→</span>
            <span className="text-2xl font-semibold" style={{ color: saffirColor(forecast24Kt) }}>
              {forecast24Kt} kt
            </span>
            <span
              className={`ml-1 rounded-xs px-1.5 py-0.5 text-xs font-semibold ${
                delta24 > 0
                  ? "border border-emerald-600/30 bg-emerald-50 text-emerald-800"
                  : delta24 < 0
                    ? "border border-blue-600/30 bg-blue-50 text-blue-800"
                    : "border border-hairline bg-muted text-muted-foreground"
              }`}
            >
              {delta24 > 0 ? `+${delta24}` : delta24} kt / 24h
            </span>
          </div>
        </div>

        {/* 3 Model Outputs At A Glance */}
        <div className="grid grid-cols-3 gap-2 border-t border-hairline/60 pt-3 text-[11px]">
          <div>
            <span className="block text-[10px] text-muted-foreground">TREND</span>
            <span className="font-semibold text-foreground">{data.trend}</span>
          </div>
          <div>
            <span className="block text-[10px] text-muted-foreground">Δ24h FORECAST</span>
            <span className="font-semibold text-primary">
              {delta24 > 0 ? `+${delta24}` : delta24} kt
            </span>
          </div>
          <div>
            <span className="block text-[10px] text-muted-foreground">RI PROBABILITY</span>
            <span
              className="font-semibold"
              style={{ color: isRI ? "var(--risk-critical)" : "var(--risk-low)" }}
            >
              {probPercent}%
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
