import { RI_BASE_RATE, type ForecastResponse } from "@/lib/forecast-types";

interface Props {
  data: ForecastResponse;
}

export function VerdictPanel({ data }: Props) {
  const forecast24 = data.forecast["+24h"];
  const observed24 = data.actual_outcome_kt;
  const error24 = forecast24 - observed24;

  const observedDelta = data.actual_outcome_kt - data.current_wind_kt;
  const actuallyRI = observedDelta >= 30;
  const flagged = data.ri_probability >= 0.35;
  const hit = actuallyRI === flagged;
  const color = hit ? "var(--risk-low)" : "var(--risk-high)";

  const verdict = actuallyRI
    ? `Best-track analysis shows ${data.storm_name} gained ${observedDelta} kt over the following 24 hours — an RI event by the 30 kt threshold.`
    : `Best-track analysis shows a ${observedDelta >= 0 ? "gain" : "loss"} of ${Math.abs(
        observedDelta,
      )} kt over the following 24 hours — below the 30 kt RI threshold.`;

  const model = flagged
    ? `The model issued an RI signal of ${(data.ri_probability * 100).toFixed(1)}% (${(
        data.ri_probability / RI_BASE_RATE
      ).toFixed(1)}× climatology) at this analysis time.`
    : `The model held RI probability at ${(data.ri_probability * 100).toFixed(
        1,
      )}%, consistent with a non-RI period.`;

  return (
    <section className="border border-hairline bg-panel" style={{ borderLeft: `3px solid ${color}` }}>
      <div className="flex items-baseline justify-between border-b border-hairline px-5 py-2.5">
        <h2 className="text-xs font-semibold tracking-wider uppercase font-mono text-foreground/90">
          VERIFICATION (+24h OUTCOME)
        </h2>
        <span className="readout text-[11px] font-mono font-semibold" style={{ color }}>
          {hit ? "Forecast verified" : "Forecast missed"}
        </span>
      </div>

      {/* Explicit +24h Verification Breakdown */}
      <div className="border-b border-hairline bg-panel-raised/50 px-5 py-3 grid grid-cols-3 gap-4 font-mono text-xs">
        <div>
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider block">
            +24h Forecast
          </span>
          <span className="text-lg font-semibold text-cyan-400">
            {forecast24} kt
          </span>
        </div>
        <div>
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider block">
            +24h Observed
          </span>
          <span className="text-lg font-semibold text-white">
            {observed24} kt
          </span>
        </div>
        <div>
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider block">
            Forecast Error
          </span>
          <span
            className={`text-lg font-semibold ${
              Math.abs(error24) <= 10 ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {error24 > 0 ? `+${error24}` : error24} kt
          </span>
        </div>
      </div>

      <div className="space-y-2 px-5 py-4 text-[13px] leading-relaxed text-foreground/80">
        <p>{model}</p>
        <p>{verdict}</p>
        <p className="text-muted-foreground text-[11px]">
          Verification is retrospective against HURDAT2/JTWC best track for this held-out case;
          the operational threshold used here for a positive RI call is p ≥ 0.35.
        </p>
      </div>
    </section>
  );
}
