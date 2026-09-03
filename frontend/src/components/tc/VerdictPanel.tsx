import { RI_BASE_RATE, type ForecastResponse } from "@/lib/forecast-types";

interface Props {
  data: ForecastResponse;
}

export function VerdictPanel({ data }: Props) {
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
        <h2 className="text-sm font-semibold tracking-tight">Operational verdict</h2>
        <span className="readout text-[11px]" style={{ color }}>
          {hit ? "Forecast verified" : "Forecast missed"}
        </span>
      </div>
      <div className="space-y-2 px-5 py-4 text-[13px] leading-relaxed text-foreground/80">
        <p>{model}</p>
        <p>{verdict}</p>
        <p className="text-muted-foreground">
          Verification is retrospective against HURDAT2/JTWC best track for this held-out case;
          the operational threshold used here for a positive RI call is p ≥ 0.35.
        </p>
      </div>
    </section>
  );
}
