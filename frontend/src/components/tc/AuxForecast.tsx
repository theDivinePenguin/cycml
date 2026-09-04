import type { ForecastResponse } from "@/lib/forecast-types";
import { saffirAbbrev, saffirColor } from "./scale";

interface Props {
  data: ForecastResponse;
}

export function AuxForecast({ data }: Props) {
  const entries = (["+6h", "+12h", "+24h"] as const).map((k) => ({
    k,
    v: data.forecast[k],
    delta: data.forecast[k] - data.current_wind_kt,
  }));

  return (
    <section className="border border-hairline bg-panel shadow-xs">
      <div className="border-b border-hairline bg-panel-raised px-5 py-2.5 flex items-center justify-between">
        <h2 className="text-xs font-semibold tracking-wider uppercase font-mono text-[#355872]">
          MODEL FORECAST
        </h2>
        <span className="readout text-[11px] text-muted-foreground font-mono font-medium">
          HORIZON GUIDANCE
        </span>
      </div>
      <div className="grid grid-cols-3 divide-x divide-hairline">
        {entries.map((e) => (
          <div key={e.k} className="px-5 py-4">
            <p className="readout text-xs text-muted-foreground font-semibold">{e.k}</p>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="readout text-3xl leading-none font-bold text-[#355872]">
                {e.v}
              </span>
              <span className="readout text-xs text-muted-foreground font-semibold">kt</span>
            </div>
            <p className="readout mt-1.5 text-[11px] text-muted-foreground font-mono">
              {e.delta >= 0 ? "+" : ""}
              {e.delta} kt · {saffirAbbrev(e.v)}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
