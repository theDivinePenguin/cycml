import { Wind } from "lucide-react";
import type { ForecastResponse } from "@/lib/forecast-types";
import { formatCoords, formatUTC, saffirAbbrev, saffirColor } from "./scale";

interface Props {
  data: ForecastResponse;
}

function envRows(e: ForecastResponse["environmental"]) {
  return [
    {
      k: "Sea surface temp",
      v: `${e.sst.toFixed(1)} °C`,
      tag: e.sst >= 29.5 ? "Super-warm" : e.sst >= 26.5 ? "Supportive" : "Marginal",
      good: e.sst >= 26.5,
    },
    {
      k: "Ocean heat content",
      v: `${e.ohc} kJ cm⁻²`,
      tag: e.ohc >= 80 ? "Deep warm layer" : e.ohc >= 50 ? "Adequate" : "Shallow",
      good: e.ohc >= 50,
    },
    {
      k: "Deep-layer shear",
      v: `${e.shear.toFixed(1)} kt`,
      tag: e.shear <= 5 ? "Low shear" : e.shear <= 12 ? "Moderate" : "Hostile",
      good: e.shear <= 12,
    },
    {
      k: "Mid-level RH (700–500 hPa)",
      v: `${e.rh}%`,
      tag: e.rh >= 70 ? "Moist" : e.rh >= 55 ? "Mixed" : "Dry intrusion",
      good: e.rh >= 55,
    },
    {
      k: "Central pressure",
      v: `${e.mslp} hPa`,
      tag: e.mslp <= 940 ? "Deepening core" : e.mslp <= 985 ? "Organized" : "Weak core",
      good: true,
    },
  ];
}

export function StormStatePanel({ data }: Props) {
  const color = saffirColor(data.current_wind_kt);

  return (
    <section className="flex h-full flex-col border border-hairline bg-panel shadow-xs">
      <div className="flex items-baseline justify-between border-b border-hairline bg-panel-raised px-5 py-2.5">
        <h2 className="text-xs font-semibold tracking-wider uppercase font-mono text-[#355872]">
          CURRENT OBSERVATION
        </h2>
        <span className="readout text-[11px] text-muted-foreground font-mono font-medium">
          {formatUTC(data.timestamp)}
        </span>
      </div>

      <div className="border-b border-hairline px-5 py-5" style={{ borderLeft: `3px solid ${color}` }}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Wind className="h-3.5 w-3.5" />
          <span className="text-xs font-semibold text-[#355872]/80">Maximum sustained wind (1-min)</span>
        </div>
        <div className="mt-1 flex items-end gap-2">
          <span className="readout text-6xl leading-none font-bold text-[#355872]">
            {data.current_wind_kt}
          </span>
          <span className="readout mb-1 text-lg text-muted-foreground font-semibold">kt</span>
          <span
            className="mb-1.5 ml-2 rounded-xs px-2 py-0.5 text-xs font-semibold font-mono border border-[#7AAACE]/40"
            style={{ background: "#9CD5FF", color: "#355872" }}
          >
            {saffirAbbrev(data.current_wind_kt)}
          </span>
        </div>
        <p className="mt-2 text-sm text-[#355872] font-semibold">{data.category}</p>
        <p className="readout mt-2 text-xs text-muted-foreground font-medium">
          {formatCoords(data.coordinates.lat, data.coordinates.lon)} · {data.storm_name}
        </p>
      </div>

      <div className="px-5 py-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider font-mono text-[#355872]/80">
          Environmental diagnostics
        </h3>
        <dl>
          {envRows(data.environmental).map((row) => (
            <div
              key={row.k}
              className="flex items-baseline justify-between gap-3 border-b border-hairline/60 py-2 last:border-0"
            >
              <dt className="text-[13px] text-[#355872] font-medium leading-tight">{row.k}</dt>
              <dd className="flex items-baseline gap-3">
                <span className="readout text-[13px] whitespace-nowrap text-[#355872] font-semibold">{row.v}</span>
                <span
                  className="w-[7.5rem] text-right text-[11px] font-mono font-semibold"
                  style={{ color: row.good ? "#355872" : "#A33B2E" }}
                >
                  {row.tag}
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
