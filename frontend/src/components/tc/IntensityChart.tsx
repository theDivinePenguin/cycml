import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastResponse } from "@/lib/forecast-types";
import { saffirColor } from "./scale";

interface Props {
  data: ForecastResponse;
  nowHour: number;
}

function smooth(values: (number | null)[]): (number | null)[] {
  return values.map((v, i) => {
    if (v === null) return null;
    const win = [values[i - 1], v, values[i + 1]].filter(
      (x): x is number => typeof x === "number",
    );
    return Math.round(win.reduce((a, b) => a + b, 0) / win.length);
  });
}

export function IntensityChart({ data, nowHour }: Props) {
  const [mode, setMode] = useState<"raw" | "smoothed">("raw");

  const series = useMemo(() => {
    const rawPredicted = data.timeline.map((p) =>
      p.t >= nowHour && p.t <= nowHour + 24 ? p.predicted_kt : null,
    );
    const predicted = mode === "smoothed" ? smooth(rawPredicted) : rawPredicted;
    return data.timeline.map((p, i) => ({
      t: p.t,
      observed: p.t <= nowHour ? p.observed_kt : null,
      predicted: predicted[i],
      verification: p.t >= nowHour ? p.observed_kt : null,
    }));
  }, [data.timeline, nowHour, mode]);

  const current = data.current_wind_kt;

  return (
    <section className="flex h-full flex-col border border-hairline bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-hairline px-5 py-2.5">
        <h2 className="text-sm font-semibold tracking-tight">
          Intensity evolution &amp; forecast window
        </h2>
        <div className="flex items-center gap-4">
          <Legend swatch="var(--saffir-td)" label="Observed best track" />
          <Legend swatch="var(--signal)" label="Model forecast (+24 h)" dashed />
          <Legend swatch="var(--muted-foreground)" label="Verifying analysis" dashed />
          <div className="flex overflow-hidden rounded-xs border border-hairline">
            {(["raw", "smoothed"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`px-2.5 py-1 text-[11px] transition-colors ${
                  mode === m
                    ? "bg-primary text-primary-foreground"
                    : "bg-panel-raised text-muted-foreground hover:text-foreground"
                }`}
              >
                {m === "raw" ? "Raw output" : "Smoothed"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="h-[360px] px-2 py-4">
        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
          <LineChart data={series} margin={{ top: 12, right: 24, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="var(--hairline)" strokeOpacity={0.5} vertical={false} />
            <ReferenceArea
              x1={nowHour}
              x2={nowHour + 24}
              fill="var(--signal)"
              fillOpacity={0.07}
            />
            <XAxis
              dataKey="t"
              tick={{ fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              tickFormatter={(v) => `T+${v}`}
              stroke="var(--hairline)"
            />
            <YAxis
              domain={[0, "dataMax + 20"]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              stroke="var(--hairline)"
              width={44}
              label={{
                value: "kt",
                angle: 0,
                position: "top",
                fill: "var(--muted-foreground)",
                fontSize: 11,
              }}
            />
            {[64, 96, 137].map((th) => (
              <ReferenceLine
                key={th}
                y={th}
                stroke={saffirColor(th)}
                strokeOpacity={0.35}
                strokeDasharray="2 4"
              />
            ))}
            <ReferenceLine
              x={nowHour}
              stroke="var(--foreground)"
              strokeOpacity={0.7}
              label={{
                value: "NOW",
                position: "top",
                fill: "var(--foreground)",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--panel-raised)",
                border: "1px solid var(--hairline)",
                borderRadius: 2,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
              }}
              labelFormatter={(v) => `T+${v} h`}
            />
            <Line
              type="monotone"
              dataKey="verification"
              stroke="var(--muted-foreground)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
              connectNulls
              name="Verifying analysis"
            />
            <Line
              type="monotone"
              dataKey="observed"
              stroke="var(--saffir-td)"
              strokeWidth={2.5}
              dot={false}
              name="Observed"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="var(--signal)"
              strokeWidth={2.5}
              strokeDasharray="5 3"
              dot={{ r: 2.5, fill: "var(--signal)", strokeWidth: 0 }}
              connectNulls
              name="Forecast"
            />
            <ReferenceDot
              x={nowHour}
              y={current}
              r={4}
              fill={saffirColor(current)}
              stroke="var(--background)"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Legend({ swatch, label, dashed }: { swatch: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <span
        className="inline-block h-0 w-5"
        style={{
          borderTop: `2px ${dashed ? "dashed" : "solid"} ${swatch}`,
        }}
      />
      {label}
    </span>
  );
}
