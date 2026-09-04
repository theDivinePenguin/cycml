/**
 * Data access layer.
 *
 * Connected directly to DeepCycloNet real cyclone forecast telemetry and predictions.
 */
import { categoryLabel, type ForecastResponse, type StormOption } from "./forecast-types";
import stormDataRaw from "../data/storm_data.json";

export const API_BASE = import.meta.env["VITE_FORECAST_API"] ?? "/api";

interface RawTimestep {
  step_index: number;
  timestamp: string;
  elapsed_hours: number;
  vmax_curr: number;
  vmax_plus_24h: number;
  actual_delta_24: number;
  actual_trend: string;
  actual_ri: number;
  category: string;
  predicted_trend: string;
  predicted_trend_probs?: {
    WEAKENING: number;
    STABLE: number;
    INTENSIFYING: number;
  };
  ri_probability: number;
  risk_level: string;
  predicted_plus_6h: number;
  predicted_plus_12h: number;
  predicted_plus_24h: number;
  latitude: number;
  longitude: number;
  environmental: {
    sst: number;
    ohc: number;
    shear: number;
    rh: number;
    mslp: number;
  };
}

interface RawStorm {
  id: string;
  name: string;
  basin: string;
  peak_intensity: number;
  category: string;
  split: string;
  description: string;
  n_timesteps: number;
  timesteps: RawTimestep[];
}

type StormDataMap = Record<string, RawStorm>;

const STORMS_DATA: StormDataMap = stormDataRaw as unknown as StormDataMap;

function formatTimestamp(ts: string): string {
  if (!ts) return new Date().toISOString();
  // Format YYYYMMDDHH to ISO 8601
  if (ts.length === 10) {
    const y = ts.substring(0, 4);
    const m = ts.substring(4, 6);
    const d = ts.substring(6, 8);
    const h = ts.substring(8, 10);
    return `${y}-${m}-${d}T${h}:00:00Z`;
  }
  return ts;
}

function parseSeason(id: string, ts?: string): number {
  if (ts && ts.length >= 4) {
    const y = parseInt(ts.substring(0, 4), 10);
    if (!isNaN(y)) return y;
  }
  if (id && id.length >= 4) {
    const y = parseInt(id.substring(0, 4), 10);
    if (!isNaN(y)) return y;
  }
  return 2020;
}

function normalizeTrend(trend: string): "Weakening" | "Stable" | "Intensifying" {
  const upper = (trend || "").toUpperCase();
  if (upper.includes("WEAK")) return "Weakening";
  if (upper.includes("STAB") || upper.includes("STEAD")) return "Stable";
  return "Intensifying";
}

export const STORMS: StormOption[] = Object.keys(STORMS_DATA).map((id) => {
  const s = STORMS_DATA[id]!;
  const firstTs = s.timesteps?.[0]?.timestamp;
  return {
    id: s.id,
    label: `${s.name} — Peak ${s.peak_intensity} kt`,
    basin: s.basin,
    season: parseSeason(s.id, firstTs),
    steps: Math.max(1, (s.timesteps?.length ?? 1) - 1),
  };
});

const EMA_ALPHA = 0.35;

function calcEma(arr: number[], a: number = EMA_ALPHA): number[] {
  const res: number[] = [];
  let s = arr[0] ?? 0;
  for (let i = 0; i < arr.length; i++) {
    s = a * (arr[i] ?? 0) + (1 - a) * s;
    res.push(s);
  }
  return res;
}

export function buildForecastFromRealData(stormId: string, stepIdx: number): ForecastResponse {
  const storm = STORMS_DATA[stormId] ?? STORMS_DATA[Object.keys(STORMS_DATA)[0]!]!;
  const timesteps = storm.timesteps || [];
  const safeIdx = Math.max(0, Math.min(timesteps.length - 1, stepIdx));
  const curr = timesteps[safeIdx] || {
    step_index: 0,
    timestamp: "2010101212",
    elapsed_hours: 0,
    vmax_curr: 35,
    vmax_plus_24h: 35,
    actual_delta_24: 0,
    actual_trend: "STABLE",
    actual_ri: 0,
    category: "Tropical Storm",
    predicted_trend: "STABLE",
    ri_probability: 5,
    risk_level: "LOW",
    predicted_plus_6h: 35,
    predicted_plus_12h: 35,
    predicted_plus_24h: 35,
    latitude: 15.0,
    longitude: 130.0,
    environmental: { sst: 29.0, ohc: 60.0, shear: 10.0, rh: 70.0, mslp: 1000.0 },
  };

  const raw6 = timesteps.map((t) => t.predicted_plus_6h);
  const raw12 = timesteps.map((t) => t.predicted_plus_12h);
  const raw24 = timesteps.map((t) => t.predicted_plus_24h);
  const ema6 = calcEma(raw6);
  const ema12 = calcEma(raw12);
  const ema24 = calcEma(raw24);

  const lifecycle = timesteps.map((st, i) => ({
    step_index: st.step_index,
    elapsed_hours: Math.round(st.elapsed_hours ?? st.step_index * 3),
    observed_kt: Math.round(st.vmax_curr),
    actual_plus_24h: Math.round(st.vmax_plus_24h),
    pred_6h: Math.round(st.predicted_plus_6h),
    pred_12h: Math.round(st.predicted_plus_12h),
    pred_24h: Math.round(st.predicted_plus_24h),
    ema_6h: Math.round(ema6[i] ?? st.predicted_plus_6h),
    ema_12h: Math.round(ema12[i] ?? st.predicted_plus_12h),
    ema_24h: Math.round(ema24[i] ?? st.predicted_plus_24h),
  }));

  const timeline = timesteps.map((st) => ({
    t: Math.round(st.elapsed_hours ?? st.step_index * 3),
    observed_kt: Math.round(st.vmax_curr),
    predicted_kt: Math.round(st.predicted_plus_24h),
  }));

  const windKt = Math.round(curr.vmax_curr);

  return {
    storm_name: storm.name,
    timestamp: formatTimestamp(curr.timestamp),
    current_wind_kt: windKt,
    category: categoryLabel(windKt),
    coordinates: {
      lat: Math.round((curr.latitude ?? 0) * 10) / 10,
      lon: Math.round((curr.longitude ?? 0) * 10) / 10,
    },
    environmental: {
      sst: Math.round((curr.environmental?.sst ?? 28.5) * 10) / 10,
      ohc: Math.round(curr.environmental?.ohc ?? 50),
      shear: Math.round((curr.environmental?.shear ?? 10.0) * 10) / 10,
      rh: Math.round(curr.environmental?.rh ?? 65),
      mslp: Math.round(curr.environmental?.mslp ?? 1005),
    },
    trend: normalizeTrend(curr.predicted_trend),
    ri_probability: Math.round((curr.ri_probability / 100.0) * 1000) / 1000,
    forecast: {
      "+6h": Math.round(curr.predicted_plus_6h),
      "+12h": Math.round(curr.predicted_plus_12h),
      "+24h": Math.round(curr.predicted_plus_24h),
    },
    timeline,
    lifecycle,
    actual_outcome_kt: Math.round(curr.vmax_plus_24h),
  };
}

export async function fetchStorms(): Promise<StormOption[]> {
  return STORMS;
}

/** GET /forecast?storm_id=X&t=Y */
export async function fetchForecast(stormId: string, t: number): Promise<ForecastResponse> {
  return buildForecastFromRealData(stormId, t);
}
