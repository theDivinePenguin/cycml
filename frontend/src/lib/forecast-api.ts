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

export interface ModelOption {
  id: string;
  category?: string;
  name: string;
  badge: string;
  tag: string;
  lead_mae: string;
  ri_mae: string;
  ri_precision: string;
  slope: string;
  modalities?: string[];
}

interface MultiModelPayload {
  models: ModelOption[];
  storms: Record<string, Record<string, RawStorm>>;
}

type StormDataMap = Record<string, RawStorm>;

const rawPayload = stormDataRaw as unknown as (MultiModelPayload | StormDataMap);
const isMultiModel = "models" in rawPayload && "storms" in rawPayload;

export const AVAILABLE_MODELS: ModelOption[] = isMultiModel
  ? (rawPayload as MultiModelPayload).models
  : [
      {
        id: "default",
        category: "Production",
        name: "Active Model",
        badge: "Standard",
        tag: "Default",
        lead_mae: "",
        ri_mae: "",
        ri_precision: "",
        slope: "",
        modalities: ["IR1 Thermal Infrared", "Atmospheric SHIPS Reanalysis"],
      },
    ];

export const DEFAULT_MODEL_ID = AVAILABLE_MODELS.some(m => m.id === "exp2_ultra")
  ? "exp2_ultra"
  : (AVAILABLE_MODELS[0]?.id || "default");

const ALL_STORMS: Record<string, StormDataMap> = isMultiModel
  ? (rawPayload as MultiModelPayload).storms
  : { default: rawPayload as StormDataMap };

const defaultStorms = ALL_STORMS[DEFAULT_MODEL_ID] || Object.values(ALL_STORMS)[0] || {};

function formatTimestamp(ts: string): string {
  if (!ts) return new Date().toISOString();
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

export const STORMS: StormOption[] = Object.keys(defaultStorms).map((id) => {
  const s = defaultStorms[id]!;
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

export function buildForecastFromRealData(
  stormId: string,
  stepIdx: number,
  modelId: string = DEFAULT_MODEL_ID,
): ForecastResponse {
  const modelStorms = ALL_STORMS[modelId] || ALL_STORMS[DEFAULT_MODEL_ID] || Object.values(ALL_STORMS)[0]!;
  const storm = modelStorms[stormId] ?? modelStorms[Object.keys(modelStorms)[0]!]!;
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
    environmental: { sst: 29.0, ohc: 65, shear: 8.5, rh: 72, mslp: 998 },
  };

  const rawPred6 = timesteps.map((t) => t.predicted_plus_6h);
  const rawPred12 = timesteps.map((t) => t.predicted_plus_12h);
  const rawPred24 = timesteps.map((t) => t.predicted_plus_24h);

  const ema6 = calcEma(rawPred6);
  const ema12 = calcEma(rawPred12);
  const ema24 = calcEma(rawPred24);

  const lifecycle = timesteps.map((t, idx) => ({
    step_index: idx,
    elapsed_hours: t.elapsed_hours,
    observed_kt: t.vmax_curr,
    actual_plus_24h: t.vmax_plus_24h,
    pred_6h: t.predicted_plus_6h,
    pred_12h: t.predicted_plus_12h,
    pred_24h: t.predicted_plus_24h,
    ema_6h: Math.round(ema6[idx]! * 10) / 10,
    ema_12h: Math.round(ema12[idx]! * 10) / 10,
    ema_24h: Math.round(ema24[idx]! * 10) / 10,
  }));

  const timeline = [
    { t: -12, observed_kt: timesteps[Math.max(0, safeIdx - 4)]?.vmax_curr ?? curr.vmax_curr, predicted_kt: timesteps[Math.max(0, safeIdx - 4)]?.vmax_curr ?? curr.vmax_curr },
    { t: -6, observed_kt: timesteps[Math.max(0, safeIdx - 2)]?.vmax_curr ?? curr.vmax_curr, predicted_kt: timesteps[Math.max(0, safeIdx - 2)]?.vmax_curr ?? curr.vmax_curr },
    { t: 0, observed_kt: curr.vmax_curr, predicted_kt: curr.vmax_curr },
    { t: 6, observed_kt: timesteps[Math.min(timesteps.length - 1, safeIdx + 2)]?.vmax_curr ?? curr.vmax_curr, predicted_kt: curr.predicted_plus_6h },
    { t: 12, observed_kt: timesteps[Math.min(timesteps.length - 1, safeIdx + 4)]?.vmax_curr ?? curr.vmax_curr, predicted_kt: curr.predicted_plus_12h },
    { t: 24, observed_kt: curr.vmax_plus_24h, predicted_kt: curr.predicted_plus_24h },
  ];

  const modelMeta = AVAILABLE_MODELS.find((m) => m.id === modelId) ?? AVAILABLE_MODELS[0];

  return {
    storm_name: storm.name,
    timestamp: formatTimestamp(curr.timestamp),
    current_wind_kt: curr.vmax_curr,
    category: curr.category || categoryLabel(curr.vmax_curr),
    coordinates: {
      lat: curr.latitude,
      lon: curr.longitude,
    },
    environmental: curr.environmental,
    trend: normalizeTrend(curr.predicted_trend),
    ri_probability: curr.ri_probability / 100.0,
    forecast: {
      "+6h": curr.predicted_plus_6h,
      "+12h": curr.predicted_plus_12h,
      "+24h": curr.predicted_plus_24h,
    },
    timeline,
    actual_outcome_kt: curr.vmax_plus_24h,
    lifecycle,
    model_id: modelId,
    model_name: modelMeta?.name ?? modelId,
    active_modalities: modelMeta?.modalities ?? ["IR1 Thermal Infrared (10.8 µm)", "Temporal Sequence History"],
  };
}

export async function fetchForecast(
  stormId: string,
  stepIdx: number,
  modelId: string = DEFAULT_MODEL_ID,
): Promise<ForecastResponse> {
  return Promise.resolve(buildForecastFromRealData(stormId, stepIdx, modelId));
}
