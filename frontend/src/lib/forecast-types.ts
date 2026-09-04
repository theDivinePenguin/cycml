export interface ForecastResponse {
  storm_name: string;
  timestamp: string;
  current_wind_kt: number;
  category: string;
  coordinates: { lat: number; lon: number };
  environmental: {
    sst: number;
    ohc: number;
    shear: number;
    rh: number;
    mslp: number;
  };
  trend: "Weakening" | "Stable" | "Intensifying";
  ri_probability: number;
  forecast: { "+6h": number; "+12h": number; "+24h": number };
  timeline: Array<{ t: number; observed_kt: number; predicted_kt: number }>;
  actual_outcome_kt: number;
  lifecycle?: Array<{
    step_index: number;
    elapsed_hours: number;
    observed_kt: number;
    actual_plus_24h: number;
    pred_6h: number;
    pred_12h: number;
    pred_24h: number;
    ema_6h: number;
    ema_12h: number;
    ema_24h: number;
  }>;
}

export interface StormOption {
  id: string;
  label: string;
  basin: string;
  season: number;
  /** number of 6-hourly synoptic steps available for this storm */
  steps: number;
}

/** Dataset-wide base rate of rapid intensification (SHIPS/HURDAT2 climatology). */
export const RI_BASE_RATE = 0.068;

export type RiskTier = "Low Risk" | "Elevated Risk" | "High Risk" | "Critical Risk";

export function riskTier(multiplier: number): RiskTier {
  if (multiplier < 1.5) return "Low Risk";
  if (multiplier < 3) return "Elevated Risk";
  if (multiplier < 6) return "High Risk";
  return "Critical Risk";
}

/** Saffir–Simpson bucket keys used for the intensity color scale. */
export type SaffirKey = "td" | "ts" | "cat1" | "cat2" | "cat3" | "cat4" | "cat5";

export function saffirKey(windKt: number): SaffirKey {
  if (windKt < 34) return "td";
  if (windKt < 64) return "ts";
  if (windKt < 83) return "cat1";
  if (windKt < 96) return "cat2";
  if (windKt < 113) return "cat3";
  if (windKt < 137) return "cat4";
  return "cat5";
}

export function categoryLabel(windKt: number): string {
  const map: Record<SaffirKey, string> = {
    td: "Tropical Depression",
    ts: "Tropical Storm",
    cat1: "Category 1 Hurricane",
    cat2: "Category 2 Hurricane",
    cat3: "Category 3 Major Hurricane",
    cat4: "Category 4 Major Hurricane",
    cat5: "Category 5 Super Typhoon",
  };
  return map[saffirKey(windKt)];
}
