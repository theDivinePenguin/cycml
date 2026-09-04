import { saffirKey, type SaffirKey, type RiskTier } from "@/lib/forecast-types";

/** CSS custom-property reference for a wind speed's Saffir–Simpson bucket. */
export function saffirColor(windKt: number): string {
  const map: Record<SaffirKey, string> = {
    td: "var(--saffir-td)",
    ts: "var(--saffir-ts)",
    cat1: "var(--saffir-cat1)",
    cat2: "var(--saffir-cat2)",
    cat3: "var(--saffir-cat3)",
    cat4: "var(--saffir-cat4)",
    cat5: "var(--saffir-cat5)",
  };
  return map[saffirKey(windKt)];
}

/** Short product-style abbreviation, e.g. "TS", "CAT 4". */
export function saffirAbbrev(windKt: number): string {
  const map: Record<SaffirKey, string> = {
    td: "TD",
    ts: "TS",
    cat1: "Cat 1",
    cat2: "Cat 2",
    cat3: "Cat 3",
    cat4: "Cat 4",
    cat5: "Cat 5",
  };
  return map[saffirKey(windKt)];
}

export function riskColor(tier: RiskTier): string {
  const map: Record<RiskTier, string> = {
    "Low Risk": "var(--risk-low)",
    "Elevated Risk": "var(--risk-elevated)",
    "High Risk": "var(--risk-high)",
    "Critical Risk": "var(--risk-critical)",
  };
  return map[tier];
}

export function formatUTC(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(
    d.getUTCHours(),
  )}${pad(d.getUTCMinutes())}Z`;
}

export function formatCoords(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(1)}°${ns}  ${Math.abs(lon).toFixed(1)}°${ew}`;
}
