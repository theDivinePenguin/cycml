import type { ModelOption } from "@/lib/forecast-api";
import { Cpu, Layers, Radio, Sparkles } from "lucide-react";

interface HeaderBarProps {
  status: "Operational" | "Offline";
  modelId?: string;
  onModelChange?: (id: string) => void;
  models?: ModelOption[];
}

export function HeaderBar({
  status,
  modelId = "exp2_ultra",
  onModelChange,
  models = [],
}: HeaderBarProps) {
  const live = status === "Operational";
  const activeModel = models.find((m) => m.id === modelId) ?? models[0];

  // Group models by category
  const categories = Array.from(new Set(models.map((m) => m.category || "General")));

  return (
    <header className="border-b border-hairline bg-panel shadow-xs">
      {/* Top Banner */}
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-x-8 gap-y-3 px-6 py-3.5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-sans text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
              DeepCycloNet{" "}
              <span className="text-[#355872] bg-[#9CD5FF]/35 border border-[#7AAACE]/60 font-mono text-[11px] tracking-wider px-2 py-0.5 rounded-xs font-bold">
                OPERATIONAL CONSOLE
              </span>
            </h1>
            <span className="readout rounded-xs border border-hairline bg-[#F7F8F0] px-1.5 py-0.5 text-[10px] text-muted-foreground font-mono font-semibold">
              SIH 26070
            </span>
          </div>
          <p className="mt-0.5 max-w-xl text-xs text-muted-foreground font-medium">
            Multi-modal deep learning tropical cyclone rapid intensification warning &amp; 24-hour intensity trajectory forecasting.
          </p>
        </div>

        {/* Operational Status & Basins */}
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-muted-foreground font-semibold">
            <Radio className="h-3.5 w-3.5 text-[#355872] animate-pulse" />
            <span>GLOBAL BASINS: WPAC · ATLN · IO · EPAC · SH</span>
          </div>

          <div className="flex items-center gap-2 border-l border-hairline pl-4">
            <span
              className="relative flex h-2.5 w-2.5"
              style={{ color: live ? "#355872" : "var(--destructive)" }}
            >
              {live && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#9CD5FF] opacity-75" />
              )}
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-[#355872]" />
            </span>
            <span className="readout text-xs font-bold text-foreground">SYSTEM {status.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* Interactive Model Switcher & Telemetry Bar */}
      <div className="border-t border-hairline bg-[#EAF3FB]/60 px-6 py-2.5">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-4">
          {/* Left: Model Selector Dropdown */}
          <div className="flex items-center gap-3">
            <label
              htmlFor="model-selector"
              className="flex items-center gap-1.5 font-mono text-xs font-bold uppercase tracking-wider text-[#355872]"
            >
              <Cpu className="h-4 w-4 text-[#355872]" />
              <span>Active Model:</span>
            </label>

            <select
              id="model-selector"
              value={modelId}
              onChange={(e) => onModelChange?.(e.target.value)}
              className="rounded-xs border-2 border-[#355872]/40 bg-white px-3 py-1 text-xs font-bold text-[#355872] shadow-xs outline-none transition-all hover:border-[#355872] focus:border-[#355872] focus:ring-2 focus:ring-[#9CD5FF] cursor-pointer"
            >
              {categories.map((cat) => (
                <optgroup key={cat} label={`─── ${cat} ───`}>
                  {models
                    .filter((m) => (m.category || "General") === cat)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} [{m.badge}]
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>

            {activeModel && (
              <span className="hidden sm:inline-flex items-center gap-1 rounded-xs bg-[#355872] px-2 py-0.5 text-[11px] font-bold text-white font-mono shadow-2xs">
                <Sparkles className="h-3 w-3 text-[#9CD5FF]" />
                {activeModel.badge}
              </span>
            )}
          </div>

          {/* Right: Active Model Multi-Modal & Performance Telemetry */}
          {activeModel && (
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 font-mono text-[11px]">
              {activeModel.lead_mae && (
                <div className="flex items-center gap-1.5">
                  <span className="text-muted-foreground font-semibold">LEAD MAE:</span>
                  <span className="font-bold text-[#355872] bg-white px-1.5 py-0.5 rounded-xs border border-hairline">
                    {activeModel.lead_mae}
                  </span>
                </div>
              )}

              {activeModel.ri_mae && (
                <div className="flex items-center gap-1.5">
                  <span className="text-muted-foreground font-semibold">RI PERFORMANCE:</span>
                  <span className="font-bold text-[#355872] bg-white px-1.5 py-0.5 rounded-xs border border-hairline">
                    {activeModel.ri_mae}
                  </span>
                </div>
              )}

              {activeModel.modalities && activeModel.modalities.length > 0 && (
                <div className="hidden lg:flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5 text-[#355872]" />
                  <span className="text-muted-foreground font-semibold">MODALITIES:</span>
                  <div className="flex items-center gap-1">
                    {activeModel.modalities.map((mod, i) => (
                      <span
                        key={i}
                        className="bg-[#9CD5FF]/30 border border-[#7AAACE]/60 px-1.5 py-0.5 rounded-xs text-[10px] font-bold text-[#355872]"
                        title={mod}
                      >
                        {typeof mod === "string" ? mod.split(" ")[0] : String(mod)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
