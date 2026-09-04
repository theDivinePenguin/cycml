interface HeaderBarProps {
  status: "Operational" | "Offline";
}

const MODEL_META = [
  { k: "Architecture", v: "CycML (ResNet-18 + Temporal Transformer · K=7 Env Fusion)" },
  { k: "Global Basins", v: "WPAC · ATLN · IO · EPAC · SH" },
  { k: "Trend Accuracy", v: "87.3%" },
  { k: "Benchmark Split", v: "14 Showcase Cyclones (Held-Out)" },
];

export function HeaderBar({ status }: HeaderBarProps) {
  const live = status === "Operational";
  return (
    <header className="border-b border-[#1C4D8D] bg-panel/90 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-8 gap-y-4 px-6 py-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-sans text-2xl font-semibold tracking-tight text-white flex items-center gap-2.5">
              CycML{" "}
              <span className="text-[#BDE8F5] font-mono text-xs tracking-wider border border-[#4988C4] px-1.5 py-0.5 rounded-xs bg-[#1C4D8D]/60 font-semibold shadow-xs">
                OPERATIONAL CONSOLE
              </span>
            </h1>
            <span className="readout rounded-xs border border-[#1C4D8D] px-1.5 py-0.5 text-[11px] text-[#8CB8E8] font-mono bg-[#0A1C3C]">
              SIH 26070
            </span>
          </div>
          <p className="mt-1 max-w-xl text-sm text-[#8CB8E8]">
            Multi-modal deep learning tropical cyclone rapid intensification warning &amp; 24-hour intensity trajectory forecasting.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <dl className="flex flex-wrap gap-x-6 gap-y-2">
            {MODEL_META.map((m) => (
              <div key={m.k}>
                <dt className="text-[11px] font-medium tracking-wide text-[#8CB8E8] uppercase font-mono">
                  {m.k}
                </dt>
                <dd className="readout text-xs text-[#EEF8FC] font-semibold">{m.v}</dd>
              </div>
            ))}
          </dl>
          <div className="flex items-center gap-2 border-l border-[#1C4D8D] pl-6">
            <span
              className="relative flex h-2 w-2"
              style={{ color: live ? "var(--risk-low)" : "var(--destructive)" }}
            >
              {live && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-60" />
              )}
              <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
            </span>
            <span className="readout text-xs text-[#EEF8FC]">Model {status.toLowerCase()}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
