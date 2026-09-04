interface HeaderBarProps {
  status: "Operational" | "Offline";
}

const MODEL_META = [
  { k: "Architecture", v: "DeepCycloNet (ResNet-18 + GRU · K=7 Env Fusion)" },
  { k: "Global Basins", v: "WPAC · ATLN · IO · EPAC · SH" },
  { k: "Trend Accuracy", v: "87.3%" },
  { k: "Benchmark Split", v: "14 Showcase Cyclones (Held-Out)" },
];

export function HeaderBar({ status }: HeaderBarProps) {
  const live = status === "Operational";
  return (
    <header className="border-b border-hairline bg-panel">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-8 gap-y-4 px-6 py-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-sans text-2xl font-semibold tracking-tight text-foreground flex items-center gap-2">
              DeepCycloNet <span className="text-[#355872] bg-[#9CD5FF]/40 border border-[#7AAACE]/60 font-mono text-xs tracking-wider px-2 py-0.5 rounded-xs font-semibold">OPERATIONAL CONSOLE</span>
            </h1>
            <span className="readout rounded-xs border border-hairline bg-[#F7F8F0] px-1.5 py-0.5 text-[11px] text-muted-foreground font-mono">
              SIH 26070
            </span>
          </div>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            Multi-modal deep learning tropical cyclone rapid intensification warning &amp; 24-hour intensity trajectory forecasting.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <dl className="flex flex-wrap gap-x-6 gap-y-2">
            {MODEL_META.map((m) => (
              <div key={m.k}>
                <dt className="text-[11px] font-medium tracking-wide text-muted-foreground">
                  {m.k}
                </dt>
                <dd className="readout text-xs text-foreground font-semibold">{m.v}</dd>
              </div>
            ))}
          </dl>
          <div className="flex items-center gap-2 border-l border-hairline pl-6">
            <span
              className="relative flex h-2 w-2"
              style={{ color: live ? "#355872" : "var(--destructive)" }}
            >
              {live && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#9CD5FF] opacity-75" />
              )}
              <span className="relative inline-flex h-2 w-2 rounded-full bg-[#355872]" />
            </span>
            <span className="readout text-xs font-medium text-foreground">Model {status.toLowerCase()}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
