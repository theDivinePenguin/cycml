interface HeaderBarProps {
  status: "Operational" | "Offline";
}

const MODEL_META = [
  { k: "Architecture", v: "ConvNeXt-T + GRU (satellite-IR ⊕ SHIPS)" },
  { k: "PR-AUC", v: "0.412" },
  { k: "Trend accuracy", v: "87.3%" },
  { k: "Held-out seasons", v: "2010–2019" },
];

export function HeaderBar({ status }: HeaderBarProps) {
  const live = status === "Operational";
  return (
    <header className="border-b border-hairline bg-panel/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-end justify-between gap-x-8 gap-y-4 px-6 py-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-sans text-2xl font-semibold tracking-tight">
              Tropical Cyclone <span className="text-primary">AI</span>
            </h1>
            <span className="readout rounded-xs border border-hairline px-1.5 py-0.5 text-[11px] text-muted-foreground">
              PS-1742
            </span>
          </div>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            Rapid intensification guidance — probability of a ≥30 kt increase in maximum
            sustained wind over the next 24 hours.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <dl className="flex flex-wrap gap-x-6 gap-y-2">
            {MODEL_META.map((m) => (
              <div key={m.k}>
                <dt className="text-[11px] font-medium tracking-wide text-muted-foreground">
                  {m.k}
                </dt>
                <dd className="readout text-xs text-foreground/85">{m.v}</dd>
              </div>
            ))}
          </dl>
          <div className="flex items-center gap-2 border-l border-hairline pl-6">
            <span
              className="relative flex h-2 w-2"
              style={{ color: live ? "var(--risk-low)" : "var(--destructive)" }}
            >
              {live && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-60" />
              )}
              <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
            </span>
            <span className="readout text-xs">Model {status.toLowerCase()}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
