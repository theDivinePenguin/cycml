import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import type { StormOption } from "@/lib/forecast-types";

interface StormSelectorProps {
  storms: StormOption[];
  stormId: string;
  onStormChange: (id: string) => void;
  step: number;
  steps: number;
  onStepChange: (t: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  timestamp?: string;
}

export function StormSelector({
  storms,
  stormId,
  onStormChange,
  step,
  steps,
  onStepChange,
  playing,
  onTogglePlay,
}: StormSelectorProps) {
  const active = storms.find((s) => s.id === stormId);

  return (
    <section className="border-b border-hairline bg-panel/90">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-4 px-6 py-3">
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-foreground/80 font-mono uppercase tracking-wider" htmlFor="storm">
            Case
          </label>
          <Select value={stormId} onValueChange={onStormChange}>
            <SelectTrigger
              id="storm"
              className="h-9 w-[320px] rounded-xs border border-[#7AAACE]/50 bg-panel font-mono text-sm text-foreground shadow-xs"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-xs border border-[#7AAACE]/50 bg-panel shadow-sm">
              {storms.map((s) => (
                <SelectItem key={s.id} value={s.id} className="font-mono text-sm text-foreground hover:bg-[#9CD5FF]/20">
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {active && (
            <span className="readout text-xs text-muted-foreground font-semibold">
              {active.basin} · {active.season}
            </span>
          )}
        </div>

        <span className="rounded-xs border border-[#7AAACE]/60 bg-[#9CD5FF]/30 px-2 py-1 text-[11px] font-semibold text-[#355872] font-mono">
          Held-out test set — never seen during training
        </span>

        <div className="ml-auto flex min-w-[420px] flex-1 items-center gap-3">
          <div className="flex items-center gap-1">
            <ScrubButton
              label="Step back"
              onClick={() => onStepChange(Math.max(0, step - 1))}
            >
              <SkipBack className="h-3.5 w-3.5" />
            </ScrubButton>
            <ScrubButton label={playing ? "Pause" : "Play"} onClick={onTogglePlay} accent>
              {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </ScrubButton>
            <ScrubButton
              label="Step forward"
              onClick={() => onStepChange(Math.min(steps, step + 1))}
            >
              <SkipForward className="h-3.5 w-3.5" />
            </ScrubButton>
          </div>
          <Slider
            value={[step]}
            min={0}
            max={steps}
            step={1}
            onValueChange={(v) => onStepChange(v[0] ?? 0)}
            className="flex-1"
            aria-label="Storm lifecycle position"
          />
          <span className="readout w-28 text-right text-xs text-muted-foreground font-semibold font-mono">
            T+{String(step * 3).padStart(3, "0")} h / {steps * 3} h
          </span>
        </div>
      </div>
    </section>
  );
}

function ScrubButton({
  children,
  onClick,
  label,
  accent,
}: {
  children: React.ReactNode;
  onClick: () => void;
  label: string;
  accent?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className={`flex h-8 w-8 items-center justify-center rounded-xs border transition-colors cursor-pointer ${
        accent
          ? "bg-[#355872] text-[#F7F8F0] hover:bg-[#355872]/90 border-[#355872] shadow-xs"
          : "bg-panel text-[#355872] border-[#7AAACE]/50 hover:bg-[#9CD5FF]/20"
      }`}
    >
      {children}
    </button>
  );
}
