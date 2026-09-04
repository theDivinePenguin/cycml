import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { HeaderBar } from "@/components/tc/HeaderBar";
import { StormSelector } from "@/components/tc/StormSelector";
import { StormStatePanel } from "@/components/tc/StormStatePanel";
import { DecisionSummary } from "@/components/tc/DecisionSummary";
import { RiskHeadline } from "@/components/tc/RiskHeadline";
import { IntensityChart } from "@/components/tc/IntensityChart";
import { AuxForecast } from "@/components/tc/AuxForecast";
import { VerdictPanel } from "@/components/tc/VerdictPanel";
import { STORMS, fetchForecast } from "@/lib/forecast-api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Tropical Cyclone AI — Rapid Intensification Forecast Console" },
      {
        name: "description",
        content:
          "Deep-learning rapid intensification guidance for tropical cyclones: 24-hour RI probability, relative risk versus climatology, and verified best-track outcomes.",
      },
      { property: "og:title", content: "Tropical Cyclone AI — RI Forecast Console" },
      {
        property: "og:description",
        content:
          "24-hour rapid intensification probability, Saffir–Simpson intensity tracking, and forecast verification for held-out cyclone cases.",
      },
    ],
  }),
  component: Console,
});

function Console() {
  const [stormId, setStormId] = useState(STORMS[0]!.id);
  const [step, setStep] = useState(16);
  const [playing, setPlaying] = useState(false);

  const steps = STORMS.find((s) => s.id === stormId)!.steps;

  const { data } = useQuery({
    queryKey: ["forecast", stormId, step],
    queryFn: () => fetchForecast(stormId, step),
    placeholderData: (prev) => prev,
  });

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= steps) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 900);
    return () => clearInterval(id);
  }, [playing, steps]);

  function selectStorm(id: string) {
    setStormId(id);
    setPlaying(false);
    setStep(Math.min(step, STORMS.find((s) => s.id === id)!.steps));
  }

  return (
    <div className="min-h-screen">
      <HeaderBar status="Operational" />
      <StormSelector
        storms={STORMS}
        stormId={stormId}
        onStormChange={selectStorm}
        step={step}
        steps={steps}
        onStepChange={setStep}
        playing={playing}
        onTogglePlay={() => setPlaying((p) => !p)}
      />

      {data ? (
        <main className="mx-auto max-w-[1500px] px-6 py-6">
          <div className="grid gap-5 xl:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
            <div className="flex flex-col gap-5">
              <StormStatePanel data={data} />
              <DecisionSummary data={data} />
              <AuxForecast data={data} />
            </div>
            <div className="flex flex-col gap-5">
              <RiskHeadline data={data} />
              <IntensityChart
                data={data}
                nowHour={step * 3}
                currentStep={step}
                onStepChange={setStep}
              />
              <VerdictPanel data={data} />
            </div>
          </div>
          <p className="readout mt-6 text-[11px] text-muted-foreground">
            Guidance product — retrospective replay of held-out cases. Not an official warning
            product; consult NHC/JTWC advisories for operational decisions.
          </p>
        </main>
      ) : (
        <div className="readout px-6 py-24 text-sm text-muted-foreground">
          Acquiring analysis…
        </div>
      )}
    </div>
  );
}
