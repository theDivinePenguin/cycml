# Tropical Intensity Watch

Build a web dashboard called "Tropical Cyclone AI" — a rapid intensification (RI) 

forecasting tool for tropical cyclones, built for a hackathon demo (judges will be 

meteorology/tech evaluators, so it needs to look credible and professional, not like 

a generic SaaS admin panel).

CONTEXT: 

This is a real-time-style forecasting dashboard. A deep learning model analyzes 

satellite imagery + environmental data for a tropical cyclone and predicts whether 

it will "rapidly intensify" (wind speed jump of 30+ knots in 24 hours) — a rare, 

dangerous event. The UI should feel like a scientific/meteorological command center — 

think weather agency operations center or mission control — NOT a generic dark 

developer dashboard with boxy cards everywhere.

DESIGN DIRECTION:

- Avoid the cliché "AI dashboard" look: no identical rounded cards with the same 

  border-radius repeated everywhere, no ALL-CAPS labels on every element, no generic 

  neon-on-black gradient buttons.

- Lean into real meteorological visual language: use the actual Saffir-Simpson 

  hurricane category color scale (blues/greens for TD/TS, yellow-orange for Cat 1-2, 

  red for Cat 3, deep red/magenta for Cat 4-5) for intensity indicators, since this 

  is an established, credible palette real forecasters use — not arbitrary UI colors.

- Typography: a technical/monospace font for data readouts and coordinates (feels 

  like real telemetry), paired with a clean grotesque sans for headings and body text.

- Give the layout a clear visual hierarchy: one dominant "headline" element (the RI 

  risk result) should visually outweigh everything else on the screen — right now 

  the risk is buried among a dozen equal-weight boxes.

KEY SCREENS/SECTIONS TO INCLUDE:

1. Header bar: 

   - Project name + problem statement ID tag

   - Live model status indicator ("Operational" / "Offline")

   - Model metadata (architecture name, PR-AUC score, trend accuracy) — small, 

     secondary, not competing with the headline

2. Storm selector:

   - Dropdown to pick a test-set cyclone (e.g. "Super Typhoon Megi — Peak 160kt")

   - A note badge indicating this is held-out/unseen test data

   - A timeline scrubber with play/pause to step through the storm's lifecycle 

     hour by hour

3. Current storm state panel:

   - Current wind speed (large, prominent number) + storm category label

   - Coordinates, timestamp

   - Environmental readouts: sea surface temp, ocean heat content, wind shear, 

     humidity, central pressure — each with a short qualitative tag (e.g. "Super-Warm", 

     "Low Shear")

4. THE HEADLINE RESULT — this should be the visual focal point of the whole page:

   - Macro trend prediction (Weakening / Stable / Intensifying) as a bold state

   - RI probability shown as a large percentage

   - Directly below/beside it, a computed "relative risk" line: 

     multiplier = probability ÷ 0.068 (0.068 is the dataset's baseline RI rate)

     Show this as "2.2× baseline risk" alongside a risk tier badge:

       < 1.5x = "Low Risk" (calm color)

       1.5-3x = "Elevated Risk" (amber)

       3-6x = "High Risk" (orange)

       6x+ = "Critical Risk" (red)

   - Make clear BOTH numbers are shown — never hide the raw probability, the 

     multiplier is additional context, not a replacement

5. Forecast timeline chart:

   - Line chart showing intensity over time: observed track (past → now), 

     active forecast window (next 24h), and the actual outcome for comparison

   - Toggle between "raw model output" and "smoothed" view of the line

   - Clear visual marker for "NOW" on the timeline

6. Auxiliary forecast: +6h / +12h / +24h predicted wind speed, shown as three 

   compact stat blocks

7. Operational verdict panel: plain-English summary of whether the prediction 

   matched the real outcome for this historical storm (for demo credibility)

TECHNICAL REQUIREMENTS:

- This will later be connected to a real backend API by another developer — 

  structure the frontend to fetch data from a REST endpoint (e.g. GET /forecast?storm_id=X&t=Y) 

  rather than hardcoding values, using placeholder/mock data for now that matches 

  this shape:

  {

    "storm_name": string,

    "timestamp": string,

    "current_wind_kt": number,

    "category": string,

    "coordinates": {"lat": number, "lon": number},

    "environmental": {"sst": number, "ohc": number, "shear": number, "rh": number, "mslp": number},

    "trend": "Weakening" | "Stable" | "Intensifying",

    "ri_probability": number,

    "forecast": {"+6h": number, "+12h": number, "+24h": number},

    "timeline": [{"t": number, "observed_kt": number, "predicted_kt": number}],

    "actual_outcome_kt": number

  }

- Keep components modular and cleanly separated so someone else can swap the mock 

  data fetch for a real API call easily.

- Make it responsive but the primary target is a laptop screen for a live demo — 

  don't over-invest in mobile layout.

Build this as a clean, well-organized React app. AVOID GENERIC AI-GENERATED LOOK — be specific about avoiding these common tells:

- Don't use the default "glassmorphism dark dashboard" template: no uniform 

  translucent cards with identical blur/border-radius stacked in a grid.

- Don't center everything with equal visual weight — real interfaces have 

  intentional asymmetry and a clear focal point, not a symmetric grid of 

  same-sized boxes.

- Avoid purple-to-blue gradients, neon glow effects, and generic "tech" 

  iconography (no glowing circuit-board patterns, no floating 3D orbs).

- Avoid filler/placeholder-feeling copy — every label should sound like it 

  was written by someone who understands meteorology, not generic dashboard 

  boilerplate like "Stats" or "Overview."

- Don't over-decorate with unnecessary icons next to every single label — 

  use icons only where they add real meaning (e.g. a wind icon next to wind 

  speed), not as decoration.

- Make deliberate typography choices (real hierarchy in size/weight between 

  headline numbers, labels, and body text) rather than everything being the 

  same size with just color/opacity differences.

- The color palette should come FROM the domain (hurricane category colors, 

  as specified above) rather than an arbitrary "AI dashboard" purple/cyan/pink 

  palette that has nothing to do with weather.

- It should look like it was designed by someone who studied real NOAA/NHC 

  storm tracking tools and weather.com-style forecast UIs — reference that 

  kind of real-world precedent, not generic dashboard templates.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/325cfa83-5ae7-458b-a6c4-daf8a8b2e99d).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
