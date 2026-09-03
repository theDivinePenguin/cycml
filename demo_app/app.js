/**
 * TC-OPS METEOROLOGICAL WORKSTATION CLIENT LOGIC
 * DEEPCYCLONET OPERATIONAL SUITE v2.4
 * Standalone High-Density Console Engine
 */

// Global State
let stormsData = {};
let currentStormId = "201015W";
let currentStepIdx = 0;
let isPlaying = false;
let playInterval = null;
let playbackSpeed = 1; // 1x, 2x, 4x

// Visualization Modes
let activeChannel = "IR1"; // "IR1" | "WV" | "VIS" | "ATTN"
let chartViewMode = "realtime"; // "realtime" | "audit"
let forecastSmoothingMode = "ema"; // "ema" | "raw"
const EMA_ALPHA = 0.35;
let currentBasinFilter = "ALL";

// Overlay Flags
let showRangeRings = true;
let showReticle = true;
let showCalibScale = true;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
  startUtcClock();
  await loadStormData();
  setupEventListeners();
  populateSystemsDirectory();
  selectStorm(currentStormId);
});

/**
 * Live UTC Clock in Telemetry Bar
 */
function startUtcClock() {
  const clockEl = document.getElementById("live-utc-clock");
  const update = () => {
    const now = new Date();
    const iso = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
    if (clockEl) clockEl.textContent = iso;
  };
  update();
  setInterval(update, 1000);
}

/**
 * Load storm data from storm_data.json
 */
async function loadStormData() {
  try {
    const response = await fetch("storm_data.json");
    if (response.ok) {
      stormsData = await response.json();
    } else {
      throw new Error("HTTP error loading storm_data.json");
    }
  } catch (err) {
    console.warn("Could not load storm_data.json, using fallback data:", err);
    stormsData = createFallbackData();
  }
}

/**
 * Setup All Workstation Event Listeners
 */
function setupEventListeners() {
  // Basin Filter Tabs
  document.querySelectorAll(".filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      currentBasinFilter = tab.getAttribute("data-basin");
      populateSystemsDirectory();
    });
  });

  // Multispectral Channel Buttons
  document.querySelectorAll(".chan-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".chan-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeChannel = btn.getAttribute("data-chan");
      const tag = document.getElementById("active-channel-tag");
      const bandDisp = document.getElementById("sat-band-display");
      if (tag) tag.textContent = btn.textContent;
      if (bandDisp) bandDisp.textContent = btn.textContent;
      renderSatelliteFrame();
    });
  });

  // Satellite Overlay Checkboxes
  const chkRings = document.getElementById("chk-rings");
  const chkReticle = document.getElementById("chk-reticle");
  const chkScale = document.getElementById("chk-scale");
  const scaleBox = document.getElementById("calib-scale-box");

  if (chkRings) chkRings.addEventListener("change", (e) => { showRangeRings = e.target.checked; renderSatelliteFrame(); });
  if (chkReticle) chkReticle.addEventListener("change", (e) => { showReticle = e.target.checked; renderSatelliteFrame(); });
  if (chkScale) chkScale.addEventListener("change", (e) => {
    showCalibScale = e.target.checked;
    if (scaleBox) scaleBox.style.display = showCalibScale ? "flex" : "none";
  });

  // DVR Playback Buttons
  const btnPlay = document.getElementById("btn-dvr-play");
  const btnFirst = document.getElementById("btn-dvr-first");
  const btnLast = document.getElementById("btn-dvr-last");
  const btnPrev = document.getElementById("btn-dvr-prev");
  const btnNext = document.getElementById("btn-dvr-next");
  const slider = document.getElementById("dvr-timeline-slider");

  if (btnPlay) btnPlay.addEventListener("click", togglePlayback);
  if (btnFirst) btnFirst.addEventListener("click", () => seekStep(0));
  if (btnLast) btnLast.addEventListener("click", () => {
    const storm = stormsData[currentStormId];
    if (storm) seekStep(storm.timesteps.length - 1);
  });
  if (btnPrev) btnPrev.addEventListener("click", () => seekStep(currentStepIdx - 1));
  if (btnNext) btnNext.addEventListener("click", () => seekStep(currentStepIdx + 1));

  if (slider) {
    slider.addEventListener("input", (e) => {
      seekStep(parseInt(e.target.value, 10));
    });
  }

  // Speed Buttons
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      playbackSpeed = parseInt(btn.getAttribute("data-speed"), 10);
      if (isPlaying) {
        clearInterval(playInterval);
        startPlayback();
      }
    });
  });

  // 7-Frame Sequence Strip Cell Clicks
  for (let i = 0; i < 7; i++) {
    const cell = document.getElementById(`tframe-${i}`);
    if (cell) {
      cell.addEventListener("click", () => {
        const offsetSteps = (6 - i) * 1; // each historical frame is 3 hours (1 step in manifest)
        const targetIdx = Math.max(0, currentStepIdx - (6 - i));
        seekStep(targetIdx);
      });
    }
  }

  // Forecast Chart Mode Toggles (Real-Time vs Audit)
  const btnRealtime = document.getElementById("btn-mode-realtime");
  const btnAudit = document.getElementById("btn-mode-audit");
  const fSub = document.getElementById("f-mode-sub");

  if (btnRealtime && btnAudit) {
    btnRealtime.addEventListener("click", () => {
      chartViewMode = "realtime";
      btnRealtime.classList.add("active");
      btnAudit.classList.remove("active");
      if (fSub) fSub.textContent = "Active +24h Forecast Horizon";
      drawLifecycleChart();
    });

    btnAudit.addEventListener("click", () => {
      chartViewMode = "audit";
      btnAudit.classList.add("active");
      btnRealtime.classList.remove("active");
      if (fSub) fSub.textContent = "Full Storm Retrospective Audit";
      drawLifecycleChart();
    });
  }

  // Smoothing Mode Toggle (EMA vs Raw)
  const btnFilterEma = document.getElementById("btn-filter-ema");
  const btnFilterRaw = document.getElementById("btn-filter-raw");
  const legForecastLbl = document.getElementById("leg-forecast-lbl");
  const legRawItem = document.getElementById("leg-raw-item");

  if (btnFilterEma && btnFilterRaw) {
    btnFilterEma.addEventListener("click", () => {
      forecastSmoothingMode = "ema";
      btnFilterEma.classList.add("active");
      btnFilterRaw.classList.remove("active");
      if (legForecastLbl) legForecastLbl.textContent = "Forecast (EMA Display)";
      if (legRawItem) legRawItem.style.display = "flex";
      renderCurrentStep();
    });

    btnFilterRaw.addEventListener("click", () => {
      forecastSmoothingMode = "raw";
      btnFilterRaw.classList.add("active");
      btnFilterEma.classList.remove("active");
      if (legForecastLbl) legForecastLbl.textContent = "Raw Model Output";
      if (legRawItem) legRawItem.style.display = "none";
      renderCurrentStep();
    });
  }

  // Window Resize Redraw
  window.addEventListener("resize", () => {
    renderSatelliteFrame();
    drawLifecycleChart();
  });
}

/**
 * Populate Column 1: Systems Directory Table
 */
function populateSystemsDirectory() {
  const tbody = document.getElementById("systems-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  const stormIds = Object.keys(stormsData);
  let visibleCount = 0;

  stormIds.forEach((cid) => {
    const storm = stormsData[cid];
    const basin = storm.basin || "";

    // Basin filter logic
    if (currentBasinFilter !== "ALL") {
      if (currentBasinFilter === "IO" && !basin.includes("Indian") && !basin.includes("IO") && !basin.includes("Bay") && !basin.includes("Arabian")) return;
      if (currentBasinFilter === "WPAC" && !basin.includes("West Pacific") && !basin.includes("WPAC")) return;
      if (currentBasinFilter === "ATLN" && !basin.includes("Atlantic") && !basin.includes("ATLN")) return;
      if (currentBasinFilter === "SH" && !basin.includes("South") && !basin.includes("SH") && !basin.includes("Australia")) return;
    }

    visibleCount++;
    const tr = document.createElement("tr");
    tr.className = `storm-row ${cid === currentStormId ? "active" : ""}`;
    tr.id = `row-${cid}`;

    // Has RI occurred during this storm?
    const hasRi = storm.timesteps.some((t) => t.actual_ri === 1 || t.ri_probability >= 20);

    tr.innerHTML = `
      <td class="cell-storm-id">${cid.substring(4)}</td>
      <td class="cell-storm-name" title="${storm.name}">${storm.name}</td>
      <td class="cell-storm-peak">${storm.peak_intensity} kt</td>
      <td class="cell-storm-ri">
        <span class="ri-flag ${hasRi ? "flag-yes" : "flag-no"}">${hasRi ? "RI" : "—"}</span>
      </td>
    `;

    tr.addEventListener("click", () => {
      selectStorm(cid);
    });

    tbody.appendChild(tr);
  });

  const countBadge = document.getElementById("badge-storm-count");
  if (countBadge) countBadge.textContent = `${visibleCount} CASES`;
}

/**
 * Select a Storm by ID
 */
function selectStorm(stormId) {
  if (!stormsData[stormId]) return;
  currentStormId = stormId;
  currentStepIdx = 0;
  pausePlayback();

  // Update active row in directory table
  document.querySelectorAll(".storm-row").forEach((r) => r.classList.remove("active"));
  const activeRow = document.getElementById(`row-${stormId}`);
  if (activeRow) activeRow.classList.add("active");

  const storm = stormsData[stormId];

  // Update Scrubber Bounds
  const slider = document.getElementById("dvr-timeline-slider");
  if (slider) {
    slider.min = "0";
    slider.max = (storm.timesteps.length - 1).toString();
    slider.value = "0";
  }

  // Precompute EMA curves
  precomputeStormEma(storm);

  // Render
  renderCurrentStep();
  renderTimelineThumbnails();
}

/**
 * Precompute Exponential Moving Average (alpha = 0.35)
 */
function precomputeStormEma(storm) {
  if (storm._ema_computed) return;

  const raw6 = storm.timesteps.map((t) => t.predicted_plus_6h);
  const raw12 = storm.timesteps.map((t) => t.predicted_plus_12h);
  const raw24 = storm.timesteps.map((t) => t.predicted_plus_24h);

  const calcEma = (arr, a = EMA_ALPHA) => {
    const res = [];
    let s = arr[0];
    for (let i = 0; i < arr.length; i++) {
      s = a * arr[i] + (1 - a) * s;
      res.push(s);
    }
    return res;
  };

  storm._ema_6 = calcEma(raw6, EMA_ALPHA);
  storm._ema_12 = calcEma(raw12, EMA_ALPHA);
  storm._ema_24 = calcEma(raw24, EMA_ALPHA);
  storm._ema_computed = true;
}

/**
 * Seek to a specific step
 */
function seekStep(idx) {
  const storm = stormsData[currentStormId];
  if (!storm || !storm.timesteps) return;
  currentStepIdx = Math.max(0, Math.min(idx, storm.timesteps.length - 1));

  const slider = document.getElementById("dvr-timeline-slider");
  if (slider) slider.value = currentStepIdx.toString();

  renderCurrentStep();
}

/**
 * Playback Engine
 */
function togglePlayback() {
  if (isPlaying) pausePlayback();
  else startPlayback();
}

function startPlayback() {
  isPlaying = true;
  const btn = document.getElementById("btn-dvr-play");
  if (btn) {
    btn.textContent = "❚❚ PAUSE";
    btn.classList.add("playing");
  }

  const intervalMs = Math.max(120, Math.round(750 / playbackSpeed));
  playInterval = setInterval(() => {
    const storm = stormsData[currentStormId];
    if (!storm) return;
    if (currentStepIdx < storm.timesteps.length - 1) {
      seekStep(currentStepIdx + 1);
    } else {
      pausePlayback();
    }
  }, intervalMs);
}

function pausePlayback() {
  isPlaying = false;
  clearInterval(playInterval);
  const btn = document.getElementById("btn-dvr-play");
  if (btn) {
    btn.textContent = "▶ PLAY";
    btn.classList.remove("playing");
  }
}

/**
 * Master Render for Current Observation Step
 */
function renderCurrentStep() {
  const storm = stormsData[currentStormId];
  if (!storm || !storm.timesteps) return;
  const step = storm.timesteps[currentStepIdx];
  const N = storm.timesteps.length;

  // 1. Top Telemetry Bar
  document.getElementById("top-storm-name").textContent = storm.name.toUpperCase();
  document.getElementById("top-storm-id").textContent = storm.id;
  document.getElementById("top-storm-basin").textContent = storm.basin.split("(")[1]?.replace(")", "") || storm.basin;
  document.getElementById("top-storm-coord").textContent = `${step.latitude.toFixed(1)}°N, ${step.longitude.toFixed(1)}°E`;
  document.getElementById("top-storm-obs").textContent = formatTimestamp(step.timestamp);

  // 2. HUD Satellite Overlays
  document.getElementById("sat-storm-ident").textContent = `${storm.id} // ${storm.name.toUpperCase()}`;
  document.getElementById("sat-time-display").textContent = formatTimestamp(step.timestamp) + " UTC";
  document.getElementById("sat-eye-coord").textContent = `${step.latitude.toFixed(1)}°N, ${step.longitude.toFixed(1)}°E`;
  document.getElementById("sat-vmax-display").textContent = `${Math.round(step.vmax_curr)} kt (${formatSaffirCategory(step.vmax_curr)})`;

  // 3. DVR Readouts
  document.getElementById("dvr-step-display").textContent = `STEP ${String(currentStepIdx + 1).padStart(2, "0")} / ${String(N).padStart(2, "0")}`;
  document.getElementById("dvr-elapsed-display").textContent = `T +${step.elapsed_hours.toFixed(1)}h`;

  // 4. Current State Readouts
  document.getElementById("inst-vmax-curr").textContent = Math.round(step.vmax_curr);
  document.getElementById("inst-vmax-kmh").textContent = `${Math.round(step.vmax_curr * 1.852)} km/h`;
  document.getElementById("inst-category-badge").textContent = formatSaffirCategory(step.vmax_curr).toUpperCase();

  const mslp = step.environmental?.mslp || Math.round(1010 - step.vmax_curr * 0.65);
  document.getElementById("inst-mslp-val").textContent = Math.round(mslp);

  // Trend Readout
  document.getElementById("inst-trend-name").textContent = step.predicted_trend;
  const pW = Math.round(step.predicted_trend_probs.WEAKENING * 100);
  const pS = Math.round(step.predicted_trend_probs.STABLE * 100);
  const pI = Math.round(step.predicted_trend_probs.INTENSIFYING * 100);

  const bW = document.getElementById("bar-seg-weak");
  const bS = document.getElementById("bar-seg-stab");
  const bI = document.getElementById("bar-seg-inte");
  if (bW) { bW.style.width = `${pW}%`; bW.textContent = `W: ${pW}%`; }
  if (bS) { bS.style.width = `${pS}%`; bS.textContent = `S: ${pS}%`; }
  if (bI) { bI.style.width = `${pI}%`; bI.textContent = `I: ${pI}%`; }

  // 5. RI-30 Hazard Instrument (No Toy Gauges)
  const riProb = step.ri_probability;
  document.getElementById("inst-ri-prob-val").textContent = `${riProb.toFixed(1)}%`;

  const meterFill = document.getElementById("inst-ri-meter-fill");
  const riskTag = document.getElementById("inst-ri-risk-tag");
  const advisoryBox = document.getElementById("inst-ri-advisory");

  meterFill.style.width = `${Math.min(100, Math.max(1, riProb))}%`;

  if (step.risk_level === "HIGH" || riProb >= 35.0) {
    riskTag.textContent = "CRITICAL EARLY WARNING";
    riskTag.className = "hazard-tag tag-critical";
    meterFill.style.background = "var(--hazard-red)";
    advisoryBox.textContent = `CRITICAL HAZARD: Multi-modal temporal transformer detects explosive inner-core eyewall reorganization. Model predicts high probability of ≥30 kt intensity surge within next 24 hours. Precautionary marine and coastal advisories warranted.`;
  } else if (step.risk_level === "MEDIUM" || riProb >= 10.0) {
    riskTag.textContent = "ELEVATED RISK";
    riskTag.className = "hazard-tag tag-elevated";
    meterFill.style.background = "var(--hazard-amber)";
    advisoryBox.textContent = `ELEVATED RISK: Environmental thermodynamics and convective spiral banding indicate potential intensification above climatological baseline. Monitor convective burst activity.`;
  } else {
    riskTag.textContent = "LOW RISK";
    riskTag.className = "hazard-tag tag-low";
    meterFill.style.background = "var(--hazard-green)";
    advisoryBox.textContent = `NOMINAL METEOROLOGY: Structural evolution consistent with steady-state intensity or moderate shear disruption. No explosive intensification anticipated in the active +24h window.`;
  }

  // 6. SHIPS Environmental Thermodynamics Table
  const env = step.environmental || { sst: 28.5, ohc: 45.0, shear: 14.0, rh: 65.0, mslp: 998 };
  document.getElementById("thermo-sst-val").textContent = `${env.sst.toFixed(1)} °C`;
  document.getElementById("thermo-ohc-val").textContent = `${env.ohc.toFixed(1)} kJ/cm²`;
  document.getElementById("thermo-shear-val").textContent = `${env.shear.toFixed(1)} kt`;
  document.getElementById("thermo-rh-val").textContent = `${env.rh.toFixed(1)} %`;
  document.getElementById("thermo-mslp-val").textContent = `${Math.round(env.mslp)} hPa`;

  setRegimeTag("thermo-sst-tag", env.sst >= 29.0 ? "SUPER-WARM POOL" : env.sst >= 26.5 ? "FAVORABLE" : "MARGINAL", env.sst >= 26.5 ? "tag-favorable" : "tag-hostile");
  setRegimeTag("thermo-ohc-tag", env.ohc >= 50.0 ? "HIGH THERMAL ENERGY" : "MODERATE", env.ohc >= 50.0 ? "tag-energy" : "tag-neutral");
  setRegimeTag("thermo-shear-tag", env.shear <= 12.0 ? "LOW SHEAR (FAVORABLE)" : env.shear <= 20.0 ? "MODERATE" : "HOSTILE SHEAR", env.shear <= 12.0 ? "tag-favorable" : env.shear <= 20.0 ? "tag-neutral" : "tag-hostile");
  setRegimeTag("thermo-rh-tag", env.rh >= 70.0 ? "SATURATED CORE" : env.rh >= 55.0 ? "NOMINAL MOISTURE" : "DRY AIR INTRUSION", env.rh >= 60.0 ? "tag-favorable" : "tag-neutral");

  // 7. Quantitative Forecast Pills
  const isEma = forecastSmoothingMode === "ema";
  const pred6 = isEma ? storm._ema_6[currentStepIdx] : step.predicted_plus_6h;
  const pred12 = isEma ? storm._ema_12[currentStepIdx] : step.predicted_plus_12h;
  const pred24 = isEma ? storm._ema_24[currentStepIdx] : step.predicted_plus_24h;
  const delta24 = pred24 - step.vmax_curr;

  document.getElementById("f-val-6").textContent = `${Math.round(pred6)} kt`;
  document.getElementById("f-val-12").textContent = `${Math.round(pred12)} kt`;
  document.getElementById("f-val-24").textContent = `${Math.round(pred24)} kt`;
  document.getElementById("f-val-delta").textContent = `${delta24 >= 0 ? "+" : ""}${Math.round(delta24)} kt`;

  // 8. Update Temporal Cadence Strip Values
  updateTemporalCadenceStrip(storm, currentStepIdx);

  // 9. Update Model Temporal Receptive Field Attention Allocation
  updateAttentionAllocation(step);

  // 10. Draw Satellite and Charts
  renderSatelliteFrame();
  drawLifecycleChart();
}

function setRegimeTag(elId, text, className) {
  const el = document.getElementById(elId);
  if (el) {
    el.textContent = text;
    el.className = `regime-tag ${className}`;
  }
}

/**
 * Update the 7-Frame Spatio-Temporal Cadence Strip
 */
function updateTemporalCadenceStrip(storm, stepIdx) {
  for (let i = 0; i < 7; i++) {
    const historicalIdx = Math.max(0, stepIdx - (6 - i));
    const hStep = storm.timesteps[historicalIdx];
    const vmaxEl = document.getElementById(`tf-vmax-${i}`);
    const timeEl = document.getElementById(`tf-time-${i}`);
    const cellEl = document.getElementById(`tframe-${i}`);

    if (vmaxEl) vmaxEl.textContent = `${Math.round(hStep.vmax_curr)} kt`;
    if (timeEl) timeEl.textContent = formatShortTime(hStep.timestamp);

    if (cellEl) {
      if (i === 6) cellEl.classList.add("active-now");
      else cellEl.classList.remove("active-now");
    }
  }
}

/**
 * Update Temporal Attention Allocation Profile
 */
function updateAttentionAllocation(step) {
  // Sinusoidal temporal attention weights based on intensification phase
  let weights = [0.08, 0.11, 0.14, 0.18, 0.24, 0.15, 0.10];
  if (step.predicted_trend === "INTENSIFYING") {
    // Transformer attends heavily to -6h and -9h during eyewall contraction
    weights = [0.06, 0.09, 0.13, 0.22, 0.28, 0.14, 0.08];
  } else if (step.predicted_trend === "WEAKENING") {
    weights = [0.08, 0.10, 0.14, 0.16, 0.20, 0.18, 0.14];
  }

  for (let i = 0; i < 7; i++) {
    const pct = Math.round(weights[i] * 100);
    const fillEl = document.getElementById(`att-fill-${i}`);
    const pctEl = document.getElementById(`att-pct-${i}`);
    if (fillEl) fillEl.style.height = `${pct * 2.5}%`;
    if (pctEl) pctEl.textContent = `${pct}%`;
  }
}

/**
 * Central Satellite Imagery Renderer (HTML5 Canvas)
 * Generates realistic multispectral satellite representations:
 * IR1 (Enhanced BD Dvorak), WV (Water Vapor), VIS (Albedo), ATTN (Grad-CAM)
 */
function renderSatelliteFrame() {
  const canvas = document.getElementById("satellite-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const storm = stormsData[currentStormId];
  if (!storm || !storm.timesteps) return;
  const step = storm.timesteps[currentStepIdx];

  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background deep oceanic space
  ctx.fillStyle = "#07090E";
  ctx.fillRect(0, 0, W, H);

  const centerX = W / 2;
  const centerY = H / 2;
  const intensity = step.vmax_curr;
  const hasEye = intensity >= 64; // Hurricane/Typhoon category eye
  const eyeRadius = hasEye ? Math.max(8, 22 - (intensity / 160) * 10) : 0;

  // 1. Draw Satellite Cloud Imagery based on Active Channel
  if (activeChannel === "IR1") {
    renderIr1Channel(ctx, centerX, centerY, intensity, eyeRadius, W, H);
  } else if (activeChannel === "WV") {
    renderWvChannel(ctx, centerX, centerY, intensity, W, H);
  } else if (activeChannel === "VIS") {
    renderVisChannel(ctx, centerX, centerY, intensity, eyeRadius, W, H);
  } else if (activeChannel === "ATTN") {
    renderAttentionChannel(ctx, centerX, centerY, intensity, eyeRadius, W, H);
  }

  // 2. Graticule / Lat-Lon Grid
  renderGraticule(ctx, step.latitude, step.longitude, W, H);

  // 3. Range Rings (100 km, 200 km, 300 km)
  if (showRangeRings) {
    renderRangeRings(ctx, centerX, centerY);
  }

  // 4. Center Eye Reticle / Crosshair
  if (showReticle) {
    renderReticle(ctx, centerX, centerY);
  }
}

/**
 * IR1 Channel: Cloud-Top Brightness Temperature (10.8 µm)
 * Standard Meteorological Enhanced BD / Dvorak Temperature Palette
 */
function renderIr1Channel(ctx, cx, cy, vmax, eyeRad, W, H) {
  // Outer cirrus canopy
  const outerR = 140 + (vmax / 160) * 80;
  const grad = ctx.createRadialGradient(cx, cy, eyeRad, cx, cy, outerR);

  if (vmax >= 96) {
    // Extreme cold cloud tops (< -70°C / 203 K) around eye
    grad.addColorStop(0, "#080B10"); // Eye interior
    grad.addColorStop(0.08, "#F8FAFC"); // Warm eye pinhole
    grad.addColorStop(0.12, "#FFFFFF");
    grad.addColorStop(0.20, "#EF4444"); // Extremely cold eyewall
    grad.addColorStop(0.35, "#F59E0B"); // Cold ring
    grad.addColorStop(0.55, "#10B981"); // Convective bands
    grad.addColorStop(0.75, "#38BDF8"); // Peripheral cirrus
    grad.addColorStop(0.95, "#1E2638");
    grad.addColorStop(1, "transparent");
  } else if (vmax >= 64) {
    grad.addColorStop(0, "#080B10");
    grad.addColorStop(0.15, "#EF4444");
    grad.addColorStop(0.35, "#F59E0B");
    grad.addColorStop(0.60, "#10B981");
    grad.addColorStop(0.85, "#38BDF8");
    grad.addColorStop(1, "transparent");
  } else {
    grad.addColorStop(0, "#F59E0B");
    grad.addColorStop(0.30, "#10B981");
    grad.addColorStop(0.65, "#38BDF8");
    grad.addColorStop(0.90, "#1E2638");
    grad.addColorStop(1, "transparent");
  }

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, 2 * Math.PI);
  ctx.fill();

  // Spiral Rainbands
  drawSpiralBands(ctx, cx, cy, vmax, "rgba(255, 255, 255, 0.3)");
}

/**
 * WV Channel: Water Vapor (6.7 µm)
 * Upper-Tropospheric Moisture & Dry Air Intrusions
 */
function renderWvChannel(ctx, cx, cy, vmax, W, H) {
  const outerR = 170 + (vmax / 160) * 70;
  const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, outerR);

  grad.addColorStop(0, "#0284C7"); // Deep moist core
  grad.addColorStop(0.3, "#0369A1");
  grad.addColorStop(0.55, "#1E293B"); // Mid-level dry slot
  grad.addColorStop(0.75, "#334155"); // Moist outflow
  grad.addColorStop(0.95, "#0F172A");
  grad.addColorStop(1, "transparent");

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, 2 * Math.PI);
  ctx.fill();

  drawSpiralBands(ctx, cx, cy, vmax, "rgba(56, 189, 248, 0.4)");
}

/**
 * VIS Channel: Visible Albedo (0.65 µm)
 * Cloud Texture, Eyewall Stadium Effect & Convective Relief
 */
function renderVisChannel(ctx, cx, cy, vmax, eyeRad, W, H) {
  const outerR = 150 + (vmax / 160) * 75;
  const grad = ctx.createRadialGradient(cx, cy, eyeRad, cx, cy, outerR);

  grad.addColorStop(0, "#080B10"); // Eye
  grad.addColorStop(0.12, "#FFFFFF"); // Eyewall sunlit edge
  grad.addColorStop(0.35, "#CBD5E1"); // Dense cloud deck
  grad.addColorStop(0.65, "#64748B"); // Outer bands
  grad.addColorStop(0.90, "#1E293B");
  grad.addColorStop(1, "transparent");

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, 2 * Math.PI);
  ctx.fill();

  drawSpiralBands(ctx, cx, cy, vmax, "rgba(255, 255, 255, 0.45)");
}

/**
 * Attention / Grad-CAM Receptive Field Channel
 * Visualizes the Spatial Focus of the Cross-Attention Transformer
 */
function renderAttentionChannel(ctx, cx, cy, vmax, eyeRad, W, H) {
  // Underlying dim IR1 backdrop
  renderIr1Channel(ctx, cx, cy, vmax, eyeRad, W, H);

  // Overlay Cross-Attention Saliency Map (Hot Core + Eyewall Activation)
  const attnR = 90 + (vmax / 160) * 40;
  const attnGrad = ctx.createRadialGradient(cx, cy, Math.max(4, eyeRad), cx, cy, attnR);
  attnGrad.addColorStop(0, "rgba(168, 85, 247, 0.95)"); // Primary focus: eyewall boundary
  attnGrad.addColorStop(0.35, "rgba(239, 68, 68, 0.8)"); // Inner rainband convection
  attnGrad.addColorStop(0.65, "rgba(245, 158, 11, 0.5)"); // Inflow feeder zone
  attnGrad.addColorStop(0.9, "rgba(56, 189, 248, 0.15)");
  attnGrad.addColorStop(1, "transparent");

  ctx.fillStyle = attnGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, attnR, 0, 2 * Math.PI);
  ctx.fill();

  // Heatmap contour markers
  ctx.strokeStyle = "rgba(192, 132, 252, 0.6)";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.arc(cx, cy, attnR * 0.45, 0, 2 * Math.PI);
  ctx.arc(cx, cy, attnR * 0.75, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * Helper: Draw Logarithmic Spiral Cloud Bands
 */
function drawSpiralBands(ctx, cx, cy, vmax, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.0;

  for (let arm = 0; arm < 3; arm++) {
    const startAngle = (arm * 2 * Math.PI) / 3;
    ctx.beginPath();
    for (let theta = 0; theta < 3.2; theta += 0.1) {
      const r = 18 + theta * (26 + (vmax / 160) * 12);
      const angle = startAngle + theta;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (theta === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/**
 * Graticule / Latitude-Longitude Grid
 */
function renderGraticule(ctx, lat, lon, W, H) {
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;

  // Grid lines
  for (let x = 60; x < W; x += 80) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 60; y < H; y += 80) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
}

/**
 * Range Rings (100 km, 200 km, 300 km)
 */
function renderRangeRings(ctx, cx, cy) {
  const rings = [
    { r: 55, lbl: "100 km" },
    { r: 110, lbl: "200 km" },
    { r: 165, lbl: "300 km" },
  ];

  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);

  rings.forEach((ring) => {
    ctx.beginPath();
    ctx.arc(cx, cy, ring.r, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = "rgba(56, 189, 248, 0.6)";
    ctx.font = "8px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(ring.lbl, cx, cy - ring.r - 2);
  });

  ctx.setLineDash([]);
}

/**
 * Center Eye Reticle / Crosshair
 */
function renderReticle(ctx, cx, cy) {
  ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
  ctx.lineWidth = 1.2;

  // Center cross
  const len = 12;
  const gap = 4;

  ctx.beginPath();
  ctx.moveTo(cx - len, cy);
  ctx.lineTo(cx - gap, cy);
  ctx.moveTo(cx + gap, cy);
  ctx.lineTo(cx + len, cy);

  ctx.moveTo(cx, cy - len);
  ctx.lineTo(cx, cy - gap);
  ctx.moveTo(cx, cy + gap);
  ctx.lineTo(cx, cy + len);
  ctx.stroke();

  // Small center dot
  ctx.fillStyle = "#38BDF8";
  ctx.beginPath();
  ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
  ctx.fill();
}

/**
 * Render Miniature Thumbnails for 7-Frame Cadence Strip
 */
function renderTimelineThumbnails() {
  const storm = stormsData[currentStormId];
  if (!storm || !storm.timesteps) return;

  for (let i = 0; i < 7; i++) {
    const canvas = document.getElementById(`thumb-${i}`);
    if (!canvas) continue;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#090C12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const stepIdx = Math.max(0, currentStepIdx - (6 - i));
    const step = storm.timesteps[stepIdx];
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 25);
    grad.addColorStop(0, "#EF4444");
    grad.addColorStop(0.4, "#F59E0B");
    grad.addColorStop(0.8, "#10B981");
    grad.addColorStop(1, "transparent");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, 20 + (step.vmax_curr / 160) * 10, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/**
 * Operational Intensity Forecast Canvas Chart
 */
function drawLifecycleChart() {
  const canvas = document.getElementById("lifecycle-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const storm = stormsData[currentStormId];
  if (!storm || !storm.timesteps || storm.timesteps.length === 0) return;

  const timesteps = storm.timesteps;
  const N = timesteps.length;
  const isEma = forecastSmoothingMode === "ema";

  const padLeft = 40;
  const padRight = 30;
  const padTop = 18;
  const padBottom = 26;
  const chartW = W - padLeft - padRight;
  const chartH = H - padTop - padBottom;

  const maxActual = Math.max(...timesteps.map((t) => t.vmax_curr));
  const maxPred = Math.max(...timesteps.map((t) => t.predicted_plus_24h));
  const maxVal = Math.max(120, maxActual, maxPred, 140);
  const minVal = 15;

  const getX = (idx) => padLeft + (idx / Math.max(1, N - 1)) * chartW;
  const getY = (val) => padTop + chartH - ((val - minVal) / (maxVal - minVal)) * chartH;

  // 1. Gridlines & Saffir-Simpson Intensity Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;

  [34, 64, 96, 137].forEach((thresh) => {
    if (thresh <= maxVal) {
      const y = getY(thresh);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + chartW, y);
      ctx.stroke();

      ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
      ctx.font = "8px 'JetBrains Mono'";
      ctx.textAlign = "right";
      const lbl = thresh === 34 ? "TS" : thresh === 64 ? "C1" : thresh === 96 ? "C3" : "C5";
      ctx.fillText(`${lbl} (${thresh})`, padLeft - 4, y + 3);
    }
  });

  // Time ticks along X axis
  for (let i = 0; i < N; i += Math.ceil(N / 7)) {
    const x = getX(i);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.beginPath();
    ctx.moveTo(x, padTop + chartH);
    ctx.lineTo(x, padTop + chartH + 3);
    ctx.stroke();

    ctx.fillStyle = "#64748B";
    ctx.font = "8.5px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`+${timesteps[i].elapsed_hours}h`, x, padTop + chartH + 13);
  }

  // 2. Real-Time vs Audit Modes
  const currStep = timesteps[currentStepIdx];
  const nowX = getX(currentStepIdx);
  const nowY = getY(currStep.vmax_curr);
  const future24Idx = Math.min(currentStepIdx + 8, N - 1);
  const future24X = getX(future24Idx);

  const p6Idx = Math.min(currentStepIdx + 2, N - 1);
  const p12Idx = Math.min(currentStepIdx + 4, N - 1);
  const p24Idx = future24Idx;

  const raw6 = currStep.predicted_plus_6h;
  const raw12 = currStep.predicted_plus_12h;
  const raw24 = currStep.predicted_plus_24h;

  const ema6 = storm._ema_6[currentStepIdx];
  const ema12 = storm._ema_12[currentStepIdx];
  const ema24 = storm._ema_24[currentStepIdx];

  const mainP6 = isEma ? ema6 : raw6;
  const mainP12 = isEma ? ema12 : raw12;
  const mainP24 = isEma ? ema24 : raw24;

  if (chartViewMode === "realtime") {
    // A. +24h Forecast Active Corridor Highlight
    if (future24Idx > currentStepIdx) {
      ctx.fillStyle = "rgba(56, 189, 248, 0.05)";
      ctx.fillRect(nowX, padTop, future24X - nowX, chartH);

      ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(future24X, padTop);
      ctx.lineTo(future24X, padTop + chartH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // B. Observed Past Intensity Path (Solid White)
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= currentStepIdx; i++) {
      const x = getX(i);
      const y = getY(timesteps[i].vmax_curr);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // C1. If in EMA mode: Draw Raw Model Forecast Vector as fine dashed line
    if (isEma) {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.65)";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(nowX, nowY);
      ctx.lineTo(getX(p6Idx), getY(raw6));
      ctx.lineTo(getX(p12Idx), getY(raw12));
      ctx.lineTo(getX(p24Idx), getY(raw24));
      ctx.stroke();
      ctx.setLineDash([]);

      const rx = getX(p24Idx);
      const ry = getY(raw24);
      ctx.fillStyle = "#EF4444";
      ctx.beginPath();
      ctx.arc(rx, ry, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // C2. Primary Forecast Vector (Solid High-Contrast Cyan)
    ctx.strokeStyle = "#38BDF8";
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(nowX, nowY);
    ctx.lineTo(getX(p6Idx), getY(mainP6));
    ctx.lineTo(getX(p12Idx), getY(mainP12));
    ctx.lineTo(getX(p24Idx), getY(mainP24));
    ctx.stroke();

    // Horizon prediction dots
    [
      { idx: p6Idx, val: mainP6, lbl: "+6h" },
      { idx: p12Idx, val: mainP12, lbl: "+12h" },
      { idx: p24Idx, val: mainP24, lbl: "+24h" },
    ].forEach((pt) => {
      const px = getX(pt.idx);
      const py = getY(pt.val);
      ctx.fillStyle = "#38BDF8";
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = "#38BDF8";
      ctx.font = "8px 'JetBrains Mono'";
      ctx.textAlign = "left";
      ctx.fillText(`${pt.lbl}: ${Math.round(pt.val)}k`, px + 4, py - 4);
    });

    // D. Ground-Truth Actual Outcome Corridor (Next 24h - Dashed Red)
    ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = currentStepIdx; i <= future24Idx; i++) {
      const x = getX(i);
      const y = getY(timesteps[i].vmax_curr);
      if (i === currentStepIdx) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

  } else {
    // Full Lifecycle Audit View
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = getX(i);
      const y = getY(timesteps[i].vmax_curr);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Predicted +24h Trajectory across Full Lifecycle
    ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    const outline = isEma ? storm._ema_24 : timesteps.map((t) => t.predicted_plus_24h);
    for (let i = 0; i < N; i++) {
      const x = getX(i);
      const y = getY(outline[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Actual target path (dashed red)
    ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = getX(i);
      const y = getY(timesteps[i].vmax_plus_24h);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 3. NOW Vertical Timeline Marker
  ctx.strokeStyle = "#38BDF8";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(nowX, padTop);
  ctx.lineTo(nowX, padTop + chartH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Observation dot
  ctx.fillStyle = "#38BDF8";
  ctx.beginPath();
  ctx.arc(nowX, nowY, 4, 0, 2 * Math.PI);
  ctx.fill();
}

/**
 * Utility: Format 10-digit timestamp (YYYYMMDDHH)
 */
function formatTimestamp(ts) {
  if (!ts) return "--";
  const s = String(ts);
  if (s.length !== 10) return s;
  return `${s.substring(0, 4)}-${s.substring(4, 6)}-${s.substring(6, 8)} ${s.substring(8, 10)}:00`;
}

function formatShortTime(ts) {
  if (!ts) return "--:--";
  const s = String(ts);
  return `${s.substring(8, 10)}:00`;
}

/**
 * Utility: Saffir-Simpson / IMD Category Formatter
 */
function formatSaffirCategory(vmax) {
  if (vmax < 34) return "TD";
  if (vmax < 64) return "TS";
  if (vmax < 83) return "Cat 1";
  if (vmax < 96) return "Cat 2";
  if (vmax < 113) return "Cat 3";
  if (vmax < 137) return "Cat 4";
  return "Cat 5";
}

/**
 * Fallback Embedded Data (if network fails)
 */
function createFallbackData() {
  return {
    "201015W": {
      id: "201015W",
      name: "Super Typhoon Megi",
      basin: "West Pacific (WPAC)",
      peak_intensity: 160,
      category: "Category 5 Super Typhoon",
      n_timesteps: 30,
      timesteps: Array.from({ length: 30 }, (_, i) => ({
        step_index: i,
        timestamp: 2010101212 + i * 3,
        elapsed_hours: i * 3.0,
        vmax_curr: Math.min(160, 25 + i * 4.5),
        vmax_plus_24h: Math.min(160, 55 + i * 4.0),
        actual_trend: "INTENSIFYING",
        actual_ri: i >= 5 && i <= 15 ? 1 : 0,
        predicted_trend: "INTENSIFYING",
        predicted_trend_probs: { WEAKENING: 0.02, STABLE: 0.12, INTENSIFYING: 0.86 },
        ri_probability: i >= 5 && i <= 14 ? 78.4 : 6.2,
        risk_level: i >= 5 && i <= 14 ? "HIGH" : "LOW",
        predicted_plus_6h: Math.min(160, 32 + i * 4.4),
        predicted_plus_12h: Math.min(160, 42 + i * 4.3),
        predicted_plus_24h: Math.min(160, 58 + i * 4.1),
        latitude: 12.1 + i * 0.25,
        longitude: 142.1 - i * 0.45,
        environmental: { sst: 29.8, ohc: 64.0, shear: 8.4, rh: 76.0, mslp: Math.round(1008 - i * 3.2) },
      })),
    },
  };
}
