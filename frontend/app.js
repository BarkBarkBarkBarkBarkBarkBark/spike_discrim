/* app.js — spike_discrim dashboard frontend
 *
 * Pure vanilla JS + Chart.js (loaded from CDN).
 * Communicates exclusively with /api/* endpoints — no direct data access.
 * Fully replaceable without touching the backend.
 */

const API = "";          // same-origin; override to "http://localhost:8000" for dev
const CHART_DEFAULTS = {
  animation: false,
  plugins: { legend: { labels: { color: "#d4d8f0", font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: "#6b7099", font: { size: 10 } }, grid: { color: "#2a2d3e" } },
    y: { ticks: { color: "#6b7099", font: { size: 10 } }, grid: { color: "#2a2d3e" } },
  },
};

// ── Chart instances (kept so we can destroy & re-draw on run change) ─────── //
let charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); charts[id] = null; }
}

// ── API helpers ───────────────────────────────────────────────────────────── //
async function apiFetch(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${path}`);
  return r.json();
}

// ── Initialise ────────────────────────────────────────────────────────────── //
document.addEventListener("DOMContentLoaded", async () => {
  await loadRuns();
  document.getElementById("refresh-btn").addEventListener("click", loadRuns);
  document.getElementById("run-select").addEventListener("change", onRunChange);
  document.getElementById("run-btn").addEventListener("click", triggerPipeline);
  document.querySelectorAll(".validate-btn").forEach(btn => {
    btn.addEventListener("click", () => runValidation(btn.dataset.action));
  });
});

// ── Run list ──────────────────────────────────────────────────────────────── //
async function loadRuns() {
  const sel = document.getElementById("run-select");
  try {
    const runs = await apiFetch("/api/runs");
    sel.innerHTML = runs.length
      ? runs.map(r => `<option value="${r}">${r}</option>`).join("")
      : `<option value="">— no runs found —</option>`;
    if (runs.length) onRunChange();
  } catch (e) {
    sel.innerHTML = `<option value="">Error: ${e.message}</option>`;
  }
}

function selectedRun() {
  return document.getElementById("run-select").value;
}

async function onRunChange() {
  const run = selectedRun();
  if (!run) return;
  // Update CSV export link
  document.getElementById("csv-export-link").href = `/api/runs/${run}/export/csv`;
  // Load all panels in parallel
  await Promise.allSettled([
    loadSummary(run),
    loadFeatureChart(run),
    loadClassifierChart(run),
    loadProfilingChart(run),
    loadWaveformChart(run),
  ]);
}

// ── Summary cards ─────────────────────────────────────────────────────────── //
async function loadSummary(run) {
  const section = document.getElementById("summary-section");
  try {
    const s = await apiFetch(`/api/runs/${run}/summary`);
    section.classList.remove("hidden");
    document.getElementById("summary-cards").innerHTML = [
      stat("WeightBank AUC", fmt(s.weight_bank_auc, 3)),
      stat("Top feature", s.top_feature || "—", ""),
      stat("Top feature set", s.top_feature_set || "—", ""),
      stat("Snippets", s.n_snippets, `${s.n_spikes} spike / ${s.n_noise} noise`),
      stat("Features", s.n_features),
      stat("Finished", (s.finished_at || "").replace("T", " ").slice(0, 19), "UTC"),
    ].join("");
  } catch {
    section.classList.add("hidden");
  }
}

function stat(label, value, sub = "") {
  return `<div class="stat-card">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
    ${sub ? `<div class="sub">${sub}</div>` : ""}
  </div>`;
}

function fmt(v, decimals = 3) {
  return (v == null) ? "—" : Number(v).toFixed(decimals);
}

// ── Feature ranking chart (horizontal bar) ────────────────────────────────── //
async function loadFeatureChart(run) {
  destroyChart("feature");
  try {
    const rows = await apiFetch(`/api/runs/${run}/features/single`);
    const top  = rows.slice(0, 14);   // top 14 features for readability
    const ctx  = document.getElementById("feature-chart").getContext("2d");
    charts["feature"] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: top.map(r => r.feature),
        datasets: [{
          label: "Fisher score",
          data:  top.map(r => r.fisher_score),
          backgroundColor: top.map((_, i) => i === 0 ? "#4a9eff" : "#2a4a7f"),
          borderRadius: 3,
        }, {
          label: "AUC",
          data:  top.map(r => r.auc),
          backgroundColor: top.map((_, i) => i === 0 ? "#4ade80aa" : "#1a4a2faa"),
          borderRadius: 3,
        }],
      },
      options: {
        ...CHART_DEFAULTS,
        indexAxis: "y",
        scales: {
          x: { ...CHART_DEFAULTS.scales.x, min: 0, title: { display: true, text: "Score", color: "#6b7099" } },
          y: { ...CHART_DEFAULTS.scales.y },
        },
      },
    });
  } catch (e) { console.warn("feature chart:", e.message); }
}

// ── Classifier benchmark chart (grouped bar) ──────────────────────────────── //
async function loadClassifierChart(run) {
  destroyChart("classifier");
  try {
    const rows    = await apiFetch(`/api/runs/${run}/features/sets`);
    const setNames = [...new Set(rows.map(r => r.set_name))];
    const models   = [...new Set(rows.map(r => r.model))];
    const COLORS   = ["#4a9eff","#4ade80","#fbbf24","#a78bfa","#ff6b6b","#34d399"];

    const datasets = models.map((model, i) => ({
      label: model,
      data:  setNames.map(s => {
        const row = rows.find(r => r.set_name === s && r.model === model);
        return row ? row.balanced_acc_mean : null;
      }),
      backgroundColor: COLORS[i % COLORS.length] + "cc",
      borderRadius: 3,
    }));

    const ctx = document.getElementById("classifier-chart").getContext("2d");
    charts["classifier"] = new Chart(ctx, {
      type: "bar",
      data: { labels: setNames, datasets },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: { ...CHART_DEFAULTS.scales.x },
          y: { ...CHART_DEFAULTS.scales.y, min: 0.5, max: 1.0,
               title: { display: true, text: "Balanced Acc.", color: "#6b7099" } },
        },
      },
    });
  } catch (e) { console.warn("classifier chart:", e.message); }
}

// ── Kernel profiling chart (horizontal bar, M snippets/s) ─────────────────── //
async function loadProfilingChart(run) {
  destroyChart("profiling");
  try {
    const data   = await apiFetch(`/api/runs/${run}/profiling`);
    const labels = Object.keys(data);
    const values = labels.map(k => data[k].throughput_ksnippets_per_sec / 1000); // → M/s
    const ops    = labels.map(k => data[k].total_arith_ops_per_sample);

    const ctx = document.getElementById("profiling-chart").getContext("2d");
    charts["profiling"] = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "M snippets / s",
          data: values,
          backgroundColor: "#4a9effcc",
          borderRadius: 3,
        }],
      },
      options: {
        ...CHART_DEFAULTS,
        indexAxis: "y",
        plugins: {
          ...CHART_DEFAULTS.plugins,
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => `arith ops/sample: ${ops[ctx.dataIndex]}`,
            },
          },
        },
        scales: {
          x: { ...CHART_DEFAULTS.scales.x, min: 0,
               title: { display: true, text: "Throughput (M snippets/s)", color: "#6b7099" } },
          y: { ...CHART_DEFAULTS.scales.y },
        },
      },
    });
  } catch (e) { console.warn("profiling chart:", e.message); }
}

// ── Waveform gallery (overlaid line chart) ────────────────────────────────── //
async function loadWaveformChart(run) {
  destroyChart("waveform");
  try {
    const resp   = await apiFetch(`/api/runs/${run}/waveforms?n=200`);
    const waves  = resp.waveforms;
    const labels = resp.labels;
    const T      = resp.n_samples;
    const xAxis  = Array.from({ length: T }, (_, i) => i);

    // Show first 12 spikes + first 12 noise (max 24 lines to keep chart readable)
    const spikes = waves.filter((_, i) => labels[i] === 1).slice(0, 12);
    const noise  = waves.filter((_, i) => labels[i] === 0).slice(0, 12);

    const spikeDS = spikes.map((w, i) => ({
      label:       i === 0 ? "spike" : null,
      data:        w,
      borderColor: "#4a9eff55",
      borderWidth: 1,
      pointRadius: 0,
      tension:     0.2,
      showLine:    true,
    }));
    const noiseDS = noise.map((w, i) => ({
      label:       i === 0 ? "noise" : null,
      data:        w,
      borderColor: "#ff6b6b55",
      borderWidth: 1,
      pointRadius: 0,
      tension:     0.2,
      showLine:    true,
    }));

    const ctx = document.getElementById("waveform-chart").getContext("2d");
    charts["waveform"] = new Chart(ctx, {
      type: "line",
      data: { labels: xAxis, datasets: [...spikeDS, ...noiseDS] },
      options: {
        ...CHART_DEFAULTS,
        animation: false,
        plugins: {
          legend: {
            labels: {
              color: "#d4d8f0",
              font: { size: 11 },
              // Only show labelled datasets (first spike, first noise)
              filter: item => item.text !== null,
            },
          },
        },
        scales: {
          x: { ...CHART_DEFAULTS.scales.x,
               title: { display: true, text: "Sample", color: "#6b7099" } },
          y: { ...CHART_DEFAULTS.scales.y,
               title: { display: true, text: "Amplitude (µV)", color: "#6b7099" } },
        },
      },
    });
  } catch (e) { console.warn("waveform chart:", e.message); }
}

// ── Pipeline trigger ──────────────────────────────────────────────────────── //
async function triggerPipeline() {
  const tier   = parseInt(document.getElementById("tier-select").value);
  const btn    = document.getElementById("run-btn");
  const status = document.getElementById("job-status");

  btn.disabled = true;
  status.textContent = "submitting…";
  status.className   = "job-status";

  try {
    const job = await fetch(API + "/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tier }),
    }).then(r => r.json());

    status.textContent = `job ${job.job_id} — queued`;
    pollJob(job.job_id, btn, status);
  } catch (e) {
    status.textContent = `error: ${e.message}`;
    status.className   = "job-status failed";
    btn.disabled       = false;
  }
}

async function pollJob(jobId, btn, statusEl) {
  const poll = setInterval(async () => {
    try {
      const job = await apiFetch(`/api/pipeline/status/${jobId}`);
      if (job.status === "running") {
        const tail = job.log_tail.at(-1) || "";
        statusEl.textContent = `job ${jobId} — ${tail.slice(0, 60)}`;
      } else if (job.status === "done") {
        clearInterval(poll);
        statusEl.textContent = `✓ done — run ${job.run_id}`;
        statusEl.className   = "job-status done";
        btn.disabled         = false;
        await loadRuns();
        // Auto-select new run
        const sel = document.getElementById("run-select");
        if (job.run_id) {
          sel.value = job.run_id;
          onRunChange();
        }
      } else if (job.status === "failed") {
        clearInterval(poll);
        statusEl.textContent = `✗ failed — see log`;
        statusEl.className   = "job-status failed";
        btn.disabled         = false;
      }
    } catch {
      clearInterval(poll);
      btn.disabled = false;
    }
  }, 1500);
}

// ── Validation panel ──────────────────────────────────────────────────────── //
async function runValidation(action) {
  const out = document.getElementById("validate-output");
  const run = selectedRun();

  out.classList.remove("hidden");
  out.innerHTML = `<span class="warn">Loading ${action}…</span>`;

  try {
    let url;
    if (action === "waveform_checksums") {
      url = `/api/validate/waveform_checksums`;
    } else if (action === "checksums") {
      url = `/api/validate/checksums/${run}`;
    } else if (action === "metrics") {
      url = `/api/validate/metrics/${run}`;
    } else if (action === "feature_stats") {
      url = `/api/validate/feature_stats/${run}`;
    } else if (action === "roundtrip") {
      url = `/api/validate/roundtrip/${run}`;
    }

    const data = await apiFetch(url);
    out.innerHTML = formatValidationResult(action, data);
  } catch (e) {
    out.innerHTML = `<span class="error">Error: ${e.message}</span>`;
  }
}

function formatValidationResult(action, data) {
  if (action === "metrics") {
    const ok = data.overall_match;
    const badge = ok
      ? `<span class="ok">✓ ALL MATCH (tolerance ${data.tolerance})</span>`
      : `<span class="error">✗ MISMATCH DETECTED</span>`;
    return `${badge}

AUC
  stored     : ${data.auc.stored.toFixed(6)}
  recomputed : ${data.auc.recomputed.toFixed(6)}
  delta      : ${data.auc.delta.toExponential(3)}  ${data.auc.match ? "✓" : "✗"}

Balanced accuracy
  stored     : ${data.balanced_accuracy.stored.toFixed(6)}
  recomputed : ${data.balanced_accuracy.recomputed.toFixed(6)}
  delta      : ${data.balanced_accuracy.delta.toExponential(3)}  ${data.balanced_accuracy.match ? "✓" : "✗"}`;
  }

  if (action === "roundtrip") {
    const ok = data.all_match;
    const badge = ok
      ? `<span class="ok">✓ CSV round-trip is lossless</span>`
      : `<span class="error">✗ Round-trip mismatches detected</span>`;
    const lines = data.files.map(f =>
      `  ${f.match ? "✓" : "✗"} ${f.file}  (${f.rows} rows, ${f.cols_checked} cols checked)`
      + (f.mismatches?.length ? "\n    mismatches: " + JSON.stringify(f.mismatches) : "")
    ).join("\n");
    return `${badge}\n\n${lines}`;
  }

  if (action === "feature_stats") {
    const lines = [`run: ${data.run_id}  |  spikes: ${data.n_spikes}  noise: ${data.n_noise}\n`];
    for (const [feat, stats] of Object.entries(data.features)) {
      lines.push(`${feat}`);
      lines.push(`  spike  mean=${stats.spike.mean.toFixed(3)}  std=${stats.spike.std.toFixed(3)}  [${stats.spike.min.toFixed(2)}, ${stats.spike.max.toFixed(2)}]`);
      lines.push(`  noise  mean=${stats.noise.mean.toFixed(3)}  std=${stats.noise.std.toFixed(3)}  [${stats.noise.min.toFixed(2)}, ${stats.noise.max.toFixed(2)}]`);
    }
    return lines.join("\n");
  }

  // Checksums and waveform_checksums
  return JSON.stringify(data, null, 2);
}
