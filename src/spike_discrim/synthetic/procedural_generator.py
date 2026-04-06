"""
procedural_generator.py — Physiologically realistic extracellular spike generator.

Waveform model
--------------
Previous model used symmetric Gaussians, which produced sinusoidal-looking
waveforms.  Real extracellular action potentials are strongly asymmetric:

  1. Fast depolarisation onset  (τ_rise ≈ 0.1–0.2 ms  →  3–6 samples at 30 kHz)
  2. Slower repolarisation      (τ_fall ≈ 0.3–0.7 ms  → 10–20 samples)
  3. Positive repolarisation peak  (current redistribution, ~30–50% of trough)
  4. After-hyperpolarisation (AHP): small, broad positive tail (~5–15 µV)

The trough shape is modelled by a double-exponential (alpha-function) kernel:

    trough(t) = -A · [exp(-(t-t0)/τ_fall) - exp(-(t-t0)/τ_rise)]  for t ≥ t0
             ≈ 0                                                    for t < t0

This gives the characteristic "shark fin" shape:  sharp rise, asymmetric decay.
The positive repolarisation peak and AHP remain Gaussian (smoother physiology).

References: Gold et al. (2006) J. Neurophysiol.; Pettersen & Einevoll (2008)
Biophys. J.; Quiroga et al. (2004) Neural Comput.

Noise model
-----------
Unchanged: Gaussian white noise, coloured noise, clipped square-wave artifacts,
and multi-peak irregular artifacts — all lacking clean biphasic morphology.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# =========================================================================== #
# Kernel helpers                                                              #
# =========================================================================== #

def _gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / (sigma + 1e-12)) ** 2)


def _double_exp_kernel(
    t:        np.ndarray,
    t0:       float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """Asymmetric double-exponential pulse, peak normalised to +1.0.

    Models the current-sink onset of an extracellular AP:
        kernel(t) = exp(-(t-t0)/τ_fall) - exp(-(t-t0)/τ_rise)   for t ≥ t0
                  = 0                                             for t < t0

    At 30 kHz (1 sample = 33 µs):
        τ_rise = 3 samples  → onset in ~0.1 ms  (fast Na+ depolarisation)
        τ_fall = 12 samples → decay  in ~0.4 ms  (K+/Na+ repolarisation)

    Peak location: t0 + τ_rise·τ_fall/(τ_fall-τ_rise) · ln(τ_fall/τ_rise)
    """
    dt  = t - t0
    out = np.where(
        dt >= 0.0,
        np.exp(-np.maximum(dt, 0) / tau_fall) - np.exp(-np.maximum(dt, 0) / tau_rise),
        np.zeros_like(t),
    )
    peak = out.max()
    if peak > 1e-12:
        out /= peak
    return out.astype(np.float32)


def _peak_offset(tau_rise: float, tau_fall: float) -> float:
    """Samples from t0 to the peak of the double-exponential kernel."""
    if tau_fall <= tau_rise:
        return 0.0
    return tau_rise * tau_fall / (tau_fall - tau_rise) * np.log(tau_fall / tau_rise)


# =========================================================================== #
# Realistic waveform builder                                                  #
# =========================================================================== #

def make_realistic_waveform(
    n_samples:    int   = 64,
    pre_peak:     int   = 20,        # sample index of the negative trough peak
    # Negative trough (double-exponential — asymmetric)
    amp_neg:      float = -100.0,    # µV  (should be negative)
    tau_rise:     float = 3.0,       # samples — onset sharpness
    tau_fall:     float = 12.0,      # samples — decay speed
    # Positive repolarisation peak (Gaussian — smoother biology)
    amp_pos:      float = 40.0,      # µV
    delay_pos:    int   = 14,        # samples after trough peak
    width_pos:    float = 7.0,       # Gaussian σ
    # After-hyperpolarisation (AHP): slow, small positive tail
    amp_ahp:      float = 8.0,       # µV  (0 = disable)
    delay_ahp:    int   = 28,        # samples after trough peak
    width_ahp:    float = 10.0,      # broad Gaussian
    # Optional small pre-potential
    amp_pre:      float = 5.0,       # µV  (0 = disable)
    pre_offset:   int   = -6,        # samples before trough onset
    width_pre:    float = 3.0,
    # Noise
    noise_std:    float = 0.0,       # µV RMS
    rng:          Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a single physiologically realistic extracellular spike snippet.

    Shape summary
    -------------
    - Samples 0..(pre_peak-τ_rise-2)  : quiet baseline
    - Samples ~(pre_peak-4)..pre_peak  : sharp depolarisation onset
    - Sample  pre_peak                  : negative trough (min)
    - Samples pre_peak..(pre_peak+12)  : asymmetric repolarisation (slower)
    - Samples (pre_peak+delay_pos)      : positive repolarisation peak
    - Samples (pre_peak+delay_ahp)      : small AHP hump

    Parameters
    ----------
    tau_rise   : Double-exp rise constant (samples).
                 3 samples ≈ 0.10 ms → narrow-spiking interneuron.
                 6 samples ≈ 0.20 ms → broad-spiking pyramidal cell.
    tau_fall   : Double-exp fall constant (samples).
                 10–14 → fast repolarisation; 16–22 → slow.
    delay_pos  : Trough-to-peak interval (samples).  Narrow cells: 8–12,
                 broad cells: 14–20.
    amp_ahp    : AHP amplitude.  Typical: 5–15 µV.  Set 0 to disable.
    """
    t = np.arange(n_samples, dtype=np.float32)

    # t0 = onset sample so that the double-exp peak falls at pre_peak
    delta = _peak_offset(tau_rise, tau_fall)
    t0    = float(pre_peak) - delta

    # ── Negative trough (double-exponential) ──────────────────────────────── #
    trough = amp_neg * _double_exp_kernel(t, t0, tau_rise, tau_fall)

    # ── Positive repolarisation peak (Gaussian) ───────────────────────────── #
    t_pos = float(pre_peak + delay_pos)
    repol = amp_pos * _gaussian(t, t_pos, width_pos)

    # ── After-hyperpolarisation (AHP) ─────────────────────────────────────── #
    ahp = np.zeros(n_samples, dtype=np.float32)
    if amp_ahp > 0.0:
        t_ahp = float(pre_peak + delay_ahp)
        ahp   = amp_ahp * _gaussian(t, t_ahp, width_ahp)

    # ── Pre-potential ─────────────────────────────────────────────────────── #
    pre = np.zeros(n_samples, dtype=np.float32)
    if amp_pre > 0.0:
        t_pre = float(pre_peak + pre_offset)
        pre   = amp_pre * _gaussian(t, t_pre, width_pre)

    w = (trough + repol + ahp + pre).astype(np.float32)

    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        w = w + rng.normal(0.0, noise_std, n_samples).astype(np.float32)

    return w


# Backward-compatible alias
def make_biphasic_waveform(
    n_samples:    int   = 64,
    pre_peak:     int   = 20,
    amp_neg:      float = -100.0,
    width_neg:    float = 5.0,      # ignored — kept for API compatibility
    amp_pos:      float = 40.0,
    delay_pos:    int   = 12,
    width_pos:    float = 8.0,
    amp_pre:      float = 10.0,
    pre_offset:   int   = -8,
    width_pre:    float = 4.0,
    noise_std:    float = 0.0,
    rng:          Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """Alias kept for test compatibility. Delegates to make_realistic_waveform."""
    return make_realistic_waveform(
        n_samples  = n_samples,
        pre_peak   = pre_peak,
        amp_neg    = amp_neg,
        amp_pos    = amp_pos,
        delay_pos  = delay_pos,
        width_pos  = width_pos,
        amp_pre    = amp_pre,
        pre_offset = pre_offset,
        width_pre  = width_pre,
        noise_std  = noise_std,
        rng        = rng,
    )


# =========================================================================== #
# Noise event generators                                                      #
# =========================================================================== #

def make_noise_waveform(
    noise_type:  str   = "gaussian",   # "gaussian" | "colored" | "clipped" | "irregular"
    n_samples:   int   = 64,
    amplitude:   float = 50.0,
    rng:         Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a noise-class waveform snippet.

    Types
    -----
    gaussian  : White Gaussian noise — high zero-crossing count.
    colored   : Pink-ish noise (first-difference of Gaussian) — more
                spatially correlated than white.
    clipped   : Saturated square-wave artifact (e.g. motion artifact).
    irregular : Multi-peak artifact with random Gaussian lobes.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_samples, dtype=np.float32)

    if noise_type == "gaussian":
        w = rng.normal(0.0, amplitude / 3.0, n_samples).astype(np.float32)

    elif noise_type == "colored":
        white = rng.normal(0.0, amplitude / 3.0, n_samples + 1)
        w = np.diff(white).astype(np.float32)
        w *= amplitude / (np.std(w) + 1e-12)

    elif noise_type == "clipped":
        # Square-wave saturated artifact
        period = rng.integers(8, 20)
        w = np.sign(np.sin(2 * np.pi * t / period)).astype(np.float32)
        w *= amplitude * rng.uniform(0.3, 1.0)

    elif noise_type == "irregular":
        # 3–6 random Gaussian bumps of random polarity and width
        w = np.zeros(n_samples, dtype=np.float32)
        n_bumps = rng.integers(3, 7)
        for _ in range(n_bumps):
            mu  = float(rng.uniform(0, n_samples))
            sig = float(rng.uniform(2, 10))
            amp = float(rng.uniform(-amplitude, amplitude))
            w   += amp * _gaussian(t, mu, sig)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    return w.astype(np.float32)


# =========================================================================== #
# Dataset generation                                                          #
# =========================================================================== #

from typing import Optional


def generate_dataset(
    output_dir:         str   = "data/synthetic",
    n_spikes_per_unit:  int   = 300,
    n_units:            int   = 3,
    n_noise:            int   = 300,
    n_samples:          int   = 64,
    noise_std_range:    tuple = (0.0, 10.0),  # µV RMS range for spike noise
    noise_types:        list  = None,
    seed:               int   = 42,
    verbose:            bool  = True,
) -> dict:
    """Generate a multi-unit + noise benchmarking dataset.

    Creates n_units synthetic units with distinct waveform morphologies and
    n_noise noise events.

    Unit morphologies are varied by:
      - trough amplitude   (different neurons have different amplitudes)
      - trough-to-peak delay  (width of the action potential)
      - repolarisation ratio  (thin fast vs broad slow neurons)

    Parameters
    ----------
    n_spikes_per_unit : Spikes per unit.  Total spikes = n_units × n_spikes.
    n_units           : Number of distinct simulated single units.
    n_noise           : Total noise events.
    noise_std_range   : (min, max) Gaussian noise RMS added to each spike.
    noise_types       : List of noise_type strings passed to make_noise_waveform.
    seed              : NumPy random seed.
    verbose           : Print progress.

    Returns
    -------
    dict:
        waveforms    : float32[N_total, n_samples]
        labels       : int32[N_total]  — unit id (1..n_units) or 0 (noise)
        unit_ids     : int32[N_total]
        source       : list[str]
    """
    from pathlib import Path
    import pandas as pd
    import json

    if noise_types is None:
        noise_types = ["gaussian", "colored", "clipped", "irregular"]

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Per-unit morphology — three biologically distinct cell classes     #
    # ------------------------------------------------------------------ #
    # Each simulated unit is drawn from one of three physiological types,
    # cycling through them so n_units=3 gives one of each.
    #
    #  Type 0 — Fast-spiking interneuron (FS/PV+)
    #    Narrow spike: τ_rise ~2–3 samples, τ_fall ~7–11 samples.
    #    Large amplitude, short trough-to-peak interval.
    #
    #  Type 1 — Regular-spiking pyramidal cell (RS)
    #    Broad spike: τ_rise ~4–6 samples, τ_fall ~14–22 samples.
    #    Smaller amplitude, long trough-to-peak, prominent AHP.
    #
    #  Type 2 — Intermediate bursting cell (IB)
    #    Between the two extremes.

    cell_type_templates = [
        # ── Type 0: Fast-spiking interneuron ──────────────────────────
        dict(
            tau_rise_lo=2.0,  tau_rise_hi=3.5,
            tau_fall_lo=7.0,  tau_fall_hi=11.0,
            amp_neg_lo=-150.0, amp_neg_hi=-100.0,
            amp_pos_lo=40.0,  amp_pos_hi=70.0,
            delay_pos_lo=8,   delay_pos_hi=12,
            width_pos_lo=5.0, width_pos_hi=8.0,
            amp_ahp_lo=3.0,   amp_ahp_hi=10.0,
            delay_ahp_lo=20,  delay_ahp_hi=26,
            amp_pre_lo=2.0,   amp_pre_hi=8.0,
        ),
        # ── Type 1: Regular-spiking pyramidal ─────────────────────────
        dict(
            tau_rise_lo=4.0,  tau_rise_hi=6.5,
            tau_fall_lo=14.0, tau_fall_hi=22.0,
            amp_neg_lo=-90.0, amp_neg_hi=-50.0,
            amp_pos_lo=20.0,  amp_pos_hi=40.0,
            delay_pos_lo=14,  delay_pos_hi=20,
            width_pos_lo=7.0, width_pos_hi=11.0,
            amp_ahp_lo=6.0,   amp_ahp_hi=16.0,
            delay_ahp_lo=28,  delay_ahp_hi=36,
            amp_pre_lo=4.0,   amp_pre_hi=14.0,
        ),
        # ── Type 2: Intermediate bursting ─────────────────────────────
        dict(
            tau_rise_lo=3.0,  tau_rise_hi=4.5,
            tau_fall_lo=11.0, tau_fall_hi=15.0,
            amp_neg_lo=-120.0, amp_neg_hi=-70.0,
            amp_pos_lo=30.0,  amp_pos_hi=55.0,
            delay_pos_lo=11,  delay_pos_hi=15,
            width_pos_lo=6.0, width_pos_hi=9.5,
            amp_ahp_lo=4.0,   amp_ahp_hi=13.0,
            delay_ahp_lo=24,  delay_ahp_hi=32,
            amp_pre_lo=3.0,   amp_pre_hi=10.0,
        ),
    ]

    unit_params = []
    for u in range(n_units):
        tmpl = cell_type_templates[u % len(cell_type_templates)]
        unit_params.append({
            "tau_rise":  float(rng.uniform(tmpl["tau_rise_lo"],  tmpl["tau_rise_hi"])),
            "tau_fall":  float(rng.uniform(tmpl["tau_fall_lo"],  tmpl["tau_fall_hi"])),
            "amp_neg":   float(rng.uniform(tmpl["amp_neg_lo"],   tmpl["amp_neg_hi"])),
            "amp_pos":   float(rng.uniform(tmpl["amp_pos_lo"],   tmpl["amp_pos_hi"])),
            "delay_pos": int(rng.integers(tmpl["delay_pos_lo"],  tmpl["delay_pos_hi"] + 1)),
            "width_pos": float(rng.uniform(tmpl["width_pos_lo"], tmpl["width_pos_hi"])),
            "amp_ahp":   float(rng.uniform(tmpl["amp_ahp_lo"],   tmpl["amp_ahp_hi"])),
            "delay_ahp": int(rng.integers(tmpl["delay_ahp_lo"],  tmpl["delay_ahp_hi"] + 1)),
            "amp_pre":   float(rng.uniform(tmpl["amp_pre_lo"],   tmpl["amp_pre_hi"])),
            "cell_type": u % len(cell_type_templates),
        })

    all_waveforms: list[np.ndarray] = []
    all_labels:    list[int]        = []
    all_unit_ids:  list[int]        = []
    all_sources:   list[str]        = []
    meta_rows:     list[dict]       = []
    snippet_id     = 0

    # Generate spikes
    for u, params in enumerate(unit_params):
        unit_label = u + 1   # 1-indexed; 0 = noise
        cell_type_names = {0: "fast-spiking", 1: "regular-spiking", 2: "intermediate"}
        ct_name = cell_type_names.get(params.get("cell_type", 0), "unknown")
        if verbose:
            print(f"  Unit {unit_label}/{n_units} [{ct_name}]: "
                  f"amp_neg={params['amp_neg']:.1f} µV, "
                  f"τ_rise={params['tau_rise']:.1f} τ_fall={params['tau_fall']:.1f} smp, "
                  f"delay_pos={params['delay_pos']} smp")
        for _ in range(n_spikes_per_unit):
            ns = float(rng.uniform(*noise_std_range))
            w  = make_realistic_waveform(
                n_samples  = n_samples,
                amp_neg    = params["amp_neg"],
                tau_rise   = params["tau_rise"],
                tau_fall   = params["tau_fall"],
                amp_pos    = params["amp_pos"],
                delay_pos  = params["delay_pos"],
                width_pos  = params["width_pos"],
                amp_ahp    = params["amp_ahp"],
                delay_ahp  = params["delay_ahp"],
                amp_pre    = params["amp_pre"],
                noise_std  = ns,
                rng        = rng,
            )
            all_waveforms.append(w)
            all_labels.append(1)       # spike class
            all_unit_ids.append(unit_label)
            all_sources.append("procedural_spike")
            meta_rows.append({
                "snippet_id":   snippet_id,
                "unit_id":      unit_label,
                "class_label":  1,
                "noise_std":    ns,
                "source":       "procedural_spike",
                **{k: float(v) if not isinstance(v, int) else v
                   for k, v in params.items()},
            })
            snippet_id += 1

    # Generate noise
    noise_per_type = max(1, n_noise // len(noise_types))
    for noise_type in noise_types:
        amp = float(rng.uniform(30.0, 120.0))
        for _ in range(noise_per_type):
            w = make_noise_waveform(
                noise_type = noise_type,
                n_samples  = n_samples,
                amplitude  = amp,
                rng        = rng,
            )
            all_waveforms.append(w)
            all_labels.append(0)     # noise class
            all_unit_ids.append(0)
            all_sources.append(f"noise_{noise_type}")
            meta_rows.append({
                "snippet_id":  snippet_id,
                "unit_id":     0,
                "class_label": 0,
                "noise_type":  noise_type,
                "amplitude":   amp,
                "source":      f"noise_{noise_type}",
            })
            snippet_id += 1

    waveforms = np.stack(all_waveforms, axis=0)
    labels    = np.array(all_labels,   dtype=np.int32)
    unit_ids  = np.array(all_unit_ids, dtype=np.int32)
    labels_df = pd.DataFrame(meta_rows)

    # Shuffle
    idx = rng.permutation(len(waveforms))
    waveforms  = waveforms[idx]
    labels     = labels[idx]
    unit_ids   = unit_ids[idx]
    labels_df  = labels_df.iloc[idx].reset_index(drop=True)

    # Save
    waveforms_path = output_dir / "waveforms.npz"
    labels_path    = output_dir / "labels.parquet"
    config_path    = output_dir / "generation_config.json"

    np.savez_compressed(
        waveforms_path,
        waveforms    = waveforms,
        class_labels = labels,
        unit_ids     = unit_ids,
    )
    labels_df.to_parquet(labels_path, index=False)

    config = {
        "generator":         "procedural_dblexp",
        "model":             "double_exponential_AHP",
        "n_units":           n_units,
        "n_spikes_per_unit": n_spikes_per_unit,
        "n_noise":           n_noise,
        "n_samples":         n_samples,
        "noise_std_range":   list(noise_std_range),
        "noise_types":       noise_types,
        "seed":              seed,
        "unit_params":       [
            {k: float(v) if not isinstance(v, int) else v
             for k, v in p.items()} for p in unit_params
        ],
    }
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)

    if verbose:
        print(f"\nProcedural dataset saved to: {output_dir}")
        print(f"  waveforms : {waveforms.shape}  float32")
        print(f"  spikes    : {int(labels.sum())}  "
              f"(noise: {int((labels == 0).sum())})")

    return {
        "waveforms": waveforms,
        "labels":    labels,
        "unit_ids":  unit_ids,
        "labels_df": labels_df,
    }
