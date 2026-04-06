"""
lfpy_generator.py — LFPy/NEURON biophysical synthetic spike generator.

Requirements
------------
    pip install LFPy neuron
    # or:  pip install -e ".[lfpy]"

Scientific purpose (docs/MANIFEST.md §12)
-----------------------------------------
LFPy simulates extracellular potentials from a Hodgkin-Huxley ball-and-stick
neuron using NEURON as the ODE solver and the line-source approximation for
the extracellular field.  By varying electrode distance and orientation we
produce a distribution of realistic waveforms whose shapes differ in exactly
the ways expected from recording geometry — amplitude (∝ 1/r), biphasic
ratio (depends on soma/dendrite angle), and waveform width (stable across
geometry, good identity feature).

Cell model
----------
  Soma  : sphere, diam=20 µm, Hodgkin-Huxley (gNa=0.12, gK=0.036 S/cm²)
  Dend  : cylinder, L=200 µm, diam=3 µm, passive leak (g_pas=5e-4 S/cm²)
  Stim  : somatic IClamp, amp=0.5 nA, dur=2 ms, delay=1 ms

Dataset
-------
  For each (distance, angle) pair the electrode is placed at:
    x = distance × cos(angle)
    y = 0
    z = distance × sin(angle)
  with the soma axis along y.

  Multiple noise levels are applied post-simulation to create noise variants
  for spike/noise discrimination benchmarks.

  Outputs saved to output_dir/:
    waveforms.npz   — {"waveforms": float32[N, T], "labels": int32[N]}
    labels.parquet  — DataFrame with per-snippet metadata

Usage
-----
    from spike_discrim.synthetic.lfpy_generator import generate_dataset
    ds = generate_dataset(output_dir="data/synthetic/lfpy_001")
"""
from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Availability check                                                          #
# --------------------------------------------------------------------------- #

try:
    import LFPy          # noqa: F401
    import neuron        # noqa: F401
    LFPY_AVAILABLE = True
except ImportError:
    LFPY_AVAILABLE = False


def _require_lfpy() -> None:
    if not LFPY_AVAILABLE:
        raise ImportError(
            "LFPy and NEURON are required for synthetic spike generation.\n"
            "Install with:  pip install LFPy neuron\n"
            "or:            pip install -e '.[lfpy]'"
        )


# --------------------------------------------------------------------------- #
# Minimal Hodgkin-Huxley ball-and-stick HOC model (inline, no external files)#
# --------------------------------------------------------------------------- #

_HOC_BALL_AND_STICK = """\
// Hodgkin-Huxley ball-and-stick cell for LFPy synthetic waveform generation
// Soma axis: y-direction (pt3d from y=0 to y=20)
// Dendrite:  y=20 to y=220

create soma, dend
connect dend(0), soma(1)

soma {{
  pt3dclear()
  pt3dadd(0, 0, 0, {soma_diam})
  pt3dadd(0, {soma_L}, 0, {soma_diam})
  nseg = 1
  Ra   = 100
  cm   = 1
  insert hh
  gnabar_hh = 0.12
  gkbar_hh  = 0.036
  gl_hh     = 0.0003
  el_hh     = -54.3
}}

dend {{
  pt3dclear()
  pt3dadd(0, {soma_L}, 0, 3)
  pt3dadd(0, {dend_end}, 0, 3)
  nseg = 20
  Ra   = 100
  cm   = 1
  insert pas
  e_pas = -65
  g_pas = 5e-4
}}
"""


def _write_hoc(
    soma_diam: float = 20.0,
    soma_L:    float = 20.0,
    dend_L:    float = 200.0,
) -> str:
    """Write HOC string to a temp file and return the path."""
    hoc = _HOC_BALL_AND_STICK.format(
        soma_diam = soma_diam,
        soma_L    = soma_L,
        dend_end  = soma_L + dend_L,
    )
    fd = tempfile.NamedTemporaryFile(
        suffix=".hoc", mode="w", delete=False, prefix="spike_discrim_cell_"
    )
    fd.write(hoc)
    fd.close()
    return fd.name


# --------------------------------------------------------------------------- #
# Single-waveform generation                                                  #
# --------------------------------------------------------------------------- #

def _simulate_cell(
    hoc_path:        str,
    electrode_x:     float = 50.0,   # µm
    electrode_y:     float = 0.0,
    electrode_z:     float = 0.0,
    sigma:           float = 0.3,    # S/m extracellular conductivity
    dt:              float = 1.0/30, # ms  (→ 30 kHz)
    tstop:           float = 8.0,    # ms  total simulation
    stim_delay:      float = 1.0,    # ms  current-clamp onset
    stim_amp:        float = 0.5,    # nA
    stim_dur:        float = 2.0,    # ms
    snippet_pre:     int   = 20,     # samples before trough
    snippet_post:    int   = 44,     # samples after trough (total = 64)
) -> Optional[np.ndarray]:
    """Run one LFPy/NEURON simulation and return float32 snippet.

    Returns None if the simulation produced no clear trough
    (no action potential elicited).
    """
    import LFPy

    cell_params = {
        "morphology":    hoc_path,
        "v_init":        -65.0,
        "dt":            dt,
        "tstart":        0.0,
        "tstop":         tstop,
        "nsegs_method":  None,
    }
    cell = LFPy.Cell(**cell_params)

    # Somatic current clamp to elicit one action potential
    stim = LFPy.StimIntElectrode(
        cell,
        idx            = cell.get_closest_idx(x=0, y=0, z=0),
        record_current = False,
        pptype         = "IClamp",
        amp            = stim_amp,
        dur            = stim_dur,
        delay          = stim_delay,
    )

    electrode = LFPy.RecExtElectrode(
        cell,
        sigma  = sigma,
        x      = np.array([electrode_x]),
        y      = np.array([electrode_y]),
        z      = np.array([electrode_z]),
        method = "linesource",
    )

    cell.simulate(rec_imem=True)
    electrode.calc_lfp()

    signal = electrode.data[0]   # µV, shape [n_timesteps]

    # Find trough (most negative point) in the second half of the simulation
    # (avoids the stimulus artefact in the first stim_delay ms)
    stim_end_idx = int((stim_delay + stim_dur) / dt)
    search_start = stim_end_idx
    search_region = signal[search_start:]

    if len(search_region) == 0:
        return None

    trough_rel = int(np.argmin(search_region))
    trough_idx = search_start + trough_rel

    # Must have a clear trough (not just noise)
    if signal[trough_idx] > -0.01:   # µV threshold — essentially flat
        return None

    start = max(0, trough_idx - snippet_pre)
    end   = min(len(signal), trough_idx + snippet_post)

    snippet = signal[start:end].astype(np.float32)

    # Zero-pad if snippet is shorter than expected (near-boundary)
    expected = snippet_pre + snippet_post
    if len(snippet) < expected:
        pad = np.zeros(expected - len(snippet), dtype=np.float32)
        snippet = np.concatenate([snippet, pad])

    return snippet


# --------------------------------------------------------------------------- #
# Dataset generation                                                          #
# --------------------------------------------------------------------------- #

def generate_dataset(
    output_dir:       str | Path = "data/synthetic",
    n_distances:      int   = 8,
    n_angles:         int   = 16,
    distance_min_um:  float = 10.0,
    distance_max_um:  float = 120.0,
    noise_levels_uv:  Optional[list[float]] = None,
    soma_diam:        float = 20.0,
    soma_L:           float = 20.0,
    dend_L:           float = 200.0,
    sigma:            float = 0.3,
    dt:               float = 1.0 / 30.0,
    snippet_pre:      int   = 20,
    snippet_post:     int   = 44,
    stim_amp:         float = 0.5,
    seed:             int   = 42,
    verbose:          bool  = True,
) -> dict:
    """Generate a labelled dataset of synthetic extracellular spike waveforms.

    Parameters
    ----------
    output_dir        : Directory to save waveforms.npz and labels.parquet.
    n_distances       : Number of distinct electrode distances.
    n_angles          : Number of distinct electrode angles (0–2π).
    distance_min/max  : Range of electrode distances in µm.
    noise_levels_uv   : RMS noise levels in µV to add (creates separate copies).
                        Default: [0.0, 1.0, 3.0, 5.0].
    soma_diam / soma_L: Soma morphology (µm).  Vary between runs for inter-unit
                        morphological diversity.
    sigma             : Extracellular conductivity (S/m). 0.3 is standard for
                        grey matter (Goto et al. 2010).
    dt                : Simulation timestep in ms.  1/30 → 30 kHz.
    snippet_pre/post  : Samples before/after trough to extract.
    seed              : NumPy random seed for noise generation.

    Returns
    -------
    dict with keys:
        waveforms  : float32[N, T]
        labels     : pandas DataFrame with columns
                     [snippet_id, distance_um, angle_deg, noise_uv_rms,
                      source, soma_diam, soma_L, dend_L, class_label]
    """
    _require_lfpy()

    import pandas as pd

    if noise_levels_uv is None:
        noise_levels_uv = [0.0, 1.0, 3.0, 5.0]

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances = np.linspace(distance_min_um, distance_max_um, n_distances)
    angles    = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    snippet_len = snippet_pre + snippet_post

    # Write HOC once per morphology
    hoc_path = _write_hoc(soma_diam=soma_diam, soma_L=soma_L, dend_L=dend_L)

    clean_snippets: list[np.ndarray]  = []
    meta_rows:      list[dict]         = []
    snippet_id      = 0

    total = n_distances * n_angles
    if verbose:
        print(f"Simulating {total} electrode positions "
              f"({n_distances} distances × {n_angles} angles)...")

    try:
        for i_d, dist in enumerate(distances):
            for i_a, angle in enumerate(angles):
                ex = float(dist * np.cos(angle))
                ey = 0.0
                ez = float(dist * np.sin(angle))

                snippet = _simulate_cell(
                    hoc_path    = hoc_path,
                    electrode_x = ex,
                    electrode_y = ey,
                    electrode_z = ez,
                    sigma       = sigma,
                    dt          = dt,
                    stim_amp    = stim_amp,
                    snippet_pre  = snippet_pre,
                    snippet_post = snippet_post,
                )

                if snippet is None:
                    if verbose:
                        print(f"  WARNING: no AP at dist={dist:.1f} µm, "
                              f"angle={np.degrees(angle):.1f}°; skipping")
                    continue

                clean_snippets.append(snippet)
                meta_rows.append({
                    "snippet_id":   snippet_id,
                    "distance_um":  float(dist),
                    "angle_deg":    float(np.degrees(angle)),
                    "noise_uv_rms": 0.0,
                    "source":       "lfpy_clean",
                    "soma_diam":    soma_diam,
                    "soma_L":       soma_L,
                    "dend_L":       dend_L,
                    "class_label":  1,   # spike
                })
                snippet_id += 1

                if verbose and (i_d * n_angles + i_a + 1) % 20 == 0:
                    pct = 100 * (i_d * n_angles + i_a + 1) / total
                    print(f"  {pct:.0f}%  ({i_d * n_angles + i_a + 1}/{total})")

    finally:
        os.unlink(hoc_path)

    if len(clean_snippets) == 0:
        raise RuntimeError(
            "No valid snippets were generated.  Check LFPy/NEURON installation "
            "and stim_amp parameter."
        )

    clean_array = np.stack(clean_snippets, axis=0)  # float32[N_clean, T]

    # ── Add noise variants ────────────────────────────────────────────────── #
    all_waveforms: list[np.ndarray] = [clean_array]
    all_meta:      list[dict]       = list(meta_rows)

    for noise_uv in noise_levels_uv:
        if noise_uv <= 0.0:
            continue
        noisy = clean_array + rng.normal(
            0.0, noise_uv, size=clean_array.shape
        ).astype(np.float32)
        all_waveforms.append(noisy)
        for base in meta_rows:
            row = base.copy()
            row["snippet_id"]   = snippet_id
            row["noise_uv_rms"] = noise_uv
            row["source"]       = f"lfpy_noise_{noise_uv:.1f}uv"
            all_meta.append(row)
            snippet_id += 1

    waveforms = np.vstack(all_waveforms)        # float32[N_total, T]
    labels_df = pd.DataFrame(all_meta)

    # ── Save outputs ──────────────────────────────────────────────────────── #
    waveforms_path = output_dir / "waveforms.npz"
    labels_path    = output_dir / "labels.parquet"
    meta_path      = output_dir / "generation_config.json"

    np.savez_compressed(
        waveforms_path,
        waveforms   = waveforms,
        class_labels = labels_df["class_label"].values.astype(np.int32),
    )
    labels_df.to_parquet(labels_path, index=False)

    import json
    config = {
        "generator":        "lfpy",
        "n_clean_snippets": len(clean_snippets),
        "n_total_snippets": len(waveforms),
        "snippet_length":   snippet_len,
        "dt_ms":            dt,
        "sampling_rate_hz": round(1.0 / dt * 1000),
        "soma_diam":        soma_diam,
        "soma_L":           soma_L,
        "dend_L":           dend_L,
        "sigma_S_per_m":    sigma,
        "distances_um":     distances.tolist(),
        "noise_levels_uv":  noise_levels_uv,
        "seed":             seed,
    }
    with open(meta_path, "w") as fh:
        json.dump(config, fh, indent=2)

    if verbose:
        print(f"\nDataset saved to: {output_dir}")
        print(f"  waveforms : {waveforms.shape}  float32")
        print(f"  labels    : {labels_path}")
        print(f"  config    : {meta_path}")

    return {"waveforms": waveforms, "labels": labels_df}
