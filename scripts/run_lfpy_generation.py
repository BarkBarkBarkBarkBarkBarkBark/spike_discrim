"""
run_lfpy_generation.py — Generate synthetic LFPy spike dataset.

Usage
-----
    python scripts/run_lfpy_generation.py [options]

    --output-dir     data/synthetic/lfpy_001
    --n-distances    8         (electrode distances to simulate)
    --n-angles       16        (electrode angles per distance)
    --dist-min       10        (µm)
    --dist-max       120       (µm)
    --noise-levels   0 1 3 5   (µV RMS; space-separated)
    --soma-diam      20        (µm)
    --soma-L         20        (µm)
    --dend-L         200       (µm)
    --seed           42

Requires:  pip install LFPy neuron
       or: pip install -e '.[lfpy]'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate LFPy/NEURON synthetic extracellular spike dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",    default="data/synthetic/lfpy",
                   help="Output directory")
    p.add_argument("--n-distances",   type=int,   default=8)
    p.add_argument("--n-angles",      type=int,   default=16)
    p.add_argument("--dist-min",      type=float, default=10.0,
                   help="Minimum electrode distance (µm)")
    p.add_argument("--dist-max",      type=float, default=120.0,
                   help="Maximum electrode distance (µm)")
    p.add_argument("--noise-levels",  type=float, nargs="+",
                   default=[0.0, 1.0, 3.0, 5.0],
                   help="Noise levels in µV RMS")
    p.add_argument("--soma-diam",     type=float, default=20.0)
    p.add_argument("--soma-L",        type=float, default=20.0)
    p.add_argument("--dend-L",        type=float, default=200.0)
    p.add_argument("--stim-amp",      type=float, default=0.5,
                   help="IClamp amplitude (nA)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    # Confirm LFPy is available before starting
    try:
        import LFPy   # noqa: F401
        import neuron  # noqa: F401
    except ImportError:
        print("ERROR: LFPy / NEURON not found.", file=sys.stderr)
        print("Install with:  pip install LFPy neuron", file=sys.stderr)
        print("           or: pip install -e '.[lfpy]'", file=sys.stderr)
        sys.exit(1)

    from spike_discrim.synthetic.lfpy_generator import generate_dataset

    print("=" * 60)
    print("spike_discrim — LFPy synthetic dataset generation")
    print("=" * 60)
    print(f"  Electrode positions: {args.n_distances} distances × "
          f"{args.n_angles} angles = {args.n_distances * args.n_angles} total")
    print(f"  Distance range:      {args.dist_min}–{args.dist_max} µm")
    print(f"  Noise levels:        {args.noise_levels} µV RMS")
    print(f"  Output dir:          {args.output_dir}")
    print()

    result = generate_dataset(
        output_dir       = args.output_dir,
        n_distances      = args.n_distances,
        n_angles         = args.n_angles,
        distance_min_um  = args.dist_min,
        distance_max_um  = args.dist_max,
        noise_levels_uv  = args.noise_levels,
        soma_diam        = args.soma_diam,
        soma_L           = args.soma_L,
        dend_L           = args.dend_L,
        stim_amp         = args.stim_amp,
        seed             = args.seed,
        verbose          = not args.quiet,
    )

    wf = result["waveforms"]
    print(f"\nDone.  {wf.shape[0]} snippets × {wf.shape[1]} samples.")
    print(f"Results saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
