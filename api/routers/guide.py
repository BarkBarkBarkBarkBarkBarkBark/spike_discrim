"""
api/routers/guide.py — Structured interpretation data for every dashboard element.

GET /api/guide/features   — full feature glossary (name, formula, tier, interpretation)
GET /api/guide/panels     — explanation of each dashboard chart panel
GET /api/guide/metrics    — definitions of AUC, Fisher score, balanced accuracy, etc.
GET /api/guide/pipeline   — what the benchmark pipeline does, step by step
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["guide"])

# ── Feature glossary ──────────────────────────────────────────────────────── #

FEATURES = [
    # ── Tier 1/2 scalar features ─────────────────────────────────────────── #
    {
        "name":    "peak_amplitude",
        "tier":    2,
        "prefix":  None,
        "formula": "max(x[t])",
        "units":   "µV",
        "short":   "Positive repolarisation peak height.",
        "detail": (
            "The largest positive voltage in the 64-sample snippet. "
            "For a typical extracellular action potential the electrode first sees "
            "a sharp negative trough (current sink near the soma) followed by a "
            "positive repolarisation lobe. Real spikes have a consistent, "
            "moderate positive peak (~20–90 µV for a nearby neuron). "
            "Noise events lack the organised repolarisation and tend to produce "
            "either very small or very large, irregular positive excursions."
        ),
        "why_discriminates": "Real spikes have a constrained, unit-specific peak height.",
    },
    {
        "name":    "trough_amplitude",
        "tier":    2,
        "prefix":  None,
        "formula": "min(x[t])",
        "units":   "µV",
        "short":   "Negative trough depth (dominant lobe of extracellular AP).",
        "detail": (
            "The most negative voltage in the snippet. The negative trough is the "
            "dominant feature of extracellular action potentials: it corresponds to "
            "the inward sodium current at the axon hillock/soma, which appears as a "
            "current sink to the recording electrode. "
            "Typical range for a well-isolated unit: −50 to −100 µV. "
            "Noise events typically show either smaller magnitudes or irregular "
            "large excursions that do not track the AP time-course."
        ),
        "why_discriminates": "Consistent trough depth is a hallmark of single-unit spikes.",
    },
    {
        "name":    "max_slope",
        "tier":    2,
        "prefix":  None,
        "formula": "max(x[t] − x[t−1])",
        "units":   "µV / sample",
        "short":   "Fastest rising edge (depolarisation front).",
        "detail": (
            "Maximum value of the first discrete derivative. Captures the fastest "
            "voltage rise in the snippet, which for a real spike corresponds to the "
            "rapid depolarisation phase of the action potential. "
            "Real spikes have a sharp, stereotyped rising edge; noise is more "
            "diffuse. Computed causally (only uses past samples) so it can be used "
            "in streaming/real-time pipelines."
        ),
        "why_discriminates": "Spikes have a fast, consistent depolarisation ramp.",
    },
    {
        "name":    "min_slope",
        "tier":    2,
        "prefix":  None,
        "formula": "min(x[t] − x[t−1])",
        "units":   "µV / sample",
        "short":   "Fastest falling edge (repolarisation front).",
        "detail": (
            "Minimum (most negative) value of the first discrete derivative. "
            "Captures the steepest downward transition, which for an extracellular "
            "spike is the rapid onset of the negative trough. "
            "Closely related to max_slope but tracks the opposite polarity transition."
        ),
        "why_discriminates": "Spikes have a sharp, unit-specific falling edge.",
    },
    {
        "name":    "max_abs_curvature",
        "tier":    2,
        "prefix":  None,
        "formula": "max|x[t] − 2·x[t−1] + x[t−2]|",
        "units":   "µV / sample²",
        "short":   "Sharpest inflection point (second derivative peak).",
        "detail": (
            "Peak of the absolute second discrete derivative — measures the sharpest "
            "change of slope in the waveform. Action potentials have abrupt inflection "
            "points at the trough and at the trough-to-peak transition. Noise tends "
            "to be either smooth (low curvature) or uniformly jagged (high but "
            "distributed curvature). The Teager energy operator also exploits this "
            "same inflection structure."
        ),
        "why_discriminates": "Spikes have one dominant sharp inflection; noise does not.",
    },
    {
        "name":    "abs_window_sum_peak",
        "tier":    2,
        "prefix":  None,
        "formula": "max_t Σ_{τ=t−w}^{t} |x[τ]|   (w = window_size_samples)",
        "units":   "µV·samples",
        "short":   "Peak local energy density (sliding absolute sum).",
        "detail": (
            "Sliding-window absolute sum, returned at its peak value. "
            "Measures energy concentration: real spikes pack most of their energy "
            "into a short, well-defined window (typically 10–20 samples around the "
            "trough). Noise distributes energy more uniformly across the snippet. "
            "Computed with a ring-buffer accumulator — O(1) per step, suitable for "
            "streaming use."
        ),
        "why_discriminates": "Spikes have a focused energy burst; noise is diffuse.",
    },
    # ── Tier 3 event features ─────────────────────────────────────────────── #
    {
        "name":    "ev_trough_amplitude",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "min(x[t])",
        "units":   "µV",
        "short":   "Same as trough_amplitude, computed in the Tier 3 event pipeline.",
        "detail": (
            "Identical in meaning to trough_amplitude but extracted by the Tier 3 "
            "batch kernel. Redundant when both tiers are run; useful as a "
            "consistency cross-check between tiers."
        ),
        "why_discriminates": "See trough_amplitude.",
    },
    {
        "name":    "ev_peak_amplitude",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "max(x[t])",
        "units":   "µV",
        "short":   "Same as peak_amplitude, computed in the Tier 3 event pipeline.",
        "detail":  "Identical to peak_amplitude; see that entry.",
        "why_discriminates": "See peak_amplitude.",
    },
    {
        "name":    "ev_trough_to_peak_time_samples",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "argmax(x) − argmin(x)   [samples after trough]",
        "units":   "samples  (1 sample = 1/30 000 s ≈ 33 µs)",
        "short":   "Time from negative trough to subsequent positive peak.",
        "detail": (
            "The interval (in samples) between the negative trough and the next "
            "positive peak of the waveform. At 30 kHz, one sample ≈ 33 µs. "
            "This is a proxy for the repolarisation rate and is correlated with "
            "cell type: narrow-spiking interneurons typically have trough-to-peak "
            "times of 4–8 samples (~130–270 µs), while broad-spiking pyramidal "
            "cells have 10–20 samples (~330–670 µs). "
            "Noise events show random, inconsistent trough-to-peak intervals."
        ),
        "why_discriminates": "Encodes cell-type information; stable within a unit.",
    },
    {
        "name":    "ev_half_width_samples",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "count{ t : x[t] ≤ 0.5 × min(x) }",
        "units":   "samples",
        "short":   "Spike width at half the trough amplitude.",
        "detail": (
            "Number of samples where the waveform voltage is at or below half the "
            "trough amplitude (i.e., where the spike is 'half as deep' as its "
            "minimum). Narrow spikes (interneurons) have small half-widths; broad "
            "spikes (pyramidal cells) have larger values. "
            "Noise waveforms may have half-width = 0 (never reach half-depth) or "
            "unexpectedly large values (sustained artifact)."
        ),
        "why_discriminates": "Encodes spike width — a classic cell-type fingerprint.",
    },
    {
        "name":    "ev_full_width_samples",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "count{ t : x[t] ≤ 0.1 × min(x) }",
        "units":   "samples",
        "short":   "Total spike duration at 10% of trough depth.",
        "detail": (
            "Wider version of half_width using a 10% threshold. Captures the full "
            "temporal extent of the waveform. Useful for distinguishing spikes from "
            "slow artifacts (e.g. LFP bleed-through or motion transients) which "
            "tend to have very long full widths."
        ),
        "why_discriminates": "Discriminates spike duration from slow artifacts.",
    },
    {
        "name":    "ev_biphasic_ratio",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "|max(x)| / (|min(x)| + ε)",
        "units":   "dimensionless",
        "short":   "Ratio of positive peak height to negative trough depth.",
        "detail": (
            "Measures the relative prominence of the positive lobe compared to the "
            "negative trough. For typical extracellular spikes the positive "
            "repolarisation peak is 30–60% of the trough magnitude, giving a ratio "
            "of 0.3–0.6. Pure noise or artifacts often have ratios near 1.0 "
            "(symmetric, no preferred polarity) or near 0 (no positive lobe). "
            "This is a shape descriptor, not amplitude-dependent."
        ),
        "why_discriminates": "Encodes waveform asymmetry — stable within a unit.",
    },
    {
        "name":    "ev_signed_area",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "Σ x[t]",
        "units":   "µV·samples",
        "short":   "Net integral of the waveform (signed).",
        "detail": (
            "Sum of all samples — equivalent to the area under the waveform curve. "
            "For extracellular APs the negative trough dominates, so real spikes "
            "have a consistently negative signed area. Biphasic noise or symmetric "
            "oscillations will have signed area near zero."
        ),
        "why_discriminates": "Real spikes have a negative net area; noise is near zero.",
    },
    {
        "name":    "ev_absolute_area",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "Σ |x[t]|",
        "units":   "µV·samples",
        "short":   "Total waveform energy (unsigned integral).",
        "detail": (
            "Sum of absolute values — total signal energy regardless of sign. "
            "Real spikes from a nearby neuron concentrate energy in a characteristic "
            "range. Very distant units have low absolute area; artifacts often have "
            "high absolute area due to large-amplitude transients."
        ),
        "why_discriminates": "Proxy for recording distance / signal strength.",
    },
    {
        "name":    "ev_max_rising_slope",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "max(x[t] − x[t−1])",
        "units":   "µV / sample",
        "short":   "Same as max_slope, computed in Tier 3.",
        "detail":  "Identical to max_slope; see that entry.",
        "why_discriminates": "See max_slope.",
    },
    {
        "name":    "ev_max_falling_slope",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "min(x[t] − x[t−1])",
        "units":   "µV / sample",
        "short":   "Same as min_slope, computed in Tier 3.",
        "detail":  "Identical to min_slope; see that entry.",
        "why_discriminates": "See min_slope.",
    },
    {
        "name":    "ev_baseline_rms",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "√( mean(x[0:8]²) )",
        "units":   "µV",
        "short":   "RMS of the first 8 samples — pre-spike quietness.",
        "detail": (
            "Root-mean-square amplitude of the first 8 samples of the snippet "
            "(samples 0–7, which occur ~12 ms before the spike trough at sample 20 "
            "at 30 kHz). "
            "\n\n"
            "WHY THIS IS THE TOP DISCRIMINATOR (Fisher score 1.69, AUC 0.90):\n"
            "Real action potentials are brief, isolated events. The 8 samples "
            "before the spike trough capture the electrode's background noise floor "
            "— typically 1–5 µV RMS. The membrane has not yet begun depolarising, "
            "so the recording is quiet.\n\n"
            "Noise events — multi-unit overlap, motion artifacts, cross-talk from "
            "adjacent channels, or electrical interference — fill the entire "
            "64-sample window with energy. There is no quiet pre-event baseline. "
            "Their baseline RMS is therefore much higher and much more variable.\n\n"
            "This one number answers the question: 'Was the electrode quiet before "
            "this event?' If yes, it looks like a real spike. If no, it looks like "
            "noise bleed-through."
        ),
        "why_discriminates": (
            "MOST DISCRIMINATING FEATURE. Real spikes have a quiet pre-event "
            "baseline (~1–5 µV RMS); noise does not. Simple, fast, causal."
        ),
    },
    {
        "name":    "ev_zero_crossing_count",
        "tier":    3,
        "prefix":  "ev_",
        "formula": "count{ t : sign(x[t]) ≠ sign(x[t−1]) }",
        "units":   "crossings",
        "short":   "Number of times the waveform crosses zero.",
        "detail": (
            "Counts sign changes in the waveform. A clean biphasic spike crosses "
            "zero exactly twice (once going negative, once returning positive). "
            "A triphasic spike crosses 3–4 times. Noise events typically cross zero "
            "many more times (>10) because they are oscillatory or multi-event "
            "superpositions. This is a simple shape complexity metric."
        ),
        "why_discriminates": "Spikes have 2–4 crossings; noise has many more.",
    },
]


PANELS = [
    {
        "id":    "summary_cards",
        "title": "Run Summary",
        "items": [
            {"label": "WeightBank AUC",
             "meaning": "Area Under the ROC Curve for the hand-coded input layer (WeightBank). "
                        "1.0 = perfect separation; 0.5 = no better than chance. "
                        "Values above 0.95 indicate the WeightBank alone can reliably "
                        "separate spikes from noise."},
            {"label": "Top feature",
             "meaning": "The single feature with the highest Fisher score — the most "
                        "statistically separated feature between spike and noise classes."},
            {"label": "Top feature set",
             "meaning": "The feature combination + classifier that achieves the highest "
                        "cross-validated balanced accuracy."},
            {"label": "Snippets",
             "meaning": "Total number of waveform snippets (spike + noise). "
                        "Each snippet is a 64-sample (≈2.1 ms) voltage trace centred "
                        "on a detected threshold crossing."},
        ],
    },
    {
        "id":    "feature_chart",
        "title": "Single-Feature Ranking",
        "description": (
            "Each bar represents ONE feature evaluated in isolation. "
            "Blue bars = Fisher score (between-class variance / within-class variance). "
            "Green bars = AUC from a logistic regression trained on that one feature. "
            "Features are sorted by Fisher score. "
            "A high Fisher score means spike and noise distributions for that feature "
            "are well separated and compact — easy to draw a decision boundary."
        ),
    },
    {
        "id":    "classifier_chart",
        "title": "Classifier Benchmark",
        "description": (
            "Grouped bars showing cross-validated balanced accuracy for each "
            "feature set (x-axis) × classifier (colour) combination. "
            "Balanced accuracy = average of (sensitivity + specificity) / 2, "
            "which is unaffected by class imbalance. "
            "Feature sets: "
            "set_A_ultra_fast uses only Tier 1/2 causal features suitable for "
            "real-time streaming; "
            "set_D_combined uses all features including Tier 3 event features."
        ),
    },
    {
        "id":    "profiling_chart",
        "title": "Kernel Throughput",
        "description": (
            "Horizontal bars show the throughput of each Numba-JIT feature kernel "
            "in millions of waveform snippets per second on the host CPU. "
            "Measured after JIT warm-up, so compile time is excluded. "
            "Tooltip shows the static arithmetic op count per sample (adds + muls), "
            "sourced from docs/features.yaml. "
            "Higher throughput = cheaper to compute in a real-time spike sorter."
        ),
    },
    {
        "id":    "waveform_chart",
        "title": "Waveform Gallery",
        "description": (
            "Overlaid raw waveform snippets: blue = spike-class, red = noise-class. "
            "Each line is one 64-sample (≈2.1 ms at 30 kHz) voltage trace. "
            "\n\nWhat you should see in blue (spikes): sharp asymmetric depolarisation "
            "onset → steep trough (~sample 20) → slower exponential repolarisation → "
            "small positive peak → shallow after-hyperpolarisation (AHP) tail. "
            "This 'shark-fin' shape is produced by a double-exponential kinetic model "
            "(fast τ_rise ≈ 0.1–0.2 ms, slow τ_fall ≈ 0.25–0.7 ms), mimicking the "
            "Na⁺ current onset (sharp) and K⁺/Na⁺ repolarisation (gradual). "
            "\n\nThree cell types are simulated: fast-spiking interneurons (narrow, "
            "large amplitude, short trough-to-peak), regular-spiking pyramidal cells "
            "(broad, smaller amplitude, long trough-to-peak, prominent AHP), and an "
            "intermediate bursting cell — visible as three distinct blue clusters. "
            "\n\nRed lines (noise) are irregular, wide-amplitude, or have too many "
            "zero-crossings — no consistent morphology. The visual separation here "
            "is what the features quantify numerically."
        ),
    },
    {
        "id":    "validate_section",
        "title": "Objective Validation",
        "description": (
            "Five independent proof methods that confirm the pipeline, storage, "
            "and API are self-consistent. These go beyond indicator lights — "
            "they re-derive values from raw artefacts and compare:"
            "\n• Re-compute metrics: re-loads weight_bank.json + feature_matrix.parquet "
            "and recomputes AUC/balanced accuracy from scratch. Delta=0 means the "
            "stored values were not fabricated."
            "\n• Feature statistics: per-feature descriptive stats split by class. "
            "A domain expert can verify trough_amplitude is negative for spikes, "
            "baseline_rms is low for spikes, etc."
            "\n• CSV round-trip: generates the CSV export in-memory, re-parses it, "
            "and checks numeric equality to float32 precision."
            "\n• File checksums: SHA-256 of every result file. Stable across runs "
            "on the same data; changes if files are modified."
            "\n• Waveform checksum: SHA-256 of waveforms.npz. If this hash changes "
            "between sessions, the dataset was regenerated."
        ),
    },
]


METRICS = [
    {
        "name":    "Fisher score",
        "formula": "(µ_spike − µ_noise)² / (σ_spike² + σ_noise²)",
        "range":   "0 to ∞  (higher = better separation)",
        "interpretation": (
            "Measures how well separated two class distributions are relative to "
            "their spread. A Fisher score of 1.0 means the class means are "
            "1 standard deviation apart. Score > 1 indicates strong separation. "
            "ev_baseline_rms scored 1.69 — unusually high for a single feature."
        ),
    },
    {
        "name":    "AUC (Area Under ROC Curve)",
        "formula": "∫ TPR d(FPR)",
        "range":   "0.5 (random) to 1.0 (perfect)",
        "interpretation": (
            "Probability that a randomly chosen spike scores higher than a randomly "
            "chosen noise event. Threshold-independent — does not depend on choosing "
            "a decision boundary. AUC ≥ 0.95 indicates a reliable discriminator."
        ),
    },
    {
        "name":    "Balanced accuracy",
        "formula": "(sensitivity + specificity) / 2",
        "range":   "0.5 (random) to 1.0 (perfect)",
        "interpretation": (
            "Average of true-positive rate and true-negative rate. Unlike raw "
            "accuracy, it is not inflated when classes are imbalanced (e.g. 3:1 "
            "spike:noise ratio). The benchmark uses 5-fold cross-validation, so "
            "reported values are held-out estimates."
        ),
    },
    {
        "name":    "Silhouette score",
        "formula": "(b − a) / max(a, b)  averaged over samples",
        "range":   "−1 (wrong cluster) to 1 (tight cluster)",
        "interpretation": (
            "Measures how similar a sample is to its own class versus the nearest "
            "other class in feature space. High silhouette means the feature set "
            "produces compact, well-separated clusters — a proxy for how easily "
            "a downstream classifier can work."
        ),
    },
    {
        "name":    "WeightBank threshold",
        "formula": "score(x) = Σ_f w_f · max_b exp(−½((x_f − c_fb)/σ_fb)²) / Σ w_f",
        "range":   "0 to 1  (threshold default: 0.5)",
        "interpretation": (
            "Population-code score from the hand-initialised input layer. "
            "Bin centres are set from quantiles of the calibration spike set, so "
            "each bin covers an equal fraction of the real-spike distribution. "
            "A real spike activates most bins (score near 1). A noise event "
            "activates few bins (score near 0). The threshold is the decision "
            "boundary — lower threshold = more permissive (fewer false negatives, "
            "more false positives)."
        ),
    },
]


PIPELINE_STEPS = [
    {
        "step": 1,
        "name": "Load waveforms",
        "detail": "Reads waveforms.npz + labels.parquet from the data directory. "
                  "Auto-generates a procedural biphasic dataset if none is found.",
    },
    {
        "step": 2,
        "name": "Profile kernels",
        "detail": "Times each Tier 1/2 Numba-JIT kernel on the full batch after "
                  "warm-up. Records wall-clock throughput and static op counts.",
    },
    {
        "step": 3,
        "name": "Extract features",
        "detail": "Runs batch_extract_scalar_features (Tier 1/2) and optionally "
                  "batch_event_features (Tier 3) on all snippets. Saves "
                  "feature_matrix.parquet.",
    },
    {
        "step": 4,
        "name": "Fit WeightBank",
        "detail": "Initialises bin centres from quantiles of the spike-class features. "
                  "Computes AUC and balanced accuracy. Saves weight_bank.json.",
    },
    {
        "step": 5,
        "name": "Single-feature benchmark",
        "detail": "Evaluates every feature column individually with Fisher score, "
                  "mutual information, and AUC. Saves single_feature_ranks.parquet.",
    },
    {
        "step": 6,
        "name": "Feature-set benchmark",
        "detail": "Evaluates each feature set × classifier combination with "
                  "5-fold cross-validated balanced accuracy. "
                  "Saves feature_set_ranks.parquet.",
    },
]


# ── Endpoints ─────────────────────────────────────────────────────────────── #

@router.get("/guide/features")
def guide_features() -> list[dict]:
    """Full feature glossary — name, formula, units, interpretation."""
    return FEATURES


@router.get("/guide/panels")
def guide_panels() -> list[dict]:
    """Explanation of each dashboard chart panel."""
    return PANELS


@router.get("/guide/metrics")
def guide_metrics() -> list[dict]:
    """Definitions of all reported evaluation metrics."""
    return METRICS


@router.get("/guide/pipeline")
def guide_pipeline() -> list[dict]:
    """Step-by-step description of the benchmark pipeline."""
    return PIPELINE_STEPS
