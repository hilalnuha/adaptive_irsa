# Load-Aware Adaptive IRSA for High-Load Wireless Sensor Networks

Reproducibility package for the article:

> M. Alif Gibran A. Bossa, Hilal Hudan Nuha, and Yusril Izza, "Load-Aware Adaptive Irregular Repetition Slotted ALOHA for Reliable Uplink Access in Dense Wireless Sensor Networks," submitted to *AUT Journal of Electrical Engineering*, 2026.

The package contains a Monte Carlo reference implementation of Slotted ALOHA, CRDSA, Standard IRSA, and the proposed load-aware Adaptive IRSA under an ideal collision channel with ideal Successive Interference Cancellation (SIC), together with validation, statistical-analysis, and load-estimation sensitivity scripts.

## Proposed mechanism

Adaptive IRSA selects a degree distribution and an admission target from the normalized offered load `G`:

| Load region | Distribution | Mean degree | Admission target |
|---|---|---|---|
| `G <= 0.60` | Λ8(x) = 0.50x² + 0.28x³ + 0.22x⁸ | 3.6000 | none |
| `0.60 < G <= 0.70` | Λ6(x) = 0.5465x² + 0.1623x³ + 0.2912x⁶ | 3.3271 | none |
| `0.70 < G <= 0.90` | Λ6 | 3.3271 | `G_target = 0.70` |
| `G > 0.90` | Λ4(x) = 0.5102x² + 0.4898x⁴ | 2.9796 | `G_target = 0.65` |

Pending packets `Q_t = B_t + A_t` are admitted with probability `min(1, G_target·M / Q_t)`. Deferred packets stay in the backlog, and packets remaining in the final backlog are counted in the primary Packet Loss Ratio (PLR).

## Repository layout

```
code/
  adaptive_irsa_simulation.py     # four-protocol Monte Carlo simulator
  validate_implementation.py      # deterministic SIC and accounting checks
  statistical_analysis.py         # run-level tests (Shapiro-Wilk, paired t/Wilcoxon, Holm)
  load_estimation_sensitivity.py  # fixed-bias load-estimation stress test
  make_figures.py                 # regenerates the article figures
  requirements.txt
results/
  run_level_results.csv           # main experiment, one row per protocol x N x G x run (2,400 rows)
  config.json                     # configuration of the main experiment
  statistics/                     # outputs of statistical_analysis.py
  sensitivity/                    # load-estimation sensitivity outputs
```

## Environment

The pinned versions are Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and Matplotlib 3.10.8. The results in `results/` were regenerated with NumPy 2.5.3, pandas 3.0.5, and SciPy 1.18.1 and match the pinned environment for the sensitivity analysis.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r code/requirements.txt
```

## Usage

Validation:

```bash
python code/validate_implementation.py
```

The checks cover probability normalization, distinct replica placement, a deterministic SIC cascade, a stopping set with no singleton, packet-accounting identities, low-load behavior, and agreement of Slotted ALOHA with `S = G·exp(-G)`.

Main experiment (50 slots/frame, 800 frames/run, 10 runs, base seed 42, N ∈ {500, 1000, 10000, 20000, 50000}, twelve load values from 0.20 to 1.00):

```bash
python code/adaptive_irsa_simulation.py \
  --output results/run_level_results.csv \
  --write-config results/config.json
python code/statistical_analysis.py results/run_level_results.csv \
  --output-dir results/statistics
```

Figures:

```bash
python code/make_figures.py results/run_level_results.csv figures
```

Load-estimation sensitivity (N = 10000, high-load region, bias ∈ {−0.10, −0.05, 0, +0.05, +0.10}, 200 frames/run):

```bash
python code/load_estimation_sensitivity.py
```

Random streams are generated with NumPy `SeedSequence`. Corresponding protocol scenarios share the same arrival stream (common random numbers), while degree selection, slot placement, and admission use separate protocol-specific streams.

## Scope and limitations

The simulator is a MAC-layer collision-channel model. It does not implement a realistic physical layer, fading, capture, residual interference after SIC, packet headers, coding, energy consumption, queueing delay, or end-to-end transport behavior. The Mbps values reported in the article are an illustrative conversion with a 1 MHz bandwidth and 1 bit/s/Hz spectral efficiency.

## Contact

Hilal Hudan Nuha, School of Computing, Telkom University, Bandung, Indonesia — hilalnuha@ieee.org
