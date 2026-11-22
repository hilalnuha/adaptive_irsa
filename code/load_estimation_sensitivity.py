#!/usr/bin/env python3
"""Supplementary sensitivity analysis for imperfect knowledge of G."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from adaptive_irsa_simulation import Config, run_experiment

BIASES = (-0.10, -0.05, 0.00, 0.05, 0.10)


def main() -> None:
    cfg = Config(nodes=(10000,), loads=(0.75, 0.80, 0.85, 0.90, 1.00), frames_per_run=200, runs=10, base_seed=4206)
    parts = [run_experiment(cfg, protocols=("adaptive_irsa",), g_bias=bias) for bias in BIASES]
    raw = pd.concat(parts, ignore_index=True)
    summary = raw.groupby("g_bias", as_index=False).agg(
        primary_plr=("primary_plr", "mean"),
        normalized_throughput=("normalized_throughput", "mean"),
        transmission_plr=("transmission_plr", "mean"),
        final_backlog_ratio=("final_backlog_ratio", "mean"),
    )
    perfect = summary.loc[summary.g_bias == 0].iloc[0]
    summary["plr_change_percent"] = 100 * (summary.primary_plr - perfect.primary_plr) / perfect.primary_plr
    summary["throughput_change_percent"] = 100 * (summary.normalized_throughput - perfect.normalized_throughput) / perfect.normalized_throughput
    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "load_estimation_sensitivity_raw.csv", index=False)
    summary.to_csv(out / "load_estimation_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
