#!/usr/bin/env python3
"""Statistical analysis matching Chapter 3 of the thesis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HIGH_LOADS = (0.75, 0.80, 0.85, 0.90, 1.00)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_summary(a: np.ndarray, b: np.ndarray, lower_is_better: bool) -> dict[str, float | str]:
    delta = a - b
    shapiro_p = float(stats.shapiro(delta).pvalue) if len(delta) >= 3 else np.nan
    if shapiro_p >= 0.05:
        statistic, p_value = stats.ttest_rel(a, b)
        effect = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else np.inf
        test = "paired t-test"
        effect_name = "Cohen dz"
    else:
        statistic, p_value = stats.wilcoxon(a, b, alternative="two-sided")
        ranks = stats.rankdata(np.abs(delta[delta != 0]))
        signs = np.sign(delta[delta != 0])
        denom = ranks.sum()
        effect = float((ranks[signs > 0].sum() - ranks[signs < 0].sum()) / denom) if denom else 0.0
        test = "Wilcoxon signed-rank"
        effect_name = "rank-biserial r"
    direction_ok = bool(a.mean() < b.mean()) if lower_is_better else bool(a.mean() > b.mean())
    return {
        "test": test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "shapiro_p": shapiro_p,
        "effect_name": effect_name,
        "effect_size": effect,
        "direction_ok": direction_ok,
        "mean_adaptive": float(a.mean()),
        "mean_baseline": float(b.mean()),
    }


def run_analysis(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    high = df[df.actual_g.isin(HIGH_LOADS)].copy()
    run_means = high.groupby(["node_count", "protocol", "run"], as_index=False).agg(
        primary_plr=("primary_plr", "mean"),
        normalized_throughput=("normalized_throughput", "mean"),
    )

    comparisons: list[dict] = []
    for node in sorted(run_means.node_count.unique()):
        node_df = run_means[run_means.node_count == node]
        adaptive = node_df[node_df.protocol == "adaptive_irsa"].sort_values("run")
        standard = node_df[node_df.protocol == "standard_irsa"].sort_values("run")
        plr = paired_summary(adaptive.primary_plr.to_numpy(), standard.primary_plr.to_numpy(), True)
        comparisons.append({"node_count": node, "metric": "primary_plr", "baseline": "standard_irsa", **plr})
        for baseline in ("slotted_aloha", "crdsa", "standard_irsa"):
            base = node_df[node_df.protocol == baseline].sort_values("run")
            th = paired_summary(adaptive.normalized_throughput.to_numpy(), base.normalized_throughput.to_numpy(), False)
            comparisons.append({"node_count": node, "metric": "normalized_throughput", "baseline": baseline, **th})

    comp = pd.DataFrame(comparisons)
    comp["holm_p_within_node_metric"] = np.nan
    for (_node, metric), idx in comp.groupby(["node_count", "metric"]).groups.items():
        comp.loc[idx, "holm_p_within_node_metric"] = holm_adjust(comp.loc[idx, "p_value"].tolist())
    comp["significant_and_correct_direction"] = (comp.holm_p_within_node_metric < 0.05) & comp.direction_ok

    output_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(output_dir / "pairwise_tests.csv", index=False)
    run_means.to_csv(output_dir / "high_load_run_means.csv", index=False)
    print(comp.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistics"))
    args = parser.parse_args()
    run_analysis(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
