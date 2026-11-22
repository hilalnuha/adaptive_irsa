#!/usr/bin/env python3
"""Deterministic unit and theoretical-behavior checks for the simulator."""
from __future__ import annotations

import math
import numpy as np

from adaptive_irsa_simulation import (
    DEGREE_DISTRIBUTIONS, Config, draw_degrees, place_replicas,
    run_experiment, sic_decode,
)


def test_distribution_normalization() -> None:
    for name, (degrees, probabilities) in DEGREE_DISTRIBUTIONS.items():
        assert len(degrees) == len(probabilities)
        assert np.all(probabilities >= 0)
        assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12), name


def test_distinct_replica_slots() -> None:
    rng = np.random.default_rng(1)
    degrees = np.array([2, 3, 8])
    occupants = place_replicas(degrees, 50, rng)
    for packet_id, degree in enumerate(degrees):
        observed = sum(packet_id in slot for slot in occupants)
        assert observed == degree


def test_sic_cascade_and_stopping_set() -> None:
    # packet 0 singleton in slot 0; cancelling it exposes packet 1 in slot 1.
    degrees = np.array([2, 1])
    cascade = [np.array([0]), np.array([0, 1])]
    assert sic_decode(degrees, cascade, 10) == 2
    stopping = [np.array([0, 1]), np.array([0, 1])]
    assert sic_decode(np.array([2, 2]), stopping, 10) == 0


def test_protocol_accounting_and_low_load_identity() -> None:
    cfg = Config(nodes=(10000,), loads=(0.50,), slots_per_frame=50, frames_per_run=200, runs=2, base_seed=123)
    df = run_experiment(cfg)
    assert np.all(df.generated >= df.decoded)
    assert np.all(df.admitted == df.decoded + df.unresolved_transmitted)
    standard = df[df.protocol == "standard_irsa"].sort_values("run")
    adaptive = df[df.protocol == "adaptive_irsa"].sort_values("run")
    # Same degree rule at low load; statistical equivalence is expected, not bitwise identity,
    # because protocol-specific slot streams intentionally remain independent.
    assert abs(standard.primary_plr.mean() - adaptive.primary_plr.mean()) < 0.08
    assert (adaptive.final_backlog == 0).all()


def test_slotted_aloha_theory() -> None:
    cfg = Config(nodes=(100000,), loads=(0.20, 0.50, 0.80), slots_per_frame=200, frames_per_run=2000, runs=2, base_seed=321)
    df = run_experiment(cfg, protocols=("slotted_aloha",))
    observed = df.groupby("actual_g").normalized_throughput.mean()
    for g, value in observed.items():
        expected = g * math.exp(-g)
        assert abs(value - expected) < 0.02, (g, value, expected)


def main() -> None:
    tests = [
        test_distribution_normalization,
        test_distinct_replica_slots,
        test_sic_cascade_and_stopping_set,
        test_protocol_accounting_and_low_load_identity,
        test_slotted_aloha_theory,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} validation checks passed.")


if __name__ == "__main__":
    main()
