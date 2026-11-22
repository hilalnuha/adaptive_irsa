#!/usr/bin/env python3
"""Reproducible Monte Carlo reference implementation for the thesis.

Implements Slotted ALOHA, CRDSA, Standard IRSA, and the proposed
load-aware Adaptive IRSA under the ideal collision-channel assumptions
stated in Chapter 3. The program writes run-level CSV results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping
from collections import deque

import numpy as np
import pandas as pd

PROTOCOLS = ("slotted_aloha", "crdsa", "standard_irsa", "adaptive_irsa")
DEGREE_DISTRIBUTIONS: Mapping[str, tuple[np.ndarray, np.ndarray]] = {
    "IRSA-8": (np.array([2, 3, 8], dtype=int), np.array([0.50, 0.28, 0.22])),
    "IRSA-6": (np.array([2, 3, 6], dtype=int), np.array([0.5465, 0.1623, 0.2912])),
    "IRSA-4": (np.array([2, 4], dtype=int), np.array([0.5102, 0.4898])),
}


@dataclass(frozen=True)
class Config:
    nodes: tuple[int, ...] = (500, 1000, 10000, 20000, 50000)
    loads: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00)
    slots_per_frame: int = 50
    frames_per_run: int = 800
    runs: int = 10
    max_sic_iterations: int = 100
    base_seed: int = 42


def seed_sequence(base_seed: int, node_count: int, load_index: int, run: int, stream: int) -> np.random.SeedSequence:
    """Create an order-independent deterministic seed for one random stream."""
    return np.random.SeedSequence([base_seed, node_count, load_index, run, stream])


def adaptive_rule(estimated_g: float) -> tuple[str, float | None]:
    """Return (degree-distribution name, effective-load target)."""
    if estimated_g <= 0.60:
        return "IRSA-8", None
    if estimated_g <= 0.70:
        return "IRSA-6", None
    if estimated_g <= 0.90:
        return "IRSA-6", 0.70
    return "IRSA-4", 0.65


def draw_degrees(protocol: str, count: int, rng: np.random.Generator, estimated_g: float) -> tuple[np.ndarray, str]:
    if count == 0:
        return np.empty(0, dtype=int), "none"
    if protocol == "slotted_aloha":
        return np.ones(count, dtype=int), "degree-1"
    if protocol == "crdsa":
        return np.full(count, 2, dtype=int), "degree-2"
    if protocol == "standard_irsa":
        name = "IRSA-8"
    elif protocol == "adaptive_irsa":
        name, _ = adaptive_rule(estimated_g)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    degrees, probabilities = DEGREE_DISTRIBUTIONS[name]
    return rng.choice(degrees, size=count, p=probabilities), name


def place_replicas(degrees: np.ndarray, slots: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Vectorized distinct-slot placement for all packets in one frame."""
    if len(degrees) == 0:
        return [np.empty(0, dtype=int) for _ in range(slots)]
    packet_parts: list[np.ndarray] = []
    slot_parts: list[np.ndarray] = []
    for degree in np.unique(degrees):
        ids = np.flatnonzero(degrees == degree)
        degree_int = int(degree)
        if degree_int > slots:
            raise ValueError("Repetition degree cannot exceed frame length")
        selected = rng.integers(0, slots, size=(len(ids), degree_int))
        # Resample rows containing duplicate slots until every packet has distinct slots.
        while degree_int > 1:
            sorted_rows = np.sort(selected, axis=1)
            bad = np.any(np.diff(sorted_rows, axis=1) == 0, axis=1)
            if not np.any(bad):
                break
            selected[bad] = rng.integers(0, slots, size=(int(bad.sum()), degree_int))
        packet_parts.append(np.repeat(ids, degree_int))
        slot_parts.append(selected.reshape(-1))
    packet_ids = np.concatenate(packet_parts)
    slot_ids = np.concatenate(slot_parts)
    return [packet_ids[slot_ids == slot].astype(int, copy=False) for slot in range(slots)]


def sic_decode(degrees: np.ndarray, slot_occupants: list[np.ndarray], max_iterations: int) -> int:
    """Ideal SIC using a singleton queue; complexity is linear in graph edges."""
    packet_count = len(degrees)
    if packet_count == 0:
        return 0
    unresolved = np.ones(packet_count, dtype=bool)
    remaining = np.asarray([len(x) for x in slot_occupants], dtype=int)
    packet_slots: list[list[int]] = [[] for _ in range(packet_count)]
    for slot_id, occupants in enumerate(slot_occupants):
        for packet_id in occupants:
            packet_slots[int(packet_id)].append(slot_id)
    queue = deque(int(i) for i in np.flatnonzero(remaining == 1))
    decoded = 0
    processed_singletons = 0
    # The iteration cap is translated to a safe cap on processed singleton events.
    event_cap = max_iterations * max(1, len(slot_occupants))
    while queue and processed_singletons < event_cap:
        slot_id = queue.popleft()
        processed_singletons += 1
        if remaining[slot_id] != 1:
            continue
        packet_id = next((int(p) for p in slot_occupants[slot_id] if unresolved[int(p)]), None)
        if packet_id is None:
            continue
        unresolved[packet_id] = False
        decoded += 1
        for affected_slot in packet_slots[packet_id]:
            if remaining[affected_slot] > 0:
                remaining[affected_slot] -= 1
                if remaining[affected_slot] == 1:
                    queue.append(affected_slot)
    return decoded


def one_protocol_run(
    protocol: str,
    node_count: int,
    actual_g: float,
    estimated_g: float,
    slots: int,
    frames: int,
    max_sic_iterations: int,
    arrival_rng: np.random.Generator,
    protocol_rng: np.random.Generator,
) -> dict[str, float | int | str]:
    generated_total = admitted_total = decoded_total = unresolved_total = replica_total = 0
    backlog = 0
    distribution_name = ""
    p_active = min(1.0, actual_g * slots / node_count)

    for _frame in range(frames):
        generated = int(arrival_rng.binomial(node_count, p_active))
        generated_total += generated
        pending = backlog + generated if protocol == "adaptive_irsa" else generated

        if protocol == "adaptive_irsa":
            distribution_name, target = adaptive_rule(estimated_g)
            if target is None:
                admitted = pending
            else:
                p_admit = min(1.0, target * slots / max(pending, 1))
                admitted = int(protocol_rng.binomial(pending, p_admit))
            backlog = pending - admitted
        else:
            admitted = pending
            backlog = 0

        degrees, distribution_name = draw_degrees(protocol, admitted, protocol_rng, estimated_g)
        replica_total += int(degrees.sum())
        admitted_total += admitted
        slot_occupants = place_replicas(degrees, slots, protocol_rng)
        if protocol == "slotted_aloha":
            decoded = sum(1 for x in slot_occupants if len(x) == 1)
        else:
            decoded = sic_decode(degrees, slot_occupants, max_sic_iterations)
        unresolved = admitted - decoded
        decoded_total += decoded
        unresolved_total += unresolved

    primary_undelivered = unresolved_total + backlog
    generated_den = max(generated_total, 1)
    admitted_den = max(admitted_total, 1)
    return {
        "protocol": protocol,
        "node_count": node_count,
        "actual_g": actual_g,
        "estimated_g": estimated_g,
        "degree_distribution": distribution_name,
        "generated": generated_total,
        "admitted": admitted_total,
        "decoded": decoded_total,
        "unresolved_transmitted": unresolved_total,
        "final_backlog": backlog,
        "replicas": replica_total,
        "mean_repetition_degree": replica_total / admitted_den,
        "primary_plr": primary_undelivered / generated_den,
        "transmission_plr": unresolved_total / admitted_den,
        "final_backlog_ratio": backlog / generated_den,
        "normalized_throughput": decoded_total / (frames * slots),
    }


def run_experiment(config: Config, protocols: Iterable[str] = PROTOCOLS, g_bias: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    protocol_list = tuple(protocols)
    for node_count in config.nodes:
        for load_index, actual_g in enumerate(config.loads):
            estimated_g = max(0.0, actual_g + g_bias)
            for run in range(config.runs):
                # Common random numbers: reinitialize the same arrival stream for each protocol.
                for protocol_index, protocol in enumerate(protocol_list):
                    arrival_rng = np.random.default_rng(seed_sequence(config.base_seed, node_count, load_index, run, 0))
                    protocol_rng = np.random.default_rng(seed_sequence(config.base_seed, node_count, load_index, run, protocol_index + 1))
                    row = one_protocol_run(
                        protocol, node_count, actual_g, estimated_g,
                        config.slots_per_frame, config.frames_per_run,
                        config.max_sic_iterations, arrival_rng, protocol_rng,
                    )
                    row["run"] = run + 1
                    row["g_bias"] = g_bias
                    rows.append(row)
    return pd.DataFrame(rows)


def parse_csv_numbers(text: str, cast=float) -> tuple:
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/run_level_results.csv"))
    parser.add_argument("--nodes", default="500,1000,10000,20000,50000")
    parser.add_argument("--loads", default="0.20,0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,1.00")
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g-bias", type=float, default=0.0)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--write-config", type=Path)
    args = parser.parse_args()

    config = Config(
        nodes=parse_csv_numbers(args.nodes, int),
        loads=parse_csv_numbers(args.loads, float),
        slots_per_frame=args.slots,
        frames_per_run=args.frames,
        runs=args.runs,
        base_seed=args.seed,
    )
    protocols = tuple(x.strip() for x in args.protocols.split(",") if x.strip())
    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise SystemExit(f"Unknown protocol(s): {sorted(unknown)}")
    frame = run_experiment(config, protocols, args.g_bias)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if args.write_config:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print(f"Wrote {len(frame):,} run-level rows to {args.output}")


if __name__ == "__main__":
    main()
