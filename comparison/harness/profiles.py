"""Benchmark profiles: which workloads run on which simulators, at which thread counts.

Each profile is declared before it runs, repetition policy included, so a
reader can see what was asked for as well as what came back. A simulator that
cannot run a workload is still listed against it; the run records why.

`smoke` finishes in a few minutes on a laptop. `reference` is the published
run. `full` is every size in the matrix; on a 4-core desktop it takes most of a
day and needs 32 GB for the 28-qubit rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import runners
from . import workloads as w

FP64_DENSE = ["prismq", "aer-statevector", "quest", "spinoza", "qip"]
FP32_DENSE = ["aer-statevector-f32", "qsim"]
SEED_VARIANTS = (w.CIRCUIT_SEED, w.CIRCUIT_SEED + 1, w.CIRCUIT_SEED + 2)
PRISMQ_ALTERNATIVES = [
    "prismq:statevector",
    "prismq:mps-64",
    "prismq:factored",
    "prismq:sparse",
    "prismq:stabilizer",
    "prismq:factored-stabilizer",
    "prismq:tensor-network",
]


@dataclass
class Cell:
    group: str
    workload: w.Workload
    simulators: list[str]
    threads: list[int] = field(default_factory=lambda: [8])
    iterations: int = 5
    min_iterations: int = 3
    budget_s: float = 30.0
    timeout_s: float = 600.0
    sim_threads: dict[str, list[int]] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    profile_simulators: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.profile_simulators = list(self.simulators)

    def threads_for(self, name: str) -> list[int]:
        """Thread counts `name` runs at: 1 for a single-threaded simulator, its `sim_threads` entry, else the cell's."""
        if runners.resolve(name).single_threaded:
            return [1]
        return self.sim_threads.get(name, self.threads)


@dataclass
class Profile:
    name: str
    description: str
    cells: list[Cell]


def _dispatch_sims(wl: w.Workload) -> list[str]:
    """Every PRISM-Q backend, plus the fp64 reference a dense row validates against."""
    sims = ["prismq", *PRISMQ_ALTERNATIVES]
    return sims + ["aer-statevector"] if wl.mode == "probs" else sims


def _dense_sims(n: int, slow_limit: int, quest_limit: int) -> list[str]:
    sims = ["prismq", "aer-statevector"]
    if n <= quest_limit:
        sims.append("quest")
    if n <= slow_limit:
        sims += ["spinoza", "qip"]
    return sims + FP32_DENSE


def _thread_counts(candidates: tuple[int, ...], threads: int) -> list[int]:
    """The candidates the host has logical cores for, plus the headline count."""
    cores = os.cpu_count() or 1
    return sorted({t for t in candidates if t <= cores} | {threads})


QFT_CONTROL = "prismq:no-qft-block"


def _with_one_thread(threads: int) -> dict[str, list[int]]:
    """PRISM-Q at 1 thread beside the headline count, so a single-threaded peer has a partner."""
    return {"prismq": sorted({1, threads})}


TN_PEERS = ["prismq", "prismq:tensor-network", "prismq:mps-256", "quimb-tn", "aer-mps"]
DEEP_MPS_PEERS = ["prismq:mps-256", "aer-mps:256", "quimb-mps:256"]
DEEP_MPS_SHOTS = 100
# Depth 12 rows also run PRISM-Q and Aer at this many shots, which shrinks the smallest shift
# their check detects from about 0.23 to 0.07; depth 16 rows cost too much per call to repeat.
DEEP_MPS_POWER_SHOTS = 1000
SWEEP_CAPS = (2, 4, 8, 16, 32)
# Cost of one quimb CircuitMPS call (gates plus 100 samples, bond cap 256) per qubit, measured on
# the reference host, and the widest row it runs at so cold call plus warm calls stay near two minutes.
QUIMB_MPS_COST = {12: (0.07, 500), 16: (0.54, 50)}


def _sweep_sims() -> list[str]:
    return [f"{sim}{c}" for sim in ("prismq:mps-", "aer-mps:", "quimb-mps:") for c in SWEEP_CAPS]


def _quimb_mps_excluded(n: int, depth: int) -> dict[str, str]:
    rate, limit = QUIMB_MPS_COST[depth]
    if n <= limit:
        return {}
    return {
        "quimb-mps:256": (
            f"one quimb CircuitMPS call costs about {rate:g} s per qubit at depth {depth} on the reference "
            f"host; rows past {limit} qubits would exceed the row budget"
        )
    }


def _wide_cells(threads: int) -> list[Cell]:
    """Tensor network expectation values, deep MPS, the wide bond-cap sweep and high-shot stabilizer rows."""
    t = [threads]
    cells = [Cell("tn", w.expval(n, 8, 4), list(TN_PEERS), t) for n in (32, 64, 128, 256)]
    for depth in (12, 16):
        for n in (50, 100, 200, 500, 1000):
            heavy = depth == 16 and n >= 500
            policy = {"iterations": 2 if heavy else 3, "min_iterations": 1 if heavy else 2, "timeout_s": 1800.0}
            cells.append(
                Cell(
                    "mpsdeep",
                    w.lowent(n, depth, DEEP_MPS_SHOTS),
                    list(DEEP_MPS_PEERS),
                    t,
                    excluded=_quimb_mps_excluded(n, depth),
                    **policy,
                )
            )
            if depth == 12:
                sims = [s for s in DEEP_MPS_PEERS if not s.startswith("quimb")]
                cells.append(Cell("mpsdeep", w.lowent(n, depth, DEEP_MPS_POWER_SHOTS), sims, t, **policy))
    for n in (32, 48):
        cells.append(Cell("bondsweep", w.expval(n, 8), _sweep_sims(), t, 3, 2, 10.0, 300.0))
    for n, shots in ((100, 10**6), (1000, 10**6), (10000, 10**6), (100, 10**7), (1000, 10**7)):
        cells.append(
            Cell(
                "highshot",
                w.clifford(n, shots, mode="stats"),
                ["prismq", "stim"],
                t,
                timeout_s=900.0,
                sim_threads=_with_one_thread(threads),
            )
        )
    return cells


def smoke(runner: str, threads: int) -> Profile:
    t = [threads]
    cells = [
        Cell("headline", w.rcs(12, 10), _dense_sims(12, 12, 12), t),
        Cell("headline", w.qft(runner, 12), _dense_sims(12, 12, 12) + [QFT_CONTROL], t),
        Cell("headline", w.exported(runner, "qv", 12), _dense_sims(12, 12, 12), t),
        Cell("headline", w.clifford(100, 200), ["prismq", "stim"], t, sim_threads=_with_one_thread(threads)),
        Cell("headline", w.lowent(50, 4, 200), ["prismq", "aer-mps", "aer-automatic"], t),
        Cell("headline", w.disjoint(40, 200), ["prismq", "aer-mps", "aer-automatic", "aer-statevector"], t),
        Cell("headline", w.sampling(12, 1000), ["prismq", "aer-statevector", "qsim"], t),
        Cell("headline", w.midcircuit(12, 500), ["prismq", "qsim"], t),
        Cell("threads", w.rcs(16, 10), ["prismq", "aer-statevector", "qsim", "quest"], [1, threads]),
        Cell("dispatch", w.rcs(12, 10), _dispatch_sims(w.rcs(12, 10)), t),
        Cell(
            "accuracy",
            w.rcs(12, 10),
            ["aer-statevector", "prismq:mps-4", "prismq:mps-64", "aer-mps:4", "aer-mps:64"],
            t,
        ),
        Cell("tn", w.expval(32, 4, 4), list(TN_PEERS), t),
        Cell("mpsdeep", w.lowent(16, 12, 100), list(DEEP_MPS_PEERS), t),
        Cell("bondsweep", w.expval(16, 6), ["prismq:mps-4", "aer-mps:4", "quimb-mps:4"], t),
        Cell(
            "highshot",
            w.clifford(100, 10_000, mode="stats"),
            ["prismq", "stim"],
            t,
            sim_threads=_with_one_thread(threads),
        ),
    ]
    for cell in cells:
        cell.iterations, cell.min_iterations, cell.budget_s, cell.timeout_s = 3, 2, 5.0, 120.0
    return Profile("smoke", "every family at a small size, every adapter, a few seconds per row", cells)


def reference(runner: str, threads: int) -> Profile:
    t = [threads]
    cells: list[Cell] = []
    for n in (20, 24, 26):
        for depth in (10, 20):
            cells.append(Cell("headline", w.rcs(n, depth), _dense_sims(n, 20, 24), t))
        cells.append(Cell("headline", w.qft(runner, n), _dense_sims(n, 20, 24) + [QFT_CONTROL], t))
    for n in (16, 20, 24):
        cells.append(Cell("headline", w.exported(runner, "qv", n), _dense_sims(n, 20, 24), t))

    stab = ["prismq", "stim"]
    for n in (100, 1000, 10000):
        cells.append(
            Cell(
                "headline",
                w.clifford(n, 1000),
                stab,
                t,
                timeout_s=120.0 if n > 1000 else 300.0,
                sim_threads=_with_one_thread(threads),
            )
        )
    mps = ["prismq", "aer-mps", "aer-automatic"]
    for n in (50, 100, 200):
        for depth in (4, 8):
            cells.append(Cell("headline", w.lowent(n, depth, 1000), mps, t, timeout_s=300.0))
    for n in (40, 100, 200):
        cells.append(Cell("headline", w.disjoint(n, 1000), mps, t, timeout_s=300.0))
    cells.append(Cell("headline", w.disjoint(40, 1000, layout="strided"), mps, t, timeout_s=300.0))

    shot_sims = ["prismq", "aer-statevector", "aer-statevector-f32", "qsim"]
    for shots in (1, 1000, 100_000, 1_000_000):
        cells.append(Cell("sampling", w.sampling(20, shots), shot_sims, t))
    for n in (16, 24):
        for shots in (1, 1_000_000):
            cells.append(Cell("sampling", w.sampling(n, shots), shot_sims, t))
    cells.append(Cell("sampling", w.sampling(20, 1_000_000, measured="half"), shot_sims, t))

    mid = ["prismq", "qsim"]
    for shots in (1000, 10_000):
        cells.append(Cell("headline", w.midcircuit(12, shots), mid, t, timeout_s=600.0))
    cells.append(Cell("headline", w.midcircuit(18, 1000), mid, t, min_iterations=2, timeout_s=600.0))

    scaling = ["prismq", "aer-statevector", "qsim", "quest"]
    for wl in (w.rcs(22, 10), w.qft(runner, 22), w.exported(runner, "qv", 20)):
        cells.append(Cell("threads", wl, scaling, _thread_counts((1, 2, 4), threads)))

    for wl in (
        w.rcs(20, 10),
        w.qft(runner, 20),
        w.clifford(1000, 1000),
        w.lowent(100, 4, 1000),
        w.disjoint(100, 1000),
        w.sampling(20, 100_000),
        w.midcircuit(12, 1000),
    ):
        cells.append(Cell("dispatch", wl, _dispatch_sims(wl), t, budget_s=15.0, timeout_s=180.0))

    bonds = (2, 4, 8, 16, 32)
    acc = ["aer-statevector", "prismq:statevector"]
    acc += [f"prismq:mps-{b}" for b in bonds] + [f"aer-mps:{b}" for b in bonds]
    cells.append(Cell("accuracy", w.rcs(20, 10), acc, t))

    for seed in SEED_VARIANTS[1:]:
        cells.append(Cell("seeds", w.rcs(20, 10, seed), ["prismq", "aer-statevector", "qsim"], t))
        cells.append(Cell("seeds", w.exported(runner, "qv", 20, seed), ["prismq", "aer-statevector", "qsim"], t))
    cells += _wide_cells(threads)
    return Profile(
        "reference", "the published subset: every family, sizes that fit about three hours on a 4-core host", cells
    )


def full(runner: str, threads: int) -> Profile:
    """The reference profile with its dense headline rows widened to every simulator and size."""
    t = [threads]
    dense: list[Cell] = []
    for n in (20, 24, 26, 28):
        for depth in (10, 20):
            dense.append(Cell("headline", w.rcs(n, depth), _dense_sims(n, 28, 28), t, timeout_s=3600.0))
    for n in (20, 24, 26):
        dense.append(Cell("headline", w.qft(runner, n), _dense_sims(n, 26, 26) + [QFT_CONTROL], t, timeout_s=3600.0))
    for n in (16, 20, 24):
        dense.append(Cell("headline", w.exported(runner, "qv", n), _dense_sims(n, 24, 24), t, timeout_s=3600.0))
    widened = {c.workload.workload_id for c in dense}
    base = reference(runner, threads)
    cells = dense + [
        c
        for c in base.cells
        if c.group != "threads" and not (c.group == "headline" and c.workload.workload_id in widened)
    ]
    scaling = ["prismq", "aer-statevector", "qsim", "quest"]
    for wl in (w.rcs(24, 10), w.rcs(26, 10), w.qft(runner, 26), w.exported(runner, "qv", 24)):
        cells.append(Cell("threads", wl, scaling, _thread_counts((1, 2, 4, 8, 16), threads)))
    return Profile("full", "every size in the matrix, all thread counts up to 16 where the host has them", cells)


PROFILES = {"smoke": smoke, "reference": reference, "full": full}
