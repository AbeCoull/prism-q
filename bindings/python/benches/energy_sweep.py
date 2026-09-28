"""Time a Python energy sweep with the Hamiltonian as a term list and as a PauliObservable.

The Python counterpart of the ``prepared/hea_l2_energy/prepared`` Criterion row: a
two-layer hardware-efficient ansatz held in a ``PreparedCircuit``, twenty bindings
per sweep, and an energy evaluated at each. ``list`` passes the term list, which
the bindings parse and group on every call; ``held`` passes a ``PauliObservable``
built once. The ``many`` modes make the same sweep one ``observable_expectation_many``
call. Build the wheel with the ``python-release`` profile and run on an idle host.

    python bindings/python/benches/energy_sweep.py --repeats 30
"""

import argparse
import json
import math
import statistics
import time

import numpy as np

from prism_q import Parameters, PauliObservable, PreparedCircuit, circuits

SEED = 0xDEAD_BEEF
POINTS = 20


def tfim(n):
    """``ZZ`` chain plus an ``X`` field, the Criterion row's Hamiltonian."""
    chain = [(1.0, [(q, "Z"), (q + 1, "Z")]) for q in range(n - 1)]
    field = [(0.5, [(q, "X")]) for q in range(n)]
    return chain + field


def heisenberg(n):
    """Couplings at distance one and two plus fields on all three axes, ``9n - 9`` terms."""
    terms = []
    for d in (1, 2):
        for q in range(n - d):
            terms.append((1.0, [(q, "X"), (q + d, "X")]))
            terms.append((0.9, [(q, "Y"), (q + d, "Y")]))
            terms.append((0.8, [(q, "Z"), (q + d, "Z")]))
    for q in range(n):
        terms.append((0.3, [(q, "X")]))
        terms.append((0.2, [(q, "Y")]))
        terms.append((0.1, [(q, "Z")]))
    return terms


HAMILTONIANS = {"tfim": tfim, "heisenberg": heisenberg}


def sweeps(prepared, points, terms):
    held = PauliObservable(terms)

    def loop(h):
        for values in points:
            prepared.observable_expectation(values, h, SEED)

    def many(h):
        prepared.observable_expectation_many(points, h, SEED)

    return {
        "list": lambda: loop(terms),
        "held": lambda: loop(held),
        "many_list": lambda: many(terms),
        "many_held": lambda: many(held),
    }


def time_sweep(sweep, repeats):
    sweep()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        sweep()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[10, 12, 14])
    parser.add_argument(
        "--hamiltonian", choices=sorted(HAMILTONIANS), nargs="+", default=["tfim", "heisenberg"]
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json", help="write the per-row samples to this path")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'row':<44} {'terms':>5} {'min ms':>9} {'median ms':>10} {'us/point':>9}")
    for n in args.qubits:
        template = circuits.hardware_efficient_ansatz(n, 2, SEED)
        params = Parameters.all_rotations(template)
        points = rng.uniform(0.0, 2.0 * math.pi, size=(POINTS, params.num_slots))
        prepared = PreparedCircuit(template, params)
        for name in args.hamiltonian:
            terms = HAMILTONIANS[name](n)
            for mode, sweep in sweeps(prepared, points, terms).items():
                samples = time_sweep(sweep, args.repeats)
                row = f"python/hea_l2_energy/{name}/{mode}/{n}"
                best, median = min(samples), statistics.median(samples)
                print(
                    f"{row:<44} {len(terms):>5} {best * 1e3:>9.3f} {median * 1e3:>10.3f}"
                    f" {median / POINTS * 1e6:>9.1f}"
                )
                rows.append({"row": row, "terms": len(terms), "seconds": samples})

    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(rows, out, indent=1)


if __name__ == "__main__":
    main()
