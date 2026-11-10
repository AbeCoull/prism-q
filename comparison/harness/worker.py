"""Python-driven simulators behind the same process protocol as the native runners.

    python worker.py <simulator> exec --mode probs|counts|expval|stats --shots S --iterations K
                     --threads T [--out PATH] [--budget-ms B] [--min-iterations M]
                     [--keys PATH] < gate_list.txt
    python worker.py <simulator> version

Stim runs single-threaded; `--threads` is recorded but has no effect on it.
One process per measurement, so peak memory and import cost belong to the
workload that caused them. Prints one JSON line. The timed region ends at the
simulator's own result object (Aer's `Result` with its saved probabilities or
expectation values, qsim's probability vector or measurement array, Stim's bit-packed
sample array, quimb's sample list or values); converting that to the harness's file format
happens after the timer. In `stats` mode Stim samples bit-packed buffers of at most
64 MiB and only the sampling is timed; reducing each buffer to the `--keys` counts is not.

quimb's tensor network contracts each observable over its reverse light cone with a
fresh cotengra `auto-hq` optimizer per observable and cleared circuit storage per call, so path finding
is inside every timed call. quimb's CircuitMPS applies gates as they are added, so its
timed call builds the MPS from the gate list.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

_T0 = time.perf_counter()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from workloads import parse  # noqa: E402

SIM_SEED = 42
STATS_CHUNK_BYTES = 1 << 26


def counts_text_from_ints(pairs: list[tuple[int, int]], num_bits: int) -> str:
    words = max(1, (num_bits + 63) // 64)
    mask = (1 << 64) - 1
    lines = []
    for value, count in sorted(pairs, key=lambda kv: [(kv[0] >> (64 * w)) & mask for w in range(words)]):
        key = ",".join(f"{(value >> (64 * w)) & mask:x}" for w in range(words))
        lines.append(f"{key} {count}")
    return "\n".join(lines) + "\n"


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def values_text(keys: list[str], values: list[float]) -> bytes:
    return "".join(f"{k} {float(v)!r}\n" for k, v in zip(keys, values)).encode()


def z_terms(key: str) -> list[int]:
    return [int(x) for x in key.split("Z")[1:]]


def stat_totals(packed: np.ndarray, keys: list[str], record_of_bit: dict[int, int], totals: np.ndarray) -> None:
    """Add each key's count of ones (`bA`) or odd parities (`bA^bB`) over a bit-packed shot block.

    `packed` is (shots, bytes) with measurement record `r` at bit `r % 8` of byte `r // 8`.
    """
    columns = [[record_of_bit[int(part[1:])] for part in key.split("^")] for key in keys]
    records = sorted({r for cols in columns for r in cols})
    where = {r: i for i, r in enumerate(records)}
    first = np.asarray([where[c[0]] for c in columns])
    second = np.asarray([where[c[-1]] for c in columns])
    single = np.asarray([len(c) == 1 for c in columns])
    byte = np.asarray([r // 8 for r in records])
    shift = np.asarray([r % 8 for r in records], dtype=np.uint8)
    for start in range(0, packed.shape[0], 1 << 14):
        bits = (packed[start : start + (1 << 14)][:, byte] >> shift) & 1
        ones = bits[:, first].sum(axis=0, dtype=np.int64)
        odd = (bits[:, first] ^ bits[:, second]).sum(axis=0, dtype=np.int64)
        totals += np.where(single, ones, odd)


class Aer:
    def __init__(self, name: str, threads: int) -> None:
        from qiskit_aer import AerSimulator

        base, _, option = name.partition(":")
        method = base.removeprefix("aer-")
        options: dict = {"max_parallel_threads": threads}
        self.precision = "double"
        if method == "statevector-f32":
            method = "statevector"
            options["precision"] = "single"
            self.precision = "single"
        if method == "mps" and option:
            options["matrix_product_state_max_bond_dimension"] = int(option)
        if method in ("mps", "automatic"):
            options["mps_lapack"] = True
            options["mps_omp_threads"] = threads
        if method == "mps":
            method = "matrix_product_state"
        self.method = method
        self.backend = AerSimulator(method=method, **options)

    @staticmethod
    def version() -> str:
        from importlib import metadata

        return metadata.version("qiskit-aer")

    def build(self, text: str, mode: str):
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

        n, clbits, instructions = parse(text)
        qreg = QuantumRegister(n, "q")
        creg = ClassicalRegister(max(clbits, 1), "c")
        qc = QuantumCircuit(qreg, creg)

        def gate(name, qubits, angle):
            if name in ("h", "x", "z", "s", "sdg"):
                getattr(qc, name)(qubits[0])
            elif name == "ry":
                qc.ry(angle, qubits[0])
            elif name == "rz":
                qc.rz(angle, qubits[0])
            elif name == "cx":
                qc.cx(qubits[0], qubits[1])
            elif name == "cz":
                qc.cz(qubits[0], qubits[1])
            elif name == "swap":
                qc.swap(qubits[0], qubits[1])
            elif name == "cp":
                qc.cp(angle, qubits[0], qubits[1])

        for inst in instructions:
            kind = inst[0]
            if kind == "gate":
                gate(*inst[1:])
            elif kind == "measure":
                qc.measure(inst[1], inst[2])
            elif kind == "reset":
                qc.reset(inst[1])
            elif kind == "if":
                with qc.if_test((creg[inst[1]], inst[2])):
                    gate(*inst[3:])
        if mode == "probs":
            qc.save_probabilities()
        elif mode == "expval":
            from qiskit.quantum_info import Pauli

            for key in self.keys:
                qubits = z_terms(key)
                qc.save_expectation_value(Pauli("Z" * len(qubits)), qubits, label=key)
        self.num_bits = clbits
        return qc

    def execute(self, circuit, mode: str, shots: int):
        if mode in ("probs", "expval"):
            result = self.backend.run(circuit, shots=1).result()
        else:
            result = self.backend.run(circuit, shots=shots, seed_simulator=SIM_SEED).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        return result

    def output(self, result, mode: str) -> bytes:
        if mode == "probs":
            return np.asarray(result.data()["probabilities"], dtype="<f8").tobytes()
        if mode == "expval":
            data = result.data()
            return values_text(self.keys, [np.real(data[k]) for k in self.keys])
        counts = result.data()["counts"]
        pairs = [(int(k, 16), int(v)) for k, v in counts.items()]
        return counts_text_from_ints(pairs, self.num_bits).encode()

    def details(self, result) -> dict:
        entry = result.results[0]
        meta = getattr(entry, "metadata", {}) or {}
        out = {
            "resolved_backend": meta.get("method"),
            "internal_ms": float(getattr(entry, "time_taken", 0.0)) * 1000.0,
            "fusion_applied": (meta.get("fusion") or {}).get("applied"),
        }
        opts = self.backend.options
        if self.method in ("matrix_product_state", "automatic"):
            out["details"] = {"mps_lapack": opts.mps_lapack, "mps_omp_threads": opts.mps_omp_threads}
        if self.method == "matrix_product_state":
            out["details"]["truncation_threshold"] = opts.matrix_product_state_truncation_threshold
        return out


class Qsim:
    precision = "single"

    def __init__(self, name: str, threads: int) -> None:
        import qsimcirq

        self.qsimcirq, self.threads = qsimcirq, threads

    @staticmethod
    def fused_gate_size(num_qubits: int) -> int:
        """qsimcirq's documented `max_fused_gate_size` guidance: 3 below 22 qubits, 4 from 22."""
        return 3 if num_qubits < 22 else 4

    @staticmethod
    def version() -> str:
        from importlib import metadata

        return metadata.version("qsimcirq")

    def build(self, text: str, mode: str):
        import cirq

        n, clbits, instructions = parse(text)
        self.fused = self.fused_gate_size(n)
        options = self.qsimcirq.QSimOptions(max_fused_gate_size=self.fused, cpu_threads=self.threads)
        self.simulator = self.qsimcirq.QSimSimulator(options, seed=SIM_SEED)
        qubits = cirq.LineQubit.range(n)
        ops = []
        measured: dict[int, int] = {}
        for inst in instructions:
            kind = inst[0]
            if kind == "measure":
                if inst[2] in measured:
                    raise RuntimeError(f"qsim adapter measures each classical bit once; bit {inst[2]} repeats")
                measured[inst[2]] = inst[1]
                continue
            if kind != "gate":
                raise RuntimeError("qsim adapter carries no reset or classically controlled gate")
            if measured:
                raise RuntimeError("qsim adapter samples terminal measurements only")
            name, targets, angle = inst[1:]
            q = [qubits[i] for i in targets]
            if name == "h":
                ops.append(cirq.H(q[0]))
            elif name == "x":
                ops.append(cirq.X(q[0]))
            elif name == "z":
                ops.append(cirq.Z(q[0]))
            elif name == "s":
                ops.append(cirq.S(q[0]))
            elif name == "sdg":
                ops.append(cirq.S(q[0]) ** -1)
            elif name == "ry":
                ops.append(cirq.ry(angle)(q[0]))
            elif name == "rz":
                ops.append(cirq.rz(angle)(q[0]))
            elif name == "cx":
                ops.append(cirq.CNOT(q[0], q[1]))
            elif name == "cz":
                ops.append(cirq.CZ(q[0], q[1]))
            elif name == "swap":
                ops.append(cirq.SWAP(q[0], q[1]))
            elif name == "cp":
                ops.append(cirq.CZPowGate(exponent=angle / np.pi)(q[0], q[1]))
        self.num_bits = clbits
        if mode == "counts":
            if sorted(measured) != list(range(clbits)):
                raise RuntimeError("qsim adapter needs every classical bit measured exactly once")
            order = [qubits[measured[b]] for b in range(clbits)]
            ops.append(cirq.measure(*order, key="m"))
        # Cirq orders the state index with the first qubit most significant;
        # reversing puts qubit 0 in the least significant bit like every adapter.
        return cirq.Circuit(ops), list(reversed(qubits))

    def execute(self, prepared, mode: str, shots: int):
        circuit, order = prepared
        if mode == "probs":
            _, state, _ = self.simulator.simulate_into_1d_array(circuit, qubit_order=order)
            parts = state.view(np.float32)
            np.square(parts, out=parts)
            probs = parts[0::2]
            return np.add(probs, parts[1::2], out=probs)
        return self.simulator.run(circuit, repetitions=shots).measurements["m"]

    def output(self, result, mode: str) -> bytes:
        if mode == "probs":
            return result.astype("<f8").tobytes()
        packed = np.packbits(result.astype(np.uint8), axis=1, bitorder="little")
        rows, counts = np.unique(packed, axis=0, return_counts=True)
        pairs = [(int.from_bytes(row.tobytes(), "little"), int(c)) for row, c in zip(rows, counts)]
        return counts_text_from_ints(pairs, self.num_bits).encode()

    def details(self, result) -> dict:
        # The simulator is seeded once, so successive calls draw fresh samples.
        return {
            "resolved_backend": "statevector",
            "details": {"max_fused_gate_size": self.fused, "cpu_threads": self.threads},
            "reseeds_per_call": False,
        }


class Stim:
    """Stim's compiled sampler; the timed call compiles the circuit and samples it."""

    precision = "exact (tableau)"
    _NAMES = {"h": "H", "x": "X", "z": "Z", "s": "S", "sdg": "S_DAG", "cx": "CX", "cz": "CZ", "swap": "SWAP"}

    def __init__(self, name: str, threads: int) -> None:
        import stim

        self.stim = stim

    @staticmethod
    def version() -> str:
        import stim

        return stim.__version__

    def build(self, text: str, mode: str):
        if mode not in ("counts", "stats"):
            raise RuntimeError("stim samples measurements only")
        _, clbits, instructions = parse(text)
        lines: list[str] = []
        record: list[int] = []
        for inst in instructions:
            kind = inst[0]
            if kind == "gate":
                name, qubits, _ = inst[1:]
                if name not in self._NAMES:
                    raise RuntimeError(f"stim runs Clifford circuits only; got {name}")
                lines.append(f"{self._NAMES[name]} {' '.join(map(str, qubits))}")
            elif kind == "measure":
                lines.append(f"M {inst[1]}")
                record.append(inst[2])
            elif kind == "reset":
                lines.append(f"R {inst[1]}")
            else:
                _, bit, value, name, qubits, _ = inst
                if value != 1 or name not in ("x", "z"):
                    raise RuntimeError("stim feed-forward carries classically controlled X and Z only")
                back = len(record) - max(i for i, b in enumerate(record) if b == bit)
                lines.append(f"C{name.upper()} rec[-{back}] {qubits[0]}")
        if sorted(record) != list(range(clbits)):
            raise RuntimeError("stim adapter needs every classical bit measured exactly once")
        self.num_bits = clbits
        # Column j of a sample is the j-th measurement; reorder to classical bit order.
        self.order = np.argsort(np.asarray(record))
        self.record_of_bit = {bit: j for j, bit in enumerate(record)}
        return self.stim.Circuit("\n".join(lines))

    def execute(self, circuit, mode: str, shots: int):
        return circuit.compile_sampler(seed=SIM_SEED).sample(shots, bit_packed=True)

    def timed_execute(self, circuit, mode: str, shots: int):
        """Run and time one call; `stats` reduces each packed block to the key totals outside the timer."""
        if mode != "stats":
            start = time.perf_counter()
            result = self.execute(circuit, mode, shots)
            return result, (time.perf_counter() - start) * 1000.0
        chunk = max(1, STATS_CHUNK_BYTES // max(1, (circuit.num_measurements + 7) // 8))
        totals = np.zeros(len(self.keys), dtype=np.int64)
        start = time.perf_counter()
        sampler = circuit.compile_sampler(seed=SIM_SEED)
        elapsed = time.perf_counter() - start
        left = shots
        while left > 0:
            take = min(left, chunk)
            start = time.perf_counter()
            packed = sampler.sample(take, bit_packed=True)
            elapsed += time.perf_counter() - start
            stat_totals(packed, self.keys, self.record_of_bit, totals)
            del packed
            left -= take
        return (shots, totals), elapsed * 1000.0

    def output(self, result, mode: str) -> bytes:
        if mode == "stats":
            shots, totals = result
            return (f"shots {shots}\n" + "".join(f"{k} {int(t)}\n" for k, t in zip(self.keys, totals))).encode()
        bits = np.unpackbits(result, axis=1, count=self.num_bits, bitorder="little")[:, self.order]
        packed = np.packbits(bits, axis=1, bitorder="little")
        rows, counts = np.unique(packed, axis=0, return_counts=True)
        pairs = [(int.from_bytes(row.tobytes(), "little"), int(c)) for row, c in zip(rows, counts)]
        return counts_text_from_ints(pairs, self.num_bits).encode()

    def details(self, result) -> dict:
        return {"resolved_backend": "compiled sampler"}


class Quimb:
    """quimb's exact tensor network (`quimb-tn`) or its CircuitMPS under a bond cap (`quimb-mps:<cap>`)."""

    precision = "double"
    MPS_CUTOFF = 1e-16

    def __init__(self, name: str, threads: int) -> None:
        import cotengra
        import quimb
        import quimb.tensor as qtn

        self.cotengra, self.qtn = cotengra, qtn
        self.z = quimb.pauli("Z")
        self.zz = np.kron(self.z, self.z)
        base, _, option = name.partition(":")
        self.mps = base == "quimb-mps"
        self.max_bond = int(option) if option else None

    @staticmethod
    def version() -> str:
        from importlib import metadata

        return metadata.version("quimb")

    def build(self, text: str, mode: str):
        n, clbits, instructions = parse(text)
        gates, measured = [], {}
        for inst in instructions:
            if inst[0] == "measure":
                measured[inst[2]] = inst[1]
                continue
            if inst[0] != "gate":
                raise RuntimeError("quimb adapter carries no reset or classically controlled gate")
            if measured:
                raise RuntimeError("quimb adapter samples terminal measurements only")
            name, qubits, angle = inst[1:]
            gates.append((name.upper(), (angle,) if angle is not None else (), tuple(qubits)))
        self.n, self.num_bits, self.measured = n, clbits, measured
        if self.mps:
            return gates
        circuit = self.qtn.Circuit(n)
        for name, params, qubits in gates:
            circuit.apply_gate(name, *params, *qubits)
        return circuit

    def _optimizer(self):
        return self.cotengra.AutoHQOptimizer(cache=False)

    def execute(self, prepared, mode: str, shots: int):
        if not self.mps:
            prepared.clear_storage()
            # A hyper optimizer keeps the trees it found, so each light cone gets its own.
            return [
                prepared.local_expectation(self.z if len(qs) == 1 else self.zz, tuple(qs), optimize=self._optimizer())
                for qs in map(z_terms, self.keys)
            ]
        circuit = self.qtn.CircuitMPS(self.n, max_bond=self.max_bond, cutoff=self.MPS_CUTOFF)
        for name, params, qubits in prepared:
            circuit.apply_gate(name, *params, *qubits)
        if mode == "counts":
            return list(circuit.sample(shots, seed=SIM_SEED))
        return [
            circuit.local_expectation(self.z if len(qs) == 1 else self.zz, tuple(qs), normalized=True)
            for qs in map(z_terms, self.keys)
        ]

    def output(self, result, mode: str) -> bytes:
        if mode == "expval":
            return values_text(self.keys, [np.real(v) for v in result])
        tally: dict[int, int] = {}
        for config in result:
            value = sum(1 << bit for bit, q in self.measured.items() if config[q] == "1")
            tally[value] = tally.get(value, 0) + 1
        return counts_text_from_ints(list(tally.items()), self.num_bits).encode()

    def details(self, result) -> dict:
        if self.mps:
            return {
                "resolved_backend": f"CircuitMPS, max_bond {self.max_bond}",
                "details": {"cutoff": self.MPS_CUTOFF, "normalized_expectations": True},
                "reseeds_per_call": True,
            }
        from cotengra.hyperoptimizers.hyper import get_default_hq_methods, get_default_optlib

        return {
            "resolved_backend": "tensor network, reverse light cone per observable",
            "details": {"optimizer": optimizer_details(get_default_hq_methods(), get_default_optlib())},
            "reseeds_per_call": False,
        }


def optimizer_details(methods, optlib) -> dict:
    import importlib.util

    return {
        "preset": "auto-hq",
        "cache": False,
        "hyper_methods": list(methods),
        "optlib": optlib,
        "kahypar_available": importlib.util.find_spec("kahypar") is not None,
        "cotengrust_available": importlib.util.find_spec("cotengrust") is not None,
        "exact_below_hardness": 650,
        "storage_cleared_per_call": True,
    }


def make(name: str, threads: int):
    if name.startswith("aer-"):
        return Aer(name, threads)
    if name == "qsim":
        return Qsim(name, threads)
    if name == "stim":
        return Stim(name, threads)
    if name.startswith("quimb-"):
        return Quimb(name, threads)
    raise SystemExit(f"unknown simulator {name}")


def _class(name: str):
    if name.startswith("aer-"):
        return Aer
    if name.startswith("quimb-"):
        return Quimb
    return Stim if name == "stim" else Qsim


def _timed(sim, prepared, mode: str, shots: int):
    if hasattr(sim, "timed_execute"):
        return sim.timed_execute(prepared, mode, shots)
    start = time.perf_counter()
    result = sim.execute(prepared, mode, shots)
    return result, (time.perf_counter() - start) * 1000.0


def main() -> int:
    name = sys.argv[1]
    command = sys.argv[2]
    if command == "version":
        print(json.dumps({"simulator": name, "version": _class(name).version()}))
        return 0
    args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    mode = args.get("--mode", "probs")
    shots = int(args.get("--shots", "0"))
    iterations = int(args.get("--iterations", "5"))
    threads = int(args.get("--threads", "1"))
    out_path = args.get("--out")
    budget_ms = float(args.get("--budget-ms", "inf"))
    min_iterations = int(args.get("--min-iterations", "1"))

    import_start = time.perf_counter()
    sim = make(name, threads)
    import_ms = (time.perf_counter() - import_start) * 1000.0
    text = sys.stdin.read()
    sim.keys = []
    if "--keys" in args:
        with open(args["--keys"], encoding="utf-8") as handle:
            sim.keys = [line.strip() for line in handle if line.strip()]

    start = time.perf_counter()
    prepared = sim.build(text, mode)
    build_ms = (time.perf_counter() - start) * 1000.0

    result, first_ms = _timed(sim, prepared, mode, shots)
    data = sim.output(result, mode)
    details = sim.details(result)
    first_internal = details.pop("internal_ms", None)
    hashes = [_digest(data)]
    if out_path:
        with open(out_path, "wb") as handle:
            handle.write(data)
    del data, result

    times_ms: list[float] = []
    internal_ms: list[float] = []
    for i in range(iterations):
        if i >= min_iterations and sum(times_ms) >= budget_ms:
            break
        result, ms = _timed(sim, prepared, mode, shots)
        times_ms.append(ms)
        info = sim.details(result)
        if "internal_ms" in info:
            internal_ms.append(info["internal_ms"])
        hashes.append(_digest(sim.output(result, mode)))
        del result

    print(
        json.dumps(
            {
                "schema": "prismq-compare-worker/1",
                "simulator": name,
                "version": sim.version(),
                "precision": sim.precision,
                "threads": threads,
                "startup_ms": (import_start - _T0) * 1000.0,
                "import_ms": import_ms,
                "build_ms": build_ms,
                "first_ms": first_ms,
                "times_ms": times_ms,
                "internal_ms": internal_ms,
                "first_internal_ms": first_internal,
                "output_hashes": hashes,
                **details,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
