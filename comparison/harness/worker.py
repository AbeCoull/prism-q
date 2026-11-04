"""Python-driven simulators behind the same process protocol as the native runners.

    python worker.py <simulator> exec --mode probs|counts --shots S --iterations K
                     --threads T [--out PATH] [--budget-ms B] [--min-iterations M]
                     < gate_list.txt
    python worker.py <simulator> version

Stim runs single-threaded; `--threads` is recorded but has no effect on it.
One process per measurement, so peak memory and import cost belong to the
workload that caused them. Prints one JSON line. The timed region ends at the
simulator's own result object (Aer's `Result` with its saved probabilities,
qsim's probability vector or measurement array, Stim's sample array);
converting that to the harness's file format happens after the timer.
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
        self.num_bits = clbits
        return qc

    def execute(self, circuit, mode: str, shots: int):
        if mode == "probs":
            result = self.backend.run(circuit, shots=1).result()
        else:
            result = self.backend.run(circuit, shots=shots, seed_simulator=SIM_SEED).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        return result

    def output(self, result, mode: str) -> bytes:
        if mode == "probs":
            return np.asarray(result.data()["probabilities"], dtype="<f8").tobytes()
        counts = result.data()["counts"]
        pairs = [(int(k, 16), int(v)) for k, v in counts.items()]
        return counts_text_from_ints(pairs, self.num_bits).encode()

    def details(self, result) -> dict:
        entry = result.results[0]
        meta = getattr(entry, "metadata", {}) or {}
        return {
            "resolved_backend": meta.get("method"),
            "internal_ms": float(getattr(entry, "time_taken", 0.0)) * 1000.0,
            "fusion_applied": (meta.get("fusion") or {}).get("applied"),
        }


class Qsim:
    precision = "single"

    def __init__(self, name: str, threads: int) -> None:
        import qsimcirq

        self.simulator = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=threads), seed=SIM_SEED)

    @staticmethod
    def version() -> str:
        from importlib import metadata

        return metadata.version("qsimcirq")

    def build(self, text: str, mode: str):
        import cirq

        n, clbits, instructions = parse(text)
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
            probs = np.abs(self.simulator.simulate(circuit, qubit_order=order).final_state_vector)
            return np.square(probs, out=probs)
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
        return {"resolved_backend": "statevector", "reseeds_per_call": False}


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
        if mode != "counts":
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
        return self.stim.Circuit("\n".join(lines))

    def execute(self, circuit, mode: str, shots: int):
        return circuit.compile_sampler(seed=SIM_SEED).sample(shots)

    def output(self, result, mode: str) -> bytes:
        bits = result[:, self.order].astype(np.uint8)
        packed = np.packbits(bits, axis=1, bitorder="little")
        rows, counts = np.unique(packed, axis=0, return_counts=True)
        pairs = [(int.from_bytes(row.tobytes(), "little"), int(c)) for row, c in zip(rows, counts)]
        return counts_text_from_ints(pairs, self.num_bits).encode()

    def details(self, result) -> dict:
        return {"resolved_backend": "compiled sampler"}


def make(name: str, threads: int):
    if name.startswith("aer-"):
        return Aer(name, threads)
    if name == "qsim":
        return Qsim(name, threads)
    if name == "stim":
        return Stim(name, threads)
    raise SystemExit(f"unknown simulator {name}")


def main() -> int:
    name = sys.argv[1]
    command = sys.argv[2]
    if command == "version":
        cls = Aer if name.startswith("aer-") else (Stim if name == "stim" else Qsim)
        print(json.dumps({"simulator": name, "version": cls.version()}))
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

    start = time.perf_counter()
    prepared = sim.build(text, mode)
    build_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    result = sim.execute(prepared, mode, shots)
    first_ms = (time.perf_counter() - start) * 1000.0
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
        start = time.perf_counter()
        result = sim.execute(prepared, mode, shots)
        times_ms.append((time.perf_counter() - start) * 1000.0)
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
