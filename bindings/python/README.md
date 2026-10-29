# PRISM-Q for Python

A quantum circuit simulator with a Rust core. PRISM-Q reads the structure of each
circuit and picks how to store its state: a stabilizer tableau for Clifford circuits, a
dense statevector for general ones, and a sparse map, matrix product state, product state
or factored form where the circuit allows. The result reports which one ran.

- Source, issues and discussions: <https://github.com/AbeCoull/prism-q>
- Documentation: <https://abecoull.github.io/prism-q/>

## Install

```bash
pip install prism-q
```

Wheels cover CPython 3.11 and newer on Linux (x86_64, aarch64), macOS (arm64) and
Windows (x64). Other platforms build from the source distribution, which needs a Rust
toolchain.

## Quick start

```python
import prism_q

# Exact probabilities of a Bell state.
bell = prism_q.CircuitBuilder(2).h(0).cx(0, 1).build()
print(prism_q.simulate(bell).seed(42).run().probabilities)  # [0.5 0.  0.  0.5]

# Parse OpenQASM and sample shots. Counts are sampled, so they move with the seed.
qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
result = prism_q.simulate(prism_q.parse_qasm(qasm)).seed(42).shots(1000)
print(result.counts())  # {'11': 493, '00': 507}

# The exact statevector as a NumPy array.
sv = prism_q.simulate(prism_q.circuits.ghz(3)).seed(42).state_vector()
print(sv.dtype, sv.shape)  # complex128 (8,)
```

## Why the backend choice matters

```python
from prism_q import CircuitBuilder, circuits, simulate

def backend(circuit):
    return simulate(circuit).seed(42).run().metadata.backend

print(backend(circuits.qft(16)))    # Statevector
print(backend(circuits.ghz(1000)))  # Stabilizer

ghz_t = CircuitBuilder(60).h(0)
for q in range(59):
    ghz_t.cx(q, q + 1)
print(backend(ghz_t.t(0).build()))  # Sparse
```

A 1,000-qubit GHZ circuit is all Clifford gates, so a stabilizer tableau holds it. Add
one T gate to a 60-qubit GHZ circuit and the tableau no longer applies, but only two of
its 2^60 basis states are ever nonzero, so a sparse map does. The selection follows
structural rules and the host's memory, not timed trials, so naming a backend with
`.backend(prism_q.BackendKind.statevector())` and its siblings can beat it on a circuit
the rules misjudge. `metadata.reason` says which rule fired.

## What the package covers

- `CircuitBuilder`, OpenQASM 3.0 parsing and export (`Circuit.to_qasm`), and text, SVG
  and Jupyter circuit drawing.
- Classical control on measured bits (`ClassicalCondition`, `guarded` with an else
  branch) and measurement in a Pauli basis or of a Pauli product.
- Circuit generators in `prism_q.circuits`: QFT, GHZ, QAOA, hardware-efficient ansatz,
  quantum volume and more.
- Parameter sweeps without a rebuild (`Parameters`, `PreparedCircuit`) and adjoint
  gradients.
- Noise models (`NoiseModel`, `NoiseChannel`, two-qubit Kraus channels included) for
  shot sampling.
- Compiled Clifford samplers (`CompiledSampler`) with streamed marginals, parities and
  correlators, plus analytic noisy marginals (`noisy_marginals_analytical`).
- QEC programs (`QecProgram`) with detector and observable sampling and feed-forward
  corrections.
- NumPy output for probabilities, statevectors and QEC bit matrices, and pickling for
  circuits, parameters, noise models and QEC programs, so they cross `multiprocessing`,
  Ray and Dask boundaries.

## GPU

The Linux and Windows wheels include the CUDA backends; the macOS wheel does not. Nothing
CUDA-related loads at import, so the package runs on the CPU on any machine. A GPU run
needs an NVIDIA device with a CUDA 12 driver, installed through the system, and NVRTC,
which the `cuda12` extra installs from PyPI:

```bash
pip install "prism-q[cuda12]"
```

Without a driver or NVRTC, only `GpuContext(...)` raises. `prism_q.gpu_info()` reports
the device, or why it cannot be used. MPI-distributed execution is a Rust feature and is
not in the wheels.

## Bit ordering

Count keys and measurement bit indices are LSB-first: in a bitstring key the leftmost
character is classical bit 0, and `q[0]` is the least significant qubit of a statevector
index. This is the reverse of Qiskit. Measuring `q[i]` into `c[i]`, the state where only
`q[0]` is set reads as `"10..."`, not `"...01"`.

## Documentation

- [Python guide](https://abecoull.github.io/prism-q/guides/python.html)
- [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) and
  [example scripts](https://github.com/AbeCoull/prism-q/tree/main/bindings/python/examples)
- [Choosing a backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html)
- [Capability matrix](https://abecoull.github.io/prism-q/guides/capabilities.html), for
  what runs where and what is experimental
- [Benchmarks](https://abecoull.github.io/prism-q/benchmarks.html) and the
  [method behind them](https://abecoull.github.io/prism-q/guides/benchmarking.html)
- [Changelog](https://github.com/AbeCoull/prism-q/blob/main/CHANGELOG.md)

## Contributing

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/AbeCoull/prism-q). If the package is useful in your work, a
star on the repository helps other people find it.

## License

MIT OR Apache-2.0
