# PRISM-Q (Python)

Python bindings for [PRISM-Q](https://github.com/AbeCoull/prism-q), a
performance-first quantum circuit simulator written in Rust.

## Install

```bash
pip install prism-q
```

The Linux and Windows wheels include the CUDA backends. They need an NVIDIA driver
and NVRTC, which the `cuda12` extra installs:

```bash
pip install "prism-q[cuda12]"
```

`prism_q.gpu_info()` reports the device, or why it cannot be used.

## Quick start

```python
import prism_q

# Build a Bell state and run it.
circuit = prism_q.CircuitBuilder(2).h(0).cx(0, 1).build()
outcome = prism_q.simulate(circuit).seed(42).run()
print(outcome.probabilities)        # array([0.5, 0., 0., 0.5])

# Parse OpenQASM and sample shots.
qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""
result = prism_q.simulate(prism_q.parse_qasm(qasm)).seed(7).shots(1000)
print(result.counts())              # {'00': ~500, '11': ~500}

# Exact statevector as a NumPy array.
sv = prism_q.simulate(prism_q.circuits.ghz(3)).seed(1).state_vector()
print(sv.dtype, sv.shape)           # complex128 (8,)
```

## Bit ordering

Count keys and measurement bit indices are **LSB-first**: in a bitstring key,
the leftmost character is classical bit 0, and `q[0]` is the least-significant
qubit. This is reversed relative to Qiskit. For example, the state where only
`q[0]` is set reads as `"10..."`, not `"...01"`.

## Features

- Fluent `CircuitBuilder` and OpenQASM 3.0 parsing.
- Classical control on measured bits (`ClassicalCondition`, `guarded` with an else
  branch) and measurement in a Pauli basis or of a Pauli product.
- OpenQASM 3.0 export (`Circuit.to_qasm`) and text, SVG, and Jupyter circuit drawing.
- Reusable circuit generators (`prism_q.circuits`): QFT, GHZ, QAOA,
  hardware-efficient ansatz, quantum volume, and more.
- Backend selection via `BackendKind` (statevector, stabilizer, sparse, MPS,
  Pauli propagation, ...).
- Parameter sweeps without a rebuild (`Parameters`, `PreparedCircuit`) plus
  adjoint gradients.
- Noise models (`NoiseModel`, `NoiseChannel`) for shot sampling.
- Native QEC programs (`QecProgram`) with detector and observable sampling.
- NumPy output for probabilities, statevectors, and QEC bit matrices.

## License

MIT OR Apache-2.0
