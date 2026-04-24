# Architecture Glossary

This glossary defines key terms used throughout the PRISM-Q architecture and documentation.

## Terms

**Backend**
A simulation strategy implementing the `Backend` trait (e.g., statevector, stabilizer, MPS). Backends are swappable and selected automatically or explicitly per circuit.

**Clifford**
A class of quantum gates (H, S, CNOT, etc.) that can be simulated efficiently on stabilizer tableaus in O(n²) time.

**Count**
A frequency histogram of measured bitstrings across multiple shots, returned by `run_counts`.

**Fusion**
The pre-simulation optimization pipeline that merges, cancels, and reorders gates to reduce execution cost.

**Marginal**
Per-qubit probabilities of measuring `|1⟩`, extracted without computing the full multi-qubit distribution.

**MPS (Matrix Product State)**
A tensor-network backend that stores the quantum state as a chain of rank-3 tensors with bounded bond dimension, trading exactness for polynomial memory.

**Noise model**
A collection of probabilistic error channels (e.g., depolarizing noise) applied during noisy multi-shot simulation.

**OpenQASM**
An assembly-like language for describing quantum circuits. PRISM-Q parses a practical subset of OpenQASM 3.0.

**Shot**
A single execution of a quantum circuit from initialization to measurement, producing one bitstring sample.

**State vector**
A flat vector of 2^n complex amplitudes representing the full quantum state, used by the statevector backend.

**Stabilizer**
A representation of a Clifford state as a bit-packed tableau of Pauli generators, enabling simulation of thousands of qubits.

**Tensor network**
A backend that represents gates as tensors and defers contraction until measurement or probability extraction.
