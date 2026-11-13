# Roadmap

Open work on PRISM-Q, grouped by theme in rough priority order, with what each item
waits on. These are intentions, not commitments, and carry no dates. A shipped item
leaves the page for the
[changelog](https://github.com/AbeCoull/prism-q/blob/main/CHANGELOG.md).

1.0 has no date. Until then, [Compatibility and Versioning](./guides/compatibility.md)
says what each release promises.

| Status | Meaning |
| --- | --- |
| Planned | Open, with nothing outside the project in the way |
| Needs hardware | Waits on a machine the project has no access to; see [Hardware that would help](#hardware-that-would-help) |
| Needs a use case | Waits on a circuit or caller that needs it. A [discussion](https://github.com/AbeCoull/prism-q/discussions) describing one moves it up |
| Research | The approach is not settled |

## QEC and noise

| Item | Status |
| --- | --- |
| Shorter time to first shot on large surface code memory experiments, where parsing and compiling the program take most of a single-shot run | Planned |
| Readout faults on mid-circuit measurements that flip the recorded bit, and every conditional that reads it, without touching the state | Planned |
| Raise two [declared limits](./guides/capabilities.md#declared-limits): `EXP_VAL` past 64 qubits on the reference QEC runner, and analytical postselection past 12 rows | Planned |
| Error mitigation primitives on the expectation-value API | Needs a use case |

## Stabilizer and Clifford+T

| Item | Status |
| --- | --- |
| Exact stabilizer-rank simulation when the T gates reach more than 20 qubits, from decompositions that cover several T gates at once rather than one | Research |
| A dense tableau mode for wide Clifford circuits whose generators fill in | Research |

## Large structured circuits

| Item | Status |
| --- | --- |
| Per-slice engine choice when a tensor-network contraction is sliced, so a Clifford slice runs on the stabilizer tableau | Research |
| Reduced density matrices on several qubits from the tensor network, which would also let it take two-qubit Kraus noise | Planned |
| Entanglement entropy and Schmidt values on the sparse, tensor-network and distributed backends | Needs a use case |
| Device SVD for MPS runs at high bond dimension | Research |

## Input and interop

| Item | Status |
| --- | --- |
| OpenQASM 3 constructs that decline today: private measurement storage in subroutines and the `complex` type. [The subset](./guides/openqasm.md#the-subset) lists the rest | Planned |
| Conversion from Python circuit frameworks, behind an optional dependency | Planned |
| Checkpoint and resume across processes, which needs a state format first | Research |
| Gate channels from a pulse-level device model. A closed-system compile to a unitary and leakage rates ships behind the `unstable-pulse` feature (see [Noise Models](./guides/noise.md#gates-compiled-from-pulses)); a Lindblad channel during the gate and OpenPulse input wait on a caller | Needs a use case |

## GPU on one device

| Item | Status |
| --- | --- |
| Stabilizer measurement that stays on the device instead of reading back to the host each time; the device tableau runs slower than the CPU today | Research |
| Two-qubit Kraus noise and dense three- and four-qubit unitaries on the device statevector, which declines both | Needs a use case |

## CPU reach

| Item | Status |
| --- | --- |
| AVX-512 on by default. The kernels ship behind `PRISM_AVX512=1`, verified under emulation and not yet timed on AVX-512 hardware | Needs hardware |
| Tile sizes and the NEON two-qubit kernel measured on ARM64 and Apple silicon; both were tuned on x86-64 | Needs hardware |
| Cache-tile and thread floors measured on many-core AMD Zen hosts, and NUMA placement for 28 qubits and up on multi-socket machines | Needs hardware |

## Beyond one device

| Item | Status |
| --- | --- |
| ROCm ports of the CUDA statevector and stabilizer kernels | Needs hardware |
| Multi-GPU statevector sharded across peer-connected devices on one host. The design is written; the build waits on a host with peer access | Needs hardware |
| Multi-node GPU execution, after multi-GPU on one host | Research |
| Distributed runs validated across hosts; every test so far runs its ranks on one machine | Needs hardware |
| An out-of-core statevector that streams tiles past host memory | Needs a use case |

## Hardware that would help

Thresholds and tile sizes were tuned on one desktop, an Intel i7-6700K with a GTX
1080 Ti (see [Performance](./performance/index.md)). The Needs hardware items wait on one of
these:

- an x86-64 CPU with AVX-512
- an Apple silicon Mac or another ARM64 machine
- a many-core AMD Zen host, or a multi-socket server
- an AMD GPU
- two or more NVIDIA GPUs with peer access
- two or more hosts joined by an MPI fabric

If you have one, say so in
[Discussions](https://github.com/AbeCoull/prism-q/discussions).
`scripts/tile_sweep.sh` runs the cache-tile sweep the CPU items need from one bench
binary.

## Not planned

- A command-line tool. PRISM-Q is a Rust library and a Python package.
- Single-precision amplitudes. Every backend computes in `f64` complex.
- A C ABI, WASM bindings or QIR, until someone needs one. A discussion with the use case
  reopens it.
