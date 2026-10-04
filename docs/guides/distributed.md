# Distributed Statevector and MPI

The distributed statevector backend splits one exact `2^n` amplitude vector across
`2^p` ranks, for registers that do not fit a single host. Each rank holds a `2^(n-p)`
slice in an ordinary statevector backend, so local gates run on the same SIMD kernels
and the same [fusion pipeline](../architecture/fusion.md) as a single-host run. The
result does not depend on the rank count: one rank is bitwise the `Statevector`
backend, and `P` ranks gather to the same vector.

```admonish info
Two feature flags. `distributed` adds the backend and the single-rank and thread
loopback transports, with no external dependency. `distributed-mpi` adds the MPI
transport over `rsmpi`, which needs a system MPI installation and libclang for its
bindgen step. The published Python wheels carry neither; build from source to enable
them (see [Python Bindings](./python.md#distributed-backend)).
```

## Running

```rust,ignore
use prism_q::{distributed::DistributedContext, simulate};

let context = DistributedContext::world()?;
let result = simulate(&circuit).distributed(context).seed(42).run()?;
```

`DistributedContext::world()` calls `MPI_Init` at `MPI_THREAD_FUNNELED` and captures
the world communicator; dropping the context on the thread that created it runs
`MPI_Finalize`. When another component already owns MPI (an embedding interpreter with
mpi4py, for instance), `DistributedContext::attached_world()` attaches without taking
ownership of the MPI lifetime. `DistributedContext::serial()` gives a single rank with
no MPI at all, which is how the tests and the `distributed`-only build run.

`simulate(&circuit).distributed(context)` is shorthand for
`.backend(BackendKind::StatevectorDistributed { context })`. Automatic dispatch never
selects this backend: a run is distributed only when the caller asks.

Launch the binary under the MPI launcher with a power-of-two rank count:

```bash
mpiexec -n 4 ./target/release/my_simulation
```

The contract is SPMD. Every rank runs the same program and enters every collective
inside the backend, so a program that branches on `context.rank()` before a simulation
call deadlocks the others.

## Memory layout

Global index `rank * 2^(n-p) + local_index`. Qubit `q` below `n - p` is bit `q` of
the local index; qubit `q` at or above `n - p` is bit `q - (n - p)` of the rank id.
Qubit 0 is the least significant bit, as everywhere in PRISM-Q, and `|0...0>` is index
0 on rank 0.

Every rank needs enough local qubits for the slice to be worth the exchanges.
`PRISM_DIST_MIN_LOCAL_QUBITS` (default 10) sets the floor, and a split that leaves
fewer is rejected at `init`.

## Qubit relabeling

At more than one rank the backend keeps a map from circuit qubits to physical
positions. A SWAP becomes a map update with no amplitude movement. Before a gate acts
non-diagonally on a qubit that currently sits in a rank bit, the backend relabels that
qubit into a local position by exchanging the half slice whose local bit differs from
the rank bit, evicting the least recently used local qubit. Later gates on the relabeled
qubit then run locally with no communication until it is evicted again. Diagonal gates
and control bits are free on global qubits and never trigger a relabel.

Relabeling wins when gate activity has locality: SWAP networks, repeated gates on the
same qubits, and working sets that fit the local positions. A cyclic scan over more hot
qubits than there are local positions evicts on every layer; `PRISM_DIST_RELABEL=0`
switches the map off and falls back to a direct exchange per gate.
`PRISM_DIST_EXCHANGE_CHUNK` tiles each exchange into messages of that many amplitudes
and bounds the transfer buffers; by default each exchange is one message.

## What runs distributed

- Gates, including fused and batched payloads, measurement, reset, and classical
  conditionals. Measurement probabilities sum with one `Allreduce`, and every rank
  draws from the same seeded stream, so the ranks agree on the outcome without
  exchanging it.
- Per-qubit probabilities, Pauli expectation values, and multi-shot terminal sampling
  answer from rank-local sums plus one reduction, at any register width, without
  gathering the dense state.
- `probabilities()` and `export_statevector()` gather and so carry the dense output
  cap; a register past the cap is rejected before the run.
- A caller-supplied `initial_state`: every rank receives the full vector and keeps its
  own slice.

Not supported: noise models, which the backend rejects because trajectory execution is
not lockstep across ranks; GPU execution, which is single device; and selection by
`BackendKind::Auto`.

## Testing and measuring

The rank logic is covered without an MPI runtime by the thread loopback transport:

```bash
cargo nextest run --features "parallel distributed" -E 'test(distributed)'
```

`scripts/test-mpi.ps1` (Windows, MS-MPI) and `scripts/test-mpi.sh` (Linux, Open MPI
flags) build `examples/dist_mpi_check.rs` with the `distributed-mpi` feature and launch
it at 1, 2 and 4 ranks under three configurations (default, tiled exchange, relabeling
off). Rank 0 compares the gathered result against a one-process statevector and fails on
mismatch, and a three-rank launch checks that the power-of-two rule is enforced. The
timed arm of each script prints per-rank timings for comparing exchange pipelines
across hosts.

`cargo bench --bench bench_distributed --features "parallel distributed bench-internal"`
runs the loopback benchmarks. Their ranks share one memory system, so the rows price
packing, copying and per-gate dispatch, not network latency; see
[Benchmark Methodology](./benchmarking.md).

The exchange paths and the relabeling algorithm are documented on the
[`distributed_statevector` module](https://docs.rs/prism-q/latest/prism_q/backend/distributed_statevector/index.html),
and the per-backend support in the [Capability and Support Matrix](./capabilities.md).
