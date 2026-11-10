# Distributed Statevector and MPI

The distributed statevector backend splits one exact `2^n` amplitude vector across
`2^p` ranks, for registers that do not fit one host. Each rank holds a `2^(n-p)` slice
in an ordinary statevector backend, so local gates run on the same SIMD kernels and
[fusion pipeline](../architecture/fusion.md) as a single-host run. The result does not
depend on the rank count: one rank is bitwise the `Statevector` backend, and `P` ranks
gather to the same vector.

```admonish info
Two feature flags. `distributed` adds the backend and the single-rank and thread
loopback transports, with no external dependency. `distributed-mpi` adds the MPI
transport over `rsmpi`, which needs a system MPI installation and libclang for its
bindgen step. The published Python wheels carry neither; build from source to enable
them (see [Python Bindings](./python-running.md#distributed-backend)).
```

## Running

```rust,ignore
use prism_q::{distributed::DistributedContext, simulate};

let context = DistributedContext::world()?;
let result = simulate(&circuit).distributed(context).seed(42).run()?;
```

`DistributedContext::world()` and `attached_world()` need the `distributed-mpi` feature.
`world()` calls `MPI_Init` at `MPI_THREAD_FUNNELED` and captures the world communicator.
Dropping the context on the thread that created it runs `MPI_Finalize`. When another
component already owns MPI, such as an embedding interpreter with mpi4py,
`attached_world()` attaches without taking over the MPI lifetime. It returns `Ok(None)`
when MPI is not initialized, and an error when MPI runs below the thread level the
backend needs. `DistributedContext::serial()` gives a single rank with no MPI, which is
how the tests and a build with `distributed` alone run.

`simulate(&circuit).distributed(context)` is shorthand for
`.backend(BackendKind::StatevectorDistributed { context })`. Automatic dispatch never
selects this backend.

Launch the binary under the MPI launcher with a power-of-two rank count:

```bash
mpiexec -n 4 ./target/release/my_simulation
```

Every rank runs the same program and enters every collective inside the backend (SPMD),
so branching on `context.rank()` before a simulation call deadlocks the other ranks.

## Other transports

`DistributedContext::from_comm(Arc<dyn RankComm>)` wraps any transport. `RankComm`
(`prism_q::distributed::RankComm`) is the whole contract the backend uses: `rank`,
`size`, allgathers of `Complex64`, `f64` and `u64` blocks, a scalar `allreduce_sum_f64`,
a paired `sendrecv_c64` exchange, and `barrier`. Every rank must issue the same sequence
of calls, as under MPI. `SerialComm` is the single-rank implementation and `MpiComm` the
MPI one.

The thread loopback in `prism_q::distributed::loopback`, compiled under `bench-internal`,
runs ranks as threads in one process. `run_ranks(size, f)` calls `f` once per rank with a
context whose transport reaches the others and returns the results in rank order:

```rust,ignore
use prism_q::{distributed::loopback::run_ranks, simulate};

let probabilities = run_ranks(4, |context| {
    simulate(&circuit).distributed(context).seed(42).run().unwrap().probabilities
});
```

## Memory layout

Global index `rank * 2^(n-p) + local_index`. Qubit `q` below `n - p` is bit `q` of
the local index, and qubit `q` at or above `n - p` is bit `q - (n - p)` of the rank id.
Qubit 0 is the least significant bit, as everywhere in PRISM-Q, and `|0...0>` is index
0 on rank 0.

`PRISM_DIST_MIN_LOCAL_QUBITS` (default 10) sets the fewest local qubits a rank may hold;
a split that leaves fewer is rejected at `init`.

## Qubit relabeling

At more than one rank the backend maps circuit qubits to physical positions, so a SWAP
is a map update with no amplitude movement. Before a gate acts non-diagonally on a qubit
in a rank bit, the backend moves that qubit to a local position by exchanging the half
slice whose local bit differs from the rank bit, evicting the least recently used local
qubit. Later gates on it run without communication until it is evicted. Diagonal gates
and control bits are free on global qubits and never trigger a relabel.

Relabeling pays off when gate activity has locality: SWAP networks, repeated gates on
the same qubits, working sets that fit the local positions. A cyclic scan over more hot
qubits than local positions evicts on every layer. `PRISM_DIST_RELABEL=0` switches the
map off and falls back to a direct exchange per gate. `PRISM_DIST_EXCHANGE_CHUNK` tiles
each exchange into messages of that many amplitudes and bounds the transfer buffers. The
default, 2^29 amplitudes, sends a slice of up to 8 GiB as one message and keeps every
message within the `int` count MPI calls take.

## What runs distributed

- Gates, including fused and batched payloads, measurement, reset, and classical
  conditionals. Measurement probabilities gather one scalar per rank and sum in rank
  order, and every rank draws from the same seeded stream, so ranks agree on the outcome
  without exchanging it.
- Per-qubit probabilities, Pauli expectation values, and multi-shot terminal sampling
  answer from rank-local sums plus one reduction, at any register width, without
  gathering the dense state.
- Reduced density matrices, summed from rank-local partial traces. Kept qubits on rank
  bits move to local positions first, one half-slice exchange each, and stay there with
  relabeling on. With `PRISM_DIST_RELABEL=0` the layout is restored afterwards. A
  subsystem wider than the local slice reads the gathered state. Entanglement entropy
  needs a global SVD and declines.
- `probabilities()` and `export_statevector()` gather, so they carry the dense output
  cap; a register past the cap is rejected before the run.
- A caller-supplied `initial_state`: every rank receives the full vector and keeps its
  own slice.
- Noisy shots through `.noise(&model)`, for every channel the trajectory engine runs:
  Pauli and depolarizing, amplitude and phase damping, thermal relaxation, one- and
  two-qubit Kraus, leakage, and quasi-static drift. Ranks run the shots one at a time in
  shot order on one sharded state, re-initialized per shot, and seed each shot's noise
  stream from the run seed and shot index, so they draw the same branches without
  exchanging them. What each event reads from the state:

  | Event | Read |
  | --- | --- |
  | Pauli | nothing |
  | Damping or thermal | the qubit's excited population, one scalar gathered per rank |
  | One-qubit Kraus | the qubit's reduced density matrix, four scalars per rank, plus a half-slice exchange when the qubit sits on a rank bit |
  | Two-qubit Kraus | the pair's block, after relabeling the pair local |

  With relabeling off, that relabel is undone after the read, two more exchanges. With
  fewer than two local qubits the event gathers the whole state, under the export cap.
  Ranks sum the gathered parts in rank order, so each weighs a draw against the same
  value. The records match the host statevector's trajectory engine run serially at the
  same seed. For a Pauli-only model with terminal measurements the host's default route
  samples from shared error prefixes instead, so its records differ.
- Before the first noisy shot the ranks compare the circuit, noise model, shot count and
  seed, and whether each rank accepted the run under the limits it resolved. A run any
  rank rejects fails on every rank instead of leaving the others waiting at a
  collective.

Not supported: GPU execution, which is single device; shot parallelism and the shared
Pauli error prefixes the host trajectory engine uses; and selection by
`BackendKind::Auto`.

## Testing and measuring

The thread loopback transport covers the rank logic without an MPI runtime:

```bash
cargo nextest run --features "parallel distributed" -E 'test(distributed)'
```

`scripts/test-mpi.ps1` (Windows, MS-MPI) and `scripts/test-mpi.sh` (Linux, Open MPI
flags) build `tools/dist_mpi_check.rs` with the `distributed-mpi` feature and launch
it at 1, 2 and 4 ranks under three configurations (default, tiled exchange, relabeling
off). Rank 0 compares the gathered result against a one-process statevector and fails on
mismatch, and a three-rank launch checks the power-of-two rule. Each script's timed arm
prints per-rank timings for comparing exchange pipelines across hosts.

`cargo bench --bench bench_distributed --features "parallel distributed bench-internal"`
runs the loopback benchmarks. Their ranks share one memory system, so the rows price
packing, copying and per-gate dispatch rather than network latency (see
[Benchmark Methodology](./benchmarking.md)).

The exchange paths and relabeling algorithm are documented on the
[`distributed_statevector` module](https://docs.rs/prism-q/latest/prism_q/backend/distributed_statevector/index.html),
and per-backend support is in the [Capability and Support Matrix](./capabilities.md).
