# PRISM-Q Benchmarks

## Framework

Criterion.rs. Most rows live in two binaries: `bench_driver` holds microbenchmarks for
gate kernels, measurement, and end-to-end QASM, and `circuits` sweeps circuit families
across qubit counts and depths.

Eight more targets cover narrower surfaces: `bench_shots_perf` (shot and count
sampling), `svd_bench` (the MPS decomposition kernels), `qec_t_strategies` and
`qec_decoder` (the QEC runner), the feature-gated `bench_gpu`, `bench_distributed` and
`bench_pulse`, and `iai`, which counts instructions under callgrind instead of timing.
List a Criterion target's rows with `cargo bench --bench <name> -- --list`.

### Run configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `PRISM_BENCH_PLOTS` | unset | Set to render the Criterion HTML report. Off by default: on the reference host a five-row group took 335s with plots and 47s without, about 58s of rendering per row against 9s of measurement. Gating reads stdout and `target/criterion/**/estimates.json`, which the plots do not feed. |
| `PRISM_BENCH_SAMPLES` | 30, or the group's own count | Samples per row outside the `bench-fast` tier, overriding every group's default (10 or 20 in `bench_shots_perf`, `qec_t_strategies` and the `scalability_d5` groups, 100 in `svd_bench`); values below 10 are clamped. Criterion divides `measurement_time` across the samples, so the count is free while one iteration fits in `measurement_time / samples`. Past that it sets the cost: `density_matrix/unitary_layers/12` runs 3.87s per iteration, so 100 samples cost 39 minutes across the six passes of an adjacent A/B against 4 minutes at 10. Raise it for a tighter interval and lower it for triage; the CI gate runs the `bench-fast` tier for the same reason. |

Sample count sets the precision of one run's mean. It does not remove drift between
runs: on the reference host, back-to-back runs of identical code at 100 samples still
moved -9.5% to +7.9%. A comparison held to the 5% gate goes through the adjacent-binary
A/B below, which keeps the build and the source edit out from between the two
measurements and reports each row against its own same-code control.

### Build cost

The relink dominates the edit-to-number loop. Touching `src/lib.rs` rebuilds the
`circuits` target in about 202s because `[profile.bench]` inherits `lto = "fat"` and
`codegen-units = 1`. Filtering to a few rows does not avoid it, so batch the rows into
one run.

Cargo fingerprints feature sets separately, so once `--features parallel` and
`--features "parallel gpu distributed"` have each been built, switching between them
costs 10s against 190s for a set's first cold build. A separate `--target-dir` per
feature set is not worth the disk.

## Benchmark categories

### Microbenchmarks (bench_driver)

| Group | What it measures |
|-------|-----------------|
| `single_qubit_gates` | H, Rx, T gate kernels across qubit counts (4 to 20) |
| `two_qubit_gates` | CX, CZ, SWAP kernels across qubit counts (4 to 20) |
| `two_qubit_gate_kernels` | The `two_qubit_gates` CX and CZ placements at 22 qubits, past that group's widest row |
| `measurement` | Probability read, 1q reduced density matrix, measure, and reset on a 20-qubit register, plus one end-to-end `simulate()` row |
| `e2e_qasm` | Full parse + simulate from OpenQASM string |
| `high_target_qubit` | H on qubit 13 at 16 qubits and on qubits 15 and 17 at 20, and Rz on qubits 17 and 19 at 20, where the diagonal pass leaves four tiles or one |
| `controlled_gates` | Controlled-H, Toffoli and CCCX kernels across qubit counts (4 to 20) |
| `diagonal_parametric_gates` | Rz, Ry and P on qubit 0 across qubit counts (4 to 20) |
| `cphase_kernel` | Controlled phase with the control below and above the target (4 to 20) |
| `new_gate_types` | SX and SXdg kernels across qubit counts (4 to 20) |
| `classical_only` | H on every qubit, a CX chain and one measurement through `simulate()` (4 to 20) |

### Macrobenchmarks (circuits)

| Group | What it measures |
|-------|-----------------|
| `statevector/random_d10` | Seeded random circuits, depth 10, at 4, 8, 12, 15, 16, 17, 20, 22 and 24 qubits, plus 28 and 30 up to `PRISM_BENCH_HIGH_QUBITS` |
| `statevector/qft_like` | QFT-structured circuits, 4 to 16 qubits |
| `statevector/hea_l5` | Hardware-efficient ansatz, 5 layers, 4 to 20 qubits |
| `statevector/clifford_d10` | Clifford-heavy circuits, depth 10, 4 to 20 qubits |
| `statevector/depth_sweep_12q` | 12-qubit random circuits, depth 5 to 100 |
| `statevector/entanglement_16q_d10` | Sparse vs dense entanglement, 16 qubits |
| `statevector/dense_unitary/k{3,4}/{20,24}` | Two layers of seeded Haar-random 3- and 4-qubit `Gate::Unitary` blocks tiling the register: the dense gather-scatter kernel |
| `product/expectation/{pauli,marginals}/{64,128,256,512}` | Product-state Pauli evaluation after 10 layers of seeded rotations: `Z` on every qubit and `XX` on every bond, and the per-qubit `Z` marginal route |
| `product/noisy_trajectory/{64,128,256,512}` | The same circuit measured, 100 shots of 1% depolarizing trajectories on the product backend |
| `noisy_sampling/trajectory_{amplitude,phase}_damping/hea_12q_g1e-3_1000` | 1000 shots of a measured three-layer hardware-efficient ansatz at 12 qubits, amplitude or phase damping at `gamma = 1e-3` on every gate target: one statevector trajectory per shot, with a probability read and a no-jump pass per event. `trajectory_pauli/hea_12q_p1e-3_1000` runs the same circuit under depolarizing noise, whose shots share one simulation per error pattern |
| `statevector/trotter_builder/heisenberg/{16,20}` | One second-order product-formula step of the `9n - 9` term Heisenberg chain from `Circuit::add_trotter`: `XX` and `YY` couplings as native `PauliRot`, `ZZ` as `Rzz`, fields as named rotations, under the default pipeline |
| `stabilizer_rank` | Clifford+T shot sampling: terminal rows (including 1000q chi2) and mid-circuit rows with measurement, reset, and conditional gates |
| `auto/clifford_t_price` | 1024 Clifford+T shots of one circuit on each engine that takes it, to price the register against the statevector: `register` (explicit `StabilizerRank`) on layered, scrambled and single-layer fixtures from 20 to 32 qubits with register widths 8 to 20, `statevector` on the same families at 20 to 26 qubits (28 under `PRISM_BENCH_HIGH_QUBITS`), and `auto` beside five of them. `distinct_17q_t16` (16 T gates on distinct qubits of 17, a 16-qubit register) runs on all three arms as the statevector side of the register's cost constant, and `onelayer_20q_t80` adds `factored` and `auto` arms. `rank_mid/mid_{26,32}q_t{6,8}` time the stabilizer-rank branches on a measure, guarded X, reset tail at 16 and 4 shots, and `statevector_mid/layers_{22,24,26}q_t8` the statevector walk on that tail at 64 shots. The counts behind the rows print from `clifford_t_price_census` under `bench-internal` |
| `tn/scalar_hea_l2` | Tensor-network scalar contraction, hardware-efficient ansatz, 2 layers, 20 to 50 qubits (`bench-internal`) |
| `tn/scalar_depth_20q` | Same contraction at 20 qubits, 4 to 7 layers, where intermediates grow large enough to reach the parallel contraction arms (`bench-internal`) |
| `tn/scalar_hea_l6` | Same contraction at 6 layers, 20 to 50 qubits, the only rows where qubit count drives how many contractions reach the faer arm (`bench-internal`) |
| `tn/scalar_hea_l7` | Same contraction at 7 layers, 30 to 50 qubits; the rows that measure contraction tree quality. The plain greedy tree peaks at 16.8M elements at 30 and 50 qubits, and the restarts and fused plan bring it to 1.05M, 1.05M and 2.1M (`bench-internal`) |
| `tn/rdm_chain` | Single-qubit reduced density matrix on CZ chains at 40 and 60 qubits, query-shaped rows past the dense ceiling; depth 4 weighs per-contraction overhead, depth 8 the arithmetic |
| `tn/midmeasure_chain` | CZ chain with one measurement and one reset at half depth, 16 and 20 qubits, where measurement path cost lands mid-run |
| `tn/noisy_chain` | Depolarizing trajectories on the measured CZ chain at 16 qubits, 100 shots; moves with the measurement path as much as the noise machinery |
| `tn/sample_chain` | Terminal shot sampling on measured CZ chains at 16 and 40 qubits, 32 shots; 16 guards the dense arm, 40 prices the conditional sweep past the dense ceiling |

The `tn/scalar_*` groups compile only under `bench-internal`, and `bench_ab.sh`
defaults to `--features parallel`, so a gating run over them needs
`--features "parallel bench-internal"` or it reports zero rows without complaint. The
first such run pays a cold build in both the working tree and the reference worktree.

### GPU benchmarks (bench_gpu)

Built with `--features "parallel gpu"` and skipped when no device answers. The groups
mirror the CPU sweeps above on the device path.

`gpu_stab/clifford_d10` and `gpu_stab/ghz_measure` run only when
`PRISM_STABILIZER_GPU_MIN_QUBITS` is at most the group's widest row (5000 and 2000
qubits, 1024 and 512 under `bench-fast`). The default crossover of 100,000 qubits keeps
stabilizer runs on the host, so a default run prints SKIP for both; set the variable to
`0` to time them.

These rows have no CPU twin:

| Group | What it measures |
|-------|-----------------|
| `gpu/pauli_expect` | Sixteen two-qubit Pauli strings evaluated on the device state after a depth-2 random circuit, 20, 22, and 24 qubits; the per-evaluation cost a shift-rule gradient multiplies by `2P` |
| `gpu/pauli_sum_grouped/{tfim,heisenberg}/{20,22,24}` | The weighted-observable route on the device state after a two-layer HEA, with the CPU `observable` Hamiltonians; every group is past the pair budget, so each evaluation is the moments pass per group |
| `gpu/export/{22,24}` | `export_statevector` on a prepared uniform superposition: one device-to-host copy of `2^n` amplitudes into the returned vector, deferred norm at one |
| `gpu/export/scaled/{22,24}` | The same readback after one measurement, so the in-place `pending_norm` pass runs on top of the copy |

### Shot and QEC benchmarks (bench_shots_perf)

| Group | What it measures |
|-------|-----------------|
| `run_counts_terminal/clifford_random_pairs_d20_100/{1000,2000}` | 100 shots of `sample_counts` on depth-20 random long-range Clifford pairs measured whole, where the forward compile over the full tableau is the cost |
| `clifford_register_counts` | `sample_counts(1000)` of a 40-qubit Clifford+T circuit whose T gates reach 20 qubits, on the Clifford register route at its 2^20 width cap |
| `qec_clifford_runner` | Packed native Clifford QEC execution |
| `qec_noisy_runner` | Packed native QEC execution with Pauli-noise rows |
| `qec_noisy_runner_split` | Parse, compile, sample, noise, detector, postselection, logical count, and total QEC timings (`bench-internal`) |
| `qec_feedforward_runner` | Packed native QEC execution of a repetition memory whose Pauli feed-forward corrections land as record flips |
| `noisy_dynamic_shots` | The circuit form of that memory, ancillas measured and reset each round and corrections in guarded bodies, sampled under depolarizing noise through the `Auto` noisy shot route |
| `qec_chunked_noise` | 100k-shot repetition memories on the record and leakage samplers at `chunk_size` 64 against the unchunked run, so a split noise unit's cost shows against one drawn whole |
| `qec_sample_one_shot` | One shot per call from compiled distance-5 and distance-25 memories at p = 1e-3, on the parity route and, with measurements kept, the record route, where walking every noise event dominates the call |
| `qec_postselection` | A 1M-shot repetition memory postselected on 20 records, unchunked and at 16384-shot chunks, where counting accepted shots and their logical errors is a large share of the run |
| `qec_parse` | Parsing distance-15 and distance-25 surface and color memory text written with `REPEAT` blocks |
| `qec_compile` | Compiling distance-5, 15 and 25 surface and color memories at p = 1e-3 to their sampler without kept measurements, where the detector parity walk dominates |
| `qec_first_shot` | Parsing a distance-5 or 25 surface memory at p = 1e-3 from text, compiling it, and drawing one shot, with measurements kept and dropped |

### Decoder benchmarks (qec_decoder)

| Group | What it measures |
|-------|-----------------|
| `qec_decoder/rep_d3_r3`, `qec_decoder/rep_d5_r5` | Bulk union-find decode of sampled repetition-memory detector batches, 1k and 20k shots |
| `qec_decoder/surface_d3_r3`, `qec_decoder/surface_d5_r5` | Bulk union-find decode of sampled rotated-surface-memory detector batches, 1k and 20k shots |
| `qec_decoder_compare/{uf,mwpm}/surface_d{5,9,13}_*` | Union-find against exact matching on the same 4096 sampled shots of rotated-surface memories, d rounds at distance d, p = 1e-3 and 5e-3 |
| `qec_decoder_compare/bposd/color_d3_r3_*` | BP+OSD on 4096 shots of a distance-3 color-code memory, whose model is a hypergraph, p = 1e-3 and 1e-2 |
| `qec_dem` | Detector error model derivation alone for repetition and rotated-surface memories, up to distance 5 with 25 rounds |

### Distributed loopback benchmarks (bench_distributed)

Requires `--features "parallel distributed bench-internal"`. Rank pairs run on the
thread loopback transport, so exchanges move at memcpy speed and the rows resolve
packing, copying, and per-gate dispatch cost rather than network latency. Outside
`bench-fast`, every group except `sample_indices` and `noisy_shots` runs at two register
widths. The larger one puts the per-rank shard above the Rayon threshold, so the combine
loops after an exchange fan out.

| Group | What it measures |
|-------|-----------------|
| `distributed/steady_state_batched` | Fused QAOA behind one SWAP at 16 and 20 qubits: batched-payload dispatch with a non-identity qubit map |
| `distributed/boundary_swap_direct` | Repeated boundary SWAPs at 18 and 20 qubits with relabeling off: the half-slice pack path |
| `distributed/global_1q_wall` | H and Rx repeated on the global qubit at 18 and 20 qubits with relabeling off: the full-slice exchange and combine path |
| `distributed/controlled_star_direct` | Controlled gates and a Multi2q star onto one global qubit at 18 and 20 qubits with relabeling off |
| `distributed/fused_2q_two_global` | Dense Fused2q repeated on both rank bits at 18 and 20 qubits across four ranks with relabeling off: the two step butterfly exchange for a two qubit gate over two global qubits |
| `distributed/sample_indices` | 4096 terminal shots of a 20 qubit uniform state at two and four ranks with relabeling off: the per-rank cumulative distribution, owner draws, and the index gather |
| `distributed/cyclic_wall/ranks{2,4}` | Three layers of a seeded Rx or Ry on every qubit and a nearest neighbour CX ring at 20 and 22 qubits across two and four ranks with relabeling on: the cyclic scan whose working set exceeds the local positions, so eviction choice sets the exchange count |
| `distributed/noisy_shots` | 16 shots of a two-layer cyclic wall with every qubit measured, at 16 qubits (12 under `bench-fast`) on two and four ranks, with depolarizing, amplitude damping, or a one-qubit Kraus channel after every gate: serial trajectories in lockstep. `noiseless` runs the same wall without a model, one fused run and terminal sampling, the floor the per-shot trajectories are measured against |

### Pulse compile benchmarks (bench_pulse)

Requires `--features "parallel unstable-pulse"`. Each row times one `pulse::compile`
call, which pays one Hermitian eigensolve and two dense products per segment, so cost
grows with segments times dimension cubed.

| Group | What it measures |
|-------|-----------------|
| `pulse_compile/cr_two_transmon_d3/300ns` | A 300 ns `GaussianSquare` cross-resonance drive on two coupled three-level transmons: dimension 9, 1,200 segments |
| `pulse_compile/x_transmon_d3/20ns` | A 20 ns DRAG pi pulse on one three-level transmon at 16 segments per ns: dimension 3, 320 segments, the control |

### Instruction counts (iai)

Eight single-thread rows under callgrind, one kernel each, for the deterministic CI
gate described under [Instruction-count gate](#instruction-count-gate). Linux only.

### Python binding timings

`bindings/python/benches/energy_sweep.py` times the `prepared/hea_l2_energy` sweep
through the installed wheel at 10, 12 and 14 qubits, with the Hamiltonian passed as a
term list and as a `PauliObservable`, per point and through
`PreparedCircuit.expectation_many`. It is a plain `perf_counter` script, neither a
Criterion target nor a gate: read its rows beside `prepared/hea_l2_energy/prepared` for
the cost the binding adds.

## Circuit families

All seeded with `0xDEAD_BEEF`.

- **Random**: Mix of single-qubit gates + 50% CX on even-odd pairs per layer.
- **QFT-like**: Hadamard + controlled phase (`Gate::cphase`) on every later qubit.
- **Hardware-efficient ansatz**: Ry/Rz layers + linear CX chain.
- **Clifford-heavy**: H, S, X, Y, Z + CX only.
- **Sparse entanglement**: H on all qubits + CX on every fourth bond of the chain per
  layer, the first bond advancing one per layer.
- **Dense entanglement**: H on all qubits + linear CX chain per layer.
- **Sparse walk** (`circuits::sparse_walk_circuit`): H on the low k qubits,
  then layers of seeded diagonal phases and basis permutations over the whole
  register, holding the amplitude map at exactly 2^k entries. The
  `sparse/walk_k12`, `sparse/sampling_k12` and `sparse/noisy_walk_k12` rows fix
  k = 12, the last under trajectory noise that can only shrink the map. The
  `sparse/densify` rows sweep k at 20 qubits against a dense arm on the same
  workload, tracing the load-factor crossover. The entry count and the routing
  (the register must not split to the decomposed path) are pinned in
  `tests/integration/bench_fixture_routing.rs`.

## Running benchmarks

```bash
# Full suite
cargo bench

# Specific benchmark
cargo bench --bench bench_driver -- "single_qubit_gates/h_gate/16"

# Quick smoke (fast settings)
cargo bench --bench bench_driver -- --warm-up-time 1 --measurement-time 3 --sample-size 10

# Every row once in Criterion's test mode, no timing (the CI smoke job)
bash scripts/bench_smoke.sh

# Save baseline
cargo bench -- --save-baseline my_baseline

# Compare against baseline
cargo bench -- --baseline my_baseline
```

## Gating comparisons: adjacent-binary A/B

Any before/after claim held to the 5% gate goes through this method instead of saving a
baseline and comparing later.

```bash
./scripts/bench_ab.sh --filter '^factored/noise_kraus/'
./scripts/bench_ab.sh -f '^density_matrix/' -r main -b circuits
./scripts/bench_ab.sh -f '^sparse/' --ref-dir /tmp/prism-q-ref   # cache the reference build
./scripts/bench_ab.sh --build-only /tmp/ref-circuits             # build a reference to reuse
./scripts/bench_ab.sh -f '^sparse/' --ref-exe /tmp/ref-circuits  # skip the reference build
./scripts/bench_ab.sh -f '^x/affected/' -c '^x/control/'         # controls at 10 samples
./scripts/bench_ab.sh -f '^statevector/' --light                 # triage tier, about a fifth of the time
./scripts/bench_ab.sh -f '^statevector/' --light --escalate      # then the full tier on rows that moved
./scripts/bench_ab.sh -f '^sparse/' -c '^mps/random_d10/16$' --interleave  # interleaved rounds, below
```

Without `--ref` the reference is the merge-base of `HEAD` and `origin/main` (local `main`
when there is no remote), and the report's `Reference` line names the resolved commit.
A clean tree sitting at that commit is an error, since both sides would be the same code;
an uncommitted change against it is a normal run.

The script builds the working tree and a reference git ref (in a separate worktree),
checks the working tree did not change between the builds, then runs the two
executables back to back. One discarded warmup pass per binary at 10 samples precedes
four measured passes in the order ref, new, new, ref. Both means centre on the same
point in time, so linear drift cancels, and each binary runs twice, so every row reports
a same-code control beside its change.

Without the warmups, the first measured pass absorbs the cold start after two builds:
three rows read -15% to -18% on identical code, all against the binary in pass 1.

```text
| Benchmark                      | Ref      | New      | Change | Control (ref) | Control (new) | Verdict |
| single_qubit_gates/h_gate/12   | 29.11 us | 27.64 us | -5.0%  | +11.5%        | -0.4%         | noise   |
```

Both binaries in that row carried identical code: the -5.0% "improvement" is drift, and
the +11.5% control says so. Do not report a change without its control column. A change
no larger than the row's control spread is noise, and rows whose control spread exceeds
the threshold are listed separately as unresolvable on this host.

`--build-only` produces a bench binary through the A/B's own build path and `--ref-exe`
consumes one, skipping the reference build. `--new-exe` does the same for the working
tree, and `--build-dir` points `--build-only` at another checkout. Nothing checks that a
supplied executable was built from what it claims, so whatever stores it owns that
claim; the report records how each binary arrived.

`--interleave` keeps the builds and hands the smoke test and the measurement to the
interleaved runner below in its two-binary mode: short invocations in A B B A rounds,
each read at its fastest sample, with disturbed rounds dropped and an early stop. The
four passes need an idle host; on a desktop that never drops under 12% load they either
wait for hours or read RERUN with same-code controls spread 60% to 120%. The mode needs
`--control`, forwards the filters, `--light`, `--threshold`, `--min-rows`, the host
guards and `--out`, and keeps the exit codes. `--escalate` is refused, and the options
that shape passes (`--control-samples`, the slow-row and warmup counts,
`--max-row-seconds`, `--no-confirm`) do not apply. `MEASURE_METHOD` near the top of the
script sets which method a run without `--interleave` or `--passes` takes.

### Host and verdict guards

The script names each of these failure patterns instead of reporting a number.

**A busy host.** Before the first measured pass the script reads host CPU load for two
seconds and stops with exit 2, listing the busiest processes, when other work holds more
than `--max-host-load` (default 25% of all CPUs; one busy thread on an 8-thread host is
12.5%, and a desktop with chat and editor apps open idles at 12% to 19%, so 10% never
cleared there). `--wait-idle SECONDS` waits for the load to fall instead. `--settle SECONDS`
asks for that many seconds of consecutive quiet readings first, for a job that builds
just before it measures: a build leaves the host disturbed after its processes exit. The
reading before each measured pass goes into the report, and a FAIL with a busy pass
behind it reads RERUN. Background load once produced a 15% to 82% same-code control
spread.

**A lane that moved.** When one binary's two passes disagree in the same direction
across most rows, the shift lands in every row's change column. The script takes the
signed median of each binary's control column, and past half the threshold the verdict
is RERUN (exit 3). It follows cold builds: one run read its working-tree binary 5% to
25% faster in its second pass than its first on 14 of 20 rows, with the reference lane
flat.

**A copy that landed slow.** A copied binary can run a row about 18% slow for a whole
run and fast in the next, invisible to controls that compare a binary with itself. On
the full tier a FAIL re-runs the regressed rows once from fresh copies of both binaries,
and stays a FAIL only if the confirmation's own verdict is FAIL; a PASS, a RERUN, or a
confirmation that stops without a verdict leaves the run at RERUN. `--no-confirm` turns
this off.

**A row that cannot run.** Before measuring, each binary runs every row of the filter
and the control filter once in Criterion's test mode (`--test`). A binary that panics or
exits non-zero stops the run with exit 4, naming the lane (`ref` or `new`), the row, and
the tail of its output, instead of surfacing mid-pass after the warmups have been paid
for.

**A row on one side only.** A row the filter matched in some passes but not others, as
a benchmark added or renamed on one side is, has no change to report. The report lists
it as NO DATA under the table, naming the passes it is missing from, and the verdict
comes from the remaining rows. `--min-rows` still turns a filter that lost its rows into
a NO DATA verdict.

**The same code twice.** With an explicit `--ref HEAD` on a clean tree, both binaries
carry the same code. The report says so in its settings table, and a run that would
PASS reads `CONTROL (same code)` instead, still exit 0, since it measured the host's
noise floor rather than a change. The default reference refuses that case instead.

`--escalate` joins the two tiers: after a light run, every row that moved past half the
threshold is measured again on the full tier against the same two binaries, and that run
decides the verdict.

Exit status: 0 PASS or CONTROL (same code), 1 FAIL, 2 host too busy to start or another
measurement holding the marker below, 3 RERUN, 4 a binary failed the smoke test, 5 NO
DATA or any other error (a bad option, a failed build, a bench binary that exited
mid-pass). Exit 1 always means FAIL.

### One measurement at a time

While a measurement runs, `bench_results/MEASURING` exists in the main checkout,
whichever worktree started it. A session that sees the file must not run `cargo`, a test
binary, or anything else that loads the host until it is gone: ten queued runs in one
week came back RERUN because a build in another session overlapped a measured pass. The
file names its holder:

```text
pid: 1234
by: bench_queue.sh
task: PrismBenchQueue
out: /c/work/prism-q/bench_results/queue-2026-10-17_101010
started: 2026-10-17 10:10:10
```

`bench_queue.sh` writes it once every build has finished and removes it at `ALLDONE`;
`bench_ab.sh` and `bench_interleave.sh` run alone write it before the host load check
and remove it at the end. All three remove it on any exit, including a `kill -INT` or
`kill -TERM` of the script's pid, which also ends the bench binary or queue step running
at the time. All three refuse to start while a live process holds it (`LOCKED` in the
queue's status file, exit 2 from the other two), and the two A/B scripts wait up to
`--wait-idle` for it to clear before their first build, so they do not compile under
someone else's measurement. The marker is created exclusively, so two runs that start
together cannot both take it. The next run replaces a stale marker, one whose pid is
gone, and logs that. `--force-lock` builds and measures regardless. A queue step
inherits `PRISM_BENCH_LOCK_HELD=1` and writes no marker of its own, and
`PRISM_BENCH_LOCK` names another path for the file.

### Keeping an A/B affordable

Each row runs six passes, and a pass is mostly Criterion's fixed windows whatever the
row's iteration time. The full tier uses 1s of warm-up and 3s of measurement: a
microsecond row collects thousands of iterations per sample in that window, and a row
whose one iteration outlasts it is priced by its sample count. The two discarded warmup
passes run at 10 samples, since warming the binary and pricing the rows do not depend on
the count. Rows the first warmup projects past `--slow-row-seconds` (default 30) per
pass at the full count run at `--slow-samples` (default 15); they keep their verdict and
set the noise floor, and the report names them. Before these three changes a 25-row MPS
sweep with three rows over a second each cost 43 minutes per run: about 20 in the
windows and 19 in the slow rows.

Four levers, in the order worth reaching for them.

**Run the light tier first.** `--light` runs three measured passes (ref, new, ref),
windows of 0.5s warm-up and 1.5s measurement, and 10 samples. A row costs about a third
of the full tier, so a 24-row sweep needs about four minutes of measurement. The report
names the tier in its header and verdict line, and the new binary's control column
reads `n/a` because it runs once. The tier says which rows moved and roughly how far; a
gate claim still needs the full tier on those rows. Run it on the exact rows the full
tier will run, and if it shows a cost, hold the full tier until the change settles.
Groups that pin their own `measurement_time` (the shots groups pin 3s) keep it, so their
rows shrink less.

**Pick controls by cost.** A control has to read flat, not precisely.
`density_matrix/rzz_layers_fused/10` runs 86ms where `/12` runs 1.97s, and answers the
same question about lane drift.

**Measure controls at a lower sample count.** `--control` takes a second regex whose
rows run at `--control-samples` (default 10, Criterion's floor). The two regexes must be
disjoint; a row matched by both is an error. Reduced rows report as `control` or
`control moved`, never as faster or slower, and stay out of the noise floor and the
unresolvable list: ten samples resolve whether a control is flat, not a percentage. A
control that moves past the threshold still fails the run, since collateral damage
counts whichever row it lands on.

**Lower the sample count on slow rows.** The full tier does this itself past
`--slow-row-seconds`; `--slow-samples` sets the count and `--slow-row-seconds 0` turns
it off. `PRISM_BENCH_SAMPLES=10` lowers every row instead, saving nothing on a
millisecond row and about two thirds on a multi-second one (see the table above). It
stacks with `--light`.

`--max-row-seconds` (default 240, 0 disables) aborts the run when Criterion projects one
row past that many seconds in one pass. The projection prints before collection, so the
check costs one iteration: the corpus row that provoked the option projected 4200s per
pass, seven hours across the six, and was caught after one 140s iteration. The first
warmup pass also prints the projected total, so a long A/B can be abandoned early.

Reuse one `--ref-dir` per repository. A fresh directory is a cold build of the crate and
its dependencies; the same directory against a different ref rebuilds incrementally.

### Why separate invocations do not work

Baseline and after runs minutes apart drift 8-20% on microsecond rows and up to +98% on
a byte-identical control group, because the rebuild and the editor's re-index land
between them. Every measured optimization in the archive used the adjacent-binary method
instead: density matrix -29% to -72%, sparse -15% to -66%, tensor network -15% to -22%.

The method removes the rebuild's drift but cannot quiet a busy host:
`density_matrix/neutrality/22` has read a control spread of -15.9% to +37.0% with the
editor taking about half the CPU. Read the control column on every run. When it is wide,
the answer is "not measurable right now".

### What the reference worktree cannot resolve

The build directory does not, by itself, change code layout. The same commit built in
the project directory and in a worktree produces byte-identical `.text` (9323520 bytes)
and `.data`; the images differ in 24 of 10797056 bytes, the COFF `TimeDateStamp` and a
16-byte PDB GUID. On MSVC `debug = "line-tables-only"` writes paths into the `.pdb`, so
the executable holds no package path for the directory to change. This was measured on
x86_64-pc-windows-msvc only; Linux keeps line tables in the binary, so the ELF case is
not assumed. Do not blame a moved row on the build directory without measuring it. Those
24 bytes differ on every link, which is why the script decides "same code" from git
rather than `cmp`.

Linked code is the real hazard: a change that adds or removes it can move rows that
never execute it, and controls that compare each binary with itself cannot see that. A
change to a hot loop's arithmetic is unaffected; a change to linked code can invert its
result. Adding a faer matmul call to `tensornetwork.rs` measured -10.3% to -13.2% on the
four `tn/scalar_hea_l2` rows under this script. Rebuilt with both commits compiled from
one directory, the same rows read +11.3% to +13.2%, so the change cost 13% and the
script had reported a 13% gain. A build with the crossover raised past every reachable
size, linking the new code but never calling it, carries the same +13%: layout, not
execution.

When a change adds a dependency call, instantiates a large generic, or deletes a
sizeable function, take repeated runs before trusting this script. `#[inline(never)]`
and `#[cold]` on the added function both left the difference intact, and rebuilding
from one directory does not help. To remove the term, put both implementations in one
binary behind a switch read once at startup and measure it with the interleaved runner
below. One binary has one layout, so the term cancels exactly, the branch is paid on
both sides, and it costs one build instead of two.

The first `--ref-dir` build is a cold build of the crate in a second package path (about
2m30s here); later runs against the same ref reuse it. Dependencies are shared, since
the worktree builds into the same target directory. `bench_ab.sh` needs only git, awk,
and cargo, and avoids `jq` and `bc`, which the reference host lacks.

## Unattended queues

`scripts/bench_queue.sh` runs a list of A/Bs and other measurements one after another,
building every binary before measuring anything:

```bash
cat > queue.sh <<'EOF'
ab  fstab  ../pq-dispatch -r 315b43e -f '^auto/crossover/auto/fstab_' --light --escalate
ab  spd    ../pq-spd      -r dea250c -f '^auto/crossover/(auto|statevector)/spd_wide_'
run cold   ../pq-spd      -- ./target/release/examples/cold_marginals 16 20 0.2 auto
EOF
./scripts/bench_queue.sh --dry-run queue.sh     # the plan, and which builds are cached
./scripts/bench_queue.sh --schedule queue.sh    # detached; progress in <out>/status.txt
```

An `ab` step without `-r` compares with the same merge-base default as `bench_ab.sh`,
resolved in its checkout when the queue is planned, and a clean checkout at that commit
stops the plan. `--interleave` among a step's options measures its two cached binaries
through the interleaved runner; retries and `NAME-tryN/` work the same.

Binaries are cached under `bench_results/bin`, keyed on commit, features, bench target,
and toolchain, so steps that share a reference build it once and a later queue reuses it.
A checkout with uncommitted changes or untracked files adds the content hash `bench_ab.sh`
checks its own builds against. The key is read again just before each build and once
more after it, and a checkout that changed during its build caches nothing (a
`BUILD-FAILED` line in the status file). `--build-only` fills the cache and stops. Each
A/B waits up to an hour for an idle host.

`--schedule` matters on Windows, where a child of a terminal session dies with that
session's job object even under `nohup`, so the queue runs from a scheduled task, started
at once; its only trigger is 00:00 on the day it was created, already past, so it never
fires again. The task opens a console window; closing it ends the queue. A second queue scheduled while
one runs needs its own `--task-name`. Its build phase waits for a live `MEASURING`
marker to clear, but the first queue writes that marker only after its own builds, so
start the second once the first reaches its measure phase.

Every step logs in full to its own file. The status file carries one START and END line
per step, the verdicts, and `ALLDONE`. A `kill -INT` or `kill -TERM` of the queue's pid
ends the running step, removes the `MEASURING` marker, and writes `STOPPED` instead.

A step that exits RERUN runs again against the same cached binaries, up to `--retries`
times (default 2), since a quieter host can answer it without a rebuild. Earlier
attempts keep their reports in `NAME-tryN/`, the status file gets a `RETRY` line for
each, and the summary line counts them. FAIL, host-busy, and smoke-test exits are not
retried; `bench_ab.sh` already confirms a FAIL from fresh copies.

## Switchable changes: interleaved same-binary A/B

A change that can be turned off at one coarse point (a fusion pass, a routing decision,
a kernel choice) can be gated from a single binary holding both implementations.
`scripts/bench_interleave.sh` builds the working tree once and runs both variants from
that executable: one fat-LTO build instead of two, and no copied binary for a slow lane
to land in.

Wire the switch at the branch point, behind `bench-internal`, and remove it once the
numbers are in. Variant B carries the new code and A the old, because the report reads B
slower than A as a regression:

```rust
#[cfg(feature = "bench-internal")]
if crate::bench_variant_b() {
    return new_path(..);
}
```

`bench_variant_b()` reads `PRISM_BENCH_VARIANT` once per process, so branch on it where
a circuit or a pass is chosen, never inside a gate kernel's loop. Bench code can call
`prism_q::bench_variant_b()` too.

Several changes can share one build. Give each its own letter through
`bench_variant('c')`, `bench_variant('d')`, and so on, and pass `--variants b,c,d`.
Rounds then run `a b c d d c b a` when odd and `b c d a a d c b` when even, which keeps
every variant at the same mean position, so linear drift cancels in each comparison with
`a`. The report holds one table and verdict per variant, and the run's verdict is the
worst of them. Each extra variant adds two invocations to every round, so keep a run to
about three.

```bash
./scripts/bench_interleave.sh -f '^statevector/random_d10/(12|16|20)$' -c '^mps/random_d10/16$'
./scripts/bench_interleave.sh -f '^sparse/' --light                     # triage
./scripts/bench_interleave.sh --build-only /tmp/circuits-switch         # then --exe
```

Each round runs four short Criterion invocations, one process each, in the order
A B B A (B A A B in even rounds), and takes `(B1 + B2) / (A1 + A2) - 1` per row from
each invocation's fastest sample per iteration. The fastest sample is least touched by
other work: on the reference host its round-to-round spread was a third to half that of
the mean. A round whose two invocations of one variant disagree by more than the
threshold on the median row is dropped as disturbed and replaced, up to half as many
rounds again. That test never compares A with B, so it cannot bias the change.

A row reports the median over the undisturbed rounds, 16 at most, with a 95% interval
from the order statistics. From the sixth undisturbed round on, verdicts are computed
after every round, and the run stops once each filter interval lies wholly on one side
of the threshold, each control median sits inside half the threshold, and no control
interval reaches past it. An effect ten times the gate settles in six rounds;
`--no-early-stop` runs the full count.

Verdicts and exit codes follow `bench_ab.sh`. FAIL means a row's interval lies wholly
above the threshold. RERUN means it straddles the threshold, a `--control` row reads off
zero, or fewer than six rounds were undisturbed. A control whose interval reaches past
the threshold reads wide, and forces a RERUN only when some filter row's |change| is
under three times the control's half-width: a 45% effect with a one-point interval
stands on its own.

Two same-code runs of six `circuits` rows took 17 minutes each and held every interval
within -3.6% to +2.9%, the widest on the 180 us row. A 94 us busy wait injected into
`statevector/random_d10/16` (7.9% of its 1.19 ms) read +9.4%, interval +5.7% to +11.0%,
and failed the gate, while the five rows it did not reach and a control row read flat.
The same filter through `bench_ab.sh` costs two builds of about 7 minutes each before
its passes.

Use `bench_ab.sh` for anything a switch cannot express: a change spread across many
sites, a data layout change, a dependency or toolchain bump, or a comparison against
another commit. A queue step `ix NAME DIR [options]` runs the interleaved runner from
`bench_queue.sh`, building its one binary in the build phase.

`--base-exe PATH` runs the same rounds on two executables: variant a runs `--base-exe`,
variant b runs `--exe`, and every invocation sees `PRISM_BENCH_VARIANT=a` and otherwise
the same environment. The smoke test covers every row on both. Two binaries keep two code
layouts, which one binary cancels and this mode cannot, so it requires `--control`: a
binary sitting in a slow layout reads as a control off zero, and the run reads RERUN.
`--variants` must stay `b`. `bench_ab.sh --interleave` and an `ab` queue step given
`--interleave` measure this way.

```bash
./scripts/bench_interleave.sh --base-exe /tmp/circuits-main --exe /tmp/circuits-change \
    -f '^statevector/random_d10/(16|20)$' -c '^mps/random_d10/16$'
```

## Batches of branches: one combined A/B

`scripts/bench_batch.sh` merges several branches onto one base in a throwaway checkout
and A/Bs its tip against the base on the union of their rows, through the queue, so one
build and one measurement stand in for one A/B per branch. Each line of a batch file
names a branch, a bench target, the rows that cover it, and optional control rows:

```text
perf/fused2q-pairs  circuits          ^statevector/(qft_textbook|qv)/20$
perf/mps-center     circuits          ^mps/random_d10/16$  ^product/scaling_d10/64$
perf/qec-walk       bench_shots_perf  ^qec_detector_sampling/surf_p001_drop/5$
```

```bash
./scripts/bench_batch.sh batch.txt -- --light --escalate          # merge, print the plan
./scripts/bench_batch.sh --schedule batch.txt                      # merge and run detached
./scripts/bench_batch.sh --only 1,2 --schedule batch.txt           # bisect a FAIL by halves
```

`--only` counts entries, the lines that name a branch, so comments and blank lines do not
shift it; error messages name the file line. Windows line endings and a last line
without a newline are read as written.

A PASS clears every branch in the batch. A FAIL names rows, not branches, so run each
half with `--only` to find the branch, about log2(N) more A/Bs. A speedup in one branch
can hide a regression in another on a row both reach. Batch branches whose changes
touch separate code, and give a branch its own A/B when its deterministic counts moved
on a row another branch also changes. A claimed speedup can come from a batch only when
the claimed rows run code no other branch in it touches.

## Knob sweeps on other hardware

The cache-resident tile sizes derive from the cache topology the OS reports (see the
performance guide), and every number behind that rule was measured on one i7-6700K.
`scripts/tile_sweep.sh` checks the rule on a host with a different geometry. It builds
the bench target once, runs the same executable at each value of an environment knob
forward and then reversed so drift cancels, and reports each row's minimum over runs
against the first value.

```bash
./scripts/tile_sweep.sh -f '^statevector/(qv|qft_textbook)/2[02]$'           # PRISM_TILE_KB over 256 128 512 1024 2048
./scripts/tile_sweep.sh -f '^statevector/qv/24$' --values '256 512 1024' --runs 4
./scripts/tile_sweep.sh -f '^statevector/qv/20$' --var PRISM_MULTI_2Q_TILE_BITS --values '13 14 15 16'
```

The `Tile sweep` workflow runs it on the hosted Linux, macOS and Windows runners from
the Actions tab. Those are triage numbers: a shared runner resolves a 2x cliff, not a 5%
move. A default changes only after a full-tier `bench_ab.sh` run on real hardware
confirms the chosen value.

## Stored baselines

Point-in-time snapshots for tracking a number across weeks. They are not a gate: one job
on one runner does not make two measurements comparable when a build sits between them,
which is why the CI gate runs the A/B above.

```bash
# Save (Unix)
./scripts/bench_baseline.sh

# Save (Windows)
.\scripts\bench_baseline.ps1

# Compare (Unix)
./scripts/bench_compare.sh

# Compare (Windows)
.\scripts\bench_compare.ps1

# Custom threshold
REGRESSION_THRESHOLD=10 ./scripts/bench_compare.sh
```

`scripts/bench_check.sh` (`save`, `compare`, `table`, `list`) reads
`target/criterion/` and writes to `bench_results/baselines/`. It requires `jq` and
`bc`, so it does not run on the Windows reference host. It is a local workflow; CI gates
on `bench_ab.sh` alone.

## CI regression gate

Pull requests run a focused benchmark gate after lint and tests pass.
`scripts/bench_ci.sh` delegates to the adjacent-binary A/B with the PR base commit as
the reference, so each row carries a same-code control and fires only when it exceeds
both the threshold and its own control spread, in two runs. Until 2026-08 the two sides
ran as separate `cargo bench` invocations on a shared runner with the head build between
them; a row a hosted runner cannot resolve now says so instead of reading as a result.

The subset runs at `CI_BENCH_FEATURES=parallel,bench-fast` and covers CPU-only
parameter points already present on the base branch. That tier pins samples to 10 and
shortens the windows to keep four passes affordable: at 100 samples a single 22q
statevector row would cost half an hour, so the set tops out at 20. Narrow filters keep
noise from tiny parameter sweeps from dominating. A row missing from either side drops
out of the comparison and the row-count guard fails the run. The gate does not replace
the local suite a performance change needs.

It measures only when a pull request changes a `.rs` file under `src/` or `benches/`,
a manifest, `rust-toolchain.toml`, `.cargo/`, or `build.rs`; a `skip-bench` label opts
out. Either way it reports a result, so it is
safe as a required check. The reference worktree sits beside the checkout because
`/target/` in `.gitignore` is anchored to the repository root: a nested worktree's build
output would read as an untracked change and abort the A/B.

CI workloads, one or two rows per simulation path (`scripts/bench_ci.sh` lists every row
and the reason for it):

| Path | Rows |
|------|------|
| Dense statevector | `statevector/{scalability_d5/18, qft_textbook/20, qpe_t_gate/16q, qaoa_l3/16, diag_mixed_l6/20}` |
| Stabilizer | `stabilizer/scaling/500`, `stabilizer/measurement/wall_measure_all/1000`, `factored_stabilizer/local_blocks/10x5/50` |
| Compiled sampler | `compiled_sampler/noiseless/noiseless_500q_10k`, `compiled_sampler/noisy/noisy_500q_10k` |
| MPS and sparse | `mps/random_d10/16`, `mps/sampling/32`, `sparse/random_d10/20`, `sparse/sampling/32` |
| Density matrix | `density_matrix/rzz_layers_fused/10`, `density_matrix/noisy_channels/kraus_amp_damp/10` |
| Factored and tensor network | `factored/partial_independence/factored/20`, `tn/linear_chain/12`, `tn/midmeasure_chain/20` |
| Product state | `product/scaling_d10/64` |
| Clifford+T | `clifford_t/spd/10q_12t`, `spp/spp_10k/20q_8t`, `stabilizer_rank/approx_1024/10q_20t` |
| Analytic terminals | `gradient/qaoa_l1/adjoint/18`, `expectation/pauli_sum/18`, `auto/expectation/brickwork_d4/20` |
| Noise | `noisy_sampling/trajectory_pauli/non_clifford_12q_512`, `noisy_sampling/compiled_pauli/clifford_100q_10k`, `pauli_path/noisy_expectation/path/50` |
| Circuit level | `dynamic/guarded_region/20`, `vqe/bind/16` |

The sizes are the smallest that still exercise the pipeline that ships:
`MIN_QUBITS_FOR_DIAG_BATCH = 14` and `MIN_QUBITS_FOR_POST_PHASE_BATCH = 18` in
`circuit/fusion.rs` are the floors, and `qft_textbook` has no 18 in its size list so it
stays at 20. The job builds head just before measuring, so `bench_ci.sh` passes
`--settle 60` and the first pass waits for a minute of unbroken quiet.

The stabilizer row measures a CNOT wall because a GHZ chain collapses once and reads the
rest of the register off deterministically, pricing almost none of the measurement work.
At 500 qubits the GHZ row also sat in the microsecond band where that fixture is a known
cache cliff: one pull request saw it read +5.7% and fail, then -1.4% and pass on a rerun
of the same commit, with a tight same-code control both times, because the reference
lane moved from 113 to 165 microseconds on a byte-identical cached executable.

A reported regression has to repeat before it blocks: the set runs a second time and
only a row that regressed in both counts. Each run passes `--no-confirm`, so this
whole-set rerun is the only confirmation. Four same-code runs on the reference host
returned three passes and one failure, on `statevector/scalability_d5/18` at +5.3%
against a 4.2% control spread; their worst same-code control spreads were 20.6%, 8.5%,
21.8% and 18.3%. An earlier three-run measurement on the same host read 6.0%, 7.1% and
5.9%, so the tier's noise floor is not a fixed property of the machine. Confirmation
costs nothing on a green run and one extra set on a red one.

Local reproduction, against `main` as the reference:

```bash
PRISM_BENCH_REF=main ./scripts/bench_ci.sh

# Cache the reference build between runs:
PRISM_BENCH_REF=main PRISM_BENCH_REF_DIR=/tmp/prism-q-ref ./scripts/bench_ci.sh

# Or skip the reference build outright, given a binary already built from that ref:
PRISM_BENCH_REF=main PRISM_BENCH_REF_EXE=/tmp/ref-circuits ./scripts/bench_ci.sh
```

In CI the third form is the normal path: every push to `main` builds the bench binary and
caches it under that commit, and the gate restores it for the PR base. The key is the
commit alone, with no `restore-keys`, so a restore yields the binary for that exact base
or nothing. A miss falls back to building the reference in a worktree.

`PRISM_BENCH_REF=HEAD` on a clean tree builds both binaries from the same code, so every
row is a control row, the report is this host's noise floor, and a clean run reads
`CONTROL (same code)` rather than PASS.

## Instruction-count gate

The `iai` target (`benches/iai.rs`) counts instructions under callgrind: eight rows, one
kernel each, all on a single Rayon thread with the tile budget pinned to 256 KB. A count
depends on the binary and its input, not the host's load, so the `instruction-counts`
CI job holds the 5% threshold on a shared runner where the wall-clock gate cannot.
Counts are not bit-exact, since hash seeds, heap addresses, and path lengths vary per
process: two CI builds of the same source differed by at most 0.13% on
`fuse_circuit_qaoa_l3` and under 0.03% on the other rows.

| Row | Kernel |
|-----|--------|
| `statevector_random_d10/q12` | Fused 1q and 2q statevector kernels below the parallel floor |
| `statevector_qft_textbook/q14` | QFT block and controlled-phase path |
| `statevector_qv/q12` | SU(4) layers through 2q fusion |
| `stabilizer_clifford_shots/q32_1k` | Tableau replay over 1000 measured shots |
| `compiled_sampler/q100_1k` | Compiled frame sampler at 100 qubits |
| `density_matrix_random_d10/q8` | Density-matrix superoperator kernels |
| `mps_random_d10/q12` | MPS contraction and truncation at bond 64 |
| `fuse_circuit_qaoa_l3/qaoa_q16` | The fusion pipeline alone, no simulation |

The gate catches a change in the work a row does: a kernel that executes more per
amplitude, a fusion pass that emits a longer circuit, code-size growth that costs a hot
loop its inlined shape. It cannot see memory bandwidth or cache misses (the cache
simulation is off), SIMD throughput (a four-wide FMA and a scalar multiply are one
instruction each), or threading. A change that trades instructions for bandwidth or
moves work between threads needs the wall-clock A/B, as does anything above the
parallel floor, which these rows stay below on purpose.

Pull requests build the base commit in a worktree with its own target directory (in a
shared one cargo reads the base binary as fresh for the head), record it with
`--save-baseline`, then count the head against it and fail when any row's `Ir` grows
past `REGRESSION_THRESHOLD`. The table lands in the job summary and the callgrind
profiles upload as the `instruction-counts` artifact. A push to `main` records counts
without comparing, and so does a pull request whose base lacks the target or pins
another `iai-callgrind` version, since the one installed runner must match the crate.
The job skips pull requests that change no Rust source or manifest.

Locally, on Linux (valgrind does not run on Windows or macOS):

```bash
sudo apt-get install valgrind
cargo install --locked iai-callgrind-runner --version 0.16.1   # the Cargo.toml pin

# Record main, then count a branch against it
git checkout main && cargo bench --bench iai --features parallel -- --save-baseline=main
git checkout my-branch && cargo bench --bench iai --features parallel -- --baseline=main --callgrind-limits='ir=5%'
```

Profiles land under `target/iai/`; `callgrind_annotate` on a `callgrind.out` file there
names the functions behind a changed count.

## Regression detection

Default threshold: 5% per benchmark (configurable via `REGRESSION_THRESHOLD`).

`bench_ab.sh` calls a row a regression only when it exceeds both the threshold and the
row's own control spread. `bench_check.*` (shell and PowerShell) compares matching
Criterion means and exits 1 when a benchmark exceeds the threshold and its confidence
interval clears the baseline's. The older `bench_compare.*` wrappers still parse
Criterion console output for quick baseline checks.

### Rows this host cannot resolve

These rows carry a same-code control spread above the 5% gate. Do not gate on them;
`bench_ab.sh` lists them under the table on every run.

| Row family | Same-code spread observed |
| --- | --- |
| `sparse/densify/map/{8,14}` | map/14 flips +9% to +16% between same-code process instances with tight in-run controls (bimodal, 5.4 to 6.3 ms); map/8, a 145 us row, has control excursions to 9%. Curve points, not gate rows; the other densify rows held 3.4% or better over three runs |
| `stabilizer/scaling/10`, `factored_stabilizer/scaling/10` | +9% to +13% (L1 resident) |
| `stabilizer_rank/shots_mid_circuit` | -17.8% to +88% |
| `gpu_stab_direct/clifford_d10/{2000,5000}` | +118% to +124% (bimodal clock state) |
| `qec_t_strategies` | about 50% at 10 samples |
| `factored/independent/*` | -45.5% to +49.7% (tens of microseconds) |
| `factored/noise_kraus/*` | 12% to 16%, dominated by per-shot backend construction |
| `noisy_sampling/*` | +9.8% to +48.3% |

The last three were measured on a host at roughly 50% background load, so treat them as
an upper bound.

Gate sparse work on `sparse/random_d10`, `sparse/walk_k12`, and `compare/*/sparse/*`,
and the stabilizer families on their 50q+ rows. `sparse/random_d10` and the compare rows
reproduced consistent signs across independent pairs. `sparse/walk_k12` replaces the
`sparse/low_entanglement` family this table used to quarantine: its register split, so
the rows ran the decomposed route at microsecond scale, and on the sparse backend the
fixture densified to `2^n` entries, the opposite of what the name promised.

## Reproducibility checklist

- [ ] Fixed RNG seed (`0xDEAD_BEEF` for circuit generation, `42` for simulation)
- [ ] Criterion windows match the tier: the `bench_ab.sh` full tier runs 1s warm-up, 3s
  measurement, and 30 samples (`PRISM_BENCH_SAMPLES`); `--light` runs 0.5s, 1.5s, and
  10 samples; `bench-fast` pins 0.2s, 1s, and 10 samples; a bare `cargo bench` takes
  Criterion's 3s and 5s defaults at 30 samples. Groups that pin their own settings
  keep them
- [ ] Gating comparisons run through `scripts/bench_ab.sh`, with the control column reported
- [ ] Document CPU model, OS, Rust version, and RUSTFLAGS. `bench_ab.sh` records the first
  three and the bench profile in its table; note RUSTFLAGS by hand when set
- [ ] Disable CPU frequency scaling if possible (`performance` governor on Linux)
- [ ] Close background applications
- [ ] Consider CPU pinning (`taskset` on Linux) for reduced variance
- [ ] One `cargo bench` process at a time. Rayon pools contend and swing the numbers

## Output

Criterion stores results in `target/criterion/`. Each benchmark gets:
- `estimates.json`: statistical estimates (mean, median, std dev)
- `benchmark.json`: configuration
- HTML reports in `target/criterion/report/`
