# PRISM-Q Benchmarks

## Framework

Criterion.rs. Two benchmark binaries:

- **bench_driver**: Microbenchmarks for individual gate kernels, measurement, and end-to-end QASM.
- **circuits**: Macrobenchmarks for circuit family sweeps across qubit counts and depths.

### Run configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `PRISM_BENCH_PLOTS` | unset | Set to render the Criterion HTML report. Off by default: on the reference host a five-row group took 335s with plots and 47s without, so roughly 58s per row goes to rendering against 9s of measurement. Gating reads stdout and `target/criterion/**/estimates.json`, which the plots do not feed. |
| `PRISM_BENCH_SAMPLES` | 30 | Samples per row outside the `bench-fast` tier. Criterion divides `measurement_time` across the sample count rather than multiplying by it, so the count is free while one iteration still fits in `measurement_time / samples`. Past that it sets the cost outright: `density_matrix/unitary_layers/12` runs 3.87s per iteration, where 100 samples cost 39 minutes across the six passes of an adjacent A/B against 4 minutes at 10. Raise it for a row needing a tighter interval, lower it for triage; the CI gate runs the `bench-fast` tier for the same reason. Values below 10 are clamped. |

Sample count controls the precision of one run's mean. It does not remove
drift between runs: on the reference host, back-to-back runs of identical code
at 100 samples still moved -9.5% to +7.9%. Any comparison that has to hold to
the 5% gate goes through the adjacent-binary A/B below, which removes the build
and the source edit from between the two measurements and reports each row
against its own same-code control.

### Build cost

The relink, not the measurement, dominates the edit-to-number loop. Touching
`src/lib.rs` rebuilds the `circuits` target in about 202s because
`[profile.bench]` inherits `lto = "fat"` and `codegen-units = 1`. Filtering to a
few rows does not avoid it, so batch the rows you need into one run.

Feature sets do not thrash the build cache. Cargo fingerprints them separately,
so once `--features parallel` and `--features "parallel gpu distributed"` have
each been built, switching back to the other cost 10s against 190s for the
first cold build of a set. A separate `--target-dir` per feature set is not
worth the disk.

## Benchmark categories

### Microbenchmarks (bench_driver)

| Group | What it measures |
|-------|-----------------|
| `single_qubit_gates` | H, Rx, T gate kernels across qubit counts (4–20) |
| `two_qubit_gates` | CX, CZ, SWAP kernels across qubit counts (4–20) |
| `measurement` | Measure-after-superposition across qubit counts |
| `e2e_qasm` | Full parse + simulate from OpenQASM string |

### Macrobenchmarks (circuits)

| Group | What it measures |
|-------|-----------------|
| `qubit_sweep/random_d10` | Seeded random circuits, depth 10, 4–20 qubits |
| `qubit_sweep/qft_like` | QFT-structured circuits, 4–16 qubits |
| `qubit_sweep/hea_l5` | Hardware-efficient ansatz, 5 layers, 4–20 qubits |
| `qubit_sweep/clifford_d10` | Clifford-heavy circuits, depth 10, 4–20 qubits |
| `depth_sweep/12q_random` | 12-qubit random circuits, depth 5–100 |
| `entanglement_structure` | Sparse vs dense entanglement, 16 qubits |
| `stabilizer_rank/shots_terminal` | Clifford+T shot sampling with terminal measurements, including 1000q chi2 |
| `stabilizer_rank/shots_mid_circuit` | Clifford+T shot sampling with measurement, reset, and conditional gates |
| `tn/scalar_hea_l2` | Tensor-network scalar contraction, hardware-efficient ansatz, 2 layers, 20–50 qubits (`bench-internal`) |
| `tn/scalar_depth_20q` | Same contraction at 20 qubits, 4–7 layers, where intermediates grow large enough to reach the parallel contraction arms (`bench-internal`) |
| `tn/scalar_hea_l6` | Same contraction at 6 layers, 20–50 qubits, the only rows where qubit count drives how many contractions reach the faer arm (`bench-internal`) |
| `tn/scalar_hea_l7` | Same contraction at 7 layers, 30–50 qubits, where the greedy tree's peak intermediate is non-monotonic in width; the rows that measure contraction tree quality (`bench-internal`) |
| `tn/rdm_chain` | Single-qubit reduced density matrix on CZ chains at 40 and 60 qubits, query-shaped rows past the dense ceiling; depth 4 weighs per-contraction overhead, depth 8 the arithmetic |
| `tn/midmeasure_chain` | CZ chain with one measurement and one reset at half depth, 16 and 20 qubits; the row where measurement path cost lands mid-run |
| `tn/noisy_chain` | Depolarizing trajectories on the measured CZ chain at 16 qubits, 100 shots; moves with the measurement path as much as the noise machinery |
| `tn/sample_chain` | Terminal shot sampling on measured CZ chains at 16 and 40 qubits, 32 shots; 16 guards the dense arm, 40 prices the conditional sweep past the dense ceiling |

The `tn/scalar_*` groups are compiled out unless `bench-internal` is enabled, and
`bench_ab.sh` defaults to `--features parallel`, so a gating run over them needs
`--features "parallel bench-internal"` or it silently reports zero rows. Cargo
fingerprints feature sets separately, so the first such run pays a cold build in
both the working tree and the reference worktree.

### Shot and QEC benchmarks (bench_shots_perf)

| Group | What it measures |
|-------|-----------------|
| `qec_clifford_runner` | Packed native Clifford QEC execution |
| `qec_noisy_runner` | Packed native QEC execution with Pauli-noise rows |
| `qec_noisy_runner_split` | Parse, compile, sample, noise, detector, postselection, logical count, and total QEC timings |

### Decoder benchmarks (qec_decoder)

| Group | What it measures |
|-------|-----------------|
| `qec_decoder/rep_d3_r3`, `qec_decoder/rep_d5_r5` | Bulk union-find decode of sampled repetition-memory detector batches, 1k and 20k shots |
| `qec_decoder/surface_d3_r3`, `qec_decoder/surface_d5_r5` | Bulk union-find decode of sampled rotated-surface-memory detector batches, 1k and 20k shots |

### Distributed loopback benchmarks (bench_distributed)

Requires `--features "parallel distributed bench-internal"`. Rank pairs run on the
thread loopback transport, so exchanges move at memcpy speed: the rows resolve
packing, copying, and per-gate dispatch cost, not network latency.

| Group | What it measures |
|-------|-----------------|
| `distributed/steady_state_batched` | Fused QAOA behind one SWAP: batched-payload dispatch with a non-identity qubit map |
| `distributed/boundary_swap_direct` | Repeated boundary SWAPs with relabeling off: the half-slice pack path |
| `distributed/controlled_star_direct` | Controlled gates and a Multi2q star onto one global qubit with relabeling off |

## Circuit families

All seeded with `0xDEAD_BEEF` for reproducibility.

- **Random**: Mix of single-qubit gates + 50% CX on even-odd pairs per layer.
- **QFT-like**: Hadamard + controlled rotations (approximated with Rz + CX decomposition).
- **Hardware-efficient ansatz**: Ry/Rz layers + linear CX chain.
- **Clifford-heavy**: H, S, X, Y, Z + CX only.
- **Sparse entanglement**: H on all qubits + single CX(0, n-1) per layer.
- **Dense entanglement**: H on all qubits + linear CX chain per layer.
- **Sparse walk** (`circuits::sparse_walk_circuit`): H on the low k qubits,
  then layers of seeded diagonal phases and basis permutations over the whole
  register, holding the amplitude map at exactly 2^k entries. The
  `sparse/walk_k12` and `sparse/sampling_k12` rows fix k = 12; the
  `sparse/densify` rows sweep k at 20 qubits against a dense arm on the same
  workload, tracing the load-factor crossover. The entry count and the routing
  (the register must not split to the decomposed path) are pinned in
  `tests/bench_fixture_routing.rs`.

## Running benchmarks

```bash
# Full suite
cargo bench

# Specific benchmark
cargo bench --bench bench_driver -- "single_qubit_gates/h_gate/16"

# Quick smoke (fast settings)
cargo bench --bench bench_driver -- --warm-up-time 1 --measurement-time 3 --sample-size 10

# Save baseline
cargo bench -- --save-baseline my_baseline

# Compare against baseline
cargo bench -- --baseline my_baseline
```

## Gating comparisons: adjacent-binary A/B

**This is the required method for any before/after claim that has to hold to the
5% gate.** It supersedes save-a-baseline-then-compare-later for that purpose.

```bash
./scripts/bench_ab.sh --filter '^factored/noise_kraus/'
./scripts/bench_ab.sh -f '^density_matrix/' -r main -b circuits
./scripts/bench_ab.sh -f '^sparse/' --ref-dir /tmp/prism-q-ref   # cache the reference build
./scripts/bench_ab.sh --build-only /tmp/ref-circuits             # build a reference to reuse
./scripts/bench_ab.sh -f '^sparse/' --ref-exe /tmp/ref-circuits  # skip the reference build
```

`--build-only` and `--ref-exe` are a pair: the first produces a bench binary through the
same build path the A/B uses, the second consumes one and skips the reference build
entirely. Nothing checks that the supplied executable was built from `--ref`, so whatever
stores it owns that claim; the report records which of the two ways the reference arrived.

The script builds the bench binary from the working tree, builds the same target
from a reference git ref in a separate worktree, verifies the working tree did
not change between the two builds, then runs the two executables adjacent with no
build and no source edit in between. One discarded warmup pass per binary, then
four measured passes in the order ref, new, new, ref: both means are centred on
the same point in time so linear drift cancels, and each binary is measured twice
so every row reports a same-code control alongside its change.

The warmup passes are load bearing. Without them the first measured pass absorbs
the cold start after two builds, and whichever binary owns it looks slow: three
rows read -15% to -18% on identical code, all against the binary in pass 1.

```text
| Benchmark                      | Ref      | New      | Change | Control (ref) | Control (new) | Verdict |
| single_qubit_gates/h_gate/12   | 29.11 us | 27.64 us | -5.0%  | +11.5%        | -0.4%         | noise   |
```

That row is from a run where both binaries carried identical code. The -5.0%
"improvement" is drift, and the +11.5% control is what says so. **Do not report a
change without its control column.** A change no larger than the row's control
spread is noise, not a result, and rows whose control spread exceeds the threshold
are listed separately as unresolvable on this host.

### Why separate invocations do not work

Baseline and after runs minutes apart drift 8-20% on microsecond rows and up to
+98% on a byte-identical control group, because the rebuild and the editor's
re-index land between them. Every measured optimization in the archive was taken
with the adjacent-binary method instead: density matrix -29% to -72%, sparse -15%
to -66%, tensor network -15% to -22%.

The method removes the drift the rebuild causes. It does not make a busy host
quiet. `density_matrix/neutrality/22` has read a control spread of -15.9% to
+37.0% under this method with the editor consuming about half the CPU, so read the
control column on every run rather than assuming any row is stable. When the
controls are wide, the answer is "not measurable right now", not a number.

### What the reference worktree cannot resolve

Two claims have been filed under this heading, and only one of them survives
measurement.

The build directory does not, by itself, change code layout. The same commit built in
the project directory and in a worktree produces byte-identical `.text` (9323520 bytes)
and `.data`; the two images differ in 24 bytes out of 10797056, all metadata, being the
COFF `TimeDateStamp` and a 16-byte PDB GUID. On MSVC `debug = "line-tables-only"` writes
paths into the `.pdb` rather than the image, so there is no package path in the
executable for the directory to change. Measured on x86_64-pc-windows-msvc; the ELF case
is untested, and Linux keeps line tables in the binary, so the same is not assumed there.
Do not explain a moved row by naming the build directory without measuring it.

What does survive is the second claim, and it is the one worth acting on: a change that
adds or removes linked code can move rows that never execute it. The control columns
compare each binary against itself, so neither can see that. For a change to a hot loop's
arithmetic it does not matter. For a change that adds or removes linked code it can
invert the result.

Adding a faer matmul call to `tensornetwork.rs` measured -10.3% to -13.2% on the four
`tn/scalar_hea_l2` rows under this script. Rebuilt with both commits compiled from one
directory, the same rows read +11.3% to +13.2%: the change costs 13% there, and the
script had reported it as a 13% gain. A build with the crossover raised past every
reachable size, so the new code is linked but never called, carries the same +13%, which
is what identifies layout rather than execution as the cause.

So: when a change adds a dependency call, instantiates a large generic, or deletes a
sizeable function, take repeated runs before trusting this script. Marking the added
function `#[inline(never)]` or `#[cold]` does not recover the difference; both were tried
and both left it intact.

Rebuilding from one directory does not address this, now that the directory is known not
to matter for a fixed commit. To remove the term, put both implementations in the same
binary behind a switch read once at startup, and run that one binary twice:

```rust
static SCATTER: OnceLock<bool> = OnceLock::new();
if *SCATTER.get_or_init(|| std::env::var("PRISM_TRANSPOSE_SCATTER").is_ok()) { .. }
```

One binary means one layout, so the term cancels exactly rather than approximately, and
the branch is paid on both sides. It costs one build rather than two. Use it whenever the
change itself adds or removes linked code, which is the case this section's caveat exists
for.

The two binaries are never byte identical even from identical sources, but not for the
reason once given here: the difference is the link timestamp and the PDB GUID, 24 bytes
in total, not an embedded package path. The script decides "same code" from git rather
than from `cmp` because those bytes differ on every link.

Reference-build cost: the first `--ref-dir` build is a cold build of the crate in
a second package path (about 2m30s here); later runs against the same ref reuse
it. Dependencies are shared, since the worktree builds into the same target
directory.

`bench_ab.sh` needs only git, awk, and cargo. It deliberately avoids `jq` and
`bc`, neither of which is present on the reference host.

## Stored baselines

Point-in-time snapshots for tracking a number across weeks. Not a gating
mechanism: one job on one runner does not make two measurements comparable when
a build sits between them, which is why the CI gate runs the A/B above.

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
`target/criterion/` and writes to `bench_results/baselines/`. It requires `jq`
and `bc`, so it runs in CI but not on the Windows reference host.

## CI regression gate

Pull requests run a focused benchmark gate after lint and tests pass.
`scripts/bench_ci.sh` delegates to the adjacent-binary A/B above, with the PR
base commit as the reference, so the gate reports a same-code control column per
row and fires only when a row exceeds both the threshold and its own control
spread.

Until 2026-08, the two sides ran as separate `cargo bench` invocations with the
head build between them, on a shared runner. A row a hosted runner cannot resolve
now says so instead of reading as a result.

The subset runs at `CI_BENCH_FEATURES=parallel,bench-fast` and covers CPU-only
parameter points already present on the base branch. That tier pins samples to 10
and shortens the measurement windows, which is what keeps four passes affordable:
at 100 samples, a single 22q statevector row would cost half an hour, which is why
the set above tops out at 20. The filters stay narrow so noise from tiny parameter
sweeps does not dominate. A row missing from either side drops out of the comparison and the
row-count guard fails the run. This is a regression gate, not a replacement for
the local suite a performance change needs.

It measures only when a pull request changes a `.rs` file under `src/` or
`benches/`, or a manifest; a `skip-bench` label opts out. Either way it reports a
result, so it stays safe to promote to a required check. The reference worktree
sits beside the checkout rather than inside it: `/target/` in `.gitignore` is
anchored to the repository root, so a nested worktree's build output would read
as an untracked change and abort the A/B.

Representative CI workloads:

| Filter | Coverage |
|--------|----------|
| `statevector/scalability_d5/18` | Dense statevector scaling, above the post-phase rebatch floor |
| `statevector/qft_textbook/20` | Structured controlled phase and swap workload |
| `statevector/qpe_t_gate/16q` | Phase estimation with non-Clifford gates |
| `statevector/qaoa_l3/16` | QAOA workload with ZZ rotations and mixer layers |
| `stabilizer/scaling/500` | Clifford stabilizer backend path |
| `stabilizer/measurement/ghz_measure_all/500` | GHZ preparation plus terminal measurements |
| `compiled_sampler/noiseless/noiseless_500q_10k` | Compiled shot sampling path |
| `compiled_sampler/noisy/noisy_500q_10k` | Compiled Pauli-noise shot sampling path |

The sizes are the smallest that still exercise the pipeline that ships, not the smallest
available: `MIN_QUBITS_FOR_DIAG_BATCH = 16` and `MIN_QUBITS_FOR_POST_PHASE_BATCH = 18` in
`circuit/fusion.rs` are the floors, and `qft_textbook` has no 18 in its size list so it
stays at 20. Six passes over the eight rows take about 75 seconds. On a quiet host they
resolve the 5% gate: three same-code runs read worst control spreads of 6.0%, 7.1%, and
5.9%, with no row moving more than 3.6% against an identical binary.

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
commit alone, with no `restore-keys`, so a restore is either the binary for that exact
base or nothing; a miss falls back to building the reference in a worktree.

`PRISM_BENCH_REF=HEAD` on a clean tree builds both binaries from the same code,
so every row is a control row and the report is this host's noise floor.

## Regression detection

Default threshold: **5%** per benchmark (configurable via `REGRESSION_THRESHOLD`).

`bench_ab.sh` applies the threshold per row and only calls a row a regression when
it exceeds both the threshold and that row's own control spread. `bench_check.*`
reads Criterion JSON from `target/criterion/`, compares matching benchmark means,
and exits with code 1 when a benchmark exceeds the threshold and its confidence
interval clears the baseline's, in both the shell and PowerShell forms. The older
`bench_compare.*` wrappers still parse Criterion console output for quick
baseline checks.

### Rows this host cannot resolve

Some rows carry a same-code control spread above the 5% gate. Do not gate a change
on them; `bench_ab.sh` lists them under the table on every run.

| Row family | Same-code spread observed |
| --- | --- |
| `sparse/densify/map/{8,14}` | map/14 flips +9% to +16% between same-code process instances with tight in-run controls (bimodal, 5.4 to 6.3 ms); map/8 is a 145 us row with control excursions to 9%. Curve points, not gate rows; the other densify rows held 3.4% or better over three runs |
| `stabilizer/scaling/10`, `factored_stabilizer/scaling/10` | +9% to +13% (L1 resident) |
| `stabilizer_rank/shots_mid_circuit` | -17.8% to +88% |
| `gpu_stab_direct/clifford_d10/{2000,5000}` | +118% to +124% (bimodal clock state) |
| `qec_t_strategies` | about 50% at 10 samples |
| `factored/independent/*` | -45.5% to +49.7% (tens of microseconds) |
| `factored/noise_kraus/*` | 12% to 16%, dominated by per-shot backend construction |
| `noisy_sampling/*` | +9.8% to +48.3% |

Those last three were measured on a host at roughly 50% background load, so treat
them as an upper bound rather than an intrinsic property of the row.

Gate sparse work on `sparse/random_d10`, `sparse/walk_k12`, and
`compare/*/sparse/*`, and the stabilizer families on their 50q+ rows.
`sparse/random_d10` and the compare rows reproduced consistent signs across
independent pairs. The quarantined `sparse/low_entanglement` family this table
used to carry is removed rather than repaired: its register split, so the rows
ran the decomposed route at microsecond scale, and on the sparse backend the
fixture densifies to `2^n` entries, the opposite of the regime the name
promised. `sparse/walk_k12` replaces it.

## Reproducibility checklist

- [ ] Fixed RNG seed (`0xDEAD_BEEF` for circuit generation, `42` for simulation)
- [ ] Criterion settings: 5s warm-up, 5s measurement, 30 samples (`PRISM_BENCH_SAMPLES`)
- [ ] Gating comparisons run through `scripts/bench_ab.sh`, with the control column reported
- [ ] Document CPU model, OS, Rust version, RUSTFLAGS (`bench_ab.sh` records these in its table)
- [ ] Disable CPU frequency scaling if possible (`performance` governor on Linux)
- [ ] Close background applications
- [ ] Consider CPU pinning (`taskset` on Linux) for reduced variance
- [ ] One `cargo bench` process at a time. Rayon pools contend and swing the numbers

## Output

Criterion stores results in `target/criterion/`. Each benchmark gets:
- `estimates.json`: statistical estimates (mean, median, std dev)
- `benchmark.json`: configuration
- HTML reports in `target/criterion/report/`
