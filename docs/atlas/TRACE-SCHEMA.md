# Native trace schemas

## Gate amplitudes, version 1

The exporter is an example binary. It reads state through the public backend API
and adds nothing to the simulation kernels.

```ts
type NativeFixture = {
  schemaVersion: 1;
  provenance: {
    sourceRevision: string;
    auditRevision: string;
    recordingRevision: string;
    crateVersion: string;
    backend: "StatevectorBackend";
    execution: "native";
    precision: "complex128";
    seed: 42;
    features: string[];
    exporter: string;
    exporterSource: string;
    instruction: "Gate::Fused";
    timing: null;
    rustc: string;
    os: string;
    cpu: string;
    architecture: string;
    availableParallelism: number;
    rustflags: string;
    profile: "dev";
    dirtyTree: boolean;
    sources: { path: string; sha256: string }[];
    encodedRustflags: string | null;
    environment: Record<string, string | null>;
    auditComparison: string;
    sourceHashNormalization: string;
  };
  traces: {
    id: string;
    n: number;
    target: number;
    gate: "H" | "X" | "Y" | "Z" | "Rx" | "Ry" | "Rz";
    theta: number;
    before: number[];
    steps: number[][];
  }[];
  wideTraces: {
    id: string;
    n: 16;
    target: number;
    gate: "H" | "Z" | "Rz";
    theta: number;
    normSquared: number;
    maxAbsoluteError: number;
    samples: { index: number; re: number; im: number }[];
  }[];
};
```

Arrays pack `[re(alpha0), im(alpha0), re(alpha1), im(alpha1), ...]` in
ascending basis-index order, q0 the least significant bit. Each array has
`2 * 2^n` finite numbers. `before` is the normalized initial state; `steps[k]`
is the full state after k+1 applications of the gate. Rotation conventions and
matrices are those of `Gate::matrix_2x2`. Every trace has four steps. Before
normalization the input is
`alpha_j = (j+1) + i*((7*j mod 11)-5)`.

`wideTraces` apply one gate to the same input at 16 qubits, above the threading
threshold, on targets 0, 2, 13 and 15. The exporter compares the full output with a
scalar pair loop and stores the result as `maxAbsoluteError`. `samples` holds ten
amplitudes, including both members of the first target pair.

`theta` is in radians and fixed at pi/3. It is stored for gates without an angle
too, so every case matches on the same fields. `id` is a display key; replay reads
the fields and never parses it. Values are written with 17 significant digits, so
they round-trip as f64. Nothing is measured, so the raw backend amplitudes carry no
deferred normalization.

The generator writes the checkout commit to both `recordingRevision` and
`sourceRevision`. `auditRevision` names the source audit, which is checked
separately. `dirtyTree` records whether producer files had uncommitted changes.
Verification checks the producer manifest against the sources of the commit that
last wrote the fixture, or the working tree while the fixture is uncommitted, and
lists later source changes without failing. The exporter source is embedded because
it postdates the audited revision. Source fingerprints include uncommitted producer
changes.

Replay needs no simulator. Find the trace with the matching n, target, gate,
theta, initial state and depth, read its input and output arrays, and compare them
with an independent model at a stated tolerance. The snapshots say nothing about
kernel duration, register allocation, dispatch, optimizer passes, measurement
probabilities or the instructions the CPU executed.

The gate schema has no optimizer events. Bump `schemaVersion` when the meaning of
an existing field changes.

## Compiler snapshots, version 1

`fixtures/native-fusion.json` is versioned separately. `src/models/compiler.ts`
reads it, and [native compiler traces](research/fusion-traces.md) gives the
recording contract. It contains:

- Environment, base revision, dirty-tree flag, embedded exporter source,
  producer hashes, thread count, cache capacity, and planner floor.
- Circuit family, width, profile, initial-state formula, and final-public-IR check.
- Ordered stages with a source function, active or skipped status, width floor,
  complete instruction stream, native kernel counters, and static sweep metrics.
- Per-instruction raw targets, complete sorted support, exact enum variant, and
  native matrix entries with their original target order.
- Per-stage norm, full-state maximum error against unfused execution, and six
  amplitude samples. The full states are compared during export, not stored.

```ts
type FusionFixture = {
  schemaVersion: 1;
  generator: {
    // Shared recording fields as in the gate schema, plus:
    source: "src/circuit/fusion_atlas_tests.rs";
    threads: number;
    lastLevelCacheBytes: number | null;
    stagePlannerFloor: number; // smallest width the cache stage planner takes
    initialState: string;
    counterSemantics: string;
    provenanceUnavailable: string;
    command: string;
    reorder2qEnabled: boolean; // false when PRISM_NO_REORDER was set
    tileBudgetOverrideKb: string | null; // PRISM_TILE_KB
  };
  records: {
    id: string; // `${family}-${width}-${profile}`
    family: "legal-disjoint" | "blocked-dependency" | "diagonal-pair-runs";
    width: number;
    profile: "Host" | "Device";
    finalMatchesPublicPipeline: true; // asserted against fuse_circuit_for_profile
    stages: {
      id: number;
      name: string;
      source: { path: string; symbol: string };
      enabled: boolean;
      floor: number | null;
      instructions: {
        index: number;
        gate: string;
        variant: string;
        targets: number[];
        support: number[];
        kernelSweeps: number;
        payload: { targets: number[]; matrix: [number, number][][] }[];
      }[];
      metrics: {
        instructionCount: number;
        measuredKernelSweeps: number;
        modeledFullStatePasses: number;
        tileApplications: number;
      };
      state: {
        normSquared: number;
        maxAbsoluteError: number; // against the unfused circuit
        samples: { index: number; re: number; im: number }[];
      };
    }[];
  }[];
};
```

Matrix entries are nested row-major arrays of `[real, imaginary]` pairs.
For two targets, `targets[0]` is the more significant local matrix bit; global
state indexing still uses q0 as the least significant bit. A batch's entries stay
separate operators. Stage-local indices do not track which instruction a fused
operator came from. Kernel counters skip initialization and count route entries,
not bytes, cache misses, worker tasks, GPU launches or elapsed time.
