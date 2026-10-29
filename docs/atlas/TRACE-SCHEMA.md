# Native trace schemas

## Gate amplitudes, version 1

The exporter is an opt-in example. It observes state through the existing public
backend API and adds no branches or allocations to simulation kernels.

```ts
type NativeFixture = {
  schemaVersion: 1;
  provenance: {
    sourceRevision: string;
    auditRevision: string;
    recordingRevision: string;
    crateVersion: string;
    backend: 'StatevectorBackend';
    execution: 'native';
    precision: 'complex128';
    seed: 42;
    features: string[];
    exporter: string;
    exporterSource: string;
    instruction: 'Gate::Fused';
    timing: null;
    rustc: string;
    os: string;
    cpu: string;
    architecture: string;
    availableParallelism: number;
    rustflags: string;
    profile: 'dev';
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
    gate: 'H' | 'X' | 'Y' | 'Z' | 'Rx' | 'Ry' | 'Rz';
    theta: number;
    before: number[];
    steps: number[][];
  }[];
  wideTraces: {
    id: string;
    n: 16;
    target: number;
    gate: 'H' | 'Z' | 'Rz';
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
is the complete state after k+1 repeated gate applications. Both rotation
conventions and matrices are those of `Gate::matrix_2x2`. Every trace contains
four steps. The input before normalization is
`alpha_j = (j+1) + i*((7*j mod 11)-5)`.

`wideTraces` apply one gate to the same input formula at 16 qubits, past the
threaded threshold, on targets 0, 2, 13, and 15. The full output is compared during
export with a scalar pair loop; `maxAbsoluteError` is that comparison and
`samples` holds ten amplitudes, including both members of the first target pair.

`theta` is in radians, fixed at pi/3 in these fixtures. It is retained even
for nonrotation gates to make experiment matching deterministic. `id` is an
opaque display key; replay uses the fields, not parsing the key. JSON stores
round-trippable f64 values from 17-digit scientific output. No measurement is
performed, so raw backend amplitude access has no deferred-normalization issue.

The generator records the actual checkout in `recordingRevision` and the existing
`sourceRevision` field. `auditRevision` identifies the separately checked source
audit. Dirty-tree state is disclosed rather than silently replaced with an older
revision. Verification checks the complete producer manifest against the sources
of the commit that last wrote the fixture, or the working tree while the fixture is
uncommitted, and reports later source changes without failing. The exporter source is included because it can be new relative to
the audited revision. Source fingerprints include uncommitted producer changes.

Replay requires no simulator. Select the matching n, target, gate, theta, initial
state, and depth, then read the corresponding input and output arrays. Compare
them to an independent model with a stated tolerance. Do not infer kernel
duration, register allocation, runtime dispatch, optimizer passes, measurement
probabilities, or actual CPU instruction selection from these snapshots.

This gate schema contains no optimizer events. Change `schemaVersion` when
existing field semantics change.

## Compiler snapshots, separate version 1

`fixtures/native-fusion.json` is a separate fixture family. Its typed consumer is
`src/models/compiler.ts`; the full recording contract is in
[Native compiler traces](research/fusion-traces.md). It contains:

- Environment, base revision, dirty-tree disclosure, embedded exporter source,
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
    source: 'src/circuit/fusion_atlas_tests.rs';
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
    family: 'legal-disjoint' | 'blocked-dependency' | 'diagonal-pair-runs';
    width: number;
    profile: 'Host' | 'Device';
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
state indexing still uses q0 as the least significant bit. A batch's entries
remain separate operators. Stage-local indices do not establish instruction
ancestry. Kernel counters exclude initialization and count route entries,
not bytes, cache misses, worker tasks, GPU launches, or elapsed time.
