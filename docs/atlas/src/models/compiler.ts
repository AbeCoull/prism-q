import { add, mul, type Complex } from "./quantum";

export type ComplexMatrix = readonly (readonly Complex[])[];
export type FusionPayload = { targets: number[]; matrix: ComplexMatrix };
export type FusionInstruction = {
  index: number;
  gate: string;
  targets: number[];
  support: number[];
  payload: FusionPayload[];
  kernelSweeps: number;
};
export type FusionStage = {
  id: number;
  name: string;
  source: { path: string; symbol: string };
  enabled: boolean;
  floor: number | null;
  instructions: FusionInstruction[];
  metrics: {
    instructionCount: number;
    measuredKernelSweeps: number;
    modeledFullStatePasses: number;
    tileApplications: number;
  };
  state: {
    normSquared: number;
    maxAbsoluteError: number;
    samples: { index: number; re: number; im: number }[];
  };
};
export type FusionRecord = {
  id: string;
  family: string;
  width: number;
  profile: "Host" | "Device";
  finalMatchesPublicPipeline: boolean;
  stages: FusionStage[];
};
export type FusionFixture = {
  schemaVersion: number;
  generator: {
    source: string;
    sourceRevision: string;
    seed: number;
    threads: number;
    execution: string;
    initialState: string;
    counterSemantics: string;
    provenanceUnavailable: string;
    stagePlannerFloor: number;
    reorder2qEnabled: boolean;
  };
  records: FusionRecord[];
};

const passSources: Record<string, readonly [string, string]> = {
  original: ["src/circuit/fusion_atlas_tests.rs", "family"],
  regions: ["src/circuit/fusion.rs", "fuse_region_bodies"],
  cancel: ["src/circuit/fusion.rs", "cancel_self_inverse_pairs"],
  "recognize-rzz": ["src/circuit/fusion_rzz.rs", "fuse_rzz"],
  "batch-rzz": ["src/circuit/fusion_rzz.rs", "fuse_batch_rzz"],
  "fuse-1q": ["src/circuit/fusion.rs", "fuse_single_qubit_gates"],
  "reorder-1q": ["src/circuit/fusion.rs", "reorder_1q_gates"],
  "cancel-after-reorder": [
    "src/circuit/fusion.rs",
    "cancel_self_inverse_pairs",
  ],
  "refuse-1q": ["src/circuit/fusion.rs", "fuse_single_qubit_gates"],
  "absorb-2q": ["src/circuit/fusion.rs", "fuse_2q_gates"],
  "absorb-2q-same-pair-only": ["src/circuit/fusion.rs", "fuse_2q_gates_where"],
  "same-pair-2q": ["src/circuit/fusion.rs", "fuse_same_pair_2q_blocks"],
  "stage-planner": ["src/circuit/fusion_stage.rs", "plan_stages"],
  "multi-1q": ["src/circuit/fusion.rs", "fuse_multi_1q_gates"],
  "reorder-2q-tiles": ["src/circuit/fusion.rs", "reorder_fused2q_into_tiles"],
  "multi-2q": ["src/circuit/fusion.rs", "fuse_multi_2q_gates"],
  "pauli-rotation-batches": ["src/circuit/fusion.rs", "fuse_pauli_rot_batches"],
  "controlled-phases": [
    "src/circuit/fusion_phase.rs",
    "fuse_controlled_phases",
  ],
  "diagonal-batch": ["src/circuit/fusion.rs", "fuse_diagonal_batch"],
  "rebatch-1q": ["src/circuit/fusion_phase.rs", "batch_post_phase_1q"],
};

function requireFact(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Invalid compiler fixture: ${message}`);
}

function integer(value: unknown, minimum = 0): value is number {
  return Number.isSafeInteger(value) && (value as number) >= minimum;
}

function qubits(value: unknown, width: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.every((q) => integer(q) && q < width) &&
    new Set(value).size === value.length
  );
}

function checkMatrix(
  value: unknown,
  dimension?: number,
): asserts value is ComplexMatrix {
  requireFact(Array.isArray(value) && value.length > 0, "matrix is empty");
  const size = dimension ?? value.length;
  requireFact(value.length === size, "matrix dimension differs from targets");
  for (const row of value) {
    requireFact(
      Array.isArray(row) && row.length === size,
      "matrix is not square",
    );
    for (const z of row)
      requireFact(
        Array.isArray(z) && z.length === 2 && z.every(Number.isFinite),
        "matrix coefficient is not a finite complex number",
      );
  }
}

export function validateFusionFixture(input: unknown): FusionFixture {
  requireFact(input !== null && typeof input === "object", "missing object");
  const fixture = input as FusionFixture;
  requireFact(fixture.schemaVersion === 1, "unsupported schema version");
  const generator = fixture.generator;
  requireFact(
    generator && /^[a-f0-9]{40}$/.test(generator.sourceRevision),
    "source revision",
  );
  requireFact(generator.source === passSources.original[0], "generator source");
  requireFact(
    generator.seed === 42 && integer(generator.threads, 1),
    "seed or thread count",
  );
  requireFact(integer(generator.stagePlannerFloor, 16), "cache planner floor");
  requireFact(
    typeof generator.reorder2qEnabled === "boolean",
    "reordering configuration",
  );
  requireFact(
    typeof generator.provenanceUnavailable === "string" &&
      generator.provenanceUnavailable.length > 0,
    "unavailable origins must be disclosed",
  );
  requireFact(
    Array.isArray(fixture.records) && fixture.records.length > 0,
    "missing records",
  );
  const ids = new Set<string>();
  for (const record of fixture.records) {
    requireFact(
      typeof record.id === "string" && !ids.has(record.id),
      "duplicate record ID",
    );
    ids.add(record.id);
    requireFact(integer(record.width, 1) && record.width <= 30, "record width");
    requireFact(
      ["legal-disjoint", "blocked-dependency", "diagonal-pair-runs"].includes(
        record.family,
      ),
      "unknown circuit family",
    );
    requireFact(
      record.id === `${record.family}-${record.width}-${record.profile}`,
      "record identity",
    );
    requireFact(
      record.profile === "Host" || record.profile === "Device",
      "profile",
    );
    requireFact(
      record.finalMatchesPublicPipeline === true,
      "final pipeline mismatch",
    );
    requireFact(
      Array.isArray(record.stages) && record.stages.length > 0,
      "missing stages",
    );
    requireFact(
      record.stages[0].name === "original",
      "first stage is not original",
    );
    const names = new Set<string>();
    const expected: [string, number | null][] = [
      ["original", null],
      ["regions", null],
      ["cancel", null],
      ["recognize-rzz", null],
      ...(record.profile === "Host"
        ? [["batch-rzz", 14] as [string, number]]
        : []),
      ["fuse-1q", 10],
      ["reorder-1q", 10],
      ["cancel-after-reorder", 10],
      ["refuse-1q", 10],
      [
        record.profile === "Host" ? "absorb-2q" : "absorb-2q-same-pair-only",
        12,
      ],
      ["same-pair-2q", 12],
    ];
    if (record.profile === "Host") {
      if (record.width >= generator.stagePlannerFloor)
        expected.push(["stage-planner", null]);
      else
        expected.push(
          ["multi-1q", 14],
          ["reorder-2q-tiles", 12],
          ["multi-2q", 12],
          ["pauli-rotation-batches", 16],
          ["controlled-phases", 14],
          ["diagonal-batch", 14],
          ["rebatch-1q", 14],
        );
    }
    requireFact(
      record.stages.length === expected.length,
      "incomplete compiler pipeline",
    );
    for (const [stageIndex, stage] of record.stages.entries()) {
      const [expectedName, floor] = expected[stageIndex];
      requireFact(
        stage.name === expectedName && stage.floor === floor,
        "pass order or floor differs from audited pipeline",
      );
      const enabled =
        (floor === null || record.width >= floor) &&
        (stage.name !== "reorder-2q-tiles" || generator.reorder2qEnabled);
      requireFact(
        stage.enabled === enabled,
        "pass enabled state differs from recorded configuration",
      );
      requireFact(
        stage.id === stageIndex && !names.has(stage.name),
        "inconsistent stage ID",
      );
      names.add(stage.name);
      const source = passSources[stage.name];
      requireFact(
        source &&
          stage.source?.path === source[0] &&
          stage.source?.symbol === source[1],
        `inconsistent source for ${stage.name}`,
      );
      requireFact(typeof stage.enabled === "boolean", "pass enable flag");
      requireFact(
        stage.floor === null || integer(stage.floor, 1),
        "pass floor",
      );
      requireFact(
        !stage.enabled || stage.floor === null || record.width >= stage.floor,
        "pass below floor",
      );
      requireFact(Array.isArray(stage.instructions), "missing instructions");
      for (const [i, instruction] of stage.instructions.entries()) {
        requireFact(
          instruction.index === i && typeof instruction.gate === "string",
          "instruction index",
        );
        requireFact(
          integer(instruction.kernelSweeps),
          "missing instruction kernel counter",
        );
        requireFact(
          qubits(instruction.targets, record.width),
          "instruction targets",
        );
        requireFact(
          qubits(instruction.support, record.width),
          "instruction support",
        );
        requireFact(
          Array.isArray(instruction.payload) && instruction.payload.length > 0,
          "missing payload",
        );
        for (const key of [
          "sourceIds",
          "sourceInstructionIds",
          "sourceIndices",
          "origins",
        ])
          requireFact(
            !(key in instruction),
            "snapshot indices cannot claim source origins",
          );
        const support = new Set(instruction.targets);
        for (const payload of instruction.payload) {
          requireFact(
            qubits(payload.targets, record.width) && payload.targets.length > 0,
            "payload targets",
          );
          checkMatrix(payload.matrix, 2 ** payload.targets.length);
          payload.targets.forEach((q) => support.add(q));
        }
        requireFact(
          JSON.stringify(instruction.support) ===
            JSON.stringify([...support].sort((a, b) => a - b)),
          "payload support differs from declared support",
        );
      }
      const counts = stage.metrics;
      requireFact(
        counts && Object.values(counts).every((v) => integer(v)),
        "count is not an integer",
      );
      requireFact(
        [
          counts.instructionCount,
          counts.measuredKernelSweeps,
          counts.modeledFullStatePasses,
          counts.tileApplications,
        ].every((v) => integer(v)),
        "missing count",
      );
      requireFact(
        counts.instructionCount === stage.instructions.length,
        "logical instruction count",
      );
      requireFact(
        counts.measuredKernelSweeps === counts.modeledFullStatePasses,
        "kernel counter differs from model",
      );
      requireFact(
        counts.measuredKernelSweeps ===
          stage.instructions.reduce((sum, inst) => sum + inst.kernelSweeps, 0),
        "instruction counters differ from stage counter",
      );
      requireFact(
        integer(counts.tileApplications),
        "missing tile application count",
      );
      const state = stage.state;
      requireFact(
        state &&
          Number.isFinite(state.normSquared) &&
          Math.abs(state.normSquared - 1) < 1e-12,
        "state normalization",
      );
      requireFact(
        Number.isFinite(state.maxAbsoluteError) &&
          state.maxAbsoluteError >= 0 &&
          state.maxAbsoluteError < 1e-12,
        "native equivalence tolerance",
      );
      requireFact(
        Array.isArray(state.samples) && state.samples.length > 0,
        "state samples",
      );
      for (const sample of state.samples)
        requireFact(
          integer(sample.index) &&
            sample.index < 2 ** record.width &&
            Number.isFinite(sample.re) &&
            Number.isFinite(sample.im),
          "invalid amplitude sample",
        );
      if (!stage.enabled && stageIndex > 0)
        requireFact(
          JSON.stringify(stage.instructions) ===
            JSON.stringify(record.stages[stageIndex - 1].instructions),
          "skipped pass changed its stream",
        );
    }
    requireFact(
      !(names.has("stage-planner") && names.has("multi-1q")),
      "mutually exclusive Host branches",
    );
    requireFact(
      record.profile === "Host" || !names.has("diagonal-batch"),
      "Host batching in Device profile",
    );
  }
  return fixture;
}

export type DependencyInstruction = {
  index: number;
  support: readonly number[];
  kind?:
    | "gate"
    | "barrier"
    | "measure"
    | "reset"
    | "conditional"
    | "region"
    | "save";
};

export function circuitDag(
  instructions: readonly DependencyInstruction[],
  numQubits: number,
) {
  const last = new Map<number, number>();
  const layers = new Map<number, number>();
  const nodes: { index: number; layer: number; support: number[] }[] = [];
  const edges: { from: number; to: number; qubits: number[] }[] = [];
  for (const instruction of instructions) {
    if (layers.has(instruction.index))
      throw new Error("Duplicate dependency node");
    const support =
      instruction.kind && instruction.kind !== "gate"
        ? Array.from({ length: numQubits }, (_, q) => q)
        : [...instruction.support];
    if (!qubits(support, numQubits))
      throw new Error("Invalid dependency support");
    const predecessors = new Map<number, number[]>();
    for (const q of support) {
      const previous = last.get(q);
      if (previous !== undefined)
        predecessors.set(previous, [...(predecessors.get(previous) ?? []), q]);
    }
    const layer =
      Math.max(-1, ...[...predecessors.keys()].map((i) => layers.get(i)!)) + 1;
    layers.set(instruction.index, layer);
    nodes.push({ index: instruction.index, layer, support });
    for (const [from, qubits] of predecessors)
      edges.push({ from, to: instruction.index, qubits });
    support.forEach((q) => last.set(q, instruction.index));
  }
  return { nodes, edges, depth: Math.max(-1, ...layers.values()) + 1 };
}

export function compilerMetrics(stage: FusionStage, width: number) {
  return {
    logicalInstructions: stage.metrics.instructionCount,
    measuredKernelSweeps: stage.metrics.measuredKernelSweeps,
    modeledFullStatePasses: stage.metrics.modeledFullStatePasses,
    tileApplications: stage.metrics.tileApplications,
    stateBytes: 16 * 2 ** width,
    modeledPayloadBytes: 32 * 2 ** width * stage.metrics.modeledFullStatePasses,
  };
}

export function matrixRows(matrix: ComplexMatrix) {
  checkMatrix(matrix);
  return matrix.map((coefficients, row) => ({ row, coefficients }));
}

export function matrixRowTerms(
  matrix: ComplexMatrix,
  row: number,
  input: readonly Complex[],
) {
  checkMatrix(matrix);
  if (!integer(row) || row >= matrix.length || input.length !== matrix.length)
    throw new Error("Matrix row and input dimensions differ");
  const terms = matrix[row].map((coefficient, column) => ({
    column,
    coefficient,
    input: input[column],
    product: mul(coefficient, input[column]),
  }));
  return {
    terms,
    sum: terms.reduce<Complex>((sum, term) => add(sum, term.product), [0, 0]),
  };
}

export function composeChronological(
  matrices: readonly ComplexMatrix[],
): ComplexMatrix {
  if (!matrices.length) throw new Error("Composition needs a matrix");
  const dimension = matrices[0].length;
  matrices.forEach((matrix) => checkMatrix(matrix, dimension));
  return matrices
    .slice(1)
    .reduce<ComplexMatrix>(
      (product, next) =>
        next.map((row) =>
          row.map((_, column) =>
            row.reduce<Complex>(
              (sum, coefficient, k) =>
                add(sum, mul(coefficient, product[k][column])),
              [0, 0],
            ),
          ),
        ),
      matrices[0],
    );
}
