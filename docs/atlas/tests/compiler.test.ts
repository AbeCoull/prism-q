import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  circuitDag,
  compilerMetrics,
  composeChronological,
  matrixRows,
  matrixRowTerms,
  validateFusionFixture,
  type ComplexMatrix,
  type FusionFixture,
  type FusionPayload,
  type FusionStage,
} from "../src/models/compiler";
import { matrixFor, type Complex } from "../src/models/quantum";

const native = JSON.parse(
  readFileSync(
    new URL("../fixtures/native-fusion.json", import.meta.url),
    "utf8",
  ),
) as FusionFixture;

function changed(mutator: (fixture: FusionFixture) => void) {
  const fixture = structuredClone(native);
  mutator(fixture);
  return fixture;
}

function nested(gate: "H" | "Rz"): ComplexMatrix {
  const m = matrixFor(gate);
  return [
    [m[0], m[1]],
    [m[2], m[3]],
  ];
}

function closeMatrix(actual: ComplexMatrix, expected: ComplexMatrix) {
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < actual.length; row++)
    for (let column = 0; column < actual.length; column++) {
      expect(actual[row][column][0]).toBeCloseTo(expected[row][column][0], 13);
      expect(actual[row][column][1]).toBeCloseTo(expected[row][column][1], 13);
    }
}

function initial(width: number) {
  const state = new Float64Array(2 * 2 ** width);
  let norm = 0;
  for (let j = 0; j < 2 ** width; j++) {
    state[2 * j] = ((j * 17 + 42) % 101) - 50;
    state[2 * j + 1] = ((j * 31 + 42) % 97) - 48;
    norm += state[2 * j] ** 2 + state[2 * j + 1] ** 2;
  }
  return state.map((value) => value / Math.sqrt(norm));
}

function denseApply(state: Float64Array, payload: FusionPayload) {
  const output = new Float64Array(state.length);
  const masks = payload.targets.map((q) => 2 ** q);
  for (let index = 0; index < state.length / 2; index++) {
    let row = 0,
      base = index;
    for (const mask of masks) {
      const bit = Math.floor(index / mask) % 2;
      row = row * 2 + bit;
      base -= bit * mask;
    }
    for (let column = 0; column < payload.matrix.length; column++) {
      let source = base;
      for (let bit = 0; bit < masks.length; bit++)
        source +=
          (Math.floor(column / 2 ** (masks.length - 1 - bit)) % 2) * masks[bit];
      const [re, im] = payload.matrix[row][column];
      output[2 * index] += re * state[2 * source] - im * state[2 * source + 1];
      output[2 * index + 1] +=
        re * state[2 * source + 1] + im * state[2 * source];
    }
  }
  return output;
}

function replay(stage: FusionStage, width: number) {
  let state = initial(width);
  for (const instruction of stage.instructions)
    for (const payload of instruction.payload)
      state = denseApply(state, payload);
  return state;
}

describe("native compiler evidence", () => {
  it("validates every snapshot, native pass counter, and source mapping", () => {
    expect(validateFusionFixture(native)).toBe(native);
    expect(native.records).toHaveLength(31);
    expect(new Set(native.records.map((record) => record.width))).toEqual(
      new Set([8, 10, 12, 14, 16, 20]),
    );
    expect(new Set(native.records.map((record) => record.profile))).toEqual(
      new Set(["Host", "Device"]),
    );
  });

  it.each([
    [
      "wrong source symbol",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[1].source.symbol = "unrelated";
      },
    ],
    [
      "wrong source path",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[1].source.path = "src/elsewhere.rs";
      },
    ],
    [
      "duplicate record ID",
      (fixture: FusionFixture) => {
        fixture.records[1].id = fixture.records[0].id;
      },
    ],
    [
      "stage index used as another ID",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[1].id = 0;
      },
    ],
    [
      "unsupported source origins",
      (fixture: FusionFixture) => {
        Object.assign(fixture.records[0].stages[1].instructions[0], {
          sourceIds: [0],
        });
      },
    ],
    [
      "missing hidden batch support",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[0].instructions[0].support = [];
      },
    ],
    [
      "counter substituted with instruction count",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[0].metrics.measuredKernelSweeps += 1;
      },
    ],
    [
      "infinite matrix coefficient",
      (fixture: FusionFixture) => {
        (
          fixture.records[0].stages[0].instructions[0].payload[0]
            .matrix[0][0] as number[]
        )[0] = Infinity;
      },
    ],
    [
      "wrong matrix dimension",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[0].instructions[0].payload[0].targets.push(1);
      },
    ],
    [
      "invalid native equivalence claim",
      (fixture: FusionFixture) => {
        fixture.records[0].stages[0].state.maxAbsoluteError = 1e-4;
      },
    ],
    [
      "public pipeline mismatch",
      (fixture: FusionFixture) => {
        fixture.records[0].finalMatchesPublicPipeline = false;
      },
    ],
  ])("rejects %s", (_, mutate) => {
    expect(() =>
      validateFusionFixture(
        changed(mutate as (fixture: FusionFixture) => void),
      ),
    ).toThrow();
  });

  it.each([
    [
      "missing pass",
      (f: FusionFixture) => {
        f.records[0].stages.splice(2, 1);
        f.records[0].stages.forEach((s, i) => (s.id = i));
      },
    ],
    [
      "reordered passes",
      (f: FusionFixture) => {
        const s = f.records[0].stages;
        [s[2], s[3]] = [s[3], s[2]];
        s.forEach((stage, i) => (stage.id = i));
      },
    ],
    [
      "invented floor",
      (f: FusionFixture) => {
        f.records[0].stages.find((s) => s.name === "fuse-1q")!.floor = 1;
      },
    ],
    [
      "unknown family",
      (f: FusionFixture) => {
        f.records[0].family = "unrecorded-family";
      },
    ],
    [
      "wrong planner branch",
      (f: FusionFixture) => {
        f.generator.stagePlannerFloor = 16;
      },
    ],
    [
      "incorrect skipped state",
      (f: FusionFixture) => {
        f.records[0].stages.find((s) => s.name === "fuse-1q")!.enabled = true;
      },
    ],
  ])("rejects pipeline corruption: %s", (_, mutate) => {
    expect(() =>
      validateFusionFixture(
        changed(mutate as (fixture: FusionFixture) => void),
      ),
    ).toThrow();
  });

  it("independently embeds matrices with targets[0] as the local most significant bit", () => {
    const state = new Float64Array(8);
    state[2] = 1;
    const cx = native.records.find(
      (record) => record.family === "blocked-dependency",
    )!.stages[0].instructions[1];
    const result = denseApply(state, cx.payload[0]);
    expect(result[6]).toBe(1);
    expect(result.reduce((sum, x) => sum + x * x, 0)).toBe(1);
  });

  it("replays native intermediate matrices and batch entries against recorded amplitude samples", () => {
    let checked = 0;
    for (const record of native.records) {
      const stages =
        record.width <= 12
          ? record.stages
          : record.width === 14
            ? record.stages.slice(-1)
            : [];
      for (const stage of stages) {
        const actual = replay(stage, record.width);
        for (const sample of stage.state.samples) {
          expect(
            Math.hypot(
              actual[2 * sample.index] - sample.re,
              actual[2 * sample.index + 1] - sample.im,
            ),
          ).toBeLessThan(2e-13);
          checked++;
        }
      }
    }
    expect(checked).toBeGreaterThan(1000);
  }, 30_000);

  it("keeps logical instructions separate from measured traversals and modeled traffic", () => {
    const batch = native.records
      .flatMap((record) => record.stages)
      .find((stage) =>
        stage.instructions.some(
          (instruction) => instruction.payload.length > 1,
        ),
      )!;
    expect(batch).toBeDefined();
    const metrics = compilerMetrics(batch, 14);
    expect(metrics.logicalInstructions).toBe(batch.instructions.length);
    expect(metrics.measuredKernelSweeps).toBe(
      batch.metrics.measuredKernelSweeps,
    );
    expect(metrics.stateBytes).toBe(16 * 2 ** 14);
    expect(metrics.modeledPayloadBytes).toBe(
      32 * 2 ** 14 * batch.metrics.modeledFullStatePasses,
    );
    expect(
      batch.instructions.reduce(
        (count, instruction) => count + instruction.payload.length,
        0,
      ),
    ).toBeGreaterThan(metrics.logicalInstructions);
  });
});

describe("conservative circuit dependencies", () => {
  it("retains the order of every shared qubit while disjoint gates share a layer", () => {
    const dag = circuitDag(
      [
        { index: 0, support: [0] },
        { index: 1, support: [1] },
        { index: 2, support: [0, 1] },
        { index: 3, support: [1] },
        { index: 4, support: [2] },
      ],
      3,
    );
    expect(dag.nodes.map((node) => node.layer)).toEqual([0, 0, 1, 2, 0]);
    expect(dag.edges).toEqual([
      { from: 0, to: 2, qubits: [0] },
      { from: 1, to: 2, qubits: [1] },
      { from: 2, to: 3, qubits: [1] },
    ]);
    expect(dag.depth).toBe(3);
  });

  it("merges dependency edges without losing their qubits", () => {
    const dag = circuitDag(
      [
        { index: 0, support: [1, 0] },
        { index: 1, support: [0, 1] },
      ],
      2,
    );
    expect(dag.edges).toEqual([{ from: 0, to: 1, qubits: [0, 1] }]);
  });

  it("fences classical and opaque instructions instead of inferring their dependencies", () => {
    const dag = circuitDag(
      [
        { index: 0, support: [0] },
        { index: 1, support: [0], kind: "measure" },
        { index: 2, support: [1], kind: "conditional" },
        { index: 3, support: [2] },
      ],
      3,
    );
    expect(dag.nodes.map((node) => node.layer)).toEqual([0, 1, 2, 3]);
  });

  it("preserves qubit order in every native snapshot", () => {
    for (const record of native.records)
      for (const stage of record.stages) {
        const dag = circuitDag(stage.instructions, record.width);
        for (let q = 0; q < record.width; q++) {
          const chain = stage.instructions.filter((instruction) =>
            instruction.support.includes(q),
          );
          for (let i = 1; i < chain.length; i++)
            expect(
              dag.edges.some(
                (edge) =>
                  edge.from === chain[i - 1].index &&
                  edge.to === chain[i].index &&
                  edge.qubits.includes(q),
              ),
            ).toBe(true);
        }
      }
  });

  it("represents an empty stream with zero depth", () => {
    expect(circuitDag([], 0)).toEqual({ nodes: [], edges: [], depth: 0 });
  });
});

describe("matrix inspection", () => {
  it("composes chronological noncommuting gates on the left", () => {
    const h = nested("H"),
      rz = nested("Rz");
    const product = composeChronological([h, rz]);
    const expected = h.map((row, i) =>
      row.map((z) => {
        const [a, b] = rz[i][i];
        return [a * z[0] - b * z[1], a * z[1] + b * z[0]] as Complex;
      }),
    );
    closeMatrix(product, expected);
    expect(product[0][1][1]).not.toBeCloseTo(
      composeChronological([rz, h])[0][1][1],
      8,
    );
  });

  it("matches the real fused payload to its chronological source matrices", () => {
    for (const record of native.records.filter(
      (entry) => entry.family === "legal-disjoint" && entry.width >= 10,
    )) {
      const original = record.stages[0];
      const fused = record.stages.find((stage) => stage.name === "refuse-1q")!;
      for (let q = 0; q < record.width; q++) {
        const chronological = original.instructions
          .filter(
            (instruction) =>
              instruction.targets.length === 1 && instruction.targets[0] === q,
          )
          .map((instruction) => instruction.payload[0].matrix);
        const emitted = fused.instructions.find(
          (instruction) =>
            instruction.support.length === 1 && instruction.support[0] === q,
        )!;
        expect(emitted).toBeDefined();
        closeMatrix(
          emitted.payload[0].matrix,
          composeChronological(chronological),
        );
      }
    }
  });

  it("shows each complex product in a matrix row and their simultaneous sum", () => {
    const matrix: ComplexMatrix = [
      [
        [1, 1],
        [2, -1],
      ],
      [
        [0, 0],
        [1, 0],
      ],
    ];
    const input: Complex[] = [
      [2, -1],
      [-1, 3],
    ];
    const row = matrixRowTerms(matrix, 0, input);
    expect(row.terms.map((term) => term.product)).toEqual([
      [3, 1],
      [1, 7],
    ]);
    expect(row.sum).toEqual([4, 8]);
    expect(matrixRows(matrix)).toEqual([
      { row: 0, coefficients: matrix[0] },
      { row: 1, coefficients: matrix[1] },
    ]);
    expect(input).toEqual([
      [2, -1],
      [-1, 3],
    ]);
  });

  it("rejects inconsistent dimensions and empty compositions", () => {
    expect(() => composeChronological([])).toThrow();
    expect(() =>
      matrixRowTerms(nested("H"), 2, [
        [1, 0],
        [0, 0],
      ]),
    ).toThrow();
    expect(() =>
      matrixRows([
        [
          [1, 0],
          [0, 0],
        ],
      ]),
    ).toThrow();
  });
});
