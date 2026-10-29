import { describe, expect, it } from "vitest";
import {
  executionModel,
  instructionGroups,
  laneProduct,
  phaseOf,
  type MicroscopeInput,
} from "../src/models/simd-execution";
import { diagonalRoute, kernelRoute } from "../src/models/hardware";
import { initialState, matrixFor } from "../src/models/quantum";

describe("source intrinsic arithmetic", () => {
  it("permutes each 128-bit complex lane independently in AVX2", () => {
    const product = laneProduct([2, -3, 5, 7], [4, -2]);
    expect(product.swapped).toEqual([-3, 2, 7, 5]);
    expect(product.imagCoefficients).toEqual([-2, -2, -2, -2]);
    expect(product.imagProducts).toEqual([6, -4, -14, -10]);
    expect(product.signedImagProducts).toEqual([-6, -4, 14, -10]);
    expect(product.result).toEqual([2, -16, 34, 18]);
  });

  it("uses the pre-negated real-lane imaginary coefficient for NEON", () => {
    const product = laneProduct([2, -3], [4, -2], true);
    expect(product.imagCoefficients).toEqual([2, -2]);
    expect(product.realProducts).toEqual([8, -12]);
    expect(product.imagProducts).toEqual([-6, -4]);
    expect(product.result).toEqual([2, -16]);
  });

  it("preserves signs and tiny imaginary components", () => {
    const product = laneProduct([-1e-20, 2e-20], [0, -1]);
    expect(product.result[0]).toBeCloseTo(2e-20, 30);
    expect(product.result[1]).toBeCloseTo(1e-20, 30);
    expect(phaseOf([0, 0])).toBeNull();
    expect(phaseOf([0, -1e-20])).toBe(-Math.PI / 2);
  });
});

function fixture(overrides: Partial<MicroscopeInput> = {}): MicroscopeInput {
  return {
    before: Float64Array.from({ length: 32 }, (_, i) => i - 8),
    matrix: [
      [2, -1],
      [3, 2],
      [-2, 4],
      [1, -3],
    ],
    pair: [3, 7],
    n: 4,
    target: 2,
    architecture: "avx2",
    width: 256,
    diagonal: false,
    ...overrides,
  };
}

describe("loaded registers and stores", () => {
  it("loads two adjacent amplitudes from each same half, not the target pair", () => {
    const model = executionModel(fixture());
    expect(model.indices).toEqual([
      [2, 3],
      [6, 7],
    ]);
    expect(model.loads).toEqual([
      [-4, -3, -2, -1],
      [4, 5, 6, 7],
    ]);
    expect(model.selectedLane).toBe(2);
    expect(model.rows).toEqual([
      [-9, 21, -1, 33],
      [39, -17, 35, -17],
    ]);
    expect(model.outputs).toEqual([
      [-1, 33],
      [35, -17],
    ]);
  });

  it("FMA128 loads and stores only the selected pair", () => {
    const model = executionModel(fixture({ width: 128 }));
    expect(model.indices).toEqual([[3], [7]]);
    expect(model.loads).toEqual([
      [-2, -1],
      [6, 7],
    ]);
    expect(model.outputs).toEqual([
      [-1, 33],
      [35, -17],
    ]);
  });

  it("retains the original low and high inputs for both output rows", () => {
    const input = fixture({ width: 128 });
    const original = Array.from(input.before);
    executionModel(input);
    expect(Array.from(input.before)).toEqual(original);
  });

  it("scales diagonal halves independently and bypasses identity low halves", () => {
    const model = executionModel(
      fixture({
        diagonal: true,
        matrix: [
          [1, 0],
          [0, 0],
          [0, 0],
          [0, -1],
        ],
      }),
    );
    expect(model.skipLow).toBe(true);
    expect(model.rows[0]).toEqual(model.loads[0]);
    expect(model.rows[1]).toEqual([5, -4, 7, -6]);
    expect(model.outputs).toEqual([
      [-2, -1],
      [7, -6],
    ]);
    expect(
      executionModel(
        fixture({
          diagonal: true,
          matrix: [
            [1, 1e-15],
            [0, 0],
            [0, 0],
            [1, 0],
          ],
        }),
      ).skipLow,
    ).toBe(false);
    expect(
      executionModel(
        fixture({
          diagonal: true,
          matrix: [
            [1, -9e-16],
            [0, 0],
            [0, 0],
            [1, 0],
          ],
        }),
      ).skipLow,
    ).toBe(true);
  });

  it.each([13, 14, 15])(
    "matches independent complex arithmetic across the n=%i route transition",
    (n) => {
      const target = 3;
      const route = kernelRoute(n, target, "avx2");
      expect(route.width).toBe(n === 14 ? 128 : 256);
      const input = fixture({
        n,
        target,
        before: initialState(n),
        matrix: matrixFor("Rx", -0.7),
        pair: [3, 11],
        width: route.width as 128 | 256,
      });
      const model = executionModel(input);
      for (const row of [0, 1]) {
        let re = 0,
          im = 0;
        for (const column of [0, 1]) {
          const index = input.pair[column];
          const [cr, ci] = input.matrix[2 * row + column];
          const zr = input.before[index * 2],
            zi = input.before[index * 2 + 1];
          re += cr * zr - ci * zi;
          im += cr * zi + ci * zr;
        }
        expect(model.outputs[row][0]).toBeCloseTo(re, 15);
        expect(model.outputs[row][1]).toBeCloseTo(im, 15);
      }
    },
  );

  it("uses 256-bit independent diagonal scaling only when the supplied route selects it", () => {
    expect(diagonalRoute(15, 13, "avx2").width).toBe(256);
    expect(diagonalRoute(15, 12, "avx2").width).toBe(128);
    expect(diagonalRoute(14, 13, "avx2").width).toBe(128);
  });

  it("has ten general and eight diagonal instruction groups with correct native operations", () => {
    const avx = instructionGroups("avx2", 256, false);
    expect(avx).toHaveLength(10);
    expect(avx[3].code).toContain("permute_pd");
    expect(avx[5].code).toContain("fmaddsub_pd");
    const neon = instructionGroups("neon", 128, true);
    expect(neon).toHaveLength(8);
    expect(neon[4].code).toContain("vmulq_f64");
    expect(neon[5].code).toContain("vfmaq_f64");
    expect(neon.some((group) => group.title === "Add the two row terms")).toBe(
      false,
    );
  });
});
