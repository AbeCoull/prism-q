import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  amplitude,
  applyGate,
  applyPair,
  binary,
  byteOffset,
  cacheLine,
  compose,
  gateNames,
  initialState,
  matrixFor,
  maxError,
  norm,
  pairs,
  type GateName,
  type Matrix2,
} from "../src/models/quantum";
import {
  bandwidthSeconds,
  kernelRoute,
  rankOwnership,
  stateBytes,
  sweepBytes,
} from "../src/models/hardware";

type C = [number, number];
type Dense = C[][];
const times = ([a, b]: C, [c, d]: C): C => [a * c - b * d, a * d + b * c];
const plus = ([a, b]: C, [c, d]: C): C => [a + c, b + d];
const identity: Dense = [
  [
    [1, 0],
    [0, 0],
  ],
  [
    [0, 0],
    [1, 0],
  ],
];

function independentMatrix(gate: GateName, theta: number): Dense {
  if (gate === "H")
    return [
      [
        [1 / Math.sqrt(2), 0],
        [1 / Math.sqrt(2), 0],
      ],
      [
        [1 / Math.sqrt(2), 0],
        [-1 / Math.sqrt(2), 0],
      ],
    ];
  const pauli: Record<string, Dense> = {
    X: [
      [
        [0, 0],
        [1, 0],
      ],
      [
        [1, 0],
        [0, 0],
      ],
    ],
    Y: [
      [
        [0, 0],
        [0, -1],
      ],
      [
        [0, 1],
        [0, 0],
      ],
    ],
    Z: [
      [
        [1, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [-1, 0],
      ],
    ],
  };
  if (gate in pauli) return pauli[gate];
  const generator = pauli[gate.slice(1).toUpperCase()];
  return generator.map((row, r) =>
    row.map((p, c) =>
      plus(
        [r === c ? Math.cos(theta / 2) : 0, 0],
        times([0, -Math.sin(theta / 2)], p),
      ),
    ),
  );
}

function kron(a: Dense, b: Dense): Dense {
  return Array.from({ length: a.length * b.length }, (_, r) =>
    Array.from({ length: a.length * b.length }, (_, c) =>
      times(
        a[Math.floor(r / b.length)][Math.floor(c / b.length)],
        b[r % b.length][c % b.length],
      ),
    ),
  );
}

function tensorOperator(n: number, target: number, gate: Dense): Dense {
  let result: Dense = [[[1, 0]]];
  for (let q = n - 1; q >= 0; q--)
    result = kron(result, q === target ? gate : identity);
  return result;
}

function denseApply(operator: Dense, state: ArrayLike<number>): Float64Array {
  const result = new Float64Array(state.length);
  for (let row = 0; row < operator.length; row++) {
    let sum: C = [0, 0];
    for (let col = 0; col < operator.length; col++)
      sum = plus(
        sum,
        times(operator[row][col], [state[2 * col], state[2 * col + 1]]),
      );
    result.set(sum, 2 * row);
  }
  return result;
}

function closeState(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance = 3e-14,
) {
  expect(actual.length).toBe(expected.length);
  for (let k = 0; k < actual.length; k++)
    expect(Math.abs(actual[k] - expected[k])).toBeLessThan(tolerance);
}

describe("state evolution against independent tensor operators", () => {
  for (const gate of gateNames) {
    for (const theta of [-2.4, 0, Math.PI / 3, Math.PI]) {
      it(`${gate}(${theta}) agrees on every target through five qubits`, () => {
        for (let n = 1; n <= 5; n++) {
          const state = initialState(n);
          const unchanged = Float64Array.from(state);
          for (let q = 0; q < n; q++) {
            const expected = denseApply(
              tensorOperator(n, q, independentMatrix(gate, theta)),
              state,
            );
            const actual = applyGate(state, q, matrixFor(gate, theta));
            closeState(actual, expected);
            expect(norm(actual)).toBeCloseTo(1, 13);
          }
          expect(state).toEqual(unchanged);
        }
      });
    }
    it(`${gate} is unitary including relative complex phase`, () => {
      const matrix = matrixFor(gate, 1.234);
      const inverse: Matrix2 = [
        [matrix[0][0], -matrix[0][1]],
        [matrix[2][0], -matrix[2][1]],
        [matrix[1][0], -matrix[1][1]],
        [matrix[3][0], -matrix[3][1]],
      ];
      for (let q = 0; q < 4; q++) {
        const state = initialState(4);
        closeState(applyGate(applyGate(state, q, matrix), q, inverse), state);
      }
    });
  }

  it("q0 is the LSB and each basis input maps to index XOR 2^target under X", () => {
    for (let n = 1; n <= 5; n++) {
      for (let q = 0; q < n; q++) {
        for (let basis = 0; basis < 2 ** n; basis++) {
          const state = new Float64Array(2 * 2 ** n);
          state[2 * basis] = 1;
          const result = applyGate(state, q, matrixFor("X"));
          expect(result[2 * (basis ^ (1 << q))]).toBe(1);
          expect(norm(result)).toBe(1);
        }
      }
    }
    expect(binary(1, 3)).toBe("001");
    expect(binary(4, 3)).toBe("100");
  });

  it("controlled pair selection makes Bell amplitudes and preserves the control-zero subspace", () => {
    const before = applyGate(initialState(2, true), 0, matrixFor("H"));
    const after = Float64Array.from(before);
    for (const [i, j] of pairs(2, 1)) {
      if (i & 1) {
        const [a, b] = applyPair(before, matrixFor("X"), i, j);
        after.set(a, 2 * i);
        after.set(b, 2 * j);
      }
    }
    const controlledX: Dense = [
      [
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
      ],
      [
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0],
      ],
    ];
    closeState(after, denseApply(controlledX, before));
    closeState(after, [Math.SQRT1_2, 0, 0, 0, 0, 0, Math.SQRT1_2, 0]);
    const arbitrary = initialState(2);
    const acted = denseApply(controlledX, arbitrary);
    expect(amplitude(acted, 0)).toEqual(amplitude(arbitrary, 0));
    expect(amplitude(acted, 2)).toEqual(amplitude(arbitrary, 2));
  });

  it("composition uses execution order Ulast times Ufirst", () => {
    const a = matrixFor("Rx", 0.7),
      b = matrixFor("Ry", -1.2),
      c = matrixFor("Rz", 0.4);
    const state = initialState(4);
    for (let q = 0; q < 4; q++) {
      const sequential = applyGate(
        applyGate(applyGate(state, q, a), q, b),
        q,
        c,
      );
      closeState(applyGate(state, q, compose(c, compose(b, a))), sequential);
    }
    const reversed = applyGate(state, 0, compose(a, b));
    const ordered = applyGate(applyGate(state, 0, a), 0, b);
    expect(maxError(reversed, ordered)).toBeGreaterThan(0.01);
  });

  it("complex error rejects matching populations with different relative phases", () => {
    const rx = applyGate(
      initialState(1, true),
      0,
      matrixFor("Rx", Math.PI / 2),
    );
    const ry = applyGate(
      initialState(1, true),
      0,
      matrixFor("Ry", Math.PI / 2),
    );
    expect(rx[2] ** 2 + rx[3] ** 2).toBeCloseTo(ry[2] ** 2 + ry[3] ** 2, 14);
    expect(maxError(rx, ry)).toBeCloseTo(1, 14);
    expect(() => maxError(rx, new Float64Array(8))).toThrow("dimensions");
  });
});

describe("address arithmetic and traffic units", () => {
  it("pairs partition every state and follow zero-bit insertion order", () => {
    for (let n = 1; n <= 10; n++) {
      for (let q = 0; q < n; q++) {
        const paired = pairs(n, q),
          visited = new Set<number>(),
          mask = (1 << q) - 1;
        expect(paired.length).toBe(2 ** (n - 1));
        paired.forEach(([i, j], k) => {
          expect(i).toBe(((k & ~mask) << 1) | (k & mask));
          expect(i & (1 << q)).toBe(0);
          expect(j ^ i).toBe(1 << q);
          expect(j - i).toBe(2 ** q);
          expect(byteOffset(j) - byteOffset(i)).toBe(16 * 2 ** q);
          expect(visited.has(i) || visited.has(j)).toBe(false);
          visited.add(i);
          visited.add(j);
        });
        expect(visited.size).toBe(2 ** n);
      }
    }
  });

  it("high-target partners are distant but each stream advances consecutively", () => {
    const pairList = pairs(6, 4);
    for (let k = 1; k < 16; k++) {
      expect(pairList[k][0] - pairList[k - 1][0]).toBe(1);
      expect(pairList[k][1] - pairList[k - 1][1]).toBe(1);
      expect(byteOffset(pairList[k][1]) - byteOffset(pairList[k][0])).toBe(256);
    }
  });

  it("cache line occupancy accounts for the assumed base offset and line size", () => {
    expect([0, 1, 2, 3, 4].map((i) => cacheLine(i))).toEqual([0, 0, 0, 0, 1]);
    expect([0, 1, 2, 3].map((i) => cacheLine(i, 16))).toEqual([0, 0, 0, 1]);
    expect([0, 1, 2, 3].map((i) => cacheLine(i, 0, 32))).toEqual([0, 0, 1, 1]);
    expect(byteOffset(2 ** 30)).toBe(16 * 2 ** 30);
  });

  it("complex128 state and complete read/write sweep sizes use bytes", () => {
    expect(stateBytes(20)).toBe(16 * 2 ** 20);
    expect(sweepBytes(20)).toBe(2 * stateBytes(20));
    expect(stateBytes(8, 16, true)).toBe(16 * 2 ** 16);
    expect(bandwidthSeconds(20, 3, 100)).toBeCloseTo(
      (3 * 32 * 2 ** 20) / 100e9,
      15,
    );
  });
});

describe("native kernel route boundaries from source", () => {
  it.each([
    [13, 1, 128, false],
    [13, 2, 256, false],
    [13, 12, 256, false],
    [14, 1, 128, false],
    [14, 2, 128, false],
    [14, 12, 128, false],
    [14, 13, 128, false],
    [15, 1, 128, true],
    [15, 2, 256, true],
    [15, 12, 256, true],
    [15, 13, 256, true],
  ])(
    "n=%i target=%i selects %i-bit arithmetic, threaded=%s",
    (n, q, width, threaded) => {
      const route = kernelRoute(n as number, q as number, "avx2");
      expect(route.width).toBe(width);
      expect(route.threaded).toBe(threaded);
    },
  );

  it("disabling parallel keeps large states on the sequential 128-bit FMA route", () => {
    for (const q of [1, 2, 12, 13]) {
      const route = kernelRoute(15, q, "avx2", false);
      expect(route.width).toBe(128);
      expect(route.threaded).toBe(false);
    }
  });

  it("the parallel target boundary separates full tiles from paired slice processing", () => {
    expect(kernelRoute(15, 12, "avx2").partition).toContain("8192");
    expect(kernelRoute(15, 13, "avx2").partition).toContain("paired");
    expect(kernelRoute(14, 12, "avx2").partition).toContain("sequential");
    expect(kernelRoute(15, 13, "neon").width).toBe(128);
    expect(kernelRoute(15, 13, "neon").complexPerRegister).toBe(1);
  });

  it("scalar arithmetic has no SIMD vector width or packed complex lanes", () => {
    const route = kernelRoute(4, 2, "scalar");
    expect(route.width).toBe(0);
    expect(route.complexPerRegister).toBe(0);
  });
});

describe("distributed contiguous rank ownership", () => {
  it("all amplitudes have unique local addresses and ownership reconstructs the global index", () => {
    for (let n = 1; n <= 8; n++) {
      for (let p = 0; p <= n; p++) {
        const ranks = 2 ** p,
          localCount = 2 ** (n - p),
          addresses = new Set<string>();
        for (let index = 0; index < 2 ** n; index++) {
          const result = rankOwnership(index, n, ranks);
          expect(result.rank).toBe(index >>> (n - p));
          expect(result.local).toBe(index & (localCount - 1));
          expect(result.rank * localCount + result.local).toBe(index);
          expect(result.rank).toBeLessThan(ranks);
          addresses.add(`${result.rank}:${result.local}`);
          for (let q = 0; q < n; q++)
            expect(
              rankOwnership(index ^ (1 << q), n, ranks).rank !== result.rank,
            ).toBe(q >= n - p);
        }
        expect(addresses.size).toBe(2 ** n);
      }
    }
  });
});

type NativeTrace = {
  id: string;
  n: number;
  target: number;
  gate: GateName;
  theta: number;
  before: number[];
  steps: number[][];
};
describe("native PRISM-Q golden traces", () => {
  it("every recorded step matches both the browser model and an independent dense operator", () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL("../fixtures/native-gates.json", import.meta.url),
        "utf8",
      ),
    ) as {
      schemaVersion: number;
      provenance: unknown;
      traces: NativeTrace[];
    };
    expect(fixture.schemaVersion).toBe(1);
    expect(fixture.provenance).toBeTruthy();
    expect(fixture.traces.length).toBeGreaterThan(0);
    const covered = new Set<GateName>();
    for (const trace of fixture.traces) {
      covered.add(trace.gate);
      expect(trace.before.length).toBe(2 * 2 ** trace.n);
      expect(trace.steps.length).toBe(4);
      expect(norm(trace.before)).toBeCloseTo(1, 13);
      const operator = tensorOperator(
        trace.n,
        trace.target,
        independentMatrix(trace.gate, trace.theta),
      );
      let browser = Float64Array.from(trace.before),
        dense = Float64Array.from(trace.before);
      for (const expected of trace.steps) {
        browser = applyGate(
          browser,
          trace.target,
          matrixFor(trace.gate, trace.theta),
        );
        dense = denseApply(operator, dense);
        closeState(browser, expected);
        closeState(dense, expected);
        expect(norm(expected)).toBeCloseTo(1, 12);
      }
    }
    expect([...covered].sort()).toEqual([...gateNames].sort());
  });
});

describe("native 16-qubit threaded traces", () => {
  it("parallel tile, paired-slice, and diagonal scaling routes match the browser model", () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL("../fixtures/native-gates.json", import.meta.url),
        "utf8",
      ),
    ) as {
      wideTraces: {
        n: number;
        target: number;
        gate: GateName;
        theta: number;
        normSquared: number;
        maxAbsoluteError: number;
        samples: { index: number; re: number; im: number }[];
      }[];
    };
    const routes = new Set<string>();
    for (const trace of fixture.wideTraces) {
      routes.add(
        `${kernelRoute(trace.n, trace.target, "avx2").partition}/${trace.gate}`,
      );
      expect(trace.maxAbsoluteError).toBeLessThan(1e-12);
      expect(trace.normSquared).toBeCloseTo(1, 12);
      const model = applyGate(
        initialState(trace.n),
        trace.target,
        matrixFor(trace.gate, trace.theta),
      );
      for (const sample of trace.samples) {
        const [re, im] = amplitude(model, sample.index);
        expect(Math.abs(re - sample.re)).toBeLessThan(3e-14);
        expect(Math.abs(im - sample.im)).toBeLessThan(3e-14);
      }
    }
    expect(routes.has("Rayon: 8192-amplitude tiles/H")).toBe(true);
    expect(routes.has("Rayon: paired low / high slices/H")).toBe(true);
  });
});
