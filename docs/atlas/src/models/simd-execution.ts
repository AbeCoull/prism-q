import { amplitude, type Complex, type Matrix2 } from "./quantum";
import type { Architecture } from "./hardware";

export type SimdWidth = 0 | 128 | 256;
export type Coefficient = 0 | 1 | 2 | 3;
export const skipsLowHalf = (matrix: Matrix2, diagonal: boolean) =>
  diagonal &&
  Math.abs(matrix[0][0] - 1) < 1e-15 &&
  Math.abs(matrix[0][1]) < 1e-15;
export type MicroscopeInput = {
  before: ArrayLike<number>;
  matrix: Matrix2;
  pair: readonly [number, number];
  n: number;
  target: number;
  architecture: Architecture;
  width: SimdWidth;
  diagonal: boolean;
};

export function laneProduct(
  values: readonly number[],
  coefficient: Complex,
  neon = false,
) {
  const realCoefficients = values.map(() => coefficient[0]);
  const imagCoefficients = values.map((_, lane) =>
    neon && lane % 2 === 0 ? -coefficient[1] : coefficient[1],
  );
  const swapped = values.map((_, lane) => values[lane ^ 1]);
  const realProducts = values.map((value) => coefficient[0] * value);
  const imagProducts = swapped.map(
    (value, lane) => value * imagCoefficients[lane],
  );
  const signedImagProducts = imagProducts.map((value, lane) =>
    !neon && lane % 2 === 0 ? -value : value,
  );
  const result = realProducts.map(
    (value, lane) => value + signedImagProducts[lane],
  );
  return {
    realCoefficients,
    imagCoefficients,
    swapped,
    realProducts,
    imagProducts,
    signedImagProducts,
    result,
  };
}

export function phaseOf(value: Complex): number | null {
  return value[0] === 0 && value[1] === 0
    ? null
    : Math.atan2(value[1], value[0]);
}

export function executionModel(input: MicroscopeInput) {
  const { before, matrix, pair, target, width, architecture, diagonal } = input;
  const complexCount = width === 256 ? 2 : 1;
  const half = 2 ** target;
  const low = complexCount === 2 ? pair[0] - ((pair[0] % half) % 2) : pair[0];
  const indices = [
    Array.from({ length: complexCount }, (_, k) => low + k),
    Array.from({ length: complexCount }, (_, k) => low + half + k),
  ];
  const loads = indices.map((group) =>
    group.flatMap((i) => [...amplitude(before, i)]),
  );
  const neon = architecture === "neon";
  const terms = matrix.map((coefficient, i) =>
    laneProduct(loads[i % 2], coefficient, neon),
  );
  const skipLow = skipsLowHalf(matrix, diagonal);
  const rows = [0, 1].map((row) =>
    diagonal
      ? row === 0 && skipLow
        ? loads[0]
        : terms[row * 3].result
      : terms[row * 2].result.map(
          (value, lane) => value + terms[row * 2 + 1].result[lane],
        ),
  );
  const selectedLane = (pair[0] - low) * 2;
  const outputs = rows.map(
    (row) => [row[selectedLane], row[selectedLane + 1]] as Complex,
  );
  return {
    indices,
    loads,
    terms,
    rows,
    outputs,
    selectedLane,
    complexCount,
    skipLow,
  };
}

export function instructionGroups(
  architecture: Architecture,
  width: SimdWidth,
  diagonal: boolean,
) {
  const scalar = width === 0;
  const neon = architecture === "neon";
  const prefix = width === 256 ? "_mm256" : "_mm";
  const load = scalar
    ? "read Complex64"
    : neon
      ? "vld1q_f64"
      : `${prefix}_loadu_pd`;
  const swap = scalar
    ? "select real / imaginary parts"
    : neon
      ? "vextq_f64(z, z, 1)"
      : width === 256
        ? "_mm256_permute_pd(z, 0b0101)"
        : "_mm_shuffle_pd(z, z, 0b01)";
  const multiply = scalar
    ? "multiply scalar parts"
    : neon
      ? "vmulq_f64(c_rr, z)"
      : `${prefix}_mul_pd(c_ii, z_swap)`;
  const fma = scalar
    ? "ar*br - ai*bi; ar*bi + ai*br"
    : neon
      ? "vfmaq_f64(prod, c_ii_as, z_swap)"
      : `${prefix}_fmaddsub_pd(c_rr, z, t)`;
  const store = scalar
    ? "write Complex64"
    : neon
      ? "vst1q_f64"
      : `${prefix}_storeu_pd`;
  return [
    {
      title: "Prepare coefficients",
      code: scalar
        ? "matrix coefficients"
        : neon
          ? "c_rr = [re, re]; c_ii_as = [-im, im]"
          : `${prefix}_set1_pd(re); ${prefix}_set1_pd(im)`,
      detail: scalar
        ? "Use the complex matrix coefficients directly."
        : diagonal
          ? "Prepare the diagonal coefficient broadcasts before entering the scaling loop."
          : "Broadcast once per prepared gate, before the application loop.",
    },
    {
      title: diagonal ? "Choose independent halves" : "Address the pair",
      code: diagonal
        ? "lo *= d0; hi *= d1"
        : "a = state[i0]; b = state[i0 + (1 << target)]",
      detail: diagonal
        ? "A diagonal gate scales each half independently. When d0 is within 1e-15 of 1+0i, the low half is skipped."
        : "The target bit separates the low and high halves. Both original inputs must remain available until both output rows are computed.",
    },
    {
      title: "Load original amplitudes",
      code: load,
      detail:
        width === 256
          ? "Each register contains two adjacent complex amplitudes from the same half. It does not contain both members of a target pair."
          : scalar
            ? "Read the real and imaginary parts of the selected amplitude."
            : "Each 128-bit register holds one complex amplitude: [real, imaginary].",
    },
    {
      title: "Exchange real / imaginary lanes",
      code: swap,
      detail:
        "Swap within each complex number; neighboring amplitudes stay independent.",
    },
    {
      title: neon
        ? "Multiply real coefficients"
        : "Multiply imaginary coefficients",
      code: multiply,
      detail: neon
        ? "NEON first forms c.re * z. Its imaginary coefficient vector already carries the subtract sign."
        : "Form the cross products. Even lanes will be subtracted and odd lanes added.",
    },
    {
      title: "Finish complex multiplication",
      code: fma,
      detail: scalar
        ? "Evaluate the ordinary complex product."
        : "Real lanes subtract the imaginary cross product; imaginary lanes add it. Browser arithmetic displays the algebra, without reproducing native fused rounding.",
    },
    ...(diagonal
      ? [
          {
            title: "Scale the other half independently",
            code: "complex_mul(d[other], original[other])",
            detail:
              "Each half uses its own diagonal coefficient. There is no sum between rows, and a skipped low half contributes no work.",
          },
        ]
      : [
          {
            title: "Evaluate the other row term",
            code: "complex_mul(u[row, 0], a); complex_mul(u[row, 1], b)",
            detail:
              "Use the other input with the other coefficient. Selecting a coefficient below exposes every lane of either term.",
          },
          {
            title: "Add the two row terms",
            code: scalar
              ? "u[row, 0] * a + u[row, 1] * b"
              : neon
                ? "vaddq_f64(term0, term1)"
                : `${prefix}_add_pd(term0, term1)`,
            detail:
              "This is a lane-wise sum. Each real lane and imaginary lane stays in place.",
          },
          {
            title: "Finish both output rows",
            code: "new_a = u00*a + u01*b; new_b = u10*a + u11*b",
            detail:
              "Compute both output rows from the original loaded amplitudes, before overwriting memory.",
          },
        ]),
    {
      title: "Store the output",
      code: store,
      detail: diagonal
        ? "Write only the independently scaled halves. A skipped low half has no load, multiply, or store."
        : "Store the completed low and high registers back into their original state positions.",
    },
  ];
}
