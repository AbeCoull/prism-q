export type Complex = readonly [number, number];
export type Matrix2 = readonly [Complex, Complex, Complex, Complex];
export type GateName = "H" | "X" | "Y" | "Z" | "Rx" | "Ry" | "Rz";
export const gateNames: GateName[] = ["H", "X", "Y", "Z", "Rx", "Ry", "Rz"];
export const add = (a: Complex, b: Complex): Complex => [
  a[0] + b[0],
  a[1] + b[1],
];
export const mul = (a: Complex, b: Complex): Complex => [
  a[0] * b[0] - a[1] * b[1],
  a[0] * b[1] + a[1] * b[0],
];
export const abs2 = (a: Complex) => a[0] ** 2 + a[1] ** 2;
export const amplitude = (state: ArrayLike<number>, i: number): Complex => [
  state[2 * i],
  state[2 * i + 1],
];
export const norm = (state: ArrayLike<number>) =>
  Array.from(state).reduce((s, x) => s + x * x, 0);

export function matrixFor(gate: GateName, theta = Math.PI / 3): Matrix2 {
  const c = Math.cos(theta / 2),
    s = Math.sin(theta / 2),
    h = Math.SQRT1_2;
  switch (gate) {
    case "H":
      return [
        [h, 0],
        [h, 0],
        [h, 0],
        [-h, 0],
      ];
    case "X":
      return [
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 0],
      ];
    case "Y":
      return [
        [0, 0],
        [0, -1],
        [0, 1],
        [0, 0],
      ];
    case "Z":
      return [
        [1, 0],
        [0, 0],
        [0, 0],
        [-1, 0],
      ];
    case "Rx":
      return [
        [c, 0],
        [0, -s],
        [0, -s],
        [c, 0],
      ];
    case "Ry":
      return [
        [c, 0],
        [-s, 0],
        [s, 0],
        [c, 0],
      ];
    case "Rz":
      return [
        [c, -s],
        [0, 0],
        [0, 0],
        [c, s],
      ];
  }
}

export function initialState(n: number, basis = false): Float64Array {
  const state = new Float64Array(2 * 2 ** n);
  if (basis) {
    state[0] = 1;
    return state;
  }
  for (let j = 0; j < 2 ** n; j++) {
    state[2 * j] = j + 1;
    state[2 * j + 1] = ((j * 7) % 11) - 5;
  }
  const scale = Math.sqrt(norm(state));
  return state.map((x) => x / scale);
}

export function pairs(n: number, target: number): [number, number][] {
  const stride = 2 ** target,
    out: [number, number][] = [];
  for (let base = 0; base < 2 ** n; base += 2 * stride)
    for (let k = 0; k < stride; k++) out.push([base + k, base + k + stride]);
  return out;
}

export function applyPair(
  state: ArrayLike<number>,
  matrix: Matrix2,
  i: number,
  j: number,
): [Complex, Complex] {
  const a = amplitude(state, i),
    b = amplitude(state, j);
  return [
    add(mul(matrix[0], a), mul(matrix[1], b)),
    add(mul(matrix[2], a), mul(matrix[3], b)),
  ];
}

export function applyGate(
  state: ArrayLike<number>,
  target: number,
  matrix: Matrix2,
): Float64Array {
  const out = Float64Array.from(state),
    n = Math.log2(state.length / 2);
  for (const [i, j] of pairs(n, target)) {
    const [a, b] = applyPair(state, matrix, i, j);
    out.set(a, 2 * i);
    out.set(b, 2 * j);
  }
  return out;
}

export function compose(a: Matrix2, b: Matrix2): Matrix2 {
  return [
    add(mul(a[0], b[0]), mul(a[1], b[2])),
    add(mul(a[0], b[1]), mul(a[1], b[3])),
    add(mul(a[2], b[0]), mul(a[3], b[2])),
    add(mul(a[2], b[1]), mul(a[3], b[3])),
  ];
}

export function complexText(a: Complex, digits = 4): string {
  const re = Math.abs(a[0]) < 0.5 * 10 ** -digits ? 0 : a[0];
  const im = Math.abs(a[1]) < 0.5 * 10 ** -digits ? 0 : a[1];
  if (!im) return re.toFixed(digits);
  return `${re.toFixed(digits)} ${im < 0 ? "−" : "+"} ${Math.abs(im).toFixed(digits)}i`;
}

export const binary = (i: number, n: number) => i.toString(2).padStart(n, "0");
export const byteOffset = (i: number) => 16 * i;
export const cacheLine = (i: number, baseAlignment = 0, lineBytes = 64) =>
  Math.floor((baseAlignment + byteOffset(i)) / lineBytes);

export function maxError(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error("State dimensions differ");
  let error = 0;
  for (let i = 0; i < a.length; i += 2)
    error = Math.max(error, Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]));
  return error;
}
