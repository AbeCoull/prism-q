import { expect, it } from "vitest";
import { cancellationCounterexample } from "../src/models/rewrite";

it("an intervening CX prevents cancellation of control X gates", () => {
  const result = cancellationCounterexample(0);
  expect(Array.from(result.valid)).toEqual([0, 0, 0, 0, 1, 0, 0, 0]);
  expect(Array.from(result.invalid)).toEqual([1, 0, 0, 0, 0, 0, 0, 0]);
  expect(result.fidelity).toBe(0);
});

it("an X eigenstate can hide an invalid operator rewrite on a single input", () => {
  const result = cancellationCounterexample(Math.PI / 4);
  expect(result.fidelity).toBeCloseTo(1, 14);
  expect(result.validNorm).toBeCloseTo(1, 14);
  expect(result.invalidNorm).toBeCloseTo(1, 14);
});

it("fidelity follows the independently derived sin squared of twice the input angle", () => {
  for (const angle of [0, 0.1, Math.PI / 8, Math.PI / 4, 1.2, Math.PI / 2]) {
    const result = cancellationCounterexample(angle);
    expect(result.fidelity).toBeCloseTo(Math.sin(2 * angle) ** 2, 14);
    expect(result.validNorm).toBeCloseTo(1, 14);
    expect(result.invalidNorm).toBeCloseTo(1, 14);
  }
});
