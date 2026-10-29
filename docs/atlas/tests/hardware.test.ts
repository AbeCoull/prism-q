import { describe, expect, it } from "vitest";
import { diagonalRoute, kernelRoute } from "../src/models/hardware";

describe("diagonal specialization from native block-count dispatch", () => {
  it.each([
    [14, 11, 128, false],
    [14, 12, 128, false],
    [14, 13, 128, false],
    [15, 12, 128, true],
    [15, 13, 256, true],
    [15, 14, 256, true],
    [16, 13, 128, true],
    [16, 14, 256, true],
    [16, 15, 256, true],
    [20, 17, 128, true],
    [20, 18, 256, true],
    [20, 19, 256, true],
  ])(
    "n=%i target=%i uses %i-bit scaling, threaded=%s",
    (n, target, width, threaded) => {
      const route = diagonalRoute(n as number, target as number, "avx2");
      expect(route.width).toBe(width);
      expect(route.threaded).toBe(threaded);
      expect(route.complexPerRegister).toBe((width as number) / 128);
    },
  );

  it("only the fewer-than-four-block branch selects the AVX2 bulk scaler", () => {
    for (let n = 15; n <= 26; n++) {
      for (let target = 0; target < n; target++) {
        const targetBlockCount = 2 ** n / (2 * 2 ** target);
        const route = diagonalRoute(n, target, "avx2");
        expect(route.width).toBe(targetBlockCount < 4 ? 256 : 128);
        expect(route.partition).toContain(
          targetBlockCount < 4 ? "4096" : "diagonal tiles",
        );
      }
    }
  });

  it("disabling the parallel feature leaves every diagonal target on 128-bit FMA", () => {
    for (const n of [14, 15, 16, 20, 26]) {
      for (let target = 0; target < n; target++) {
        const route = diagonalRoute(n, target, "avx2", false);
        expect(route.width).toBe(128);
        expect(route.threaded).toBe(false);
        expect(route.partition).toBe("Full sequential state");
      }
    }
  });

  it("NEON remains one complex value per vector across both parallel branches", () => {
    for (const [n, target] of [
      [14, 12],
      [15, 12],
      [15, 13],
      [20, 19],
    ]) {
      const route = diagonalRoute(n, target, "neon");
      expect(route.width).toBe(128);
      expect(route.complexPerRegister).toBe(1);
      expect(route.name).toBe("NEON diagonal");
    }
  });

  it("scalar scaling reports no SIMD register at any target", () => {
    for (const [n, target] of [
      [14, 12],
      [15, 12],
      [15, 13],
      [20, 19],
    ]) {
      const route = diagonalRoute(n, target, "scalar");
      expect(route.width).toBe(0);
      expect(route.complexPerRegister).toBe(0);
      expect(route.name).toBe("Scalar diagonal");
    }
  });

  it("general matrix and diagonal routes differ on the same small-state target", () => {
    expect(kernelRoute(13, 2, "avx2").width).toBe(256);
    expect(diagonalRoute(13, 2, "avx2").width).toBe(128);
    expect(kernelRoute(15, 2, "avx2").width).toBe(256);
    expect(diagonalRoute(15, 2, "avx2").width).toBe(128);
  });
});
