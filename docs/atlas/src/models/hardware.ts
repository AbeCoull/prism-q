export type Architecture = "avx2" | "neon" | "scalar";
export function kernelRoute(
  n: number,
  target: number,
  architecture: Architecture,
  parallel = true,
) {
  const threaded = parallel && n >= 15;
  const avx = architecture === "avx2" && target >= 2 && (n <= 13 || threaded);
  const width = architecture === "scalar" ? 0 : avx ? 256 : 128;
  return {
    width,
    complexPerRegister: width / 128,
    name:
      architecture === "scalar"
        ? "Scalar complex128"
        : architecture === "neon"
          ? "AArch64 NEON"
          : avx
            ? "AVX2 + FMA"
            : "128-bit FMA",
    threaded,
    partition: !threaded
      ? "Full sequential state"
      : target <= 12
        ? "Rayon: 8192-amplitude tiles"
        : "Rayon: paired low / high slices",
    reason:
      architecture !== "avx2"
        ? architecture === "neon"
          ? "NEON is baseline on AArch64; one complex number per 128-bit vector."
          : "Portable arithmetic model; no SIMD lanes."
        : target < 2
          ? "Target < 2: this general gate path chooses 128-bit FMA."
          : n === 14
            ? "At 14 qubits, the full sequential state exceeds 8192 amplitudes and Rayon has not started."
            : !parallel && n > 13
              ? "Without parallel, the full sequential state exceeds the 8192-amplitude AVX2 limit."
              : "Target ≥ 2 and the sequential slice is within 8192 amplitudes, or the parallel slice path is active.",
  };
}
export const stateBytes = (n: number, precision = 16, density = false) =>
  precision * 2 ** (n * (density ? 2 : 1));
export const sweepBytes = (n: number) => 32 * 2 ** n;
export function diagonalRoute(
  n: number,
  target: number,
  architecture: Architecture,
  parallel = true,
) {
  const threaded = parallel && n >= 15;
  const width =
    architecture === "scalar"
      ? 0
      : architecture === "avx2" && threaded && target >= n - 2
        ? 256
        : 128;
  return {
    width,
    complexPerRegister: width / 128,
    threaded,
    name:
      architecture === "scalar"
        ? "Scalar diagonal"
        : architecture === "neon"
          ? "NEON diagonal"
          : width === 256
            ? "AVX2 diagonal scaling"
            : "128-bit FMA diagonal",
    partition: !threaded
      ? "Full sequential state"
      : target >= n - 2
        ? "Rayon: 4096-amplitude scaling slices"
        : "Rayon: diagonal tiles",
    reason:
      "Diagonal entries scale each half independently. The AVX2 bulk scaler is used only by the parallel high-target branch; no cross-row addition is needed.",
  };
}
export const bandwidthSeconds = (n: number, passes: number, gbps: number) =>
  (sweepBytes(n) * passes) / (gbps * 1e9);
export function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
  const at = Math.min(
    Math.floor(Math.log2(Math.max(bytes, 1)) / 10),
    units.length - 1,
  );
  return `${(bytes / 1024 ** at).toLocaleString("en-US", { maximumFractionDigits: 2 })} ${units[at]}`;
}
export function rankOwnership(index: number, n: number, ranks: number) {
  const localCount = 2 ** n / ranks;
  return { rank: Math.floor(index / localCount), local: index % localCount };
}
