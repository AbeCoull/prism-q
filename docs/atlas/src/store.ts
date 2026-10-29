import { create } from "zustand";
import { gateNames, type GateName } from "./models/quantum";
import type { Architecture } from "./models/hardware";

export const layers = [
  "Mathematics",
  "Algorithm",
  "Memory",
  "Hardware",
  "Validation",
] as const;
export type Layer = (typeof layers)[number];
export type Experiment = {
  n: number;
  target: number;
  gate: GateName;
  theta: number;
  depth: number;
  pairIndex: number;
  phase: number;
  architecture: Architecture;
  layer: Layer;
  basis: boolean;
};
const params = new URLSearchParams(location.search);
const number = (
  key: string,
  fallback: number,
  min: number,
  max: number,
  integer = true,
) => {
  const raw = params.get(key),
    x = raw === null ? fallback : Number(raw);
  return Number.isFinite(x)
    ? Math.max(min, Math.min(max, integer ? Math.round(x) : x))
    : fallback;
};
const n = number("n", 4, 2, 16);
const initial: Experiment = {
  n,
  target: number("q", 2, 0, n - 1),
  depth: number("depth", 1, 1, 4),
  theta: number("theta", Math.PI / 3, 0, 2 * Math.PI, false),
  pairIndex: number("pair", 0, 0, 2 ** (n - 1) - 1),
  phase: number("stage", 0, 0, 4),
  gate: gateNames.includes(params.get("gate") as GateName)
    ? (params.get("gate") as GateName)
    : "H",
  architecture: ["avx2", "neon", "scalar"].includes(params.get("arch") ?? "")
    ? (params.get("arch") as Architecture)
    : "avx2",
  layer: layers.includes(params.get("layer") as Layer)
    ? (params.get("layer") as Layer)
    : "Mathematics",
  basis: params.get("state") === "basis",
};
export const useExperiment = create<
  Experiment & { set: (patch: Partial<Experiment>) => void }
>((set) => ({
  ...initial,
  set: (patch) =>
    set((state) => {
      const next = { ...state, ...patch };
      next.target = Math.min(next.target, next.n - 1);
      next.pairIndex = Math.min(next.pairIndex, 2 ** (next.n - 1) - 1);
      return next;
    }),
}));
export function experimentUrl(
  state: Experiment,
  renderer: "3d" | "svg",
): string {
  const url = new URL(location.href);
  for (const [key, value] of Object.entries({
    n: `${state.n}`,
    q: `${state.target}`,
    gate: state.gate,
    theta: `${state.theta}`,
    depth: `${state.depth}`,
    pair: `${state.pairIndex}`,
    stage: `${state.phase}`,
    arch: state.architecture,
    layer: state.layer,
    state: state.basis ? "basis" : "fixture",
    renderer,
  }))
    url.searchParams.set(key, value);
  url.hash = "gate-laboratory";
  return url.href;
}
