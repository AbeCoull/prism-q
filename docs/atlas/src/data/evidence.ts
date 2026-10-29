import mathematics from "./math-evidence.json";
import runtime from "./runtime-evidence.json";
import hardware from "./hardware-evidence.json";

export const revision = "7adde35d0bacaff71711148223ac8df06a008d01";
export type Source = {
  path: string;
  symbol: string;
  startLine: number;
  endLine: number;
};
export type Evidence = {
  id: string;
  chapter: string | number;
  claim: string;
  status: string;
  sources: Source[];
  derivation: string;
  limitations: string;
};
export const evidence = [...mathematics, ...runtime, ...hardware] as Evidence[];
export const sourceUrl = (source: Source) =>
  `https://github.com/AbeCoull/prism-q/blob/${revision}/${source.path}#L${source.startLine}-L${source.endLine}`;

export const chapters = [
  {
    n: 1,
    title: "The exponential state space",
    ids: ["state-layout", "hardware-state-bytes"],
    note: "Pure states store 2ⁿ complex amplitudes. Mixed-state density matrices store 4ⁿ entries; compressed representations exploit circuit structure.",
  },
  {
    n: 2,
    title: "Anatomy of a statevector gate",
    ids: ["state-index-pairs", "hardware-state-layout"],
    note: "A complete interactive chapter: operator, pair indexing, complex128 storage, implemented SIMD routing, cache assumptions, and native numerical replay.",
  },
  {
    n: 3,
    title: "Gate fusion and memory bandwidth",
    ids: ["fusion-host-pass-order", "fusion-cache-stage-planner"],
    note: "The host pipeline includes algebraic rewrites, width-dependent batching, and cache-derived stage planning. The audited pass order is not a universal static list.",
  },
  {
    n: 4,
    title: "Binary symplectic stabilizers",
    ids: [],
    note: "Clifford gates update packed binary X/Z rows and phase bits. Specialized Clifford+T engines extend the practical boundary beyond Clifford-only circuits.",
  },
  {
    n: 5,
    title: "MPS and entanglement complexity",
    ids: [],
    note: "Two-site contractions use SVD with relative singular-value cutoff and a bond cap. Cumulative local discarded weights do not certify final global infidelity.",
  },
  {
    n: 6,
    title: "AVX2 vs NEON for quantum simulation",
    ids: ["hardware-diagonal-route"],
    note: "The actual array-of-structs lane layouts differ by register width. AVX2 capability alone does not select a 256-bit kernel.",
  },
  {
    n: 7,
    title: "CUDA execution and crossover",
    ids: [],
    note: "Kernel launches, device storage, fusion tiles, and transfers are distinct costs. Source-audited geometry is available; no device timings are asserted here.",
  },
  {
    n: 8,
    title: "Distributed quantum simulation with MPI",
    ids: [],
    note: "Initially contiguous rank ownership maps high physical bits to rank bits. Default qubit relabeling can change the physical mapping and communication schedule.",
  },
  {
    n: 9,
    title: "Quantum noise and error correction",
    ids: [],
    note: "Trajectory, density-matrix, and compiled Pauli sampling have different semantics. Detector events are changes in stabilizer outcomes, not direct measurements of errors.",
  },
  {
    n: 10,
    title: "Why backends have different complexity",
    ids: [],
    note: "Dispatch depends on query semantics, circuit analysis, operation compatibility, limits, and terminal requirements. No synthetic dispatch decision is presented as native output.",
  },
  {
    n: 11,
    title: "Reproducible benchmark laboratory",
    ids: [],
    note: "Benchmark infrastructure is audited. This release contains no imported timing dataset and makes no simulator speedup claim.",
  },
];
