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
    note: "A pure state stores 2ⁿ complex amplitudes and a density matrix stores 4ⁿ entries. Compressed backends trade that size for assumptions about circuit structure.",
  },
  {
    n: 2,
    title: "Anatomy of a statevector gate",
    ids: ["state-index-pairs", "hardware-state-layout"],
    note: "Interactive. The operator, pair indexing, complex128 layout, the SIMD route PRISM-Q picks, cache assumptions, and a check against native recordings.",
  },
  {
    n: 3,
    title: "Gate fusion and memory bandwidth",
    ids: ["fusion-host-pass-order", "fusion-cache-stage-planner"],
    note: "Interactive. The host fusion pipeline, recorded pass by pass. Which passes run depends on circuit width and cache size, so there is no single fixed order.",
  },
  {
    n: 4,
    title: "Binary symplectic stabilizers",
    ids: [],
    note: "Clifford gates update packed binary X and Z rows plus phase bits. Separate Clifford+T engines handle circuits with a limited number of T gates.",
  },
  {
    n: 5,
    title: "MPS and entanglement complexity",
    ids: [],
    note: "Two-site updates use an SVD with a relative singular-value cutoff and a bond cap. Adding up the locally discarded weight does not bound the final global error.",
  },
  {
    n: 6,
    title: "AVX2 vs NEON for quantum simulation",
    ids: ["hardware-diagonal-route"],
    note: "The lane layout changes with register width. Having AVX2 does not on its own select the 256-bit kernel.",
  },
  {
    n: 7,
    title: "CUDA execution and crossover",
    ids: [],
    note: "Kernel launches, device memory, fusion tiles and transfers each cost something different. Source notes only, with no device timings.",
  },
  {
    n: 8,
    title: "Distributed quantum simulation with MPI",
    ids: [],
    note: "Each rank starts with a contiguous block, so the high bits pick the rank. Qubit relabelling can change that mapping and the communication pattern.",
  },
  {
    n: 9,
    title: "Quantum noise and error correction",
    ids: [],
    note: "Trajectories, density matrices and compiled Pauli sampling answer different questions. A detector event is a change in a stabilizer outcome, not a direct observation of an error.",
  },
  {
    n: 10,
    title: "Why backends have different complexity",
    ids: [],
    note: "Dispatch looks at what the caller asks for, the circuit structure, gate support, memory limits and the requested output.",
  },
  {
    n: 11,
    title: "Reproducible benchmark laboratory",
    ids: [],
    note: "Source notes on the benchmark tooling. This release includes no timing data and makes no speed claims.",
  },
];
