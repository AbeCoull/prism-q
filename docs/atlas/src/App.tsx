import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { scaleLog } from "d3-scale";
import type nativeFixture from "../fixtures/native-gates.json";
import { Equation } from "./components/Equation";
import {
  chapters,
  evidence,
  revision,
  sourceUrl,
  type Evidence,
} from "./data/evidence";
import {
  abs2,
  amplitude,
  applyGate,
  applyPair,
  binary,
  cacheLine,
  complexText,
  gateNames,
  initialState,
  matrixFor,
  maxError,
  mul,
  norm,
  pairs,
} from "./models/quantum";
import {
  bandwidthSeconds,
  diagonalRoute,
  formatBytes,
  kernelRoute,
  stateBytes,
  sweepBytes,
  type Architecture,
} from "./models/hardware";
import { experimentUrl, layers, useExperiment, type Layer } from "./store";
import AnatomyMemory from "./renderers/AnatomyMemory";
import { SceneBoundary } from "./components/SceneBoundary";
import { ExecutionMicroscope } from "./components/ExecutionMicroscope";
import CompilerLab from "./chapters/CompilerLab";
import { skipsLowHalf } from "./models/simd-execution";

const MemoryScene = lazy(() => import("./renderers/MemoryScene"));
const stages = [
  "Pair indices",
  "Load inputs",
  "Transform",
  "Store outputs",
  "Verify state",
];
const stageNotes = [
  "Insert a zero at the target bit to get i, and set it to get j. Pairs never overlap, so each amplitude is written once.",
  "Load both inputs before writing either. Each register holds real and imaginary parts side by side.",
  "Each output is one matrix row applied to both inputs. A diagonal matrix has no cross terms and takes a separate scaling path.",
  "Write the results back to the same two indices. The array layout does not change.",
  "Compare the whole state with the native PRISM-Q recording, when one exists, and check the norm separately.",
];

function SourceLink({ id, label }: { id: string; label?: string }) {
  const entry = evidence.find((item) => item.id === id);
  return entry ? (
    <a
      className="source-link"
      href={sourceUrl(entry.sources[0])}
      target="_blank"
      rel="noreferrer"
    >
      {label ?? entry.sources[0].symbol} <span aria-hidden="true">↗</span>
    </a>
  ) : null;
}

function Range({
  label,
  value,
  min,
  max,
  onChange,
  suffix = "",
  step = 1,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  suffix?: string;
  step?: number;
}) {
  return (
    <label className="range-control">
      <span>
        {label}
        <output>
          {Number.isInteger(value) ? value : value.toFixed(2)}
          {suffix}
        </output>
      </span>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Matrix({ values }: { values: ReturnType<typeof matrixFor> }) {
  return (
    <div className="matrix" aria-label="Unitary matrix">
      {values.map((value, index) => (
        <span key={index}>{complexText(value, 3)}</span>
      ))}
    </div>
  );
}

function AmplitudePlot({
  before,
  after,
  n,
  pair,
  onSelect,
}: {
  before: ArrayLike<number>;
  after: ArrayLike<number>;
  n: number;
  pair: [number, number];
  onSelect: (i: number) => void;
}) {
  const [page, setPage] = useState(0);
  useEffect(() => setPage(Math.floor(pair[0] / 32)), [pair[0]]);
  const total = 2 ** n,
    currentPage = Math.min(page, Math.ceil(total / 32) - 1),
    start = currentPage * 32,
    count = Math.min(32, total - start);
  let ceiling = 1e-16;
  for (let k = 0; k < count; k++)
    ceiling = Math.max(
      ceiling,
      abs2(amplitude(before, start + k)),
      abs2(amplitude(after, start + k)),
    );
  return (
    <section className="amplitude-panel" aria-labelledby="amplitudes-title">
      <div className="section-top">
        <h3 id="amplitudes-title">Resulting state</h3>
        <span className="micro">Probability |αⱼ|²</span>
      </div>
      <div
        className="probability-chart"
        role="group"
        aria-label="Basis-state probabilities"
      >
        {Array.from({ length: count }, (_, k) => {
          const index = start + k,
            old = abs2(amplitude(before, index)),
            next = abs2(amplitude(after, index));
          return (
            <button
              key={index}
              className={`prob-column ${pair.includes(index) ? "selected" : ""}`}
              aria-label={`Inspect amplitude ${index}, probability ${next.toFixed(6)}`}
              onClick={() => onSelect(index)}
              title={`|${binary(index, n)}⟩: ${next.toPrecision(5)}`}
            >
              <span className="prob-bars">
                <i style={{ height: `${(100 * old) / ceiling}%` }} />
                <b style={{ height: `${(100 * next) / ceiling}%` }} />
              </span>
              <small>{index}</small>
            </button>
          );
        })}
      </div>
      <div className="plot-footer">
        <span>
          <i className="key old" />
          Input <i className="key new" />
          Output · y-axis max {ceiling.toPrecision(3)}
        </span>
        {total > 32 && (
          <span>
            <button
              aria-label="Previous amplitude page"
              disabled={currentPage === 0}
              onClick={() => setPage(currentPage - 1)}
            >
              ←
            </button>{" "}
            {start}–{start + count - 1} / {total}{" "}
            <button
              aria-label="Next amplitude page"
              disabled={start + count >= total}
              onClick={() => setPage(currentPage + 1)}
            >
              →
            </button>
          </span>
        )}
      </div>
    </section>
  );
}

function HardwarePanel({ diagonal }: { diagonal: boolean }) {
  const e = useExperiment();
  const [probeN, setProbeN] = useState(e.n),
    [probeQ, setProbeQ] = useState(e.target),
    [parallel, setParallel] = useState(true);
  useEffect(() => {
    setProbeN(e.n);
    setProbeQ(e.target);
  }, [e.n, e.target]);
  const route = (diagonal ? diagonalRoute : kernelRoute)(
    probeN,
    Math.min(probeQ, probeN - 1),
    e.architecture,
    parallel,
  );
  return (
    <div className="hardware-content">
      <div>
        <h3>
          Kernel route: <span>{route.name}</span>
        </h3>
        <p>{route.reason}</p>
        <p className="technical">{route.partition}</p>
        <SourceLink
          id={diagonal ? "hardware-diagonal-route" : "hardware-primary-route"}
          label="Kernel source"
        />
      </div>
      <div className="probe-controls">
        <Range
          label="Kernel probe width"
          value={probeN}
          min={2}
          max={26}
          onChange={(n) => {
            setProbeN(n);
            setProbeQ(Math.min(probeQ, n - 1));
          }}
          suffix=" q"
        />
        <Range
          label="Kernel probe target"
          value={Math.min(probeQ, probeN - 1)}
          min={0}
          max={probeN - 1}
          onChange={setProbeQ}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={parallel}
            onChange={(x) => setParallel(x.target.checked)}
          />
          Built with the parallel feature
        </label>
        <p className="fine">
          This evaluates the routing rule only; no state is allocated. The CPU
          choice is an assumption, not a detection of this machine.
        </p>
      </div>
    </div>
  );
}

function PerformanceModel() {
  const [n, setN] = useState(25),
    [passes, setPasses] = useState(8),
    [group, setGroup] = useState(4),
    [bandwidth, setBandwidth] = useState(40),
    [peak, setPeak] = useState(400),
    [cache, setCache] = useState(32);
  const fused = Math.ceil(passes / group),
    bytes = sweepBytes(n),
    intensity = 14 / 32;
  const x = scaleLog().domain([0.01, 100]).range([54, 600]);
  const y = scaleLog().domain([0.1, 10000]).range([220, 20]);
  const points = Array.from(
    { length: 100 },
    (_, i) => 10 ** (-2 + (4 * i) / 99),
  );
  const roof = points
    .map(
      (ai, i) =>
        `${i ? "L" : "M"}${x(ai)},${y(Math.min(peak, bandwidth * ai))}`,
    )
    .join(" ");
  return (
    <section id="bandwidth-model" className="model-section">
      <div className="section-top">
        <div>
          <h2>How many bytes a gate moves</h2>
        </div>
      </div>
      <p className="section-intro">
        One sweep over a complex128 state reads and writes 32 × 2ⁿ bytes. The
        lower bound below assumes each sweep streams the whole state from one
        memory level. It ignores allocation, launch overhead, cache reuse, and
        the diagonal skip.
      </p>
      <div className="model-grid">
        <div className="model-controls">
          <Range
            label="Model qubit count"
            value={n}
            min={2}
            max={34}
            onChange={setN}
          />
          <Range
            label="Gate passes"
            value={passes}
            min={1}
            max={32}
            onChange={setPasses}
          />
          <Range
            label="Fusion group size"
            value={group}
            min={1}
            max={8}
            onChange={setGroup}
          />
          <Range
            label="Effective bandwidth"
            value={bandwidth}
            min={5}
            max={200}
            onChange={setBandwidth}
            suffix=" GB/s"
          />
          <Range
            label="Compute ceiling"
            value={peak}
            min={20}
            max={2000}
            step={20}
            onChange={setPeak}
            suffix=" GFLOP/s"
          />
          <Range
            label="Cache capacity"
            value={cache}
            min={1}
            max={128}
            onChange={setCache}
            suffix=" MiB"
          />
        </div>
        <div>
          <div className="model-readouts">
            <div>
              <span>Amplitude payload</span>
              <strong>{formatBytes(stateBytes(n))}</strong>
            </div>
            <div>
              <span>Unfused lower bound</span>
              <strong>
                {(bandwidthSeconds(n, passes, bandwidth) * 1e3).toFixed(3)}{" "}
                <small>ms</small>
              </strong>
            </div>
            <div>
              <span>Grouped lower bound</span>
              <strong>
                {(bandwidthSeconds(n, fused, bandwidth) * 1e3).toFixed(3)}{" "}
                <small>ms</small>
              </strong>
            </div>
          </div>
          <svg
            className="roofline"
            viewBox="0 0 640 280"
            role="img"
            aria-label="Theoretical roofline: bandwidth slope capped by compute ceiling"
          >
            <g className="chart-grid">
              {[0.1, 1, 10, 100, 1000, 10000].map((v) => (
                <g key={v}>
                  <path d={`M54 ${y(v)}H610`} />
                  <text x="44" y={y(v) + 4} textAnchor="end">
                    {v}
                  </text>
                </g>
              ))}
              {[0.01, 0.1, 1, 10, 100].map((v) => (
                <g key={v}>
                  <path d={`M${x(v)} 20V220`} />
                  <text x={x(v)} y="241" textAnchor="middle">
                    {v}
                  </text>
                </g>
              ))}
            </g>
            <path d={roof} fill="none" stroke="var(--blue)" strokeWidth="2" />
            <circle
              cx={x(intensity)}
              cy={y(Math.min(peak, bandwidth * intensity))}
              r="5"
              fill="var(--red)"
            />
            <text
              x={x(intensity) + 10}
              y={y(Math.min(peak, bandwidth * intensity)) - 12}
              className="chart-label"
            >
              General 2×2: 0.4375 FLOP/B
            </text>
            <text x="320" y="271" textAnchor="middle" className="axis-label">
              Arithmetic intensity · FLOP / byte (log scale)
            </text>
            <text x="56" y="12" className="axis-label">
              GFLOP/s · log scale
            </text>
          </svg>
          <p className="fine">
            Four complex multiplies (6 FLOPs each) and two complex adds (2 each)
            make 28 FLOPs per pair, or 14 per amplitude. Fusing g general
            one-qubit gates into one sweep gives at most 14g/32 FLOP/B. Dense
            multi-qubit fusion adds more arithmetic on top.
          </p>
          <p className="model-warning">
            {stateBytes(n) <= cache * 2 ** 20
              ? "The state fits in the selected cache, so the DRAM estimate probably overstates traffic. Fitting does not guarantee the state stays resident."
              : "The state is larger than the selected cache, so streaming from memory probably dominates. The model does not predict achieved bandwidth."}{" "}
            {formatBytes(bytes)} per sweep; {passes} sweeps become {fused} when
            grouped.
          </p>
        </div>
      </div>
      <SourceLink
        id="fusion-cache-stage-planner"
        label="The cache stage planner PRISM-Q uses"
      />
    </section>
  );
}

function EvidenceDialog({
  close,
  chapter,
}: {
  close: () => void;
  chapter: number | null;
}) {
  const ref = useRef<HTMLDialogElement>(null),
    previousFocus = useRef(document.activeElement),
    [query, setQuery] = useState("");
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => {
      dialog?.close();
      if (previousFocus.current instanceof HTMLElement)
        previousFocus.current.focus();
    };
  }, []);
  const selected =
    chapter === null ? null : chapters.find((c) => c.n === chapter);
  const relevant = evidence.filter((item) => {
    if (query)
      return `${item.claim} ${item.sources.map((s) => s.path).join(" ")}`
        .toLowerCase()
        .includes(query.toLowerCase());
    if (!selected || selected.n === 2) return true;
    const expressions: Record<number, RegExp> = {
      1: /state-layout|state-bytes|storage/,
      3: /fusion|compiler|gate-payload/,
      4: /stabilizer|clifford/,
      5: /mps|schmidt/,
      6: /simd|avx|neon|lane|diagonal-route/,
      7: /gpu|cuda/,
      8: /distributed|mpi/,
      9: /noise|qec|measurement/,
      10: /dispatch|plan|sparse|factored|tensor/,
      11: /bench|testing|correctness/,
    };
    return expressions[selected.n]?.test(`${item.id} ${item.claim}`) ?? true;
  });
  return (
    <dialog
      ref={ref}
      className="evidence-dialog"
      aria-labelledby="evidence-heading"
      onCancel={close}
      onClick={(e) => {
        if (e.target === ref.current) close();
      }}
    >
      <div className="dialog-top">
        <div>
          <h2 id="evidence-heading">{selected?.title ?? "Source index"}</h2>
        </div>
        <button autoFocus aria-label="Close source evidence" onClick={close}>
          ×
        </button>
      </div>
      {selected && (
        <>
          <p>{selected.note}</p>
          {selected.n !== 2 && (
            <p className="dossier-status">
              This chapter has source notes only. There is no interactive page
              for it yet.
            </p>
          )}
        </>
      )}
      <label className="search">
        Search claims
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="SIMD, MPS, CUDA, dispatch…"
        />
      </label>
      <p className="fine">
        {relevant.length} records, each linked to source lines at a fixed
        revision. No timing data.
      </p>
      <div className="evidence-list">
        {relevant.map((item) => (
          <article key={item.id} id={`evidence-${item.id}`}>
            <span
              className={`pill ${item.status === "model" || item.status === "theory" ? "model" : ""}`}
            >
              {item.status}
            </span>
            <p>{item.claim}</p>
            <p className="fine">{item.derivation}</p>
            <p className="caveat">{item.limitations}</p>
            {item.sources.map((s, index) => (
              <a
                key={index}
                href={sourceUrl(s)}
                target="_blank"
                rel="noreferrer"
              >
                {s.path}:{s.startLine} <span> {s.symbol} ↗</span>
              </a>
            ))}
          </article>
        ))}
      </div>
    </dialog>
  );
}

type NativeGates = typeof nativeFixture;
let nativeRequest: Promise<NativeGates> | null = null;

function useNativeGates() {
  const [fixture, setFixture] = useState<NativeGates | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    nativeRequest ??= fetch(new URL("./native-gates.json", location.href)).then(
      (response) => {
        if (!response.ok)
          throw new Error(`Trace request failed (${response.status})`);
        return response.json();
      },
    );
    let live = true;
    nativeRequest.then(
      (data) => live && setFixture(data),
      (reason: Error) => live && setError(reason.message),
    );
    return () => {
      live = false;
    };
  }, []);
  return { native: fixture, nativeError: error };
}

export default function App() {
  const e = useExperiment();
  const { native, nativeError } = useNativeGates();
  useEffect(() => {
    let mounted = true;
    document.fonts.ready.then(() => {
      if (!mounted) return;
      const target =
        location.hash === "#compiler-laboratory" ||
        location.hash === "#gate-laboratory"
          ? document.getElementById(location.hash.slice(1))
          : null;
      target?.scrollIntoView({ behavior: "instant" });
    });
    return () => {
      mounted = false;
    };
  }, []);
  const [dialog, setDialog] = useState<number | null | false>(false),
    [copied, setCopied] = useState(false),
    [renderer, setRenderer] = useState<"3d" | "svg">(
      new URLSearchParams(location.search).get("renderer") === "svg"
        ? "svg"
        : "3d",
    );
  const matrix = useMemo(() => matrixFor(e.gate, e.theta), [e.gate, e.theta]);
  const diagonal = e.gate === "Z" || e.gate === "Rz";
  const skipLow = skipsLowHalf(matrix, diagonal);
  const allPairs = useMemo(() => pairs(e.n, e.target), [e.n, e.target]);
  const pair = allPairs[e.pairIndex];
  const trace =
    !e.basis && Math.abs(e.theta - Math.PI / 3) < 1e-14
      ? native?.traces.find(
          (t) => t.n === e.n && t.target === e.target && t.gate === e.gate,
        )
      : undefined;
  const result = useMemo(() => {
    let current = initialState(e.n, e.basis),
      previous = current;
    for (let k = 0; k < e.depth; k++) {
      previous = current;
      current = applyGate(current, e.target, matrix);
    }
    const after = trace ? trace.steps[e.depth - 1] : current;
    const before = trace
      ? e.depth === 1
        ? trace.before
        : trace.steps[e.depth - 2]
      : previous;
    return {
      before,
      after,
      predicted: current,
      error: trace ? maxError(current, after) : null,
    };
  }, [e.n, e.basis, e.depth, e.target, matrix, trace]);
  const [i, j] = pair,
    a = amplitude(result.before, i),
    b = amplitude(result.before, j),
    [outA, outB] = applyPair(result.before, matrix, i, j);
  const route = (diagonal ? diagonalRoute : kernelRoute)(
    e.n,
    e.target,
    e.architecture,
  );
  const selectAmplitude = (index: number) => {
    const mask = 2 ** e.target - 1;
    e.set({
      pairIndex: ((index >> (e.target + 1)) << e.target) | (index & mask),
    });
  };
  const fallback = (
    <AnatomyMemory
      n={e.n}
      target={e.target}
      pair={pair}
      phase={e.phase}
      onSelect={selectAmplitude}
      width={route.width}
      skipLow={skipLow}
      architecture={e.architecture}
    />
  );
  const copyLink = async () => {
    const url = experimentUrl(e, renderer);
    try {
      await navigator.clipboard.writeText(url);
      history.replaceState(null, "", url);
      setCopied(true);
    } catch {
      history.replaceState(null, "", url);
      setCopied(true);
    }
  };
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to experiment
      </a>
      <header className="site-header">
        <a
          className="brand"
          href="#main-content"
          aria-label="PRISM-Q atlas home"
        >
          PRISM-Q <span>atlas</span>
        </a>
        <nav aria-label="Primary">
          <a href="#gate-laboratory">Gates</a>
          <a href="#compiler-laboratory">Fusion</a>
          <button onClick={() => setDialog(null)}>
            Sources ({evidence.length})
          </button>
          <a href="#methodology">Methods</a>
          <a
            href="https://github.com/AbeCoull/prism-q"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </nav>
      </header>
      <div className="page-shell">
        <main id="main-content">
          <section className="intro">
            <h1>Applying a one-qubit gate</h1>
            <p className="lede">
              A gate on qubit q mixes the amplitudes of a state two at a time.
              This page follows one of those pairs through the PRISM-Q
              statevector backend: where the two amplitudes sit in memory, how
              the AVX2 and NEON kernels load and combine them, and what the
              fusion passes do to a whole circuit.
            </p>
            <p>
              The settings next to the figure drive everything below it. Where
              PRISM-Q recorded the same case natively, the page checks its own
              arithmetic against that recording.
            </p>
          </section>
          <section
            id="gate-laboratory"
            className="laboratory"
            data-testid="workspace"
          >
            <div className="lab-heading">
              <div>
                <h2>Amplitude pairs and SIMD registers</h2>
              </div>
              <button
                aria-label="Source evidence"
                className="text-button"
                onClick={() => setDialog(null)}
              >
                Sources <span>↗</span>
              </button>
            </div>
            <div className="instrument">
              <div className="scene-section">
                <div className="scene-hud">
                  <span className="pill">
                    {trace ? "Native replay" : "Browser model"}
                  </span>
                  <span>
                    <b style={{ fontWeight: 400 }}>{route.name}</b> <i>·</i>{" "}
                    {formatBytes(stateBytes(e.n))}
                  </span>
                </div>
                <div
                  className="scene-canvas"
                  aria-label="Isometric memory and SIMD register anatomy"
                >
                  {renderer === "svg" ? (
                    fallback
                  ) : (
                    <SceneBoundary fallback={fallback}>
                      <Suspense fallback={fallback}>
                        <MemoryScene
                          n={e.n}
                          target={e.target}
                          pair={pair}
                          phase={e.phase}
                          onSelect={selectAmplitude}
                          architecture={e.architecture}
                          width={route.width}
                          skipLow={skipLow}
                        />
                      </Suspense>
                    </SceneBoundary>
                  )}
                  <div className="scene-annotation">
                    <div>
                      Contiguous complex128 storage
                      <br />
                      <small>
                        Byte offsets from the start of the vector. q₀ is the
                        lowest bit.
                      </small>
                    </div>
                  </div>
                  <div className="scene-coordinate">
                    Pair distance{" "}
                    <strong data-testid="pair-offset">
                      {16 * 2 ** e.target} bytes
                    </strong>
                  </div>
                </div>
                <div className="scene-footer">
                  <span>
                    <i className="key new" />
                    aᵢ <i className="key pair-b" />
                    aⱼ <i className="key old" />
                    64-byte cache lines (modelled)
                  </span>
                  <button
                    className="text-button"
                    onClick={() =>
                      setRenderer(renderer === "3d" ? "svg" : "3d")
                    }
                  >
                    {renderer === "3d" ? "Switch to SVG" : "Switch to 3D"}
                  </button>
                </div>
                <div className="timeline">
                  <button
                    aria-label="Previous stage"
                    disabled={e.phase === 0}
                    onClick={() => e.set({ phase: e.phase - 1 })}
                  >
                    ←
                  </button>
                  {stages.map((stage, index) => (
                    <button
                      key={stage}
                      className={e.phase === index ? "active" : ""}
                      aria-pressed={e.phase === index}
                      onClick={() => e.set({ phase: index })}
                    >
                      <span>{index + 1}</span>
                      {stage}
                    </button>
                  ))}
                  <button
                    aria-label="Next stage"
                    disabled={e.phase === 4}
                    onClick={() => e.set({ phase: e.phase + 1 })}
                  >
                    →
                  </button>
                </div>
                <p className="stage-note" aria-live="polite">
                  {diagonal && e.phase === 1
                    ? "Load each amplitude on its own; a diagonal gate needs no partner. When the low coefficient is exactly 1, that half is skipped."
                    : diagonal && e.phase === 2
                      ? "Multiply each amplitude by its own diagonal entry. There are no cross terms."
                      : stageNotes[e.phase]}
                </p>
              </div>
              <aside
                className="experiment-controls"
                aria-label="Experiment controls"
              >
                <div className="control-title">
                  <h3>Settings</h3>
                </div>
                <Range
                  label="Qubit count"
                  value={e.n}
                  min={2}
                  max={16}
                  onChange={(n) => e.set({ n })}
                  suffix=" q"
                />
                <div className="control-pair">
                  <label>
                    Target qubit
                    <select
                      aria-label="Target qubit"
                      value={e.target}
                      onChange={(x) =>
                        e.set({ target: Number(x.target.value) })
                      }
                    >
                      {Array.from({ length: e.n }, (_, q) => (
                        <option key={q} value={q}>
                          q{q} · stride {2 ** q}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Gate
                    <select
                      aria-label="Gate"
                      value={e.gate}
                      onChange={(x) =>
                        e.set({ gate: x.target.value as typeof e.gate })
                      }
                    >
                      {gateNames.map((g) => (
                        <option key={g}>{g}</option>
                      ))}
                    </select>
                  </label>
                </div>
                {["Rx", "Ry", "Rz"].includes(e.gate) && (
                  <>
                    <Range
                      label="Rotation angle"
                      value={e.theta}
                      min={0}
                      max={2 * Math.PI}
                      step={Math.PI / 24}
                      onChange={(theta) => e.set({ theta })}
                      suffix=" rad"
                    />
                    <button
                      className="text-button"
                      onClick={() => e.set({ theta: Math.PI / 3 })}
                    >
                      Reset to the recorded angle, π/3
                    </button>
                  </>
                )}
                <Range
                  label="Circuit depth"
                  value={e.depth}
                  min={1}
                  max={4}
                  onChange={(depth) => e.set({ depth })}
                />
                <label>
                  Architecture
                  <select
                    aria-label="Architecture"
                    value={e.architecture}
                    onChange={(x) =>
                      e.set({ architecture: x.target.value as Architecture })
                    }
                  >
                    <option value="avx2">x86 · AVX2 + FMA capable</option>
                    <option value="neon">AArch64 · NEON</option>
                    <option value="scalar">Portable · scalar model</option>
                  </select>
                </label>
                <label>
                  Input state
                  <select
                    aria-label="Input state"
                    value={e.basis ? "basis" : "fixture"}
                    onChange={(x) =>
                      e.set({ basis: x.target.value === "basis" })
                    }
                  >
                    <option value="fixture">Complex test state</option>
                    <option value="basis">Basis state |0…0⟩</option>
                  </select>
                </label>
                <div className="precision">
                  <span>Precision</span>
                  <strong>
                    complex128 <small>2 × f64</small>
                  </strong>
                  <span className="fine">The CPU backend state format</span>
                </div>
                <div className="control-stats">
                  <div>
                    <span>Amplitudes</span>
                    <b>{(2 ** e.n).toLocaleString("en-US")}</b>
                  </div>
                  <div>
                    <span>Pairs / gate</span>
                    <b>{(2 ** (e.n - 1)).toLocaleString("en-US")}</b>
                  </div>
                  <div>
                    <span>State norm²</span>
                    <b data-testid="norm">{norm(result.after).toFixed(12)}</b>
                  </div>
                </div>
                <button
                  aria-label="Copy experiment link"
                  className="share-button"
                  onClick={copyLink}
                >
                  {copied
                    ? "Link ready in address bar ✓"
                    : "Copy experiment link"}{" "}
                  <span>↗</span>
                </button>
                <span className="sr-only" role="status">
                  {copied ? "Experiment URL is ready in the address bar." : ""}
                </span>
              </aside>
            </div>
          </section>
          <section className="detail-section">
            <div className="detail-heading">
              <div>
                <h2>The selected pair, step by step</h2>
              </div>
              <div className="pair-picker">
                <label htmlFor="pair-index">Pair ordinal k</label>
                <input
                  id="pair-index"
                  aria-label="Pair ordinal"
                  type="number"
                  min={0}
                  max={allPairs.length - 1}
                  value={e.pairIndex}
                  onChange={(x) => {
                    const value = Number(x.target.value);
                    if (Number.isFinite(value))
                      e.set({
                        pairIndex: Math.max(
                          0,
                          Math.min(allPairs.length - 1, Math.round(value)),
                        ),
                      });
                  }}
                />
                <span>/ {allPairs.length - 1}</span>
              </div>
            </div>
            <div
              className="view-tabs"
              role="tablist"
              aria-label="Visualization layer"
            >
              {layers.map((layer, index) => (
                <button
                  id={`tab-${layer}`}
                  role="tab"
                  aria-selected={e.layer === layer}
                  aria-controls="layer-panel"
                  key={layer}
                  className={e.layer === layer ? "active" : ""}
                  onClick={() => e.set({ layer })}
                  onKeyDown={(event) => {
                    if (
                      event.key === "ArrowRight" ||
                      event.key === "ArrowLeft"
                    ) {
                      event.preventDefault();
                      const next =
                        layers[
                          (index + (event.key === "ArrowRight" ? 1 : 4)) % 5
                        ];
                      e.set({ layer: next });
                      document.getElementById(`tab-${next}`)?.focus();
                    }
                  }}
                  tabIndex={e.layer === layer ? 0 : -1}
                >
                  <span>{String.fromCharCode(65 + index)}</span>
                  {layer}
                </button>
              ))}
            </div>
            <div
              id="layer-panel"
              role="tabpanel"
              aria-labelledby={`tab-${e.layer}`}
              className="layer-panel"
            >
              {e.layer === "Mathematics" && (
                <div className="math-content">
                  <div>
                    <div className="operator-line">
                      <span className="operator-name">
                        {e.gate}
                        {e.gate.startsWith("R") && <small>(θ)</small>}
                      </span>
                      <Matrix values={matrix} />
                    </div>
                    <Equation
                      block
                      tex={String.raw`\begin{pmatrix}\alpha_i'\\\alpha_j'\end{pmatrix}=\begin{pmatrix}u_{00}&u_{01}\\u_{10}&u_{11}\end{pmatrix}\begin{pmatrix}\alpha_i\\\alpha_j\end{pmatrix}`}
                    />
                    <p>
                      U acts on q{e.target} and leaves every other bit alone, so
                      it mixes amplitudes two at a time. As a full operator it
                      is{" "}
                      <Equation
                        tex={String.raw`I_{2^{n-q-1}}\otimes U\otimes I_{2^q}`}
                      />{" "}
                      with bits ordered |qₙ₋₁…q₀⟩.
                    </p>
                    <SourceLink id="state-index-pairs" />
                  </div>
                  <div className="live-calculation">
                    <h3>The selected pair</h3>
                    <div>
                      <span>αᵢ · |{binary(i, e.n)}⟩</span>
                      <code>{complexText(a, 6)}</code>
                    </div>
                    <div>
                      <span>αⱼ · |{binary(j, e.n)}⟩</span>
                      <code>{complexText(b, 6)}</code>
                    </div>
                    <div className="calculation-rule">
                      U × input <span>↓</span>
                    </div>
                    <div className="result-a">
                      <span>αᵢ′</span>
                      <code>{complexText(outA, 6)}</code>
                    </div>
                    <div className="result-b">
                      <span>αⱼ′</span>
                      <code>{complexText(outB, 6)}</code>
                    </div>
                    <p className="fine">
                      Input after {e.depth - 1} applications, output after{" "}
                      {e.depth}. Probability is |α|², not the real part.
                    </p>
                  </div>
                </div>
              )}
              {e.layer === "Algorithm" && (
                <div className="algorithm-content">
                  <div>
                    <pre>
                      <code>{`half = 1 << ${e.target}       // ${2 ** e.target}\nmask = half - 1       // ${2 ** e.target - 1}\nk = ${e.pairIndex}\ni = ((k & ~mask) << 1) | (k & mask)\n  = ${i}\nj = i | half\n  = ${j}\n\na, b = state[i], state[j]\nstate[i] = u00*a + u01*b\nstate[j] = u10*a + u11*b`}</code>
                    </pre>
                    <p className="fine">
                      This is the scalar reference. The SIMD loops walk each
                      half as a contiguous stream but keep the same pairing.
                    </p>
                  </div>
                  <div>
                    <h3>
                      {diagonal
                        ? "Diagonal entries scale independently"
                        : "Read both, then write both"}
                    </h3>
                    <p>
                      With a general matrix each output depends on both inputs,
                      so writing αᵢ′ before computing αⱼ′ gives the wrong
                      answer. Pairs never overlap, so threads can take different
                      pairs without writing to the same amplitude.
                    </p>
                    <p>
                      For target q{e.target} the two streams are {2 ** e.target}{" "}
                      amplitudes apart, but each one moves through memory an
                      element at a time. The pair distance is not the stride of
                      every load.
                    </p>
                    <p className="route-note">
                      {diagonal
                        ? "This matrix is diagonal, so PRISM-Q scales each half on its own. The multiply-add pseudocode above gives the same result."
                        : "This matrix goes through the general one-qubit kernel."}
                    </p>
                    <SourceLink id="state-index-pairs" />
                  </div>
                </div>
              )}
              {e.layer === "Memory" && (
                <div className="memory-content">
                  <div>
                    <h3>Index to byte offset</h3>
                    {pair.map((index, k) => (
                      <div
                        className={`address-row ${k ? "pair-b-text" : "pair-a-text"}`}
                        key={index}
                      >
                        <span>α{k ? "ⱼ" : "ᵢ"}</span>
                        <div className="binary">
                          {binary(index, e.n)
                            .split("")
                            .map((bit, at) => (
                              <b
                                key={at}
                                className={
                                  e.n - at - 1 === e.target ? "target-bit" : ""
                                }
                              >
                                {bit}
                              </b>
                            ))}
                        </div>
                        <code>
                          {index} × 16 = +0x
                          {(index * 16)
                            .toString(16)
                            .toUpperCase()
                            .padStart(4, "0")}
                        </code>
                      </div>
                    ))}
                    <p>
                      A <code>Complex64</code> stores its real part at +16j
                      bytes and its imaginary part at +16j + 8. These are
                      offsets from the start of the vector, not process
                      addresses.
                    </p>
                    <div className="component-strip">
                      <span>
                        re(αᵢ)<small>+{i * 16} B</small>
                      </span>
                      <span>
                        im(αᵢ)<small>+{i * 16 + 8} B</small>
                      </span>
                      <i>…</i>
                      <span>
                        re(αⱼ)<small>+{j * 16} B</small>
                      </span>
                      <span>
                        im(αⱼ)<small>+{j * 16 + 8} B</small>
                      </span>
                    </div>
                  </div>
                  <div>
                    <h3>
                      {cacheLine(i) === cacheLine(j)
                        ? "Same cache line"
                        : "Different cache lines"}
                    </h3>
                    <p>
                      The drawing assumes 64-byte lines and a vector that starts
                      on a line boundary, so four amplitudes share a line. The
                      pair sits on lines {cacheLine(i)} and {cacheLine(j)}. In
                      practice the Vec makes no alignment promise, and the SIMD
                      loads are unaligned.
                    </p>
                    <p>
                      At high targets the two streams are far apart, but each is
                      contiguous and prefetches well. Address distance alone
                      says nothing about cache misses or bandwidth.
                    </p>
                    <SourceLink id="hardware-state-layout" />
                  </div>
                </div>
              )}
              {e.layer === "Hardware" && (
                <>
                  <HardwarePanel diagonal={diagonal} />
                  <div className="intrinsic-panel">
                    <h3>Intrinsics</h3>
                    {diagonal ? (
                      <>
                        <Equation
                          tex={String.raw`\alpha_i'=d_0\alpha_i,\quad \alpha_j'=d_1\alpha_j`}
                          block
                        />
                        <p>
                          Each amplitude is scaled by its own entry, with no
                          cross-row sum. For Z, d₀ = 1, so the low half is
                          skipped. The register width is the diagonal kernel's,
                          not the general kernel's.
                        </p>
                      </>
                    ) : e.architecture === "scalar" ? (
                      <>
                        <pre>
                          {"re(cz) = cr*zr − ci*zi\nim(cz) = cr*zi + ci*zr"}
                        </pre>
                        <p>
                          No SIMD registers on this path. The browser runs the
                          same f64 operations, which says nothing about native
                          scalar speed.
                        </p>
                      </>
                    ) : (
                      <>
                        <div
                          className="lane-grid"
                          style={{
                            gridTemplateColumns: `90px repeat(${route.width / 64}, 1fr)`,
                          }}
                        >
                          <span>register a</span>
                          {(route.width === 256
                            ? [
                                `re(α${i & ~1})`,
                                `im(α${i & ~1})`,
                                `re(α${(i & ~1) + 1})`,
                                `im(α${(i & ~1) + 1})`,
                              ]
                            : [`re(α${i})`, `im(α${i})`]
                          ).map((text, at) => (
                            <b key={at}>
                              <small>lane {at}</small>
                              {text}
                            </b>
                          ))}
                        </div>
                        <pre>
                          {e.architecture === "neon"
                            ? "swap = vextq_f64(z, z, 1)\nproduct = vmulq_f64(cr, z)\ncz = vfmaq_f64(product, [-ci, +ci], swap)"
                            : route.width === 256
                              ? "swap = _mm256_permute_pd(z, 0b0101)\nt = _mm256_mul_pd(ci, swap)\ncz = _mm256_fmaddsub_pd(cr, z, t)"
                              : "swap = _mm_shuffle_pd(z, z, 0b01)\nt = _mm_mul_pd(ci, swap)\ncz = _mm_fmaddsub_pd(cr, z, t)"}
                        </pre>
                        <p>
                          Lane order follows load address. With AVX2 one
                          register holds two neighbouring amplitudes from one
                          half and a second register holds their partners.
                          fmaddsub subtracts in even lanes and adds in odd ones.
                          NEON broadcasts signed imaginary coefficients [−ci,
                          +ci] instead.
                        </p>
                        <div className="complex-product">
                          <span>u₀₀ × αᵢ =</span>
                          <code>{complexText(mul(matrix[0], a), 6)}</code>
                        </div>
                      </>
                    )}
                  </div>
                </>
              )}
              {e.layer === "Validation" && (
                <div className="validation-content">
                  <div>
                    <h3>
                      {trace
                        ? "Matches the native recording"
                        : "No native recording for this case"}
                    </h3>
                    <div className="validation-metrics">
                      <div>
                        <span>maxⱼ |αⱼ(model) − αⱼ(native)|</span>
                        <strong>
                          {result.error === null
                            ? "Not available"
                            : result.error.toExponential(3)}
                        </strong>
                      </div>
                      <div>
                        <span>|Σⱼ |αⱼ|² − 1|</span>
                        <strong>
                          {Math.abs(norm(result.after) - 1).toExponential(3)}
                        </strong>
                      </div>
                    </div>
                    <p>
                      Recordings cover 2 to 6 qubits, every target, seven gates
                      and four repeated steps at θ = π/3. Sampled 16-qubit runs
                      of H, Z and Rz cover the threaded tile, paired slice and
                      diagonal paths. The exporter calls{" "}
                      <code>StatevectorBackend::apply</code> with{" "}
                      <code>Gate::Fused</code>. Z and Rz take the diagonal
                      kernel. The other gates take the general 2×2 kernel, even
                      at angles where their matrix happens to be diagonal.
                    </p>
                    <p>
                      Tests compare each recorded step with this page's model
                      and with a dense Kronecker operator built separately. FMA
                      contraction can change the last bits, so agreement is
                      checked to 10⁻¹² rather than bit for bit.
                    </p>
                  </div>
                  <div className="provenance-card">
                    <h3>Recording details</h3>
                    <dl>
                      <dt>Simulator</dt>
                      <dd>
                        {native
                          ? `PRISM-Q ${native.provenance.crateVersion}`
                          : nativeError || "Loading native fixture"}
                      </dd>
                      <dt>Revision</dt>
                      <dd>{native?.provenance.sourceRevision.slice(0, 12)}</dd>
                      <dt>Audit revision</dt>
                      <dd>{native?.provenance.auditRevision.slice(0, 12)}</dd>
                      <dt>Recording changes</dt>
                      <dd>
                        {native?.provenance.dirtyTree
                          ? "Uncommitted changes, hashed"
                          : "Clean checkout"}
                      </dd>
                      <dt>Execution</dt>
                      <dd>Native Rust · complex128</dd>
                      <dt>Seed</dt>
                      <dd>42; the input state is a fixed formula</dd>
                      <dt>Trace</dt>
                      <dd>{trace?.id ?? "None for this case"}</dd>
                      <dt>Timing</dt>
                      <dd>Not collected</dd>
                    </dl>
                    <a href="./native-gates.json" download>
                      Download the recordings ↓
                    </a>
                    <p className="fine">
                      The architecture menu changes the kernel this page draws.
                      It does not rerun the recording, and the recording does
                      not say which ISA produced it.
                    </p>
                  </div>
                </div>
              )}
            </div>
            <div className="persistent-validation">
              <span>
                {trace
                  ? `Native fixture ${trace.id} · step ${e.depth} · max amplitude error ${result.error?.toExponential(2)}`
                  : "Browser model · no native recording for this case"}
              </span>
              <span data-testid="native-error">
                {result.error !== null
                  ? `Δ ≤ 10⁻¹²: ${result.error <= 1e-12 ? "PASS" : "FAIL"}`
                  : "Not checked against native"}
              </span>
            </div>
          </section>
          <AmplitudePlot
            before={result.before}
            after={result.after}
            n={e.n}
            pair={pair}
            onSelect={selectAmplitude}
          />
          <section className="derivation-section" id="derivation">
            <div>
              <h2>Why amplitudes pair up</h2>
              <p>
                Write the state in the computational basis and group the indices
                that agree on every bit except q. Each group has two members,
                one with bit q clear and one with it set, so a one-qubit gate
                becomes 2ⁿ⁻¹ independent 2×2 products.
              </p>
              <Equation
                block
                tex={String.raw`|\psi\rangle=\sum_{k}\left(\alpha_{i(k)}|0\rangle_q+\alpha_{j(k)}|1\rangle_q\right)\otimes |k\rangle_{\neg q}`}
              />
              <p className="fine">
                The tensor product groups subsystems for the algebra; the stored
                bits are not reordered. View A gives the operator in storage
                order.
              </p>
            </div>
            <div className="derivation-facts">
              <div>
                <span>Pair stride</span>
                <Equation tex={String.raw`s=2^q`} />
                <p>Distance between partners, in amplitudes.</p>
              </div>
              <div>
                <span>Partner index</span>
                <Equation tex={String.raw`j=i\oplus(1\ll q)`} />
                <p>Valid when bit q of i is zero.</p>
              </div>
              <div>
                <span>Work and memory</span>
                <Equation
                  tex={String.raw`T(n)=\Theta(2^n),\quad M(n)=16\cdot2^n\ \mathrm{bytes}`}
                />
                <p>One gate touches every amplitude; each takes 16 bytes.</p>
              </div>
            </div>
          </section>
          <ExecutionMicroscope
            contextLabel={`${e.gate}, θ = ${e.theta.toFixed(4)} rad, application ${e.depth}, pair [${pair.join(", ")}], input: ${e.basis ? "basis |0⟩" : "fixed complex test state"}`}
            onEdit={() => {
              const control = document.querySelector<HTMLSelectElement>(
                'select[aria-label="Gate"]',
              );
              control?.focus({ preventScroll: true });
              control?.scrollIntoView({ block: "center", behavior: "instant" });
            }}
            before={result.before}
            matrix={matrix}
            pair={pair}
            n={e.n}
            target={e.target}
            architecture={e.architecture}
            width={route.width as 0 | 128 | 256}
            diagonal={diagonal}
          />
          <PerformanceModel />
          <CompilerLab />
          <section id="methodology" className="methodology">
            <h2>Where the numbers come from</h2>
            <h3>Source links</h3>
            <p>
              {evidence.length} records tie each claim on this page to source
              lines at{" "}
              <a href={`https://github.com/AbeCoull/prism-q/tree/${revision}`}>
                <code>{revision.slice(0, 7)}</code>
              </a>
              .{" "}
              <button className="inline-link" onClick={() => setDialog(null)}>
                Open the source index
              </button>
              .
            </p>
            <h3>An independent reference</h3>
            <p>
              Native outputs are checked against a dense tensor-product operator
              that shares no code with the page&apos;s pair iterator, so a bug
              in one does not hide in the other.
            </p>
            <h3>What is modelled</h3>
            <p>
              Nothing here runs native SIMD. The 3D view is a diagram of
              addresses and registers, not a CPU layout. Cache size, alignment,
              bandwidth and the roofline ceilings are inputs you set. The page
              reports no timings.
            </p>
            <h3>Other chapters</h3>
            <p>These have source notes but no interactive page yet.</p>
            <ul className="chapter-list">
              {chapters
                .filter((chapter) => ![2, 3].includes(chapter.n))
                .map((chapter) => (
                  <li key={chapter.n}>
                    <button
                      className="inline-link"
                      onClick={() => setDialog(chapter.n)}
                    >
                      {chapter.title}
                    </button>
                  </li>
                ))}
            </ul>
            <p className="fine credits">
              The SVG figure uses geometry from{" "}
              <a
                href="https://github.com/wheresryan22/anatomy"
                target="_blank"
                rel="noreferrer"
              >
                Anatomy
              </a>
              , the 3D view uses React Three Fiber, and equations use KaTeX. The
              SVG view works without WebGL.{" "}
              <a href="https://github.com/AbeCoull/prism-q/tree/main/docs/atlas">
                How to regenerate and extend the atlas
              </a>
              .
            </p>
          </section>
        </main>
      </div>
      {dialog !== false && (
        <EvidenceDialog close={() => setDialog(false)} chapter={dialog} />
      )}
    </>
  );
}
