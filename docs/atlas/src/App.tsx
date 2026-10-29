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
  "Insert a zero bit at the target position. Every pair is disjoint, so each output has one owner.",
  "Read both complex inputs before either location is overwritten. A register holds interleaved real and imaginary components.",
  "Form two matrix rows from the original inputs. Diagonal matrices take an independent scaling path.",
  "Write the new amplitudes to their original basis-index locations. The array layout does not change.",
  "Compare the full result with native PRISM-Q replay where available. Check normalization separately from amplitude agreement.",
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
        <h3 id="amplitudes-title">The resulting state</h3>
        <span className="micro">PROBABILITY · |αⱼ|²</span>
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
          Output · local vertical scale: {ceiling.toPrecision(3)}
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
        <p className="eyebrow">IMPLEMENTED ROUTE · SELECTED CPU CAPABILITIES</p>
        <h3>{route.name}</h3>
        <p>{route.reason}</p>
        <p className="technical">{route.partition}</p>
        <SourceLink
          id={diagonal ? "hardware-diagonal-route" : "hardware-primary-route"}
          label="Read the audited kernel evidence"
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
          Built with parallel
        </label>
        <p className="fine">
          Routing probe only. No state is allocated at the probe width. CPU
          choice describes a capability assumption, not detection of the browser
          host.
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
          <p className="eyebrow">ANALYTICAL MODEL · NO TIMING DATA</p>
          <h2>Count the bytes before counting the nanoseconds.</h2>
        </div>
        <span className="pill model">MODEL</span>
      </div>
      <p className="section-intro">
        A full complex128 read/write sweep moves 32 × 2ⁿ logical bytes. This
        lower-bound model assumes the entire state is streamed from the selected
        memory level. It excludes allocation, launch costs, cache reuse, and
        specialized diagonal skips.
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
            <path d={roof} fill="none" stroke="var(--sage)" strokeWidth="2" />
            <circle
              cx={x(intensity)}
              cy={y(Math.min(peak, bandwidth * intensity))}
              r="5"
              fill="var(--copper)"
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
            Four complex multiplies (6 real FLOPs each) and two complex adds (2
            each): 28 FLOPs per pair, or 14 per amplitude. A fused group of g
            general 1q gates has ideal logical intensity 14g/32 FLOP/B if it
            reuses each amplitude in one sweep. Dense multi-qubit fusion can add
            more arithmetic.
          </p>
          <p className="model-warning">
            {stateBytes(n) <= cache * 2 ** 20
              ? "Payload fits the selected cache capacity. A DRAM streaming estimate may overstate external traffic; capacity alone does not establish residency."
              : "Payload exceeds selected cache capacity. Streaming may dominate, but the model does not establish achieved bandwidth."}{" "}
            {formatBytes(bytes)} per full sweep; {passes} → {fused} ideal
            grouped sweeps.
          </p>
        </div>
      </div>
      <SourceLink
        id="fusion-cache-stage-planner"
        label="PRISM-Q uses a cache-derived stage planner"
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
          <p className="eyebrow">SOURCE DOSSIER · {revision.slice(0, 7)}</p>
          <h2 id="evidence-heading">{selected?.title ?? "Source evidence"}</h2>
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
              Source audit complete. The full interactive chapter belongs to a
              subsequent implementation phase.
            </p>
          )}
        </>
      )}
      <label className="search">
        Search implementation claims
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="SIMD, MPS, CUDA, dispatch…"
        />
      </label>
      <p className="fine">
        {relevant.length} records · pinned source links · no benchmark
        measurements in this release
      </p>
      <div className="evidence-list">
        {relevant.map((item) => (
          <article key={item.id} id={`evidence-${item.id}`}>
            <span
              className={`pill ${item.status === "model" || item.status === "theory" ? "model" : ""}`}
            >
              {item.status === "source-audited"
                ? "IMPLEMENTATION"
                : item.status.toUpperCase()}
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
  const [activeChapter, setActiveChapter] = useState(
    location.hash === "#compiler-laboratory" ? 3 : 2,
  );
  useEffect(() => {
    let mounted = true;
    const update = () => {
      const compiler = document.getElementById("compiler-laboratory");
      setActiveChapter(
        compiler && compiler.getBoundingClientRect().top < innerHeight * 0.4
          ? 3
          : 2,
      );
    };
    update();
    document.fonts.ready.then(() => {
      if (!mounted) return;
      const target =
        location.hash === "#compiler-laboratory" ||
        location.hash === "#gate-laboratory"
          ? document.getElementById(location.hash.slice(1))
          : null;
      target?.scrollIntoView({ behavior: "instant" });
      update();
    });
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      mounted = false;
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
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
  const goToChapter = (chapter: number) => {
    if (chapter !== 2 && chapter !== 3) {
      setDialog(chapter);
      return;
    }
    const id = chapter === 3 ? "compiler-laboratory" : "gate-laboratory";
    document.getElementById(id)?.scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "instant"
        : "smooth",
    });
    const url = new URL(location.href);
    url.hash = id;
    history.replaceState(null, "", url);
    setActiveChapter(chapter);
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
          aria-label="Inside PRISM-Q home"
        >
          <svg viewBox="0 0 36 36" aria-hidden="true">
            <path d="M18 4 33 30H3ZM18 4v26M3 30l15-8 15 8" />
          </svg>
          <span>
            PRISM–Q <small>ANATOMY SERIES</small>
          </span>
        </a>
        <nav aria-label="Primary">
          <a className="active" href="#gate-laboratory">
            The atlas
          </a>
          <button onClick={() => setDialog(null)}>
            Source index <span>{evidence.length}</span>
          </button>
          <a href="#methodology">Methodology</a>
        </nav>
        <a
          className="repository"
          href="https://github.com/AbeCoull/prism-q"
          target="_blank"
          rel="noreferrer"
        >
          GitHub <span>↗</span>
        </a>
      </header>
      <nav className="mobile-chapters" aria-label="Mobile chapter contents">
        <label htmlFor="chapter-selector">Contents</label>
        <select
          id="chapter-selector"
          aria-label="Atlas chapter"
          value={activeChapter}
          onChange={(event) => goToChapter(Number(event.target.value))}
        >
          {chapters.map((chapter) => (
            <option key={chapter.n} value={chapter.n}>
              {String(chapter.n).padStart(2, "0")} {chapter.title}
              {[2, 3].includes(chapter.n) ? " (live)" : ""}
            </option>
          ))}
        </select>
      </nav>
      <div className="page-shell">
        <aside className="chapter-rail" aria-label="Atlas chapters">
          <p className="eyebrow">INSIDE PRISM-Q</p>
          <p className="rail-title">
            From quantum algebra
            <br />
            to silicon.
          </p>
          <ol>
            {chapters.map((chapter) => (
              <li key={chapter.n}>
                <button
                  className={chapter.n === activeChapter ? "current" : ""}
                  aria-current={
                    chapter.n === activeChapter ? "location" : undefined
                  }
                  onClick={() => goToChapter(chapter.n)}
                >
                  <span>{`${chapter.n}`.padStart(2, "0")}</span>
                  <span>
                    {chapter.title}
                    {[2, 3].includes(chapter.n) && (
                      <small>INTERACTIVE CHAPTER</small>
                    )}
                  </span>
                  {[2, 3].includes(chapter.n) ? (
                    <b>↗</b>
                  ) : (
                    <b className="dossier-dot" title="Source dossier">
                      ·
                    </b>
                  )}
                </button>
              </li>
            ))}
          </ol>
          <div className="rail-note">
            <span className="status-dot" />
            SOURCE SNAPSHOT
            <br />
            <a href={`https://github.com/AbeCoull/prism-q/tree/${revision}`}>
              {revision.slice(0, 7)} ↗
            </a>
            <p>
              11 source dossiers.
              <br />
              Gate microscope + compiler laboratory.
            </p>
          </div>
        </aside>
        <main id="main-content">
          <section className="intro">
            <div className="breadcrumb">
              <span className="status-dot" /> INTERACTIVE RESEARCH ATLAS{" "}
              <i>/</i> CHAPTER 02
            </div>
            <div className="intro-row">
              <div>
                <h1>
                  One gate.
                  <br />
                  <em>Every amplitude.</em>
                </h1>
                <p className="subtitle">
                  Anatomy of a quantum statevector gate:
                  <br />
                  from a unitary matrix to SIMD instructions.
                </p>
                <nav
                  className="chapter-shortcuts"
                  aria-label="Interactive laboratories"
                >
                  <a href="#gate-laboratory">
                    02 <span>Gate microscope</span> ↓
                  </a>
                  <a href="#compiler-laboratory">
                    03 <span>Compiler laboratory</span> ↓
                  </a>
                </nav>
              </div>
              <div className="intro-note">
                <span className="tiny-index">[ 02 / 11 ]</span>
                <p>
                  A quantum operation is a linear map.
                  <br />A simulator turns it into memory traffic.
                  <br />
                  Trace the mechanism, one pair at a time.
                </p>
                <a href="#gate-laboratory">
                  Enter the experiment <span>↓</span>
                </a>
              </div>
            </div>
          </section>
          <section
            id="gate-laboratory"
            className="laboratory"
            data-testid="workspace"
          >
            <div className="lab-heading">
              <div>
                <span className="fig-number">FIG. 02.1</span>
                <h2>Anatomy of a statevector gate</h2>
              </div>
              <button
                aria-label="Source evidence"
                className="text-button"
                onClick={() => setDialog(null)}
              >
                Source evidence <span>↗</span>
              </button>
            </div>
            <div className="instrument">
              <div className="scene-section">
                <div className="scene-hud">
                  <span className="pill">
                    {trace ? "NATIVE REPLAY" : "EDUCATIONAL MODEL"}
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
                    <span className="annotation-index">01</span>
                    <div>
                      CONTIGUOUS COMPLEX128 STORAGE
                      <br />
                      <small>
                        Relative byte offsets · q₀ is the least significant bit
                      </small>
                    </div>
                  </div>
                  <div className="scene-coordinate">
                    PAIR DISTANCE{" "}
                    <strong data-testid="pair-offset">
                      {16 * 2 ** e.target} bytes
                    </strong>
                  </div>
                </div>
                <div className="scene-footer">
                  <span>
                    <i className="key new" />
                    aᵢ <i className="key copper" />
                    aⱼ <i className="key old" />
                    64-byte model lines
                  </span>
                  <button
                    className="text-button"
                    onClick={() =>
                      setRenderer(renderer === "3d" ? "svg" : "3d")
                    }
                  >
                    {renderer === "3d" ? "SVG view" : "3D view"} ↗
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
                      <span>{`0${index + 1}`}</span>
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
                  <span>0{e.phase + 1}</span>
                  {diagonal && e.phase === 1
                    ? "Load each amplitude independently for diagonal scaling. A unit diagonal coefficient can skip its half entirely."
                    : diagonal && e.phase === 2
                      ? "Multiply each amplitude by its own diagonal coefficient. No cross-row sum is formed."
                      : stageNotes[e.phase]}
                </p>
              </div>
              <aside
                className="experiment-controls"
                aria-label="Experiment controls"
              >
                <div className="control-title">
                  <span className="status-dot" />
                  <h3>Experiment parameters</h3>
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
                      Restore native angle π/3
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
                    <option value="fixture">Normalized complex fixture</option>
                    <option value="basis">Basis state |0…0⟩</option>
                  </select>
                </label>
                <div className="precision">
                  <span>PRECISION</span>
                  <strong>
                    complex128 <small>2 × f64</small>
                  </strong>
                  <span className="fine">Native CPU state format</span>
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
                <p className="eyebrow">FIVE VIEWS · ONE SELECTED PAIR</p>
                <h2>Follow the same two amplitudes.</h2>
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
                    <p className="eyebrow">A · LINEAR ALGEBRA</p>
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
                      Apply U to target q{e.target}, leaving every other basis
                      bit unchanged. Each pair is a two-dimensional subspace.
                      The complete operator is{" "}
                      <Equation
                        tex={String.raw`I_{2^{n-q-1}}\otimes U\otimes I_{2^q}`}
                      />{" "}
                      in the ordering |qₙ₋₁…q₀⟩.
                    </p>
                    <SourceLink id="state-index-pairs" />
                  </div>
                  <div className="live-calculation">
                    <p className="eyebrow">
                      SELECTED PAIR · LIVE COMPLEX ARITHMETIC
                    </p>
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
                      Input at depth {e.depth - 1}; output at depth {e.depth}.
                      Amplitude phase is retained. Probability is |α|², not the
                      real component.
                    </p>
                  </div>
                </div>
              )}
              {e.layer === "Algorithm" && (
                <div className="algorithm-content">
                  <div>
                    <p className="eyebrow">B · THE PAIR ITERATOR</p>
                    <pre>
                      <code>{`half = 1 << ${e.target}       // ${2 ** e.target}\nmask = half - 1       // ${2 ** e.target - 1}\nk = ${e.pairIndex}\ni = ((k & ~mask) << 1) | (k & mask)\n  = ${i}\nj = i | half\n  = ${j}\n\na, b = state[i], state[j]\nstate[i] = u00*a + u01*b\nstate[j] = u10*a + u11*b`}</code>
                    </pre>
                    <p className="fine">
                      Pseudocode follows the scalar reference. SIMD loops
                      process contiguous half-block streams while preserving
                      this pairing.
                    </p>
                  </div>
                  <div>
                    <p className="eyebrow">DEPENDENCY INVARIANT</p>
                    <h3>
                      {diagonal
                        ? "Diagonal scaling is independent."
                        : "Two reads. Then two writes."}
                    </h3>
                    <p>
                      For a general matrix, both new values depend on both
                      original values. Overwriting αᵢ before forming αⱼ′ would
                      change the operator. Disjoint pairs let different tasks
                      work without writing the same amplitude.
                    </p>
                    <p>
                      With target q{e.target}, the lower and upper streams are
                      separated by {2 ** e.target} amplitudes, but each stream
                      advances sequentially within a block. Pair separation is
                      not the traversal step of every load.
                    </p>
                    <p className="route-note">
                      {diagonal
                        ? "This matrix is diagonal. PRISM-Q detects it and scales each half independently; the full multiply/add pseudocode is a mathematically equivalent reference."
                        : "The selected Fused matrix follows the general one-qubit gate path."}
                    </p>
                    <SourceLink id="state-index-pairs" />
                  </div>
                </div>
              )}
              {e.layer === "Memory" && (
                <div className="memory-content">
                  <div>
                    <p className="eyebrow">
                      C · BINARY INDEX → RELATIVE ADDRESS
                    </p>
                    {pair.map((index, k) => (
                      <div
                        className={`address-row ${k ? "copper-text" : "sage-text"}`}
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
                      Each <code>Complex64</code> occupies adjacent real and
                      imaginary f64 components. Real is at +16j bytes; imaginary
                      is at +16j + 8 bytes. These are offsets from the vector
                      base, not process addresses.
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
                    <p className="eyebrow">CACHE PLACEMENT IS A MODEL</p>
                    <h3>
                      {cacheLine(i) === cacheLine(j)
                        ? "Same modeled cache line."
                        : "Different modeled cache lines."}
                    </h3>
                    <p>
                      The drawing assumes a 64-byte cache line and a
                      line-aligned base, giving four complex128 amplitudes per
                      line. Selected lines: {cacheLine(i)} and {cacheLine(j)}.
                      PRISM-Q uses unaligned SIMD loads; its Vec does not
                      promise this cache-line alignment.
                    </p>
                    <p>
                      At high targets, two distant but individually contiguous
                      streams can still prefetch effectively. This view does not
                      infer cache misses, TLB misses, residency, or achieved
                      bandwidth from address distance.
                    </p>
                    <SourceLink id="hardware-state-layout" />
                  </div>
                </div>
              )}
              {e.layer === "Hardware" && (
                <>
                  <HardwarePanel diagonal={diagonal} />
                  <div className="intrinsic-panel">
                    <p className="eyebrow">D · INTRINSIC-LEVEL MECHANISM</p>
                    {diagonal ? (
                      <>
                        <Equation
                          tex={String.raw`\alpha_i'=d_0\alpha_i,\quad \alpha_j'=d_1\alpha_j`}
                          block
                        />
                        <p>
                          Independent complex scaling replaces the cross-row
                          sum. For Z, d₀ = 1, so the lower half is skipped. The
                          register width follows the diagonal specialization,
                          not the generic matrix route.
                        </p>
                      </>
                    ) : e.architecture === "scalar" ? (
                      <>
                        <pre>
                          {"re(cz) = cr*zr − ci*zi\nim(cz) = cr*zi + ci*zr"}
                        </pre>
                        <p>
                          Scalar arithmetic has no SIMD register layout. The
                          browser evaluates f64 operations; this is not a native
                          scalar microbenchmark.
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
                          Ascending lanes correspond to ascending loaded f64
                          addresses. AVX2 holds two adjacent amplitudes from one
                          half, while a second register holds their partners.
                          Even FMA lanes subtract and odd lanes add. NEON
                          instead broadcasts the signed imaginary coefficients
                          [−ci, +ci].
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
                    <p className="eyebrow">E · NUMERICAL EVIDENCE</p>
                    <h3>
                      {trace
                        ? "Native replay agrees with the browser model."
                        : "Educational execution, no matching native fixture."}
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
                      Native fixtures cover n = 2…6, every target, seven gates,
                      and four repeated steps at θ = π/3, plus sampled 16-qubit
                      H, Z, and Rz steps through the threaded tile,
                      paired-slice, and diagonal scaling routes. The exporter
                      executes actual <code>StatevectorBackend::apply</code>{" "}
                      with <code>Gate::Fused</code>. Z and Rz take the diagonal
                      specialization; other named gates take the general 2×2
                      kernel even at angles where their matrix is diagonal.
                    </p>
                    <p>
                      Tests compare every recorded step against both this
                      browser model and an independent dense Kronecker operator.
                      Floating-point arithmetic may differ through FMA
                      contraction; agreement uses a 10⁻¹² amplitude tolerance,
                      not a bitwise guarantee.
                    </p>
                  </div>
                  <div className="provenance-card">
                    <p className="eyebrow">FIXTURE PROVENANCE</p>
                    <dl>
                      <dt>Simulator</dt>
                      <dd>
                        {native
                          ? `PRISM-Q ${native.provenance.crateVersion}`
                          : nativeError || "Loading native fixture"}
                      </dd>
                      <dt>Revision</dt>
                      <dd>{native?.provenance.sourceRevision.slice(0, 12)}</dd>
                      <dt>Audited implementation</dt>
                      <dd>{native?.provenance.auditRevision.slice(0, 12)}</dd>
                      <dt>Recording changes</dt>
                      <dd>
                        {native?.provenance.dirtyTree
                          ? "Uncommitted producer changes fingerprinted"
                          : "Clean recording checkout"}
                      </dd>
                      <dt>Execution</dt>
                      <dd>Native Rust · complex128</dd>
                      <dt>Seed</dt>
                      <dd>42 (input formula is deterministic)</dd>
                      <dt>Trace</dt>
                      <dd>{trace?.id ?? "No recorded match"}</dd>
                      <dt>Timing</dt>
                      <dd>Not collected</dd>
                    </dl>
                    <a href="./native-gates.json" download>
                      Download replay fixtures ↓
                    </a>
                    <p className="fine">
                      Architecture selection changes the instructional kernel
                      model. It does not rerun the native fixture on AVX2 or
                      NEON, and does not assert which ISA produced the recorded
                      amplitudes.
                    </p>
                  </div>
                </div>
              )}
            </div>
            <div className="persistent-validation">
              <span className="status-dot" />
              <span>
                {trace
                  ? `Native fixture ${trace.id} · step ${e.depth} · max amplitude error ${result.error?.toExponential(2)}`
                  : "Browser educational model · no matching native trace"}
              </span>
              <span data-testid="native-error">
                {result.error !== null
                  ? `Δ ≤ 10⁻¹²: ${result.error <= 1e-12 ? "PASS" : "FAIL"}`
                  : "UNVERIFIED BY REPLAY"}
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
              <p className="eyebrow">THE DERIVATION</p>
              <h2>
                One bit chooses the partner.
                <br />
                <em>All other bits stay fixed.</em>
              </h2>
              <p>
                Expand the state in the computational basis. Group all indices
                whose non-target bits agree. Each group contains exactly the two
                target-bit values, so a one-qubit gate reduces to 2ⁿ⁻¹
                independent matrix-vector products.
              </p>
              <Equation
                block
                tex={String.raw`|\psi\rangle=\sum_{k}\left(\alpha_{i(k)}|0\rangle_q+\alpha_{j(k)}|1\rangle_q\right)\otimes |k\rangle_{\neg q}`}
              />
              <p className="fine">
                The displayed tensor product groups logical subsystems; it does
                not reorder the stored index bits. The full ordered operator is
                given in view A.
              </p>
            </div>
            <div className="derivation-facts">
              <div>
                <span>PAIR STRIDE</span>
                <Equation tex={String.raw`s=2^q`} />
                <p>Logical separation in amplitudes.</p>
              </div>
              <div>
                <span>PARTNER INDEX</span>
                <Equation tex={String.raw`j=i\oplus(1\ll q)`} />
                <p>Valid when bit q of i is zero.</p>
              </div>
              <div>
                <span>WORK AND STORAGE</span>
                <Equation
                  tex={String.raw`T(n)=\Theta(2^n),\quad M(n)=16\cdot2^n\ \mathrm{bytes}`}
                />
                <p>Fixed-size one-qubit gate; complex128 amplitude payload.</p>
              </div>
            </div>
          </section>
          <ExecutionMicroscope
            contextLabel={`${e.gate}, θ = ${e.theta.toFixed(4)} rad, application ${e.depth}, pair [${pair.join(", ")}], initial state: ${e.basis ? "basis |0>" : "deterministic complex fixture"}`}
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
            <p className="eyebrow">METHODS, PROVENANCE & LIMITATIONS</p>
            <h2>A diagram is a claim. Make it auditable.</h2>
            <div className="method-grid">
              <article>
                <span>01</span>
                <h3>Implementation evidence</h3>
                <p>
                  {evidence.length} source records cover the eleven planned
                  chapters. Links are pinned to {revision.slice(0, 7)}. Research
                  notes document representations, limits, and the actual runtime
                  branches.
                </p>
                <button className="text-button" onClick={() => setDialog(null)}>
                  Open source-evidence matrix ↗
                </button>
              </article>
              <article>
                <span>02</span>
                <h3>Independent numerical reference</h3>
                <p>
                  Native outputs are checked against a dense tensor-product
                  operator built separately from the browser pair iterator.
                  Separate agents audited runtime and mathematics, with
                  cross-review and an adversarial pass. Hardware prose was
                  self-reviewed by its author.
                </p>
              </article>
              <article>
                <span>03</span>
                <h3>Models stay models</h3>
                <p>
                  No native SIMD runs in this browser. The 3D view is an address
                  and register schematic, not a physical CPU floorplan. Cache
                  sizes, alignment, bandwidth, and roofline ceilings are
                  assumptions. No measured performance or competitor comparison
                  is supplied.
                </p>
              </article>
            </div>
            <p className="credits">
              Isometric SVG geometry uses{" "}
              <a
                href="https://github.com/wheresryan22/anatomy"
                target="_blank"
                rel="noreferrer"
              >
                Anatomy
              </a>
              ; 3D geometry uses React Three Fiber. Equations use KaTeX. A text
              and SVG view remains available without WebGL.{" "}
              <a href="https://github.com/AbeCoull/prism-q/tree/main/docs/atlas">
                Extension and reproduction guide ↗
              </a>
            </p>
          </section>
          <footer>
            <span>
              INSIDE PRISM-Q <i>/</i> FROM QUANTUM ALGEBRA TO SILICON
            </span>
            <span>Source-grounded. Numerically checked.</span>
            <a href="#main-content">Back to top ↑</a>
          </footer>
        </main>
      </div>
      {dialog !== false && (
        <EvidenceDialog close={() => setDialog(false)} chapter={dialog} />
      )}
    </>
  );
}
