import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { SceneBoundary } from "../components/SceneBoundary";
import RewriteChallenge from "../components/RewriteChallenge";
import { Equation } from "../components/Equation";
import { revision } from "../data/evidence";
import {
  circuitDag,
  compilerMetrics,
  validateFusionFixture,
  type FusionFixture,
  type FusionInstruction,
  type FusionRecord,
  type FusionStage,
} from "../models/compiler";
import { abs2, binary, complexText } from "../models/quantum";
import { formatBytes } from "../models/hardware";
import "../styles/compiler.css";

const OperatorScene = lazy(() => import("../renderers/OperatorScene"));
const names: Record<string, string> = {
  original: "Original circuit",
  regions: "Guarded bodies",
  cancel: "Cancel independent pairs",
  "recognize-rzz": "Recognize CX–Rz–CX",
  "batch-rzz": "Batch Rzz",
  "fuse-1q": "Compose 1q matrices",
  "reorder-1q": "Reorder legal 1q operations",
  "cancel-after-reorder": "Cancel newly adjacent pairs",
  "refuse-1q": "Compose reordered runs",
  "absorb-2q": "Absorb 1q into 2q",
  "absorb-2q-same-pair-only": "Absorb within same-pair runs",
  "same-pair-2q": "Fuse same-pair blocks",
  "stage-planner": "Plan cache stages",
  "multi-1q": "Tile 1q batches",
  "reorder-2q-tiles": "Reorder 2q tiles",
  "multi-2q": "Tile 2q batches",
  "pauli-rotation-batches": "Batch Pauli rotations",
  "controlled-phases": "Collect controlled phases",
  "diagonal-batch": "Combine diagonal batches",
  "rebatch-1q": "Rebatch after phases",
};
const families: Record<string, { title: string; description: string }> = {
  "legal-disjoint": {
    title: "Disjoint gates can move",
    description:
      "X on q0 is separated by an operation on q1. The qubits are independent, so a legal reorder can expose X² = I. Repeated rotations across all qubits create matrix and tiled fusion opportunities.",
  },
  "blocked-dependency": {
    title: "A CX blocks cancellation",
    description:
      "The two X gates on the control of a CX cannot be canceled by moving through it. The native stream preserves the interaction, while neighboring single-qubit gates can still form dense blocks.",
  },
  "diagonal-pair-runs": {
    title: "Parity phases and pair blocks",
    description:
      "CX–Rz–CX becomes a parity rotation. Diagonal runs and repeated two-qubit pairs expose different batching mechanisms; Host and Device profiles intentionally retain different intermediate representations.",
  },
};
const gateNames: Record<string, string> = {
  fused: "U₂",
  fused2q: "U₄",
  multi_fused: "MULTI 1Q",
  multi2q: "MULTI 2Q",
  multi_2q: "MULTI 2Q",
  diagonal_batch: "DIAGONAL",
  batch_rzz: "RZZ × N",
  batch_phase: "PHASE × N",
};
const gateLabel = (gate: string) => gateNames[gate] ?? gate.toUpperCase();
const phaseText = (re: number, im: number) =>
  re === 0 && im === 0
    ? "undefined at zero"
    : `${((Math.atan2(im, re) * 180) / Math.PI).toFixed(2)}°`;

function Diagram({
  stage,
  width,
  selected,
  select,
  dag,
  inspectable = true,
}: {
  stage: FusionStage;
  width: number;
  selected: number;
  select: (index: number) => void;
  dag: boolean;
  inspectable?: boolean;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const frame = viewport.current;
    const node = frame?.querySelector<SVGGElement>(
      `[data-instruction="${selected}"]`,
    );
    if (!frame || !node) return;
    const box = node.getBoundingClientRect(),
      area = frame.getBoundingClientRect();
    if (box.left < area.left || box.right > area.right)
      frame.scrollLeft += box.left - area.left - 50;
  }, [stage, selected, dag]);
  const graph = useMemo(
    () => circuitDag(stage.instructions, width),
    [stage, width],
  );
  const visibleQubits = [
    ...new Set(
      stage.instructions.flatMap((instruction) => instruction.support),
    ),
  ].sort((a, b) => a - b);
  const row = (q: number) => visibleQubits.indexOf(q);
  const graphWidth = Math.max(
    720,
    (dag ? graph.depth : stage.instructions.length) * 76 + 100,
  );
  const graphHeight = Math.max(220, visibleQubits.length * 38 + 75);
  const coordinates = new Map(
    graph.nodes.map((node) => [
      node.index,
      {
        x: 85 + (dag ? node.layer : node.index) * 76,
        y:
          48 +
          (node.support.reduce((sum, q) => sum + row(q), 0) /
            Math.max(1, node.support.length)) *
            38,
      },
    ]),
  );
  return (
    <div
      className="compiler-diagram"
      ref={viewport}
      tabIndex={inspectable ? undefined : 0}
      role="region"
      aria-label={
        dag
          ? "Qubit-order dependency graph, horizontally scrollable"
          : "Native circuit stream, horizontally scrollable"
      }
    >
      <div
        className="wire-labels"
        aria-hidden="true"
        style={{ height: graphHeight }}
      >
        {visibleQubits.map((q, at) => (
          <span key={q} style={{ top: 48 + at * 38 }}>
            q{q}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${graphWidth} ${graphHeight}`}
        style={{ width: graphWidth, height: graphHeight }}
        role="group"
        aria-label={`${stage.instructions.length} native instructions at ${names[stage.name] ?? stage.name}`}
      >
        <defs>
          <marker
            id={`arrow-${dag ? "dag" : "circuit"}-${stage.name}`}
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 0 8 4 0 8Z" fill="#718b65" />
          </marker>
        </defs>
        {visibleQubits.map((q, at) => (
          <path
            key={q}
            d={`M56 ${48 + at * 38}H${graphWidth - 20}`}
            className={dag ? "wire-guide" : "wire"}
          />
        ))}
        {dag &&
          graph.edges.map((edge) => {
            const a = coordinates.get(edge.from)!,
              b = coordinates.get(edge.to)!;
            return (
              <path
                key={`${edge.from}-${edge.to}`}
                d={`M${a.x + 27} ${a.y}C${a.x + 50} ${a.y} ${b.x - 50} ${b.y} ${b.x - 29} ${b.y}`}
                className={
                  edge.to === selected || edge.from === selected
                    ? "dependency selected"
                    : "dependency"
                }
                markerEnd={`url(#arrow-dag-${stage.name})`}
              >
                <title>Dependency on qubits {edge.qubits.join(", ")}</title>
              </path>
            );
          })}
        {stage.instructions.map((instruction) => {
          const position = coordinates.get(instruction.index)!,
            active = instruction.index === selected;
          const top = 48 + Math.min(...instruction.support.map(row)) * 38,
            bottom = 48 + Math.max(...instruction.support.map(row)) * 38;
          const cx =
            !dag &&
            instruction.gate.toLowerCase() === "cx" &&
            instruction.targets.length === 2;
          return (
            <g
              key={instruction.index}
              role={inspectable ? "button" : undefined}
              tabIndex={inspectable && active ? 0 : -1}
              data-instruction={instruction.index}
              aria-label={`${inspectable ? "Inspect instruction" : "Instruction"} ${instruction.index}: ${instruction.gate}, qubits ${instruction.support.join(", ")}`}
              aria-pressed={inspectable ? active : undefined}
              onClick={
                inspectable ? () => select(instruction.index) : undefined
              }
              onKeyDown={(event) => {
                if (!inspectable) return;
                const direction =
                  event.key === "ArrowRight" || event.key === "ArrowDown"
                    ? 1
                    : event.key === "ArrowLeft" || event.key === "ArrowUp"
                      ? -1
                      : 0;
                if (direction || event.key === "Home" || event.key === "End") {
                  event.preventDefault();
                  const next =
                    event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? stage.instructions.length - 1
                        : Math.max(
                            0,
                            Math.min(
                              stage.instructions.length - 1,
                              instruction.index + direction,
                            ),
                          );
                  select(next);
                  viewport.current
                    ?.querySelector<SVGGElement>(`[data-instruction="${next}"]`)
                    ?.focus({ preventScroll: true });
                }
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  select(instruction.index);
                }
              }}
              className={`circuit-node ${active ? "active" : ""}`}
            >
              {(!dag || active) && instruction.support.length > 1 && (
                <path
                  d={`M${position.x} ${top}V${bottom}`}
                  className="gate-span"
                />
              )}
              {!dag &&
                !cx &&
                instruction.support.length > 1 &&
                instruction.support.map((q) => (
                  <circle
                    key={q}
                    cx={position.x}
                    cy={48 + row(q) * 38}
                    r="3"
                    fill={active ? "#d0e6b9" : "#7f9873"}
                  />
                ))}
              {cx ? (
                <>
                  <circle
                    cx={position.x}
                    cy={48 + row(instruction.targets[0]) * 38}
                    r="4"
                    fill={active ? "#d0e6b9" : "#91a28f"}
                  />
                  <circle
                    cx={position.x}
                    cy={48 + row(instruction.targets[1]) * 38}
                    r="10"
                    fill="var(--bg)"
                    stroke="var(--sage)"
                  />
                  <path
                    d={`M${position.x - 7} ${48 + row(instruction.targets[1]) * 38}h14M${position.x} ${41 + row(instruction.targets[1]) * 38}v14`}
                    stroke="var(--sage)"
                  />
                  <title>
                    Control q{instruction.targets[0]}, target q
                    {instruction.targets[1]}
                  </title>
                </>
              ) : (
                <>
                  <rect
                    x={position.x - 28}
                    y={position.y - 13}
                    width="56"
                    height="26"
                    rx="3"
                  />
                  <text
                    x={position.x}
                    y={position.y + 3}
                    textAnchor="middle"
                    className="gate-text"
                  >
                    {gateLabel(instruction.gate)}
                  </text>
                </>
              )}
              <text
                x={position.x}
                y={graphHeight - 22}
                textAnchor="middle"
                className="instruction-id"
              >
                {instruction.index}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function SweepChart({
  record,
  selected,
  select,
}: {
  record: FusionRecord;
  selected: number;
  select: (index: number) => void;
}) {
  const max = Math.max(
    ...record.stages.map((stage) => stage.metrics.measuredKernelSweeps),
    1,
  );
  return (
    <div
      className="compiler-sweep-chart"
      role="group"
      aria-label="Native kernel traversal counts by compiler stage"
    >
      {record.stages.map((stage, index) => (
        <button
          key={stage.name}
          onClick={() => select(index)}
          aria-label={`${names[stage.name] ?? stage.name}: ${stage.metrics.measuredKernelSweeps} kernel traversals`}
          aria-pressed={selected === index}
          className={selected === index ? "selected" : ""}
        >
          <span className="compiler-sweep-column">
            <b
              style={{
                height: `${Math.max(1, (stage.metrics.measuredKernelSweeps / max) * 100)}%`,
              }}
            />
            <i>{stage.metrics.measuredKernelSweeps}</i>
          </span>
          <small>{String(index).padStart(2, "0")}</small>
        </button>
      ))}
    </div>
  );
}

function Laboratory({ fixture }: { fixture: FusionFixture }) {
  const params = new URLSearchParams(location.search);
  const initial =
    fixture.records.find((record) => record.id === params.get("compiler")) ??
    fixture.records.find(
      (record) =>
        record.width === 14 &&
        record.profile === "Host" &&
        record.family === "diagonal-pair-runs",
    ) ??
    fixture.records[0];
  const [recordId, setRecordId] = useState(initial.id),
    [stageIndex, setStageIndex] = useState(() =>
      params.has("pass")
        ? Math.max(
            0,
            Math.min(
              initial.stages.length - 1,
              Math.trunc(Number(params.get("pass"))) || 0,
            ),
          )
        : initial.stages.length - 1,
    );
  const index = (value: string | null | undefined) =>
    Math.max(0, Math.trunc(Number(value)) || 0);
  const cell = params.get("cell")?.split(",") ?? [];
  const [selectedInstruction, setSelectedInstruction] = useState(
      index(params.get("inst")),
    ),
    [payloadIndex, setPayloadIndex] = useState(index(params.get("entry"))),
    [selectedCell, setSelectedCell] = useState<[number, number]>([
      index(cell[0]),
      index(cell[1]),
    ]),
    [dag, setDag] = useState(params.get("view") === "graph"),
    [copied, setCopied] = useState(false),
    [show3d, setShow3d] = useState(true);
  const [compare, setCompare] = useState(params.get("compare") === "1");
  const record = fixture.records.find((item) => item.id === recordId)!;
  const stage = record.stages[Math.min(stageIndex, record.stages.length - 1)],
    original = record.stages[0];
  const instruction =
    stage.instructions[
      Math.min(selectedInstruction, stage.instructions.length - 1)
    ];
  const payload =
    instruction?.payload[
      Math.min(payloadIndex, instruction.payload.length - 1)
    ];
  const coefficient =
    payload?.matrix[Math.min(selectedCell[0], payload.matrix.length - 1)][
      Math.min(selectedCell[1], payload.matrix.length - 1)
    ];
  const metrics = compilerMetrics(stage, record.width),
    baseline = compilerMetrics(original, record.width);
  const previous = record.stages[Math.max(0, stageIndex - 1)];
  const change =
    stage.metrics.instructionCount - previous.metrics.instructionCount;
  const setRecord = (id: string) => {
    const next = fixture.records.find((item) => item.id === id)!;
    setRecordId(id);
    setStageIndex(next.stages.length - 1);
    setSelectedInstruction(0);
    setPayloadIndex(0);
    setSelectedCell([0, 0]);
  };
  const setStage = (index: number) => {
    setStageIndex(index);
    setSelectedInstruction(0);
    setPayloadIndex(0);
    setSelectedCell([0, 0]);
  };
  const chooseDimension = (
    field: "family" | "width" | "profile",
    value: string | number,
  ) => {
    const exact = fixture.records.find(
      (item) =>
        item[field] === value &&
        (field === "family" || item.family === record.family) &&
        (field === "width" || item.width === record.width) &&
        (field === "profile" || item.profile === record.profile),
    );
    if (exact) setRecord(exact.id);
  };
  const share = async () => {
    const url = new URL(location.href);
    url.searchParams.set("compiler", record.id);
    url.searchParams.set("pass", `${stageIndex}`);
    url.searchParams.set("inst", `${instruction?.index ?? 0}`);
    url.searchParams.set(
      "entry",
      `${payload ? instruction.payload.indexOf(payload) : 0}`,
    );
    url.searchParams.set("cell", selectedCell.join(","));
    url.searchParams.set("view", dag ? "graph" : "circuit");
    url.searchParams.set("compare", compare ? "1" : "0");
    url.hash = "compiler-laboratory";
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url.href);
    } catch {}
    setCopied(true);
  };
  const source = `https://github.com/AbeCoull/prism-q/blob/${stage.source.path.endsWith("fusion_atlas_tests.rs") ? "main" : revision}/${stage.source.path}`;
  return (
    <>
      <div className="compiler-controls">
        <label>
          Circuit experiment
          <select
            aria-label="Compiler circuit"
            value={record.family}
            onChange={(event) => chooseDimension("family", event.target.value)}
          >
            {[...new Set(fixture.records.map((item) => item.family))].map(
              (family) => (
                <option
                  key={family}
                  value={family}
                  disabled={
                    !fixture.records.some(
                      (item) =>
                        item.family === family &&
                        item.width === record.width &&
                        item.profile === record.profile,
                    )
                  }
                >
                  {families[family]?.title ?? family}
                </option>
              ),
            )}
          </select>
        </label>
        <label>
          Native register width
          <select
            aria-label="Compiler width"
            value={record.width}
            onChange={(event) =>
              chooseDimension("width", Number(event.target.value))
            }
          >
            {[...new Set(fixture.records.map((item) => item.width))]
              .sort((a, b) => a - b)
              .map((width) => (
                <option
                  key={width}
                  value={width}
                  disabled={
                    !fixture.records.some(
                      (item) =>
                        item.width === width &&
                        item.family === record.family &&
                        item.profile === record.profile,
                    )
                  }
                >
                  {width} qubits
                </option>
              ))}
          </select>
        </label>
        <label>
          Optimization profile
          <select
            aria-label="Compiler profile"
            value={record.profile}
            onChange={(event) =>
              chooseDimension(
                "profile",
                event.target.value as "Host" | "Device",
              )
            }
          >
            {["Host", "Device"].map((profile) => (
              <option
                key={profile}
                disabled={
                  !fixture.records.some(
                    (item) =>
                      item.profile === profile &&
                      item.family === record.family &&
                      item.width === record.width,
                  )
                }
              >
                {profile}
              </option>
            ))}
          </select>
        </label>
        <button
          className="compiler-share"
          aria-label="Copy compiler experiment link"
          onClick={share}
        >
          {copied ? "Link ready ✓" : "Share this trace ↗"}
        </button>
      </div>
      <div className="compiler-experiment-note">
        <p>{families[record.family]?.description}</p>
        <span className="pill">NATIVE SNAPSHOTS</span>
      </div>
      <p className="fine">
        Unavailable combinations are disabled. Change width first to compare
        families or profiles without changing another experimental variable.
      </p>
      <div className="compiler-metrics">
        <div>
          <span>Logical instructions</span>
          <strong>
            {baseline.logicalInstructions} <i>→</i>{" "}
            {metrics.logicalInstructions}
          </strong>
        </div>
        <div>
          <span>Instrumented kernel traversals</span>
          <strong data-testid="compiler-sweeps">
            {baseline.measuredKernelSweeps} <i>→</i>{" "}
            {metrics.measuredKernelSweeps}
          </strong>
        </div>
        <div>
          <span>Maximum native amplitude error</span>
          <strong className="compiler-error" data-testid="compiler-error">
            {stage.state.maxAbsoluteError.toExponential(2)}
          </strong>
        </div>
        <div>
          <span>In-tile operator applications</span>
          <strong>{metrics.tileApplications}</strong>
        </div>
      </div>
      <div className="compiler-workbench">
        <aside
          className="compiler-pass-list"
          aria-label="Native compiler stages"
        >
          <div className="compiler-pass-title">
            <span>PASS</span>
            <span>IR COUNT</span>
          </div>
          {record.stages.map((pass, index) => (
            <button
              key={pass.name}
              className={stageIndex === index ? "active" : ""}
              aria-pressed={stageIndex === index}
              onClick={() => setStage(index)}
            >
              <span className="pass-ordinal">
                {`${index}`.padStart(2, "0")}
              </span>
              <span>
                {names[pass.name] ?? pass.name}
                {!pass.enabled && (
                  <small>SKIPPED · WIDTH FLOOR {pass.floor}</small>
                )}
              </span>
              <b>{pass.metrics.instructionCount}</b>
            </button>
          ))}
        </aside>
        <div className="compiler-stage-content">
          <div className="compiler-stage-heading">
            <div>
              <p className="eyebrow">
                PASS {`${stageIndex}`.padStart(2, "0")} /{" "}
                {record.stages.length - 1}
              </p>
              <h3>{names[stage.name] ?? stage.name}</h3>
            </div>
            <span className={`pill ${stage.enabled ? "" : "model"}`}>
              {stage.enabled ? "EXECUTED" : "SKIPPED"}
            </span>
          </div>
          <div className="compiler-stage-step">
            <button
              aria-label="Previous compiler pass"
              disabled={stageIndex === 0}
              onClick={() => setStage(stageIndex - 1)}
            >
              ←
            </button>
            <label>
              Compiler pass
              <input
                type="range"
                aria-label="Compiler pass"
                min="0"
                max={record.stages.length - 1}
                value={stageIndex}
                onChange={(event) => setStage(Number(event.target.value))}
              />
            </label>
            <button
              aria-label="Next compiler pass"
              disabled={stageIndex === record.stages.length - 1}
              onClick={() => setStage(stageIndex + 1)}
            >
              →
            </button>
          </div>
          <p className="compiler-stage-explanation">
            {!stage.enabled
              ? `The ${record.width}-qubit stream is below this pass's ${stage.floor}-qubit floor. The recorded stream is unchanged.`
              : stageIndex === 0
                ? "Original unfused circuit. The deterministic initial state is complex and normalized; all subsequent streams are executed against this reference."
                : change < 0
                  ? `${-change} fewer logical instructions than the previous snapshot. A smaller IR count does not by itself establish fewer state traversals or faster execution.`
                  : change > 0
                    ? `${change} additional logical instructions after this transformation. Follow the actual traversal counts rather than equating representation size with cost.`
                    : "Instruction count is unchanged at this stage. Inspect the stream for changed order, matrix coefficients, or batching structure."}
          </p>
          <div className="compiler-stream-heading">
            <span>RECORDED OUTPUT STREAM</span>
            <div>
              <button
                aria-pressed={!dag}
                className={!dag ? "active" : ""}
                onClick={() => setDag(false)}
              >
                Circuit
              </button>
              <button
                aria-pressed={dag}
                className={dag ? "active" : ""}
                onClick={() => setDag(true)}
              >
                Dependencies
              </button>
              <button
                aria-pressed={compare}
                disabled={stageIndex === 0}
                onClick={() => setCompare(!compare)}
              >
                Before / after
              </button>
            </div>
          </div>
          {compare && stageIndex > 0 && (
            <section
              className="compiler-before"
              aria-label="Previous native stream"
            >
              <h4>Before: {names[previous.name] ?? previous.name}</h4>
              <Diagram
                stage={previous}
                width={record.width}
                selected={0}
                select={() => {}}
                dag={dag}
                inspectable={false}
              />
            </section>
          )}
          {compare && stageIndex > 0 && (
            <h4>After: {names[stage.name] ?? stage.name}</h4>
          )}
          <Diagram
            stage={stage}
            width={record.width}
            selected={selectedInstruction}
            select={(index) => {
              setSelectedInstruction(index);
              setPayloadIndex(0);
              setSelectedCell([0, 0]);
            }}
            dag={dag}
          />
          <p className="fine" data-testid="selected-support">
            Selected instruction {instruction?.index}: qubit support [
            {instruction?.support.join(", ")}].{" "}
            {dag
              ? "Incident dependency edges preserve the qubits named in their tooltips."
              : "CX uses a filled control and a circled target; generic spans show fused or batched support."}
          </p>
          {compare && stageIndex > 0 && (
            <p className="fine">
              Previous snapshot: {previous.metrics.instructionCount}{" "}
              instructions, {previous.metrics.measuredKernelSweeps} kernel
              traversals. Current snapshot: {stage.metrics.instructionCount}{" "}
              instructions, {stage.metrics.measuredKernelSweeps} kernel
              traversals. Streams are complete recorded snapshots; corresponding
              positions do not establish instruction ancestry.
            </p>
          )}
          <p className="fine">
            {dag
              ? "Conservative qubit-order dependency graph, derived from this snapshot. Edges preserve same-qubit order; the graph does not claim to reproduce the native planner's commutation analysis."
              : "Columns are native instruction order, not physical duration. Vertical spans mark the complete support of fused or batched instructions."}{" "}
            Scroll within the figure to inspect every instruction. Tab enters
            the selected instruction; arrow keys move between instructions, and
            Home or End reaches the stream boundaries.
          </p>
          <div className="compiler-counter-heading">
            <span>KERNEL TRAVERSALS THROUGH THE PIPELINE</span>
            <b>NATIVE COUNTER · NOT TIMING</b>
          </div>
          <SweepChart record={record} selected={stageIndex} select={setStage} />
          <div className="compiler-source">
            <a href={source} target="_blank" rel="noreferrer">
              {stage.source.symbol} ↗
            </a>
            <span>{stage.source.path}</span>
          </div>
        </div>
      </div>
      {instruction && payload && coefficient && (
        <div className="operator-inspector">
          <div className="operator-inspector-title">
            <div>
              <p className="eyebrow">INSPECT THE ACTUAL MATRIX</p>
              <h3>
                Instruction {instruction.index}: <code>{instruction.gate}</code>
              </h3>
            </div>
            <span className="pill">
              {instruction.kernelSweeps} KERNEL TRAVERSALS
            </span>
          </div>
          <div className="operator-inspector-grid">
            <div className="operator-geometry">
              <div className="operator-geometry-hud">
                <span>COMPLEX OPERATOR LANDSCAPE</span>
                <button
                  className="text-button"
                  onClick={() => setShow3d(!show3d)}
                >
                  {show3d ? "Use matrix table" : "Show 3D operator"}
                </button>
              </div>
              {show3d ? (
                <SceneBoundary
                  fallback={
                    <p className="operator-fallback">
                      WebGL is unavailable. Every coefficient remains accessible
                      in the matrix table.
                    </p>
                  }
                >
                  <Suspense
                    fallback={
                      <p className="operator-fallback">
                        Loading operator geometry…
                      </p>
                    }
                  >
                    <OperatorScene
                      matrix={payload.matrix}
                      selected={selectedCell}
                      onSelect={setSelectedCell}
                    />
                  </Suspense>
                </SceneBoundary>
              ) : (
                <div className="operator-table-intro">
                  <Equation block tex={String.raw`U^\dagger U=I`} />
                  <p>
                    Coefficient height is magnitude |uᵣ𝒸|. A spoke in the local
                    complex plane encodes arg(uᵣ𝒸); the phase of a zero
                    coefficient is undefined. These values come from the native
                    output matrix.
                  </p>
                </div>
              )}
              <div className="operator-geometry-legend">
                <span>HEIGHT = |u|</span>
                <span>SPOKE = arg(u)</span>
                <span>SPOKES: |u| &gt;= 1e-14</span>
              </div>
              <p className="operator-phase-key">
                Magnitude: height = 1.8 × |u| in scene units. Phase: +Re = 0°,
                +Im = +90°, −Re = 180°, −Im = −90°. Rows are outputs; columns
                are inputs.
              </p>
            </div>
            <div className="operator-values">
              <label>
                Operator entry
                <select
                  aria-label="Operator payload"
                  value={Math.min(payloadIndex, instruction.payload.length - 1)}
                  onChange={(event) => {
                    setPayloadIndex(Number(event.target.value));
                    setSelectedCell([0, 0]);
                  }}
                >
                  {instruction.payload.map((entry, index) => (
                    <option value={index} key={index}>
                      Entry {index} · q[{entry.targets.join(", ")}] ·{" "}
                      {entry.matrix.length}×{entry.matrix.length}
                    </option>
                  ))}
                </select>
              </label>
              <div
                className="operator-coefficient-grid"
                style={{
                  gridTemplateColumns: `repeat(${payload.matrix.length}, minmax(0, 1fr))`,
                }}
                role="group"
                aria-label="Native complex matrix coefficients"
              >
                {payload.matrix.flatMap((row, r) =>
                  row.map((value, c) => (
                    <button
                      key={`${r}-${c}`}
                      className={
                        selectedCell[0] === r && selectedCell[1] === c
                          ? "selected"
                          : ""
                      }
                      aria-label={`Coefficient row ${r} column ${c}: ${complexText(value, 6)}`}
                      onClick={() => setSelectedCell([r, c])}
                    >
                      <small>
                        {r},{c}
                      </small>
                      <span>{value[0].toFixed(3)}</span>
                      <span className="matrix-imag">
                        {value[1] < 0 ? "-" : "+"}
                        {Math.abs(value[1]).toFixed(3)}i
                      </span>
                    </button>
                  )),
                )}
              </div>
              <div className="coefficient-detail">
                <span>
                  u[{selectedCell[0]},{selectedCell[1]}]
                </span>
                <code>{complexText(coefficient, 8)}</code>
                <dl>
                  <dt>Magnitude</dt>
                  <dd>{Math.sqrt(abs2(coefficient)).toFixed(8)}</dd>
                  <dt>Phase</dt>
                  <dd>{phaseText(...coefficient)}</dd>
                  <dt>Matrix support</dt>
                  <dd>q[{payload.targets.join(", ")}]</dd>
                </dl>
              </div>
              <p className="fine">
                Matrices are exported from the actual gate or batch entries. For
                a two-target matrix, local basis order is |targets[0]
                targets[1]&#x27e9;, with the first target as the matrix MSB;
                global q0 is still the statevector LSB. An expanded matrix for a
                named gate does not imply that the matrix is physically stored
                in the Gate enum. Transformed instruction ancestry was not
                exported and is not inferred.
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="compiler-proof">
        <div>
          <p className="eyebrow">CORRECTNESS IS SEPARATE FROM COST</p>
          <h3>The optimized stream preserves the state.</h3>
          <p>
            Every native stage was run on the same normalized complex input. All
            amplitudes were compared with unfused CPU execution before these
            samples were recorded. The final instruction stream also matches the
            public fusion pipeline. One deterministic input is a regression
            check, not an exhaustive proof of operator equivalence.
          </p>
          <div className="compiler-proof-badges">
            <span>Σ |α|² = {stage.state.normSquared.toFixed(12)}</span>
            <span>max |Δα| &lt; 10⁻¹²</span>
            <span>PUBLIC PIPELINE: MATCH</span>
          </div>
        </div>
        <div className="compiler-samples">
          <div className="sample-head">
            <span>BASIS INDEX</span>
            <span>NATIVE COMPLEX AMPLITUDE</span>
          </div>
          {stage.state.samples.map((sample) => (
            <div key={sample.index}>
              <span>|{binary(sample.index, record.width)}⟩</span>
              <code>{complexText([sample.re, sample.im], 6)}</code>
            </div>
          ))}
        </div>
      </div>
      <RewriteChallenge />
      <div className="compiler-traffic">
        <div>
          <p className="eyebrow">A COUNT IS NOT A BANDWIDTH MEASUREMENT</p>
          <strong>{formatBytes(metrics.modeledPayloadBytes)}</strong>
          <span>conditional full-sweep payload model at this stage</span>
          <p>
            32 × 2ⁿ bytes per modeled full read/write sweep. Actual DRAM traffic
            differs with caches, diagonal skips, partial updates, and gathered
            subcubes. Native kernel entry counts are observations; this byte
            total is a model.
          </p>
        </div>
        <div>
          <p>
            The fixture uses {fixture.generator.threads} Rayon threads.{" "}
            {record.profile === "Device"
              ? "Device is the optimizer profile. These snapshots and equivalence checks were executed by the CPU statevector backend; no CUDA performance is asserted."
              : "Host batching can choose a cache-derived stage planner. Only the branch actually reached on the recorded host is included."}
          </p>
          <a href="./native-fusion.json" download>
            Download native compiler fixture ↓
          </a>
          <a
            href="https://github.com/AbeCoull/prism-q/blob/main/docs/atlas/research/fusion-traces.md"
            target="_blank"
            rel="noreferrer"
          >
            Reproduce and audit these traces ↗
          </a>
        </div>
      </div>
    </>
  );
}

export default function CompilerLab() {
  const [fixture, setFixture] = useState<FusionFixture | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(new URL("./native-fusion.json", location.href), {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok)
          throw new Error(`Trace request failed (${response.status})`);
        return response.json();
      })
      .then((data) => setFixture(validateFusionFixture(data)))
      .catch((reason) => {
        if (reason.name !== "AbortError") setError(String(reason.message));
      });
    return () => controller.abort();
  }, []);
  return (
    <section
      id="compiler-laboratory"
      className="compiler-laboratory"
      aria-labelledby="compiler-title"
      data-testid="compiler-laboratory"
    >
      <div className="compiler-intro">
        <div>
          <p className="eyebrow">CHAPTER 03 · THE COMPILER LABORATORY</p>
          <h2 id="compiler-title">
            Fewer passes.
            <br />
            <em>Same unitary.</em>
          </h2>
        </div>
        <p>
          Watch the actual optimizer rewrite a circuit. Inspect every
          intermediate matrix, challenge a dependency, and separate a smaller
          gate list from less work on the statevector.
        </p>
      </div>
      {error ? (
        <p role="alert" className="compiler-loading">
          Native compiler evidence could not be loaded: {error}. Run the native
          trace generator before building the atlas.
        </p>
      ) : fixture ? (
        <Laboratory fixture={fixture} />
      ) : (
        <p className="compiler-loading" role="status">
          Loading and validating native compiler evidence…
        </p>
      )}
    </section>
  );
}
