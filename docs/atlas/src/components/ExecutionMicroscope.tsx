import { useEffect, useMemo, useState } from "react";
import { abs2, amplitude, complexText, type Complex } from "../models/quantum";
import {
  executionModel,
  instructionGroups,
  phaseOf,
  type Coefficient,
  type MicroscopeInput,
} from "../models/simd-execution";
import "../styles/microscope.css";

const revision = "07acc59bf9aba08b706bcc85efed35e0c456bb12";
const source = `https://github.com/AbeCoull/prism-q/blob/${revision}/src/backend/simd.rs`;
const numeric = (value: number) =>
  value === 0
    ? "0"
    : Math.abs(value) < 1e-4
      ? value.toExponential(3)
      : value.toFixed(6);

function Lanes({
  label,
  values,
  visible,
  signed = false,
}: {
  label: string;
  values: readonly number[];
  visible: boolean;
  signed?: boolean;
}) {
  return (
    <div className={`micro-register ${visible ? "is-ready" : "is-pending"}`}>
      <span className="micro-register-label">{label}</span>
      <div
        className="micro-lanes"
        style={{
          gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))`,
        }}
      >
        {values.map((value, lane) => (
          <div
            className={lane % 2 === 0 ? "micro-lane real" : "micro-lane imag"}
            key={lane}
          >
            <small>
              {signed
                ? lane % 2 === 0
                  ? "signed Re"
                  : "signed Im"
                : `${lane % 2 === 0 ? "Re" : "Im"}${values.length === 4 ? ` ${Math.floor(lane / 2)}` : ""}`}
            </small>
            <span>{visible ? numeric(value) : "pending"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Argand({
  before,
  after,
  stored,
  index,
}: {
  before: Complex;
  after: Complex;
  stored: boolean;
  index: number;
}) {
  const radius = Math.max(Math.hypot(...before), Math.hypot(...after), 1e-12);
  const point = (z: Complex) => [
    112 + (78 * z[0]) / radius,
    106 - (78 * z[1]) / radius,
  ];
  const [bx, by] = point(before),
    [ax, ay] = point(after);
  const phase = (value: Complex) =>
    phaseOf(value) === null
      ? "undefined (zero amplitude)"
      : `${((phaseOf(value)! * 180) / Math.PI).toFixed(2)}°`;
  return (
    <div className="micro-argand">
      <svg
        viewBox="0 0 240 220"
        role="img"
        aria-label={`Complex amplitude ${index}, before and after the gate`}
      >
        <circle cx="112" cy="106" r="78" className="micro-orbit" />
        <path d="M 18 106 H 212 M 112 12 V 198" className="micro-axis" />
        <text x="214" y="110">
          Re
        </text>
        <text x="117" y="17">
          Im
        </text>
        <line
          x1="112"
          y1="106"
          x2={bx}
          y2={by}
          className="micro-vector before"
        />
        <circle cx={bx} cy={by} r="4" className="micro-point before" />
        <line
          x1="112"
          y1="106"
          x2={stored ? ax : bx}
          y2={stored ? ay : by}
          className="micro-vector after"
        />
        <circle
          cx={stored ? ax : bx}
          cy={stored ? ay : by}
          r="5"
          className="micro-point after"
        />
        <text x="14" y="216">
          radius = {numeric(radius)}
        </text>
      </svg>
      <div className="micro-phase-readout">
        <strong>Amplitude |{index}⟩</strong>
        <span>
          <i className="micro-key before" />
          Before: {complexText(before, 6)}
        </span>
        <span>
          phase {phase(before)} · probability {abs2(before).toExponential(5)}
        </span>
        <span>
          <i className="micro-key after" />
          After: {complexText(after, 6)}
        </span>
        <span>
          phase {phase(after)} · probability {abs2(after).toExponential(5)}
        </span>
        <small>
          The circle is scaled to this amplitude. Probabilities are relative to
          the whole state.
        </small>
      </div>
    </div>
  );
}

export function ExecutionMicroscope(
  props: MicroscopeInput & { contextLabel?: string; onEdit?: () => void },
) {
  const { matrix, pair, before, architecture, width, diagonal, n, target } =
    props;
  const [coefficient, setCoefficient] = useState<Coefficient>(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const model = useMemo(
    () => executionModel(props),
    [before, matrix, pair[0], pair[1], target, width, architecture, diagonal],
  );
  const groups = useMemo(
    () => instructionGroups(architecture, width, diagonal),
    [architecture, width, diagonal],
  );
  const selected: Coefficient = diagonal
    ? coefficient < 2
      ? 0
      : 3
    : coefficient;
  const row = selected < 2 ? 0 : 1;
  const column = selected % 2;
  const product = model.terms[selected];
  const skipped = diagonal && row === 0 && model.skipLow;
  const last = groups.length - 1;
  const currentStep = Math.min(step, last);
  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [before, matrix, pair[0], pair[1], architecture, width, diagonal]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setStep((value) => {
          if (value >= last) {
            setPlaying(false);
            return last;
          }
          return value + 1;
        }),
      1150,
    );
    return () => window.clearInterval(timer);
  }, [playing, last]);
  const chooseStep = (value: number) => {
    setPlaying(false);
    setStep(value);
  };
  const sumVisible = diagonal ? currentStep >= 5 : currentStep >= 7;
  const sourceLine = diagonal
    ? width === 0
      ? "L1109-L1132"
      : width === 256
        ? "L1762-L1782"
        : architecture === "neon"
          ? "L1021-L1105"
          : "L971-L1105"
    : architecture === "neon"
      ? "L208-L223"
      : width === 256
        ? "L342-L384"
        : width === 128
          ? "L270-L315"
          : "L670-L683";
  return (
    <section
      className="execution-microscope"
      id="execution-microscope"
      aria-labelledby="microscope-title"
    >
      <div className="micro-heading">
        <div>
          <h2 id="microscope-title">Step through the kernel</h2>
        </div>
        <span className="micro-route">
          {n} qubits · q[{target}] ·{" "}
          {width
            ? `${width}-bit ${architecture === "neon" ? "NEON" : "FMA"}`
            : "scalar"}
        </span>
      </div>
      {props.contextLabel && (
        <div className="micro-context">
          <span>{props.contextLabel}</span>
          <button onClick={props.onEdit}>Change settings ↑</button>
          <a href="#gate-laboratory">Back to the memory view ↑</a>
        </div>
      )}
      <p>
        Each step is one group of intrinsics from the kernel source, from the
        loads to the stores, evaluated on the pair and gate chosen above. The
        browser computes the values; this is not a trace of a native run.
      </p>
      <div className="micro-transport">
        <button
          type="button"
          onClick={() => chooseStep(0)}
          aria-label="Restart execution microscope"
        >
          ↺
        </button>
        <button
          type="button"
          onClick={() => {
            if (currentStep === last) setStep(0);
            setPlaying(!playing);
          }}
        >
          {playing ? "Pause" : "Play sequence"}
        </button>
        <button
          type="button"
          disabled={currentStep === 0}
          onClick={() => chooseStep(currentStep - 1)}
        >
          Previous
        </button>
        <input
          type="range"
          min="0"
          max={last}
          value={currentStep}
          onChange={(event) => chooseStep(Number(event.target.value))}
          aria-label="Execution instruction group"
        />
        <button
          type="button"
          disabled={currentStep === last}
          onClick={() => chooseStep(currentStep + 1)}
        >
          Next
        </button>
        <span>
          {currentStep + 1} / {groups.length}
        </span>
      </div>
      <div className="micro-body">
        <ol className="micro-timeline" aria-label="Source instruction groups">
          {groups.map((group, index) => (
            <li
              key={group.title}
              className={
                index === currentStep
                  ? "active"
                  : index < currentStep
                    ? "completed"
                    : ""
              }
            >
              <button
                type="button"
                onClick={() => chooseStep(index)}
                aria-current={index === currentStep ? "step" : undefined}
              >
                <span>{index + 1}</span>
                {group.title}
              </button>
            </li>
          ))}
        </ol>
        <div className="micro-workbench">
          <div className="micro-step-heading" aria-live="polite">
            <span>Step {currentStep + 1}</span>
            <h3>{groups[currentStep].title}</h3>
            <code>{groups[currentStep].code}</code>
            <p>{groups[currentStep].detail}</p>
          </div>
          <div
            className="micro-coefficients"
            aria-label="Matrix coefficient to inspect"
          >
            {([0, 1, 2, 3] as Coefficient[])
              .filter((i) => !diagonal || i === 0 || i === 3)
              .map((i) => (
                <button
                  type="button"
                  key={i}
                  aria-pressed={selected === i}
                  onClick={() => setCoefficient(i)}
                >
                  <span>
                    {diagonal
                      ? `d${i === 0 ? 0 : 1}`
                      : `u${Math.floor(i / 2)}${i % 2}`}
                  </span>
                  <code>{complexText(matrix[i], 4)}</code>
                </button>
              ))}
          </div>
          <p className="micro-addresses">
            {diagonal
              ? `Independent ${row === 0 ? "low" : "high"} half`
              : `Output row ${row}, input ${column === 0 ? "a" : "b"}`}{" "}
            · {skipped ? "bypassed indices" : "loaded indices"} [
            {model.indices[column].join(", ")}] · byte offsets [
            {model.indices[column].map((i) => i * 16).join(", ")}]
          </p>
          {skipped ? (
            <div className="micro-bypass">
              <strong>Low half bypassed</strong>
              <p>
                d0 is within 1e-15 of 1+0i, so the kernel never loads,
                multiplies or stores the low half. This amplitude stays at{" "}
                {complexText(amplitude(before, pair[0]), 6)}.
              </p>
            </div>
          ) : (
            <div
              className="micro-registers"
              role="region"
              aria-label="SIMD register values, horizontally scrollable"
              tabIndex={0}
            >
              <Lanes
                label={width ? "c.re broadcast" : "c.re scalar"}
                values={product.realCoefficients}
                visible={currentStep >= 0}
              />
              <Lanes
                label={
                  !width
                    ? "c.im scalar"
                    : architecture === "neon"
                      ? "c.im signed broadcast"
                      : "c.im broadcast"
                }
                values={product.imagCoefficients}
                visible={currentStep >= 0}
              />
              <Lanes
                label={width ? "z · original load" : "z · original parts"}
                values={model.loads[column]}
                visible={currentStep >= 2}
              />
              <Lanes
                label="z_swap · within amplitude"
                values={product.swapped}
                visible={currentStep >= 3}
              />
              <Lanes
                label={
                  architecture === "neon"
                    ? "prod · c.re × z"
                    : "t · c.im × z_swap"
                }
                values={
                  architecture === "neon"
                    ? product.realProducts
                    : product.imagProducts
                }
                visible={currentStep >= 4}
              />
              <Lanes
                label="signed imaginary contribution"
                values={product.signedImagProducts}
                visible={currentStep >= 5}
                signed
              />
              <Lanes
                label="complex product"
                values={product.result}
                visible={currentStep >= 5}
              />
              {!diagonal && (
                <Lanes
                  label="other term in this row"
                  values={model.terms[row * 2 + (1 - column)].result}
                  visible={currentStep >= 6}
                />
              )}
              <Lanes
                label={diagonal ? "scaled output" : "two-term row sum"}
                values={model.rows[row]}
                visible={sumVisible}
              />
              <Lanes
                label={`store → [${model.indices[row].join(", ")}]`}
                values={model.rows[row]}
                visible={currentStep === last}
              />
            </div>
          )}
        </div>
        <aside className="micro-phase">
          <h3>Amplitude & phase</h3>
          <Argand
            before={amplitude(before, pair[row])}
            after={model.outputs[row]}
            stored={currentStep === last}
            index={pair[row]}
          />
          <p>
            {currentStep === last
              ? "The blue arrow now shows the stored output."
              : "The dashed red arrow is the input. The blue arrow moves to the output at the store step."}
          </p>
        </aside>
      </div>
      <div className="micro-source">
        <a href={`${source}#${sourceLine}`} target="_blank" rel="noreferrer">
          Kernel source at {revision.slice(0, 7)} ↗
        </a>
        <span>
          Lane values are computed in JavaScript doubles. Native FMA can differ
          in the last bits.
        </span>
      </div>
    </section>
  );
}
