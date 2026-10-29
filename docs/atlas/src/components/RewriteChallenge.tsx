import { useMemo, useState } from "react";
import { Equation } from "./Equation";
import { cancellationCounterexample } from "../models/rewrite";
import { abs2, amplitude, binary } from "../models/quantum";
import "../styles/rewrite.css";

export default function RewriteChallenge() {
  const [theta, setTheta] = useState(0),
    [remove, setRemove] = useState(false);
  const result = useMemo(() => cancellationCounterexample(theta), [theta]);
  const output = remove ? result.invalid : result.valid;
  return (
    <section className="rewrite-challenge" aria-labelledby="rewrite-title">
      <div className="rewrite-header">
        <div>
          <h3 id="rewrite-title">Can the two X gates disappear?</h3>
        </div>
        <span className={`pill ${remove ? "model" : ""}`}>
          {remove ? "Invalid rewrite" : "Original circuit"}
        </span>
      </div>
      <p>
        Both act on the control qubit, with a CX between them. Delete them and
        compare the output. PRISM-Q does not make this rewrite; the point is to
        see why it can't.
      </p>
      <div className="rewrite-grid">
        <div>
          <div
            className="rewrite-circuit"
            aria-label={
              remove
                ? "Candidate circuit CX only"
                : "Original circuit X on control, CX, X on control"
            }
          >
            <span>q0 · control</span>
            <b className={remove ? "removed" : ""}>X</b>
            <b>●</b>
            <b className={remove ? "removed" : ""}>X</b>
            <span>q1 · target</span>
            <i />
            <b>⊕</b>
            <i />
          </div>
          <label className="range-control">
            <span>
              Target input angle
              <output>{((theta * 180) / Math.PI).toFixed(1)}°</output>
            </span>
            <input
              type="range"
              aria-label="Rewrite input angle"
              min="0"
              max={Math.PI / 2}
              step={Math.PI / 72}
              value={theta}
              onChange={(event) => setTheta(Number(event.target.value))}
            />
          </label>
          <div className="rewrite-presets">
            <button onClick={() => setTheta(0)}>Basis |0⟩</button>
            <button onClick={() => setTheta(Math.PI / 4)}>
              X eigenstate |+⟩
            </button>
            <button onClick={() => setTheta(Math.PI / 8)}>Intermediate</button>
          </div>
          <Equation
            block
            tex={String.raw`|\psi_{\mathrm{in}}\rangle=|0\rangle_{q_0}\otimes(\cos\theta|0\rangle+\sin\theta|1\rangle)_{q_1}`}
          />
          <button
            className="rewrite-toggle"
            aria-pressed={remove}
            onClick={() => setRemove(!remove)}
          >
            {remove ? "Restore both X gates" : "Remove both X gates"}{" "}
            <span>→</span>
          </button>
        </div>
        <div>
          <div className="rewrite-probabilities">
            {Array.from({ length: 4 }, (_, index) => (
              <div key={index}>
                <span>|{binary(index, 2)}⟩</span>
                <i>
                  <b
                    style={{
                      width: `${100 * abs2(amplitude(output, index))}%`,
                    }}
                  />
                </i>
                <code>{abs2(amplitude(output, index)).toFixed(4)}</code>
              </div>
            ))}
          </div>
          <div className="rewrite-fidelity">
            <span>Fidelity of original and candidate outputs</span>
            <strong data-testid="rewrite-fidelity">
              {result.fidelity.toFixed(6)}
            </strong>
          </div>
          <p className="rewrite-verdict" aria-live="polite">
            {result.fidelity > 1 - 1e-12
              ? "This input hides the error. X leaves |+⟩ unchanged, so the two circuits agree on this one state. Pick a basis input to see them differ."
              : `The output changes. Both states are still normalized, but the fidelity is ${result.fidelity.toFixed(6)}, so a correct norm proves nothing about the circuit.`}
          </p>
          <p className="fine">
            Basis labels are |q1 q0⟩. Two circuits are equal only if they agree
            on every input, not one convenient example.
          </p>
        </div>
      </div>
    </section>
  );
}
