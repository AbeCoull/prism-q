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
          <p className="eyebrow">
            ADVERSARIAL REWRITE EXPERIMENT · EDUCATIONAL MODEL
          </p>
          <h3 id="rewrite-title">Can the two X gates disappear?</h3>
        </div>
        <span className={`pill ${remove ? "model" : ""}`}>
          {remove ? "ILLEGAL REWRITE" : "ORIGINAL OPERATOR"}
        </span>
      </div>
      <p>
        They act on the same control qubit, but a CX sits between them. Try
        deleting both and inspect the output. This sandbox evaluates the
        proposed rewrite; it does not claim that PRISM-Q performs it.
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
              ? "This input hides the error. The target |+⟩ is unchanged by X, so these two different operators agree on this particular state. Switch to a basis input to expose the invalid rewrite."
              : `The candidate changes the output. Both states remain normalized, but fidelity is ${result.fidelity.toFixed(6)}. Normalization alone cannot establish circuit equivalence.`}
          </p>
          <p className="fine">
            Basis labels use |q1 q0⟩. The input equation groups logical
            subsystems and does not change that stored ordering. Operator
            equivalence requires agreement on all inputs, not one favorable
            example.
          </p>
        </div>
      </div>
    </section>
  );
}
