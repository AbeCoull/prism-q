import { applyGate, matrixFor, norm } from "./quantum";

function cx(state: Float64Array): Float64Array {
  const out = state.slice();
  out[2] = state[6];
  out[3] = state[7];
  out[6] = state[2];
  out[7] = state[3];
  return out;
}

export function cancellationCounterexample(theta: number) {
  const input = new Float64Array(8);
  input[0] = Math.cos(theta);
  input[4] = Math.sin(theta);
  const x = matrixFor("X");
  const valid = applyGate(cx(applyGate(input, 0, x)), 0, x);
  const invalid = cx(input);
  let overlapRe = 0,
    overlapIm = 0;
  for (let i = 0; i < valid.length; i += 2) {
    overlapRe += valid[i] * invalid[i] + valid[i + 1] * invalid[i + 1];
    overlapIm += valid[i] * invalid[i + 1] - valid[i + 1] * invalid[i];
  }
  return {
    input,
    valid,
    invalid,
    fidelity: overlapRe ** 2 + overlapIm ** 2,
    validNorm: norm(valid),
    invalidNorm: norm(invalid),
  };
}
