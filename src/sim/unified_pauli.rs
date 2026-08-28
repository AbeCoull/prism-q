//! Pauli propagation engines for circuits of Clifford gates and Pauli
//! rotations: SPP samples backward Heisenberg propagation stochastically, SPD
//! carries the full weighted Pauli sum with optional truncation. Neither
//! materializes a state vector.
//!
//! Z-axis rotations (`T`, `Tdg`, `Rz`, `P`) and the two-qubit `Rzz` branch
//! natively; `Rx`, `Ry`, and multi-qubit `PauliRot` strings lower to Clifford
//! conjugation around one `Rz` before the run.

use num_complex::Complex64;
use rand::SeedableRng;
use rand::{Rng, RngExt};
use rand_chacha::ChaCha8Rng;

use std::borrow::Cow;

use crate::circuit::{Circuit, Instruction, SmallVec, pauli_rotation_lowering};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{PauliVec, flip_bit, propagate_backward};

/// Absolute ceiling on the weighted-Pauli term count, enforced even in exact
/// mode (`max_terms == 0`). Without a per-step truncation budget the term set
/// grows as `2^(in-cone branching rotations)`; this caps transient memory the
/// same way the stabilizer-rank backend does. Exceeding it is an error, not a
/// silent truncation: callers wanting bounded approximate evaluation pass a
/// nonzero `max_terms` instead.
const SPD_MAX_TERMS_CEILING: usize = 1 << 20;

#[inline]
fn check_spd_term_ceiling(len: usize) -> Result<()> {
    if len > SPD_MAX_TERMS_CEILING {
        return Err(PrismError::BackendUnsupported {
            backend: "SPD".into(),
            operation: format!(
                "weighted-Pauli sum exceeded {SPD_MAX_TERMS_CEILING} terms; pass a nonzero \
                 max_terms for bounded approximate truncation"
            ),
        });
    }
    Ok(())
}

/// Rotation angle of the Z-axis rotation a gate lowers to, if any.
///
/// `P(λ)` and `Rz(λ)` differ by the global phase `e^{-iλ/2}`, which conjugation
/// discards, so both report `λ`. Clifford Z rotations (`Z`, `S`, `Sdg`) are
/// deliberately absent: they conjugate exactly through the Clifford path and
/// would otherwise branch into a term with a rounding-noise coefficient.
#[inline]
fn z_rotation_angle(gate: &Gate) -> Option<f64> {
    match gate {
        Gate::T => Some(std::f64::consts::FRAC_PI_4),
        Gate::Tdg => Some(-std::f64::consts::FRAC_PI_4),
        Gate::Rz(theta) | Gate::P(theta) => Some(*theta),
        _ => None,
    }
}

#[inline]
fn zz_rotation_angle(gate: &Gate) -> Option<f64> {
    match gate {
        Gate::Rzz(theta) => Some(*theta),
        _ => None,
    }
}

/// Validate the circuit and lower arbitrary-axis rotations in one walk.
///
/// `Rx`, `Ry`, and multi-qubit `PauliRot` strings expand through the shared
/// CNOT-ladder lowering into Clifford conjugation around one `Rz`, so the
/// engines only ever branch on Z-axis rules; `Rzz` stays native and branches
/// directly on the `Z⊗Z` anticommutation test. The lowering tests sit on the
/// walk's rejection path, so a circuit needing none pays no second scan and
/// borrows through unchanged. The lowering emits only supported gates, so the
/// rebuilt circuit needs no second validation.
fn validate_and_lower<'c>(circuit: &'c Circuit, backend: &'static str) -> Result<Cow<'c, Circuit>> {
    let mut needs_lowering = false;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, .. } => {
                if gate.is_clifford()
                    || z_rotation_angle(gate).is_some()
                    || zz_rotation_angle(gate).is_some()
                {
                    continue;
                }
                if matches!(gate, Gate::Rx(_) | Gate::Ry(_) | Gate::PauliRot(_)) {
                    needs_lowering = true;
                    continue;
                }
                return Err(PrismError::BackendUnsupported {
                    backend: backend.to_string(),
                    operation: format!(
                        "gate `{}` is neither Clifford nor a supported Pauli rotation",
                        gate.name()
                    ),
                });
            }
            Instruction::Barrier { .. } => {}
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. }
            | Instruction::Region(_) => {
                return Err(PrismError::IncompatibleBackend {
                    backend: backend.to_string(),
                    reason: "Pauli propagation requires a unitary circuit without measurements, \
                         resets, or conditionals"
                        .to_string(),
                });
            }
        }
    }
    if !needs_lowering {
        return Ok(Cow::Borrowed(circuit));
    }

    let mut out: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len() * 2);
    for inst in &circuit.instructions {
        let (theta, targets, axes): (f64, &[usize], &[PauliAxis]) = match inst {
            Instruction::Gate {
                gate: Gate::Rx(theta),
                targets,
            } => (*theta, targets, &[PauliAxis::X]),
            Instruction::Gate {
                gate: Gate::Ry(theta),
                targets,
            } => (*theta, targets, &[PauliAxis::Y]),
            Instruction::Gate {
                gate: Gate::PauliRot(data),
                targets,
            } => (data.theta(), targets, data.axes()),
            _ => {
                out.push(inst.clone());
                continue;
            }
        };
        pauli_rotation_lowering(theta, targets, axes, |gate, tgts| {
            out.push(Instruction::Gate {
                gate,
                targets: SmallVec::from_slice(tgts),
            });
        });
    }
    Ok(Cow::Owned(circuit.with_instructions(out)))
}

// ---- Coalesced circuit representation ----

/// Marks a [`CoalescedOp::ZRot`] with no second qubit.
const ZROT_SINGLE: u32 = u32::MAX;

/// The ops slice streams through the SPP per-sample loop, so the layout is
/// deliberate on two counts, both measured on op-heavy rows: the qubits pack
/// as `u32` to hold the enum at 40 bytes (48 cost +2.6% to +5.8%), and `Rzz`
/// shares the `ZRot` variant via the `ZROT_SINGLE` sentinel instead of adding
/// a third variant, which kept an outlined call in the sample loop's match
/// and cost about the same again.
enum CoalescedOp {
    SmallCliff(Vec<(Gate, SmallVec<[usize; 4]>)>),
    ZRot {
        qubit: u32,
        pair: u32,
        branch: RotBranch,
    },
}

fn coalesce_cliffords(circuit: &Circuit) -> Vec<CoalescedOp> {
    let mut ops = Vec::new();
    let mut cliff_buf: Vec<(Gate, SmallVec<[usize; 4]>)> = Vec::new();

    for inst in &circuit.instructions {
        if let Instruction::Gate { gate, targets } = inst {
            if let Some(theta) = z_rotation_angle(gate) {
                flush_cliff_buf(&mut cliff_buf, &mut ops);
                ops.push(CoalescedOp::ZRot {
                    qubit: targets[0] as u32,
                    pair: ZROT_SINGLE,
                    branch: RotBranch::new(theta),
                });
            } else if let Some(theta) = zz_rotation_angle(gate) {
                flush_cliff_buf(&mut cliff_buf, &mut ops);
                ops.push(CoalescedOp::ZRot {
                    qubit: targets[0] as u32,
                    pair: targets[1] as u32,
                    branch: RotBranch::new(theta),
                });
            } else {
                cliff_buf.push((gate.clone(), SmallVec::from_slice(targets)));
            }
        } else {
            flush_cliff_buf(&mut cliff_buf, &mut ops);
        }
    }
    flush_cliff_buf(&mut cliff_buf, &mut ops);
    ops
}

fn flush_cliff_buf(buf: &mut Vec<(Gate, SmallVec<[usize; 4]>)>, ops: &mut Vec<CoalescedOp>) {
    if buf.is_empty() {
        return;
    }
    ops.push(CoalescedOp::SmallCliff(std::mem::take(buf)));
}

// ---- SPP (Stochastic Pauli Propagation) ----

/// Sampling data for one Z-axis rotation, built once per gate so the per-sample
/// loop stays free of trigonometry.
#[derive(Clone, Copy)]
struct RotBranch {
    p_keep: f64,
    keep_weight: f64,
    flip_weight_im: f64,
}

impl RotBranch {
    /// Branch probability and importance weights for `Rz(theta)`; `Rzz`
    /// shares them, since its expansion carries the same coefficients.
    ///
    /// Backward conjugation sends an in-support Pauli `P` to
    /// `cos(theta) P + (-i sin(theta)) P Z_q`. PauliVec stores the Y letter as
    /// the ordered product XZ and actual `Y = i XZ`, which is where the flip
    /// branch picks up its imaginary unit. Branch selection is proportional to
    /// coefficient magnitude, so the per-sample weight magnitude is
    /// `|cos| + |sin|`: 1 at the Clifford angles, `sqrt(2)` at `theta = pi/4`.
    fn new(theta: f64) -> Self {
        let (sin, cos) = theta.sin_cos();
        let norm1 = cos.abs() + sin.abs();
        Self {
            p_keep: cos.abs() / norm1,
            keep_weight: norm1 * cos.signum(),
            flip_weight_im: -norm1 * sin.signum(),
        }
    }
}

/// Stochastic branch for a `Z` or `Z⊗Z` rotation. The branch fires only when
/// the propagated Pauli anticommutes with the generator: an X or Y letter on
/// the qubit for `Rz`, exactly one such letter on the pair for `Rzz`. The
/// flip right-multiplies by the generator, a pure z-bit flip on its support
/// in the ordered `X^x Z^z` letter convention.
#[inline(always)]
fn branch_z_rotation(
    pauli: &mut PauliVec,
    qubit: u32,
    pair: u32,
    branch: &RotBranch,
    rng: &mut impl Rng,
) -> Complex64 {
    let q = qubit as usize;
    let anticommutes = if pair == ZROT_SINGLE {
        pauli.has_x_or_y(q)
    } else {
        pauli.has_x_or_y(q) != pauli.has_x_or_y(pair as usize)
    };
    if !anticommutes {
        return Complex64::new(1.0, 0.0);
    }

    if rng.random_bool(branch.p_keep) {
        return Complex64::new(branch.keep_weight, 0.0);
    }

    flip_bit(&mut pauli.z, q);
    if pair != ZROT_SINGLE {
        flip_bit(&mut pauli.z, pair as usize);
    }
    Complex64::new(0.0, branch.flip_weight_im)
}

fn backward_propagate_coalesced(
    ops: &[CoalescedOp],
    observable: &PauliVec,
    rng: &mut impl Rng,
) -> (PauliVec, Complex64) {
    let mut pauli = PauliVec {
        x: observable.x.clone(),
        z: observable.z.clone(),
    };
    let mut weight = Complex64::new(1.0, 0.0);

    for op in ops.iter().rev() {
        match op {
            CoalescedOp::SmallCliff(gates) => {
                for (gate, targets) in gates.iter().rev() {
                    // Phase track Clifford conjugation, mirroring the
                    // SPD path (`conjugate_all_backward_phased`); the
                    // bare `propagate_backward` is Pauli-frame only
                    // and drops the `-1` from e.g. `HYH = -Y`.
                    weight *= clifford_conjugation_phase(gate, targets, &pauli);
                    propagate_backward(&mut pauli, gate, targets);
                }
            }
            CoalescedOp::ZRot {
                qubit,
                pair,
                branch,
            } => {
                weight *= branch_z_rotation(&mut pauli, *qubit, *pair, branch, rng);
            }
        }
    }

    (pauli, weight)
}

fn count_branching_gates(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| match inst {
            Instruction::Gate { gate, .. } => {
                z_rotation_angle(gate).is_some() || zz_rotation_angle(gate).is_some()
            }
            _ => false,
        })
        .count()
}

/// Per-qubit `⟨Z_q⟩` estimates from a stochastic Pauli propagation run.
pub struct SppResult {
    pub expectations: Vec<f64>,
    pub std_errors: Vec<f64>,
    /// Samples drawn per qubit, not in total.
    pub num_samples: usize,
    /// Branching gates in the circuit: `T`, `Tdg`, `Rz`, `P`, and `Rzz`.
    pub t_count: usize,
    /// Fraction of samples whose propagated Pauli was diagonal (contributed a
    /// nonzero value).
    pub nonzero_fraction: f64,
}

fn estimate_qubit_expectation(
    ops: &[CoalescedOp],
    qubit: usize,
    num_words: usize,
    num_samples: usize,
    seed: u64,
) -> (f64, f64, usize) {
    let obs = PauliVec::z_on_qubit(num_words, qubit);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut nonzero = 0usize;

    for i in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            weight.re
        } else {
            0.0
        };

        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = if num_samples > 1 {
        m2 / (num_samples - 1) as f64
    } else {
        0.0
    };
    let std_error = (variance / num_samples as f64).sqrt();
    (mean, std_error, nonzero)
}

/// Pauli axis for a joint-observable term. Identity factors are omitted
/// from the term list and contribute trivially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PauliAxis {
    X,
    Y,
    Z,
}

impl PauliAxis {
    /// Uppercase letter naming the axis, the spelling Pauli strings carry.
    pub fn letter(self) -> char {
        match self {
            PauliAxis::X => 'X',
            PauliAxis::Y => 'Y',
            PauliAxis::Z => 'Z',
        }
    }

    /// Inverse of [`letter`](Self::letter), accepting either case.
    pub fn from_letter(letter: char) -> Option<Self> {
        match letter.to_ascii_uppercase() {
            'X' => Some(PauliAxis::X),
            'Y' => Some(PauliAxis::Y),
            'Z' => Some(PauliAxis::Z),
            _ => None,
        }
    }
}

/// One non-identity factor of a joint Pauli observable. Ordered by qubit,
/// then axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PauliTerm {
    pub qubit: usize,
    pub axis: PauliAxis,
}

impl PauliTerm {
    pub fn new(qubit: usize, axis: PauliAxis) -> Self {
        Self { qubit, axis }
    }

    pub fn x(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::X)
    }

    pub fn y(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::Y)
    }

    pub fn z(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::Z)
    }
}

/// Result of running SPP on a joint Pauli observable.
#[derive(Debug, Clone)]
pub struct SppObservableResult {
    pub mean: f64,
    pub std_error: f64,
    /// Sample variance of the per-sample contributions, not of the mean.
    pub variance: f64,
    pub num_samples: usize,
    /// Fraction of samples whose propagated Pauli was diagonal.
    pub nonzero_fraction: f64,
    pub t_count: usize,
}

/// Result of running SPD on a joint Pauli observable.
#[derive(Debug, Clone)]
pub struct SpdObservableResult {
    /// `⟨P⟩`; exact when nothing was truncated.
    pub mean: f64,
    pub t_count: usize,
    /// Peak size of the weighted Pauli sum during propagation.
    pub peak_terms: usize,
    /// Total coefficient magnitude discarded by truncation (0 for exact runs).
    pub total_discarded: f64,
}

fn pauli_vec_from_terms(
    num_qubits: usize,
    terms: &[PauliTerm],
) -> std::result::Result<(PauliVec, Complex64), crate::error::PrismError> {
    let num_words = num_qubits.div_ceil(64);
    let mut pv = PauliVec::new(num_words);
    let mut coeff = Complex64::new(1.0, 0.0);
    let mut seen = vec![false; num_qubits];
    for term in terms {
        if term.qubit >= num_qubits {
            return Err(crate::error::PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
        match term.axis {
            PauliAxis::X => {
                pv.x[term.qubit / 64] |= 1u64 << (term.qubit % 64);
            }
            PauliAxis::Z => {
                pv.z[term.qubit / 64] |= 1u64 << (term.qubit % 64);
            }
            PauliAxis::Y => {
                pv.x[term.qubit / 64] |= 1u64 << (term.qubit % 64);
                pv.z[term.qubit / 64] |= 1u64 << (term.qubit % 64);
                coeff *= Complex64::new(0.0, 1.0);
            }
        }
    }
    Ok((pv, coeff))
}

/// Estimate `⟨0^n| U† P U |0^n⟩` for joint Pauli observable `P` on a circuit
/// `U` of Clifford gates and Pauli rotations, via stochastic Pauli
/// propagation.
///
/// Each sample backward-propagates the observable through the circuit
/// (Clifford segments as coalesced gate runs, `Rz(theta)` and the two-qubit
/// `Rzz(theta)` via a stochastic Pauli branch that records a complex weight;
/// `Rx`, `Ry`, and multi-qubit `PauliRot` strings lower to Clifford
/// conjugation around one `Rz` first). The contribution is `Re(weight)` when
/// the final Pauli is diagonal in `{I, Z}` (i.e. evaluates trivially on
/// `|0^n⟩`), else zero.
///
/// Sample variance grows with the product of `|cos θ| + |sin θ|` over the
/// in-support rotations, so it is largest at `θ = π/4` and vanishes as the
/// angles approach a Clifford multiple of `π/2`.
pub fn run_spp_observable(
    circuit: &Circuit,
    observable: &[PauliTerm],
    num_samples: usize,
    seed: u64,
) -> Result<SppObservableResult> {
    let lowered = validate_and_lower(circuit, "SPP observable")?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let t_count = count_branching_gates(circuit);
    let ops = coalesce_cliffords(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut nonzero = 0usize;

    for i in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            (obs_coeff * weight).re
        } else {
            0.0
        };
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = if num_samples > 1 {
        m2 / (num_samples - 1) as f64
    } else {
        0.0
    };
    let std_error = (variance / num_samples.max(1) as f64).sqrt();
    let nonzero_fraction = nonzero as f64 / num_samples.max(1) as f64;

    Ok(SppObservableResult {
        mean,
        std_error,
        variance,
        num_samples,
        nonzero_fraction,
        t_count,
    })
}

/// Estimate `⟨Z_q⟩` for every qubit of a unitary circuit of Clifford gates and
/// Pauli rotations, via stochastic Pauli propagation.
///
/// Draws `num_samples` backward propagations per qubit; the cost scales with
/// samples and gate count, not `2^n`. See [`run_spp_observable`] for a single
/// joint observable.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, run_spp};
///
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::T, &[0]);
/// circuit.add_gate(Gate::H, &[0]);
///
/// let result = run_spp(&circuit, 20_000, 42)?;
/// assert_eq!(result.t_count, 1);
/// // <Z0> = cos(pi/4); qubit 1 is untouched, so <Z1> = 1 exactly.
/// assert!((result.expectations[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.05);
/// assert!((result.expectations[1] - 1.0).abs() < 1e-12);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn run_spp(circuit: &Circuit, num_samples: usize, seed: u64) -> Result<SppResult> {
    let lowered = validate_and_lower(circuit, "SPP")?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_branching_gates(circuit);
    let ops = coalesce_cliffords(circuit);

    #[cfg(feature = "parallel")]
    let results: Vec<(f64, f64, usize)> = {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|q| {
                estimate_qubit_expectation(
                    &ops,
                    q,
                    num_words,
                    num_samples,
                    seed.wrapping_add(q as u64),
                )
            })
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(f64, f64, usize)> = (0..n)
        .map(|q| {
            estimate_qubit_expectation(&ops, q, num_words, num_samples, seed.wrapping_add(q as u64))
        })
        .collect();

    let mut expectations = Vec::with_capacity(n);
    let mut std_errors = Vec::with_capacity(n);
    let mut total_nonzero = 0usize;

    for (mean, std_error, nonzero) in &results {
        expectations.push(*mean);
        std_errors.push(*std_error);
        total_nonzero += nonzero;
    }

    let nonzero_fraction = total_nonzero as f64 / (n * num_samples) as f64;

    Ok(SppResult {
        expectations,
        std_errors,
        num_samples,
        t_count,
        nonzero_fraction,
    })
}

#[cfg(test)]
fn spp_to_probabilities(result: &SppResult) -> Vec<f64> {
    super::expectations_to_marginals(&result.expectations)
        .into_iter()
        .flat_map(|(p0, p1)| [p0, p1])
        .collect()
}

// ---- Deterministic Sparse Pauli Dynamics (SPD) ----

use std::collections::HashMap;

#[inline(always)]
fn pv_get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
fn clifford_conjugation_phase(gate: &Gate, targets: &[usize], pauli: &PauliVec) -> Complex64 {
    match gate {
        Gate::H => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) && pv_get_bit(&pauli.z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::S => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Sdg => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::X => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Y => {
            let q = targets[0];
            let xq = pv_get_bit(&pauli.x, q);
            let zq = pv_get_bit(&pauli.z, q);
            if xq ^ zq {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Z => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SX => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SXdg => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            if pv_get_bit(&pauli.x, q0) && pv_get_bit(&pauli.x, q1) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        _ => Complex64::new(1.0, 0.0),
    }
}

struct WeightedPauliSum {
    terms: HashMap<PauliVec, Complex64>,
    scratch: Vec<(PauliVec, Complex64)>,
}

impl WeightedPauliSum {
    fn new() -> Self {
        Self {
            terms: HashMap::new(),
            scratch: Vec::new(),
        }
    }

    fn insert(&mut self, pauli: PauliVec, coeff: Complex64) {
        let entry = self.terms.entry(pauli).or_insert(Complex64::new(0.0, 0.0));
        *entry += coeff;
    }

    fn conjugate_all_backward_phased(&mut self, gate: &Gate, targets: &[usize]) {
        // Clifford conjugation is a bijection on Pauli strings: the symplectic
        // action on the (x, z) bits is invertible, so distinct keys map to
        // distinct keys and no coefficient merging occurs. Reuse `scratch` to
        // avoid a per-gate heap allocation, drain into it (which keeps the
        // map's bucket capacity), then re-insert directly without the
        // accumulate path.
        self.scratch.clear();
        self.scratch.extend(self.terms.drain());
        for (mut pauli, coeff) in self.scratch.drain(..) {
            let phase = clifford_conjugation_phase(gate, targets, &pauli);
            propagate_backward(&mut pauli, gate, targets);
            self.terms.insert(pauli, coeff * phase);
        }
    }

    /// Split every in-support term under backward conjugation by a Z rotation,
    /// which sends `P` to `cos(theta) P + (-i sin(theta)) P Z_q`. Takes the
    /// evaluated `(sin, cos)` because `run_spd` replays the circuit once per
    /// qubit and the trigonometry is per gate, not per replay. A branch whose
    /// coefficient is exactly zero is not inserted, so `Rz(0)` and `P(0)` leave
    /// the term count alone.
    fn branch_z_rotation(&mut self, qubit: usize, sin: f64, cos: f64) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();

        for (pauli, coeff) in old_terms {
            if !pauli.has_x_or_y(qubit) {
                self.insert(pauli, coeff);
                continue;
            }

            if cos != 0.0 {
                self.insert(pauli.clone(), coeff * cos);
            }
            if sin != 0.0 {
                let mut pauli_flip = pauli;
                flip_bit(&mut pauli_flip.z, qubit);
                self.insert(pauli_flip, Complex64::new(coeff.im * sin, -coeff.re * sin));
            }
        }
    }

    /// Two-qubit analogue of [`branch_z_rotation`](Self::branch_z_rotation)
    /// for `Rzz`: a term splits only when it anticommutes with `Z⊗Z` on the
    /// pair, i.e. exactly one of the two qubits carries an X or Y letter. The
    /// flip branch right-multiplies by `Z_a Z_b`, a pure z-bit flip on both
    /// qubits in the ordered `X^x Z^z` letter convention.
    fn branch_zz_rotation(&mut self, a: usize, b: usize, sin: f64, cos: f64) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();

        for (pauli, coeff) in old_terms {
            if pauli.has_x_or_y(a) == pauli.has_x_or_y(b) {
                self.insert(pauli, coeff);
                continue;
            }

            if cos != 0.0 {
                self.insert(pauli.clone(), coeff * cos);
            }
            if sin != 0.0 {
                let mut pauli_flip = pauli;
                flip_bit(&mut pauli_flip.z, a);
                flip_bit(&mut pauli_flip.z, b);
                self.insert(pauli_flip, Complex64::new(coeff.im * sin, -coeff.re * sin));
            }
        }
    }

    /// Backward-apply one gate: a single discriminant match, so the Clifford
    /// path pays no rotation-probe chain per instruction. Angle cases mirror
    /// [`z_rotation_angle`] and [`zz_rotation_angle`].
    #[inline(always)]
    fn apply_backward(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::T => self.branch_z_rotation_angle(targets[0], std::f64::consts::FRAC_PI_4),
            Gate::Tdg => self.branch_z_rotation_angle(targets[0], -std::f64::consts::FRAC_PI_4),
            Gate::Rz(theta) | Gate::P(theta) => self.branch_z_rotation_angle(targets[0], *theta),
            Gate::Rzz(theta) => {
                let (sin, cos) = theta.sin_cos();
                self.branch_zz_rotation(targets[0], targets[1], sin, cos);
            }
            _ => self.conjugate_all_backward_phased(gate, targets),
        }
    }

    #[inline(always)]
    fn branch_z_rotation_angle(&mut self, qubit: usize, theta: f64) {
        let (sin, cos) = theta.sin_cos();
        self.branch_z_rotation(qubit, sin, cos);
    }

    fn truncate(&mut self, epsilon: f64) -> f64 {
        let mut discarded = 0.0;
        self.terms.retain(|_, coeff| {
            if coeff.norm() < epsilon {
                discarded += coeff.norm();
                false
            } else {
                true
            }
        });
        discarded
    }

    fn diagonal_expectation(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for (pauli, coeff) in &self.terms {
            if pauli.is_diagonal() {
                sum += coeff;
            }
        }
        sum.re
    }
}

/// Per-qubit `⟨Z_q⟩` values from a deterministic sparse Pauli dynamics run.
pub struct SpdResult {
    /// `⟨Z_q⟩` per qubit; exact when nothing was truncated.
    pub expectations: Vec<f64>,
    /// Branching gates in the circuit: `T`, `Tdg`, `Rz`, `P`, and `Rzz`.
    pub t_count: usize,
    /// Peak size of the weighted Pauli sum across all per-qubit runs, not the
    /// truncation budget passed in.
    pub max_terms: usize,
    /// Total coefficient magnitude discarded by truncation (0 for exact runs).
    pub total_discarded: f64,
}

/// Deterministic SPD on a joint Pauli observable.
///
/// Starts with the single weighted term `(observable, 1.0)`, backward-
/// propagates through every gate, splits each anticommuting term at an
/// `Rz(theta)` or `Rzz(theta)` into `cos(theta)` and `-i sin(theta)`
/// branches, and truncates terms whose
/// magnitude falls below `epsilon` whenever the sum exceeds `max_terms`.
/// Returns `⟨0^n| U† P U |0^n⟩` as the sum of remaining diagonal-term
/// coefficients.
///
/// # Truncation
///
/// Every rotation with a non-Clifford angle branches, so a variational circuit
/// branches on each of its rotations rather than only on its `T` gates, and the
/// term count is bounded by `2^(in-support rotations)`. With `max_terms == 0`
/// the run is exact and errors once the sum passes an internal ceiling. With
/// `max_terms > 0`, sub-`epsilon` terms are dropped whenever the sum exceeds
/// the budget and `total_discarded` bounds the resulting error, per the
/// 1-norm bound `|error| ≤ Σ |discarded|`. An `epsilon` too small to prune the
/// growth still reaches the ceiling and errors there: the run never returns a
/// silently over-truncated value.
pub fn run_spd_observable(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    let lowered = validate_and_lower(circuit, "SPD observable")?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let t_count = count_branching_gates(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;

    let mut sum = WeightedPauliSum::new();
    sum.insert(obs, obs_coeff);
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for inst in circuit.instructions.iter().rev() {
        if let Instruction::Gate { gate, targets } = inst {
            sum.apply_backward(gate, targets);
        }
        if max_terms > 0 && sum.terms.len() > max_terms {
            total_discarded += sum.truncate(epsilon);
        }
        check_spd_term_ceiling(sum.terms.len())?;
        if sum.terms.len() > peak_terms {
            peak_terms = sum.terms.len();
        }
    }

    if epsilon > 0.0 {
        total_discarded += sum.truncate(epsilon);
    }

    Ok(SpdObservableResult {
        mean: sum.diagonal_expectation(),
        t_count,
        peak_terms,
        total_discarded,
    })
}

/// Compute `⟨Z_q⟩` for every qubit of a unitary circuit of Clifford gates and
/// Pauli rotations, via deterministic sparse Pauli dynamics.
///
/// Each qubit's `Z_q` propagates backward as a weighted Pauli sum; every
/// rotation with a non-Clifford angle doubles the in-support terms. When the
/// sum exceeds `max_terms`, terms with coefficient magnitude below `epsilon`
/// are dropped. `max_terms == 0` disables truncation: exact, but the sum must
/// stay under a hard term ceiling. See [`run_spd_observable`] for a single
/// joint observable and for the truncation contract.
pub fn run_spd(circuit: &Circuit, epsilon: f64, max_terms: usize) -> Result<SpdResult> {
    let lowered = validate_and_lower(circuit, "SPD")?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_branching_gates(circuit);

    let mut expectations = Vec::with_capacity(n);
    let mut peak_terms = 0usize;
    let mut total_discarded = 0.0;

    enum Rot {
        Single { sin: f64, cos: f64 },
        Pair { sin: f64, cos: f64 },
    }
    let rotations: Vec<Option<Rot>> = circuit
        .instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Gate { gate, .. } => {
                if let Some((sin, cos)) = z_rotation_angle(gate).map(f64::sin_cos) {
                    Some(Rot::Single { sin, cos })
                } else {
                    zz_rotation_angle(gate)
                        .map(f64::sin_cos)
                        .map(|(sin, cos)| Rot::Pair { sin, cos })
                }
            }
            _ => None,
        })
        .collect();

    for q in 0..n {
        let mut sum = WeightedPauliSum::new();
        sum.insert(PauliVec::z_on_qubit(num_words, q), Complex64::new(1.0, 0.0));

        for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
            if let Instruction::Gate { gate, targets } = inst {
                match rotations[idx] {
                    Some(Rot::Single { sin, cos }) => sum.branch_z_rotation(targets[0], sin, cos),
                    Some(Rot::Pair { sin, cos }) => {
                        sum.branch_zz_rotation(targets[0], targets[1], sin, cos)
                    }
                    None => sum.conjugate_all_backward_phased(gate, targets),
                }
            }

            if max_terms > 0 && sum.terms.len() > max_terms {
                total_discarded += sum.truncate(epsilon);
            }
            check_spd_term_ceiling(sum.terms.len())?;

            if sum.terms.len() > peak_terms {
                peak_terms = sum.terms.len();
            }
        }

        if epsilon > 0.0 {
            total_discarded += sum.truncate(epsilon);
        }

        expectations.push(sum.diagonal_expectation());
    }

    Ok(SpdResult {
        expectations,
        t_count,
        max_terms: peak_terms,
        total_discarded,
    })
}

/// Inverse light cone of a Pauli observable under a circuit, computed
/// conservatively by gate-graph reachability.
///
/// Returns, for each instruction index in `circuit.instructions`, whether the
/// gate at that index can affect the backward-propagated observable. A gate is
/// in the cone if its support intersects the current cone-qubit set when the
/// circuit is traversed in reverse from the observable.
///
/// Exactness: a gate whose target set is disjoint from the cone-qubit set at
/// its backward-traversal depth conjugates the propagated observable trivially
/// (`U_k^dag P_k U_k = P_k` because the support of `P_k` is contained in the
/// cone set at depth `k`, and `U_k` acts as identity outside its targets).
/// Removing those gates from the backward pass is therefore exact.
pub fn inverse_light_cone(circuit: &Circuit, observable: &[PauliTerm]) -> Vec<bool> {
    let mut cone: std::collections::HashSet<usize> = observable.iter().map(|t| t.qubit).collect();
    let n_inst = circuit.instructions.len();
    let mut keep = vec![false; n_inst];

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        let targets: &[usize] = match inst {
            Instruction::Gate { targets, .. } => targets,
            _ => continue,
        };
        let touches = targets.iter().any(|q| cone.contains(q));
        if touches {
            keep[idx] = true;
            for &q in targets {
                cone.insert(q);
            }
        }
    }
    keep
}

/// SPD on a joint Pauli observable, restricted to the inverse light cone.
///
/// Identical in result to `run_spd_observable`, but skips gates whose support
/// is disjoint from the propagated observable's causal cone. For QEC syndrome
/// and detector observables with bounded spatial support, this turns the SPD
/// cliff from a function of the total branching-gate count into a function of
/// the in-cone count.
pub fn run_spd_observable_light_cone(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    let lowered = validate_and_lower(circuit, "light-cone SPD observable")?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let t_count = count_branching_gates(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;
    let keep = inverse_light_cone(circuit, observable);

    let mut sum = WeightedPauliSum::new();
    sum.insert(obs, obs_coeff);
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        if !keep[idx] {
            continue;
        }
        if let Instruction::Gate { gate, targets } = inst {
            sum.apply_backward(gate, targets);
        }
        if max_terms > 0 && sum.terms.len() > max_terms {
            total_discarded += sum.truncate(epsilon);
        }
        check_spd_term_ceiling(sum.terms.len())?;
        if sum.terms.len() > peak_terms {
            peak_terms = sum.terms.len();
        }
    }

    if epsilon > 0.0 {
        total_discarded += sum.truncate(epsilon);
    }

    Ok(SpdObservableResult {
        mean: sum.diagonal_expectation(),
        t_count,
        peak_terms,
        total_discarded,
    })
}

#[cfg(test)]
fn spd_to_probabilities(result: &SpdResult) -> Vec<f64> {
    super::expectations_to_marginals(&result.expectations)
        .into_iter()
        .flat_map(|(p0, p1)| [p0, p1])
        .collect()
}

#[cfg(test)]
#[path = "unified_pauli_tests.rs"]
mod tests;
