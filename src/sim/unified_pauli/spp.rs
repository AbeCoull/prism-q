//! Stochastic Pauli propagation (SPP): each sample carries one Pauli string backward
//! and branches at a Pauli rotation with a recorded complex weight.

use num_complex::Complex64;
use rand::SeedableRng;
use rand::{Rng, RngExt};
use rand_chacha::ChaCha8Rng;

use super::{
    PauliAxis, PauliTerm, clifford_conjugation_phase, count_branching_gates, pauli_vec_from_terms,
    validate_and_lower, z_rotation_angle, zz_rotation_angle,
};
use crate::circuit::clifford_t::is_pauli_native;
use crate::circuit::{Circuit, Instruction, SmallVec, smallvec};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::compiled::{PauliVec, flip_bit, propagate_backward};

/// Marks a [`CoalescedOp::ZRot`] with no second qubit.
pub(super) const ZROT_SINGLE: u32 = u32::MAX;

/// The ops slice streams through the SPP per-sample loop, so the layout is
/// deliberate on two counts, both measured on op-heavy rows: the qubits pack
/// as `u32` to hold the enum at 40 bytes (48 cost +2.6% to +5.8%), and `Rzz`
/// shares the `ZRot` variant via the `ZROT_SINGLE` sentinel instead of adding
/// a third variant, which kept an outlined call in the sample loop's match
/// and cost about the same again. A `PauliRot` string is boxed for the same
/// size bound.
pub(super) enum CoalescedOp {
    SmallCliff(Vec<(Gate, SmallVec<[usize; 4]>)>),
    ZRot {
        qubit: u32,
        pair: u32,
        branch: RotBranch,
    },
    String(Box<StringRot>),
}

/// Gates SPP takes without lowering: the shared native set plus `PauliRot`.
fn is_spp_native(gate: &Gate) -> bool {
    is_pauli_native(gate) || matches!(gate, Gate::PauliRot(_))
}

pub(super) fn coalesce_cliffords(circuit: &Circuit) -> Vec<CoalescedOp> {
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
            } else if let Gate::PauliRot(data) = gate {
                flush_cliff_buf(&mut cliff_buf, &mut ops);
                ops.push(CoalescedOp::String(Box::new(StringRot::new(
                    data.theta(),
                    targets,
                    data.axes(),
                ))));
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

/// Sampling data for one Z-axis rotation, built once per gate so the per-sample
/// loop stays free of trigonometry.
#[derive(Clone, Copy)]
pub(super) struct RotBranch {
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
    pub(super) fn new(theta: f64) -> Self {
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
pub(super) fn branch_z_rotation(
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

/// Sampling data for one `PauliRot` string `exp(-i θ P / 2)`, with the bits of `P`
/// stored over the words its support spans, starting at `first_word`.
pub(super) struct StringRot {
    first_word: usize,
    x: SmallVec<[u64; 2]>,
    z: SmallVec<[u64; 2]>,
    branch: RotBranch,
    flip_weight: Complex64,
}

impl StringRot {
    /// `branch` carries the `Rz(θ)` probabilities, since the string conjugates to
    /// `Rz(θ)` on one qubit. With `P = i^num_y X^px Z^pz` in the ordered letter
    /// convention, the flip `Q -> Q P` scales the `Rz` flip weight by `i^num_y`, and
    /// by `-1` when an odd number of the source's z bits pass the string's x bits.
    fn new(theta: f64, targets: &[usize], axes: &[PauliAxis]) -> Self {
        let first_word = targets.iter().min().map_or(0, |&q| q / 64);
        let last_word = targets.iter().max().map_or(0, |&q| q / 64);
        let span = last_word - first_word + 1;
        let mut x: SmallVec<[u64; 2]> = smallvec![0; span];
        let mut z: SmallVec<[u64; 2]> = smallvec![0; span];
        let mut num_y = 0u32;
        for (&q, axis) in targets.iter().zip(axes) {
            let bit = q - first_word * 64;
            match axis {
                PauliAxis::X => flip_bit(&mut x, bit),
                PauliAxis::Z => flip_bit(&mut z, bit),
                PauliAxis::Y => {
                    flip_bit(&mut x, bit);
                    flip_bit(&mut z, bit);
                    num_y += 1;
                }
            }
        }
        let branch = RotBranch::new(theta);
        let f = branch.flip_weight_im;
        let flip_weight = match num_y & 3 {
            0 => Complex64::new(0.0, f),
            1 => Complex64::new(-f, 0.0),
            2 => Complex64::new(0.0, -f),
            _ => Complex64::new(f, 0.0),
        };
        Self {
            first_word,
            x,
            z,
            branch,
            flip_weight,
        }
    }
}

/// Stochastic branch for a `PauliRot` string: fires when the propagated Pauli
/// anticommutes with the string, and the flip right-multiplies by it.
#[inline]
fn branch_pauli_string(pauli: &mut PauliVec, rot: &StringRot, rng: &mut impl Rng) -> Complex64 {
    let words = rot.first_word..rot.first_word + rot.x.len();
    let (qx, qz) = (&pauli.x[words.clone()], &pauli.z[words.clone()]);
    let overlap = (qx.iter().zip(qz))
        .zip(rot.x.iter().zip(&rot.z))
        .fold(0, |acc, ((qx, qz), (px, pz))| acc ^ (qx & pz) ^ (qz & px));
    if overlap.count_ones() & 1 == 0 {
        return Complex64::new(1.0, 0.0);
    }

    if rng.random_bool(rot.branch.p_keep) {
        return Complex64::new(rot.branch.keep_weight, 0.0);
    }

    let passed = (qz.iter().zip(&rot.x)).fold(0, |acc, (qz, px)| acc ^ (qz & px));
    for (w, p) in pauli.x[words.clone()].iter_mut().zip(&rot.x) {
        *w ^= p;
    }
    for (w, p) in pauli.z[words].iter_mut().zip(&rot.z) {
        *w ^= p;
    }
    if passed.count_ones() & 1 == 1 {
        -rot.flip_weight
    } else {
        rot.flip_weight
    }
}

pub(super) fn backward_propagate_coalesced(
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
                    weight *= clifford_conjugation_phase(gate, targets, &pauli.x, &pauli.z);
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
            CoalescedOp::String(rot) => {
                weight *= branch_pauli_string(&mut pauli, rot, rng);
            }
        }
    }

    (pauli, weight)
}

/// Per-qubit `⟨Z_q⟩` estimates from a stochastic Pauli propagation run.
pub struct SppResult {
    pub expectations: Vec<f64>,
    pub std_errors: Vec<f64>,
    /// Samples drawn per qubit, not in total.
    pub num_samples: usize,
    /// Branching gates in the circuit: `T`, `Tdg`, `Rz`, `P`, `Rzz`, `Rx`, `Ry`
    /// and `PauliRot`.
    pub t_count: usize,
    /// Fraction of samples whose propagated Pauli was diagonal (contributed a
    /// nonzero value).
    pub nonzero_fraction: f64,
}

/// Running mean and unbiased sample variance by Welford's update, so the variance
/// does not cancel catastrophically when the mean sits near the sample values.
#[derive(Clone, Copy, Default)]
pub(crate) struct Welford {
    mean: f64,
    m2: f64,
    count: usize,
}

impl Welford {
    #[inline]
    pub(crate) fn push(&mut self, val: f64) {
        self.count += 1;
        let delta = val - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (val - self.mean);
    }

    pub(crate) fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance with the `n - 1` denominator; zero below two samples.
    pub(crate) fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    pub(crate) fn std_error(&self) -> f64 {
        (self.variance() / self.count.max(1) as f64).sqrt()
    }
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

    let mut stats = Welford::default();
    let mut nonzero = 0usize;

    for _ in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            weight.re
        } else {
            0.0
        };
        stats.push(val);
    }

    (stats.mean(), stats.std_error(), nonzero)
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

/// Estimate `⟨0^n| U† P U |0^n⟩` for joint Pauli observable `P` on a circuit
/// `U` of Clifford gates and Pauli rotations, via stochastic Pauli
/// propagation.
///
/// Each sample backward-propagates the observable through the circuit
/// (Clifford segments as coalesced gate runs, `Rz(theta)`, the two-qubit
/// `Rzz(theta)` and multi-qubit `PauliRot` strings via a stochastic Pauli branch
/// that records a complex weight; `Rx` and `Ry` lower to Clifford conjugation
/// around one `Rz` first). The contribution is `Re(weight)` when
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
    let lowered = validate_and_lower(circuit, "SPP observable", is_spp_native)?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let t_count = count_branching_gates(circuit);
    let ops = coalesce_cliffords(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut stats = Welford::default();
    let mut nonzero = 0usize;

    for _ in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            (obs_coeff * weight).re
        } else {
            0.0
        };
        stats.push(val);
    }

    let nonzero_fraction = nonzero as f64 / num_samples.max(1) as f64;

    Ok(SppObservableResult {
        mean: stats.mean(),
        std_error: stats.std_error(),
        variance: stats.variance(),
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
    let lowered = validate_and_lower(circuit, "SPP", is_spp_native)?;
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
                    crate::sim::mix_seed(seed, q),
                )
            })
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(f64, f64, usize)> = (0..n)
        .map(|q| {
            estimate_qubit_expectation(
                &ops,
                q,
                num_words,
                num_samples,
                crate::sim::mix_seed(seed, q),
            )
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
