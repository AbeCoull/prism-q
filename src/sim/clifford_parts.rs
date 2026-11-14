//! Exact Pauli expectation values of a unitary Clifford+T circuit as a product over the
//! parts of its register in the product gauge, the sets of qubits excess rotations link.

use num_complex::Complex64;
use std::f64::consts::FRAC_PI_8;

use super::{
    CliffordRegister, MAX_REGISTER_QUBITS, RegisterPauli, apply_rotation, bit, fold_anchor_rest,
    groups, join, mul_rows,
};
use crate::circuit::{Circuit, Instruction};
use crate::gates::Gate;
use crate::qec::camps_prefix::SignedPauli;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm, propagate_register_rotations};

/// Most register qubits the gauge holds, one bit of a word each.
const MAX_GAUGE_QUBITS: usize = 64;

/// Most excess rotations a part propagates an observable through: at most `2^20` terms,
/// the SPD term ceiling.
const MAX_PROPAGATED_EXCESS: usize = 20;

/// Register amplitude updates one propagated term costs per rotation it passes.
/// No bench row brackets this yet.
const PROPAGATED_TERM_AMPLITUDES: f64 = 16.0;

/// The state `C (|0> ⊗ E (⊗_b α_b))` of a unitary Clifford+T circuit. A T gate whose axis
/// flips a qubit in `|0>` anchors it as register bit `b` in the one-qubit state `α_b`, with
/// the rest of the axis folded into `C`; every other T gate is an excess rotation in `E`,
/// applied in circuit order.
pub(super) struct ProductGauge {
    pub(super) register: CliffordRegister,
    /// `[<0|α_b>, <1|α_b>]` per register bit.
    anchors: Vec<[Complex64; 2]>,
    /// `(R, θ)` for each excess rotation `exp(-i θ R)`, `R` over register bits.
    pub(super) excess: Vec<(RegisterPauli, f64)>,
}

impl ProductGauge {
    /// `None` past 64 register qubits or on a gate the prefix does not hold. Measurements
    /// and barriers are skipped.
    pub(super) fn build(circuit: &Circuit) -> Option<Self> {
        let mut register = CliffordRegister::new(circuit.num_qubits);
        let mut anchors = Vec::new();
        let mut excess = Vec::new();
        for inst in &circuit.instructions {
            match inst {
                Instruction::Gate {
                    gate: gate @ (Gate::T | Gate::Tdg),
                    targets,
                } => {
                    let theta = if matches!(gate, Gate::T) {
                        FRAC_PI_8
                    } else {
                        -FRAC_PI_8
                    };
                    let q = targets[0];
                    let width = register.width;
                    let p = register.admit_z(q, MAX_GAUGE_QUBITS)?;
                    if register.width == width {
                        let axis = register.restrict(&p);
                        if axis.x | axis.z != 0 {
                            excess.push((axis, theta));
                        }
                        continue;
                    }
                    let anchor = register
                        .slot
                        .iter()
                        .position(|&s| s == Some(width))
                        .expect("the admitted qubit holds the last slot");
                    fold_anchor_rest(&mut register.prefix, anchor, &p);
                    let axis = register.prefix.conjugate_z(q);
                    debug_assert!(
                        (0..circuit.num_qubits)
                            .all(|j| { j == anchor || !(bit(&axis.x, j) || bit(&axis.z, j)) })
                    );
                    let angle = if axis.phase4 == 2 { -theta } else { theta };
                    let (sin, cos) = angle.sin_cos();
                    let one = if bit(&axis.z, anchor) {
                        Complex64::new(sin, 0.0)
                    } else {
                        Complex64::new(0.0, -sin)
                    };
                    anchors.push([Complex64::new(cos, 0.0), one]);
                }
                Instruction::Gate { gate, targets } => {
                    register.prefix.apply_state_gate(gate, targets).ok()?;
                }
                Instruction::Measure { .. } | Instruction::Barrier { .. } => {}
                _ => return None,
            }
        }
        Some(Self {
            register,
            anchors,
            excess,
        })
    }

    /// `C† P C` restricted to the register, or `None` when it flips a qubit held in `|0>`,
    /// where `<P>` is zero.
    fn pull(&self, observable: &[PauliTerm]) -> Option<RegisterPauli> {
        let prefix = &self.register.prefix;
        let words = prefix.num_qubits().div_ceil(64).max(1);
        let mut pulled = SignedPauli {
            x: vec![0; words],
            z: vec![0; words],
            phase4: 0,
        };
        for term in observable {
            let (x, z) = (&prefix.inv_x[term.qubit], &prefix.inv_z[term.qubit]);
            match term.axis {
                PauliAxis::X => mul_rows(&mut pulled, x),
                PauliAxis::Z => mul_rows(&mut pulled, z),
                PauliAxis::Y => {
                    mul_rows(&mut pulled, x);
                    mul_rows(&mut pulled, z);
                    pulled.phase4 = (pulled.phase4 + 1) & 3;
                }
            }
        }
        let slot = &self.register.slot;
        if (0..slot.len()).any(|q| slot[q].is_none() && bit(&pulled.x, q)) {
            return None;
        }
        Some(self.register.restrict(&pulled))
    }

    /// Amplitudes of the part on the register bits of `mask`, lowest bit first.
    fn part_state(&self, mask: u64, excess: &[usize]) -> Vec<Complex64> {
        let mut amps = Vec::with_capacity(1 << mask.count_ones());
        amps.push(Complex64::new(1.0, 0.0));
        let mut rest = mask;
        while rest != 0 {
            let [a0, a1] = self.anchors[rest.trailing_zeros() as usize];
            rest &= rest - 1;
            let len = amps.len();
            amps.extend_from_within(..);
            for (i, amp) in amps.iter_mut().enumerate() {
                *amp *= if i < len { a0 } else { a1 };
            }
        }
        for &i in excess {
            let (p, theta) = self.excess[i];
            let local = RegisterPauli {
                x: compress(p.x, mask),
                z: compress(p.z, mask),
                phase4: p.phase4,
            };
            apply_rotation(&mut amps, local, theta);
        }
        amps
    }
}

/// Bits of `word` at the set bits of `mask`, packed from bit 0 up.
fn compress(word: u64, mask: u64) -> u64 {
    let mut packed = 0;
    let mut rest = mask;
    let mut i = 0;
    while rest != 0 {
        packed |= ((word >> rest.trailing_zeros()) & 1) << i;
        rest &= rest - 1;
        i += 1;
    }
    packed
}

enum Engine {
    /// No excess rotation: one qubit, read off its Bloch vector.
    Product,
    /// Amplitudes on the part's bits, built once before the first read.
    Dense(Vec<Complex64>),
    /// Each observable propagated backward through the part's excess rotations, then
    /// read off the Bloch vectors.
    Propagate,
}

struct Part {
    mask: u64,
    /// Indices into [`ProductGauge::excess`], in circuit order.
    excess: Vec<usize>,
    engine: Engine,
}

/// Pauli expectation values on a Clifford+T circuit, planned on its product-gauge
/// register. Each observable pulled through the Clifford is a sign times a product over
/// the parts it touches, and each part takes its cheaper exact engine.
pub(in crate::sim) struct RegisterExpectations {
    gauge: ProductGauge,
    /// `[<X>, <Y>, <Z>]` per register bit.
    bloch: Vec<[f64; 3]>,
    parts: Vec<Part>,
    pulled: Vec<Option<RegisterPauli>>,
    work: f64,
}

impl RegisterExpectations {
    /// `None` past 64 register qubits, or when a part an observable touches is both wider
    /// than [`MAX_REGISTER_QUBITS`] and holds more than [`MAX_PROPAGATED_EXCESS`] excess
    /// rotations. `circuit` must be unitary and lowered to Clifford+T.
    pub(in crate::sim) fn plan(circuit: &Circuit, observables: &[Vec<PauliTerm>]) -> Option<Self> {
        Self::plan_with(
            circuit,
            observables,
            MAX_REGISTER_QUBITS,
            MAX_PROPAGATED_EXCESS,
        )
    }

    fn plan_with(
        circuit: &Circuit,
        observables: &[Vec<PauliTerm>],
        max_dense_width: usize,
        max_excess: usize,
    ) -> Option<Self> {
        let gauge = ProductGauge::build(circuit)?;
        let mut parent: Vec<usize> = (0..gauge.register.width).collect();
        for (p, _) in &gauge.excess {
            join(&mut parent, p.x | p.z);
        }
        let mut parts: Vec<Part> = groups(&mut parent)
            .into_iter()
            .map(|mask| Part {
                mask,
                excess: Vec::new(),
                engine: Engine::Product,
            })
            .collect();
        for (i, (p, _)) in gauge.excess.iter().enumerate() {
            parts
                .iter_mut()
                .find(|part| part.mask & (p.x | p.z) != 0)
                .expect("an excess rotation lies inside one part")
                .excess
                .push(i);
        }
        let pulled: Vec<Option<RegisterPauli>> =
            observables.iter().map(|o| gauge.pull(o)).collect();

        let mut work = 0.0;
        for part in parts.iter_mut().filter(|part| !part.excess.is_empty()) {
            let uses = pulled
                .iter()
                .flatten()
                .filter(|q| (q.x | q.z) & part.mask != 0)
                .count();
            if uses == 0 {
                continue;
            }
            let width = part.mask.count_ones() as usize;
            let excess = part.excess.len();
            let dense = (width <= max_dense_width)
                .then(|| (excess + 1 + uses) as f64 * (1u64 << width) as f64);
            let propagate = (excess <= max_excess)
                .then(|| uses as f64 * PROPAGATED_TERM_AMPLITUDES * (1u64 << excess) as f64);
            let (engine, cost) = match (dense, propagate) {
                (Some(d), Some(p)) if p < d => (Engine::Propagate, p),
                (Some(d), _) => (Engine::Dense(Vec::new()), d),
                (None, Some(p)) => (Engine::Propagate, p),
                (None, None) => return None,
            };
            part.engine = engine;
            work += cost;
        }
        let bloch = gauge
            .anchors
            .iter()
            .map(|&[a, b]| {
                let ab = a.conj() * b;
                [2.0 * ab.re, 2.0 * ab.im, a.norm_sqr() - b.norm_sqr()]
            })
            .collect();
        Some(Self {
            gauge,
            bloch,
            parts,
            pulled,
            work,
        })
    }

    /// Register amplitude updates the evaluation costs, the tableau excluded.
    pub(in crate::sim) fn work(&self) -> f64 {
        self.work
    }

    pub(in crate::sim) fn values(mut self) -> Vec<f64> {
        for part in &mut self.parts {
            if let Engine::Dense(amps) = &mut part.engine {
                *amps = self.gauge.part_state(part.mask, &part.excess);
            }
        }
        self.pulled
            .iter()
            .map(|pulled| {
                let Some(q) = pulled else {
                    return 0.0;
                };
                debug_assert!(
                    q.phase4 & 1 == 0,
                    "a Hermitian observable keeps a real sign"
                );
                let support = q.x | q.z;
                let sign = if q.phase4 == 2 { -1.0 } else { 1.0 };
                self.parts
                    .iter()
                    .filter(|part| part.mask & support != 0)
                    .fold(sign, |value, part| value * self.factor(part, q))
            })
            .collect()
    }

    /// `<φ_c| Q_c |φ_c>` for the letters of `q` on `part`.
    fn factor(&self, part: &Part, q: &RegisterPauli) -> f64 {
        let (x, z) = (q.x & part.mask, q.z & part.mask);
        match &part.engine {
            Engine::Product => bloch_value(&self.bloch, x, z),
            Engine::Dense(amps) => dense_value(
                amps,
                compress(x, part.mask) as usize,
                compress(z, part.mask) as usize,
            ),
            Engine::Propagate => {
                let rotations = part.excess.iter().map(|&i| {
                    let (p, theta) = self.gauge.excess[i];
                    (p.x, p.z, p.phase4, theta)
                });
                propagate_register_rotations(x, z, rotations)
                    .into_iter()
                    .map(|(tx, tz, coeff)| coeff.re * bloch_value(&self.bloch, tx, tz))
                    .sum()
            }
        }
    }
}

/// `<⊗_b α_b| (x, z) |⊗_b α_b>` for a letter string with no phase.
fn bloch_value(bloch: &[[f64; 3]], x: u64, z: u64) -> f64 {
    let mut value = 1.0;
    let mut rest = x | z;
    while rest != 0 {
        let b = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let letter = match ((x >> b) & 1, (z >> b) & 1) {
            (1, 0) => 0,
            (1, _) => 1,
            _ => 2,
        };
        value *= bloch[b][letter];
    }
    value
}

/// `<φ| P |φ>` for the letter string `(x, z)` with no phase, using
/// `P|b> = i^#Y (-1)^|b & z| |b ^ x>`.
fn dense_value(amps: &[Complex64], x: usize, z: usize) -> f64 {
    let mut sum = Complex64::new(0.0, 0.0);
    for (b, &amp) in amps.iter().enumerate() {
        let term = amps[b ^ x].conj() * amp;
        if (b & z).count_ones() & 1 == 1 {
            sum -= term;
        } else {
            sum += term;
        }
    }
    (sum * Complex64::i().powu((x & z).count_ones())).re
}

#[cfg(test)]
#[path = "clifford_parts_tests.rs"]
mod tests;
