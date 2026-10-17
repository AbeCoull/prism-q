//! Product-formula builder: Trotter and Suzuki circuits approximating `exp(-i H t)`
//! for a Pauli-sum Hamiltonian, emitted as the Pauli rotations fusion batches.

use super::Circuit;
use crate::sim::PauliObservable;
use crate::sim::unified_pauli::PauliTerm;

/// Product formula [`Circuit::add_trotter`] expands each step with.
///
/// The error of one step of length `dt` is `O(dt^2)` for `First`, `O(dt^3)`
/// for `Second`, and `O(dt^(2k+1))` for `Suzuki(k)`, the recursive symmetric
/// formula of order `2k`; `Suzuki(1)` is `Second`. A `Suzuki(k)` step costs
/// `5^(k-1)` second-order substeps. `Suzuki(0)` panics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrotterOrder {
    First,
    Second,
    Suzuki(usize),
}

impl Circuit {
    /// Append `steps` product-formula steps approximating `exp(-i H t)` for
    /// `hamiltonian = sum_j c_j P_j`, each term evolving as the rotation
    /// `exp(-i c_j P_j dt)` with `dt = t / steps`, appended through
    /// [`add_pauli_rotation`](Self::add_pauli_rotation) with angle `2 c_j dt`.
    ///
    /// Terms follow the observable's canonical order. A zero coefficient
    /// appends nothing, and an identity term is dropped: it contributes the
    /// global phase `exp(-i c t)`, which no measurement or expectation sees.
    /// Consecutive rotations of one string merge into one, so a second-order
    /// step over `m` terms emits `2m - 1` rotations and adjacent steps share
    /// their boundary rotation.
    ///
    /// # Panics
    /// Panics if `steps` is zero, if `order` is `Suzuki(0)`, or if a term names
    /// a qubit out of bounds.
    pub fn add_trotter(
        &mut self,
        hamiltonian: &PauliObservable,
        t: f64,
        steps: usize,
        order: TrotterOrder,
    ) {
        assert!(steps > 0, "Trotter expansion needs at least one step");
        if let TrotterOrder::Suzuki(0) = order {
            panic!("Suzuki formula needs k >= 1");
        }
        let terms: Vec<(f64, &[PauliTerm])> = hamiltonian
            .terms()
            .iter()
            .filter(|(coefficient, string)| *coefficient != 0.0 && !string.is_empty())
            .map(|(coefficient, string)| (*coefficient, string.as_slice()))
            .collect();
        let mut rotations = Rotations {
            circuit: self,
            terms: &terms,
            pending: None,
        };
        let dt = t / steps as f64;
        for _ in 0..steps {
            match order {
                TrotterOrder::First => rotations.first_order(dt),
                TrotterOrder::Second => rotations.second_order(dt),
                TrotterOrder::Suzuki(k) => rotations.suzuki(k, dt),
            }
        }
        rotations.flush();
    }
}

/// Emits term evolutions as rotations, summing the evolution time of
/// consecutive evolutions of one term so the shared boundary of two symmetric
/// substeps lands as a single gate.
struct Rotations<'a> {
    circuit: &'a mut Circuit,
    terms: &'a [(f64, &'a [PauliTerm])],
    pending: Option<(usize, f64)>,
}

impl Rotations<'_> {
    fn evolve(&mut self, index: usize, time: f64) {
        match &mut self.pending {
            Some((pending, total)) if *pending == index => *total += time,
            _ => {
                self.flush();
                self.pending = Some((index, time));
            }
        }
    }

    fn flush(&mut self) {
        if let Some((index, time)) = self.pending.take() {
            let (coefficient, factors) = self.terms[index];
            self.circuit
                .add_pauli_rotation(2.0 * coefficient * time, factors);
        }
    }

    fn first_order(&mut self, dt: f64) {
        for index in 0..self.terms.len() {
            self.evolve(index, dt);
        }
    }

    fn second_order(&mut self, dt: f64) {
        let half = 0.5 * dt;
        for index in 0..self.terms.len() {
            self.evolve(index, half);
        }
        for index in (0..self.terms.len()).rev() {
            self.evolve(index, half);
        }
    }

    // S_2k(dt) = S_2k-2(p dt)^2 S_2k-2((1 - 4p) dt) S_2k-2(p dt)^2 with
    // p = 1 / (4 - 4^(1 / (2k - 1))), the Suzuki fractal recursion.
    fn suzuki(&mut self, k: usize, dt: f64) {
        if k == 1 {
            self.second_order(dt);
            return;
        }
        let p = 1.0 / (4.0 - 4f64.powf(1.0 / (2 * k - 1) as f64));
        let inner = p * dt;
        self.suzuki(k - 1, inner);
        self.suzuki(k - 1, inner);
        self.suzuki(k - 1, (1.0 - 4.0 * p) * dt);
        self.suzuki(k - 1, inner);
        self.suzuki(k - 1, inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Instruction;
    use crate::gates::Gate;

    fn heisenberg(n: usize) -> PauliObservable {
        let mut terms = Vec::new();
        for q in 0..n - 1 {
            terms.push((1.0, vec![PauliTerm::x(q), PauliTerm::x(q + 1)]));
            terms.push((0.8, vec![PauliTerm::y(q), PauliTerm::y(q + 1)]));
            terms.push((0.6, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
        }
        for q in 0..n {
            terms.push((0.3, vec![PauliTerm::x(q)]));
        }
        PauliObservable::from_terms(terms).unwrap()
    }

    fn angles(circuit: &Circuit) -> Vec<f64> {
        circuit
            .instructions
            .iter()
            .map(|inst| match inst {
                Instruction::Gate {
                    gate: Gate::PauliRot(data),
                    ..
                } => data.theta(),
                Instruction::Gate {
                    gate: Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) | Gate::Rzz(theta),
                    ..
                } => *theta,
                other => panic!("unexpected instruction {other:?}"),
            })
            .collect()
    }

    #[test]
    fn first_order_emits_one_rotation_per_term_per_step() {
        let h = heisenberg(4);
        let mut circuit = Circuit::new(4, 0);
        circuit.add_trotter(&h, 0.5, 3, TrotterOrder::First);
        assert_eq!(circuit.instructions.len(), 3 * h.num_terms());
        let (coefficient, _) = &h.terms()[0];
        assert!((angles(&circuit)[0] - 2.0 * coefficient * 0.5 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn second_order_merges_the_middle_and_the_step_boundary() {
        let h = heisenberg(4);
        let m = h.num_terms();
        let mut one = Circuit::new(4, 0);
        one.add_trotter(&h, 0.4, 1, TrotterOrder::Second);
        assert_eq!(one.instructions.len(), 2 * m - 1);
        let mut three = Circuit::new(4, 0);
        three.add_trotter(&h, 0.4, 3, TrotterOrder::Second);
        assert_eq!(three.instructions.len(), 3 * (2 * m - 1) - 2);

        let dt = 0.4;
        let angles = angles(&one);
        let (first, _) = &h.terms()[0];
        let (last, _) = &h.terms()[m - 1];
        assert!(
            (angles[0] - first * dt).abs() < 1e-15,
            "half step at the edge"
        );
        assert!(
            (angles[m - 1] - 2.0 * last * dt).abs() < 1e-15,
            "merged middle"
        );
        assert!((angles[2 * m - 2] - first * dt).abs() < 1e-15);
    }

    #[test]
    fn suzuki_one_is_second_order_and_suzuki_two_costs_five_substeps() {
        let h = heisenberg(3);
        let m = h.num_terms();
        let mut second = Circuit::new(3, 0);
        second.add_trotter(&h, 0.3, 2, TrotterOrder::Second);
        let mut suzuki = Circuit::new(3, 0);
        suzuki.add_trotter(&h, 0.3, 2, TrotterOrder::Suzuki(1));
        assert_eq!(angles(&second), angles(&suzuki));

        let mut fourth = Circuit::new(3, 0);
        fourth.add_trotter(&h, 0.3, 1, TrotterOrder::Suzuki(2));
        assert_eq!(fourth.instructions.len(), 5 * (2 * m - 1) - 4);
        let total: f64 = angles(&fourth).iter().sum();
        let total_coefficient: f64 = h.terms().iter().map(|(c, _)| c).sum();
        assert!(
            (total - 2.0 * total_coefficient * 0.3).abs() < 1e-12,
            "substep lengths sum to the step"
        );
    }

    #[test]
    fn zero_and_identity_terms_append_nothing() {
        let h = PauliObservable::from_terms([
            (0.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
            (2.5, vec![]),
            (0.7, vec![PauliTerm::z(0)]),
        ])
        .unwrap();
        let mut circuit = Circuit::new(2, 0);
        circuit.add_trotter(&h, 1.0, 2, TrotterOrder::First);
        // The surviving term is the only term, so its two steps merge too.
        assert!(matches!(
            circuit.instructions.as_slice(),
            [Instruction::Gate {
                gate: Gate::Rz(theta),
                ..
            }] if (theta - 1.4).abs() < 1e-15
        ));
    }

    #[test]
    #[should_panic(expected = "at least one step")]
    fn zero_steps_panics() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_trotter(&heisenberg(2), 1.0, 0, TrotterOrder::First);
    }

    #[test]
    #[should_panic(expected = "k >= 1")]
    fn suzuki_zero_panics() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_trotter(&heisenberg(2), 1.0, 1, TrotterOrder::Suzuki(0));
    }
}
