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
#[path = "trotter_tests.rs"]
mod tests;
