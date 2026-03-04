//! Fluent circuit builder with method chaining.
//!
//! ```
//! use prism_q::CircuitBuilder;
//!
//! let result = CircuitBuilder::new(2)
//!     .h(0).cx(0, 1)
//!     .run(42)
//!     .expect("simulation failed");
//! let probs = result.probabilities.expect("no probabilities").to_vec();
//! assert!((probs[0] - 0.5).abs() < 1e-10);
//! assert!((probs[3] - 0.5).abs() < 1e-10);
//! ```

use num_complex::Complex64;

use super::{Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::gates::Gate;

/// Fluent builder for quantum circuits.
///
/// Provides method-chaining syntax for circuit construction. Each gate
/// method returns `&mut Self`, allowing compact one-liner circuits.
/// Call [`build`](Self::build) to extract the finished [`Circuit`], or
/// use [`run`](Self::run) / [`run_with`](Self::run_with) for direct execution.
pub struct CircuitBuilder {
    circuit: Circuit,
}

macro_rules! gate_1q {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self, q: usize) -> &mut Self {
            self.circuit.add_gate(Gate::$variant, &[q]);
            self
        }
    };
}

macro_rules! gate_1q_param {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self, theta: f64, q: usize) -> &mut Self {
            self.circuit.add_gate(Gate::$variant(theta), &[q]);
            self
        }
    };
}

impl CircuitBuilder {
    /// Create a builder for a circuit with `num_qubits` qubits and no classical bits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits, 0),
        }
    }

    /// Create a builder with explicit qubit and classical bit counts.
    pub fn new_with_classical(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits, num_classical_bits),
        }
    }

    gate_1q!(id, Id);
    gate_1q!(x, X);
    gate_1q!(y, Y);
    gate_1q!(z, Z);
    gate_1q!(h, H);
    gate_1q!(s, S);
    gate_1q!(sdg, Sdg);
    gate_1q!(t, T);
    gate_1q!(tdg, Tdg);
    gate_1q!(sx, SX);
    gate_1q!(sxdg, SXdg);

    gate_1q_param!(rx, Rx);
    gate_1q_param!(ry, Ry);
    gate_1q_param!(rz, Rz);
    gate_1q_param!(p, P);

    pub fn rzz(&mut self, theta: f64, q0: usize, q1: usize) -> &mut Self {
        self.circuit.add_gate(Gate::Rzz(theta), &[q0, q1]);
        self
    }

    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.circuit.add_gate(Gate::Cx, &[control, target]);
        self
    }

    pub fn cz(&mut self, q0: usize, q1: usize) -> &mut Self {
        self.circuit.add_gate(Gate::Cz, &[q0, q1]);
        self
    }

    pub fn swap(&mut self, q0: usize, q1: usize) -> &mut Self {
        self.circuit.add_gate(Gate::Swap, &[q0, q1]);
        self
    }

    pub fn cu(&mut self, mat: [[Complex64; 2]; 2], control: usize, target: usize) -> &mut Self {
        self.circuit.add_gate(Gate::cu(mat), &[control, target]);
        self
    }

    pub fn cphase(&mut self, theta: f64, control: usize, target: usize) -> &mut Self {
        self.circuit
            .add_gate(Gate::cphase(theta), &[control, target]);
        self
    }

    pub fn mcu(
        &mut self,
        mat: [[Complex64; 2]; 2],
        controls: &[usize],
        target: usize,
    ) -> &mut Self {
        let mut targets: SmallVec<[usize; 4]> = controls.into();
        targets.push(target);
        self.circuit.instructions.push(Instruction::Gate {
            gate: Gate::mcu(mat, controls.len() as u8),
            targets,
        });
        self
    }

    pub fn measure(&mut self, qubit: usize, classical_bit: usize) -> &mut Self {
        self.circuit.add_measure(qubit, classical_bit);
        self
    }

    /// Measure all qubits into classical bits with matching indices.
    ///
    /// Expands `num_classical_bits` if needed to accommodate all qubits.
    pub fn measure_all(&mut self) -> &mut Self {
        let n = self.circuit.num_qubits;
        if self.circuit.num_classical_bits < n {
            self.circuit.num_classical_bits = n;
        }
        for q in 0..n {
            self.circuit.add_measure(q, q);
        }
        self
    }

    pub fn barrier(&mut self, qubits: &[usize]) -> &mut Self {
        self.circuit.add_barrier(qubits);
        self
    }

    pub fn conditional(
        &mut self,
        condition: ClassicalCondition,
        gate: Gate,
        targets: &[usize],
    ) -> &mut Self {
        self.circuit.instructions.push(Instruction::Conditional {
            condition,
            gate,
            targets: targets.into(),
        });
        self
    }

    pub fn gate(&mut self, gate: Gate, targets: &[usize]) -> &mut Self {
        self.circuit.add_gate(gate, targets);
        self
    }

    /// Extract the finished circuit, replacing the builder's internal circuit with an empty one.
    pub fn build(&mut self) -> Circuit {
        std::mem::replace(&mut self.circuit, Circuit::new(0, 0))
    }

    /// Borrow the circuit without consuming the builder.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Execute with automatic backend selection.
    pub fn run(&self, seed: u64) -> crate::Result<crate::sim::SimulationResult> {
        crate::sim::run(&self.circuit, seed)
    }

    /// Execute with explicit backend selection.
    pub fn run_with(
        &self,
        kind: crate::sim::BackendKind,
        seed: u64,
    ) -> crate::Result<crate::sim::SimulationResult> {
        crate::sim::run_with(kind, &self.circuit, seed)
    }

    /// Execute multi-shot sampling.
    pub fn run_shots(&self, num_shots: usize, seed: u64) -> crate::Result<crate::sim::ShotsResult> {
        crate::sim::run_shots(&self.circuit, num_shots, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn builder_bell_state() {
        let c = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        assert_eq!(c.instructions.len(), 2);
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.num_classical_bits, 0);
    }

    #[test]
    fn builder_parametric() {
        let c = CircuitBuilder::new(2).rx(PI, 0).rz(PI / 2.0, 1).build();
        assert_eq!(c.instructions.len(), 2);
        match &c.instructions[0] {
            Instruction::Gate { gate, targets } => {
                assert!(matches!(gate, Gate::Rx(_)));
                assert_eq!(targets.as_slice(), &[0]);
            }
            _ => panic!("expected Gate instruction"),
        }
    }

    #[test]
    fn builder_measure_all() {
        let c = CircuitBuilder::new(3).h(0).measure_all().build();
        assert_eq!(c.num_classical_bits, 3);
        let measures: Vec<_> = c
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .collect();
        assert_eq!(measures.len(), 3);
    }

    #[test]
    fn builder_conditional() {
        let c = CircuitBuilder::new_with_classical(2, 1)
            .x(0)
            .measure(0, 0)
            .conditional(ClassicalCondition::BitIsOne(0), Gate::X, &[1])
            .build();
        assert_eq!(c.instructions.len(), 3);
        assert!(matches!(
            &c.instructions[2],
            Instruction::Conditional { .. }
        ));
    }

    #[test]
    fn builder_run_matches_direct() {
        let builder_result = CircuitBuilder::new(2)
            .h(0)
            .cx(0, 1)
            .run(42)
            .expect("builder run failed");
        let bp = builder_result.probabilities.expect("no probs").to_vec();

        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let direct_result = crate::sim::run(&c, 42).expect("direct run failed");
        let dp = direct_result.probabilities.expect("no probs").to_vec();

        assert_eq!(bp.len(), dp.len());
        for (b, d) in bp.iter().zip(dp.iter()) {
            assert!((b - d).abs() < 1e-12);
        }
    }

    #[test]
    fn builder_generic_gate() {
        let c = CircuitBuilder::new(2).gate(Gate::Swap, &[0, 1]).build();
        assert_eq!(c.instructions.len(), 1);
        match &c.instructions[0] {
            Instruction::Gate { gate, targets } => {
                assert!(matches!(gate, Gate::Swap));
                assert_eq!(targets.as_slice(), &[0, 1]);
            }
            _ => panic!("expected Gate instruction"),
        }
    }

    #[test]
    fn builder_cphase() {
        let c = CircuitBuilder::new(2).cphase(PI / 4.0, 0, 1).build();
        assert_eq!(c.instructions.len(), 1);
        match &c.instructions[0] {
            Instruction::Gate { gate, targets } => {
                assert!(matches!(gate, Gate::Cu(_)));
                assert_eq!(targets.as_slice(), &[0, 1]);
            }
            _ => panic!("expected Gate instruction"),
        }
    }

    #[test]
    fn builder_mcu() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let x_mat = [[zero, one], [one, zero]];
        let c = CircuitBuilder::new(3).mcu(x_mat, &[0, 1], 2).build();
        assert_eq!(c.instructions.len(), 1);
        match &c.instructions[0] {
            Instruction::Gate { gate, targets } => {
                assert!(matches!(gate, Gate::Mcu(_)));
                assert_eq!(targets.as_slice(), &[0, 1, 2]);
            }
            _ => panic!("expected Gate instruction"),
        }
    }
}
