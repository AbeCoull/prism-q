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

use super::parameter::Parameters;
use super::{Circuit, ClassicalCondition, Instruction, SmallVec, guarded};
use crate::gates::Gate;
use crate::sim::unified_pauli::PauliTerm;

/// Fluent builder for quantum circuits.
///
/// Provides method-chaining syntax for circuit construction. Each gate
/// method returns `&mut Self`, allowing compact one-liner circuits.
/// Call [`build`](Self::build) to extract the finished [`Circuit`], or
/// use [`run`](Self::run) / [`run_with`](Self::run_with) for direct execution.
///
/// [`param`](Self::param) binds the most recently appended gate to a parameter
/// slot. Retrieve the recorded set with [`parameters`](Self::parameters) or
/// [`build_parametric`](Self::build_parametric).
///
/// Gate and measurement methods panic when a qubit or classical bit index is
/// out of bounds.
///
/// # Examples
///
/// ```
/// use prism_q::{CircuitBuilder, simulate};
///
/// let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
/// let result = simulate(&circuit).seed(42).run().expect("simulation failed");
/// let probs = result.probabilities.expect("no probabilities").to_vec();
/// assert!((probs[0] - 0.5).abs() < 1e-10);
/// assert!((probs[3] - 0.5).abs() < 1e-10);
/// ```
pub struct CircuitBuilder {
    circuit: Circuit,
    params: Parameters,
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

macro_rules! gate_2q {
    ($name:ident, $variant:ident, $a:ident, $b:ident) => {
        pub fn $name(&mut self, $a: usize, $b: usize) -> &mut Self {
            self.circuit.add_gate(Gate::$variant, &[$a, $b]);
            self
        }
    };
}

impl CircuitBuilder {
    /// Create a builder for a circuit with `num_qubits` qubits and no classical bits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits, 0),
            params: Parameters::new(0),
        }
    }

    /// Create a builder with explicit qubit and classical bit counts.
    pub fn new_with_classical(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            circuit: Circuit::new(num_qubits, num_classical_bits),
            params: Parameters::new(0),
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

    /// Append the Pauli rotation `exp(-i θ P / 2)` for the Pauli string given
    /// as one factor per qubit, lowering as
    /// [`Circuit::add_pauli_rotation`] does.
    ///
    /// # Panics
    /// Panics if `factors` is empty or names a qubit twice, as well as on the
    /// out-of-bounds index every gate method panics on.
    pub fn pauli_rotation(&mut self, theta: f64, factors: &[PauliTerm]) -> &mut Self {
        self.circuit.add_pauli_rotation(theta, factors);
        self
    }

    /// Bind the most recently appended gate to parameter `slot`. Several gates
    /// may share a slot: binding writes one angle to each and the adjoint
    /// gradient accumulates them. Example: `builder.rz(theta, q).param(0)`.
    ///
    /// The declared slot count grows to cover the highest slot named here.
    ///
    /// # Panics
    /// Panics if no gate has been appended yet, or if the last instruction is
    /// not a gate carrying an angle (`Rx`, `Ry`, `Rz`, `Rzz`, `P`).
    pub fn param(&mut self, slot: usize) -> &mut Self {
        let last = self
            .circuit
            .instructions
            .len()
            .checked_sub(1)
            .expect("param() called before any gate was appended");
        match &self.circuit.instructions[last] {
            Instruction::Gate { gate, .. } if gate.pauli_generator().is_some() => {}
            other => panic!(
                "param() requires the last instruction to be a gate carrying an angle \
                 (rx, ry, rz, rzz, p), got {other:?}"
            ),
        }
        self.params.link_growing(last, slot);
        self
    }

    gate_2q!(cx, Cx, control, target);
    gate_2q!(cz, Cz, q0, q1);
    gate_2q!(swap, Swap, q0, q1);

    /// Append a controlled unitary applying `mat` to `target` when `control` is |1⟩.
    pub fn cu(&mut self, mat: [[Complex64; 2]; 2], control: usize, target: usize) -> &mut Self {
        self.circuit.add_gate(Gate::cu(mat), &[control, target]);
        self
    }

    /// Append a controlled-phase gate applying phase `e^{i theta}` to |11⟩.
    pub fn cphase(&mut self, theta: f64, control: usize, target: usize) -> &mut Self {
        self.circuit
            .add_gate(Gate::cphase(theta), &[control, target]);
        self
    }

    /// Append a multi-controlled unitary applying `mat` to `target` when every
    /// qubit in `controls` is |1⟩.
    ///
    /// # Panics
    /// Panics if `controls` holds more than `u8::MAX` entries, the width
    /// [`McuData::num_controls`](crate::gates::McuData::num_controls) can carry.
    pub fn mcu(
        &mut self,
        mat: [[Complex64; 2]; 2],
        controls: &[usize],
        target: usize,
    ) -> &mut Self {
        let num_controls =
            u8::try_from(controls.len()).expect("mcu supports at most u8::MAX control qubits");
        let mut targets: SmallVec<[usize; 4]> = controls.into();
        targets.push(target);
        self.circuit.instructions.push(Instruction::Gate {
            gate: Gate::mcu(mat, num_controls),
            targets,
        });
        self
    }

    pub fn measure(&mut self, qubit: usize, classical_bit: usize) -> &mut Self {
        self.circuit.add_measure(qubit, classical_bit);
        self
    }

    /// Reset `qubit` to |0⟩.
    pub fn reset(&mut self, qubit: usize) -> &mut Self {
        self.circuit.add_reset(qubit);
        self
    }

    /// Measure all qubits into classical bits with matching indices.
    ///
    /// Expands `num_classical_bits` if needed to accommodate all qubits.
    pub fn measure_all(&mut self) -> &mut Self {
        self.circuit.measure_all();
        self
    }

    /// Append a barrier over `qubits` (scheduling hint, no physical operation).
    pub fn barrier(&mut self, qubits: &[usize]) -> &mut Self {
        self.circuit.add_barrier(qubits);
        self
    }

    /// Append `gate` on `targets`, executed only when `condition` holds at runtime.
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

    /// Append a guarded region: everything `body` appends runs only when
    /// `condition` holds at runtime, measurement and reset included.
    ///
    /// The body builder starts empty and shares this circuit's dimensions, so
    /// its qubit and classical-bit indices are the enclosing circuit's. Nesting
    /// is allowed to [`MAX_REGION_DEPTH`]; an empty body appends nothing.
    ///
    /// [`MAX_REGION_DEPTH`]: crate::circuit::MAX_REGION_DEPTH
    pub fn guarded(
        &mut self,
        condition: ClassicalCondition,
        body: impl FnOnce(&mut CircuitBuilder),
    ) -> &mut Self {
        let mut inner = CircuitBuilder::new(self.circuit.num_qubits);
        inner.circuit.num_classical_bits = self.circuit.num_classical_bits;
        body(&mut inner);
        if let Some(inst) = guarded(condition, inner.circuit.instructions) {
            self.circuit.instructions.push(inst);
        }
        self
    }

    /// Append an arbitrary [`Gate`]; panics if the gate's arity does not
    /// match `targets.len()`.
    pub fn gate(&mut self, gate: Gate, targets: &[usize]) -> &mut Self {
        self.circuit.add_gate(gate, targets);
        self
    }

    /// Extract the finished circuit, replacing the builder's internal circuit with an empty one.
    pub fn build(&mut self) -> Circuit {
        self.params = Parameters::new(0);
        std::mem::replace(&mut self.circuit, Circuit::new(0, 0))
    }

    /// Extract the finished circuit together with the recorded parameters,
    /// resetting the builder.
    pub fn build_parametric(&mut self) -> (Circuit, Parameters) {
        let circuit = std::mem::replace(&mut self.circuit, Circuit::new(0, 0));
        let params = std::mem::replace(&mut self.params, Parameters::new(0)).pinned_to(&circuit);
        (circuit, params)
    }

    /// Borrow the circuit without consuming the builder.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// The parameters recorded by [`param`](Self::param). Not yet pinned to the
    /// circuit; [`build_parametric`](Self::build_parametric) pins on the way out.
    pub fn parameters(&self) -> &Parameters {
        &self.params
    }

    /// Execute with automatic backend selection.
    pub fn run(&self, seed: u64) -> crate::Result<crate::sim::RunOutcome> {
        crate::sim::simulate(&self.circuit).seed(seed).run()
    }

    /// Execute with explicit backend selection.
    pub fn run_with(
        &self,
        kind: crate::sim::BackendKind,
        seed: u64,
    ) -> crate::Result<crate::sim::RunOutcome> {
        crate::sim::simulate(&self.circuit)
            .backend(kind)
            .seed(seed)
            .run()
    }

    /// Execute multi-shot sampling.
    pub fn run_shots(&self, num_shots: usize, seed: u64) -> crate::Result<crate::sim::ShotsResult> {
        crate::sim::simulate(&self.circuit)
            .seed(seed)
            .shots(num_shots)
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
    fn builder_reset_emits_instruction_and_chains() {
        let c = CircuitBuilder::new(2).x(0).reset(0).h(1).build();
        assert_eq!(c.instructions.len(), 3);
        assert!(matches!(
            c.instructions.as_slice(),
            [
                Instruction::Gate {
                    gate: Gate::X,
                    targets: x_targets,
                },
                Instruction::Reset { qubit: 0 },
                Instruction::Gate {
                    gate: Gate::H,
                    targets: h_targets,
                },
            ] if x_targets.as_slice() == [0] && h_targets.as_slice() == [1]
        ));
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
        let direct_result = crate::sim::simulate(&c)
            .seed(42)
            .run()
            .expect("direct run failed");
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
