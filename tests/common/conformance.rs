//! Seeded cross-backend differential conformance harness.
//!
//! A generator turns one fixed seed into a corpus of small circuits covering
//! Clifford and non-Clifford mixes, mid-circuit measurement, reset (including
//! reset of one half of an entangled pair), and classical conditioning. Every
//! circuit runs on every participant the capability matrix declares eligible:
//! the explicit backends, the `Auto` route, the subsystem-decomposition route,
//! and the stabilizer-rank route.
//!
//! # Oracle policy
//!
//! The statevector is a participant, never the reference. `tests/common/mod.rs`
//! records why: `sv_reference_probs` can only report that two implementations
//! disagree, and it names the other one as the failure. When `reset` was a
//! renormalized projection in the statevector and the channel elsewhere, a
//! statevector-referenced matrix reported the four correct backends as broken.
//!
//! Blame is assigned in three layers, in this order:
//!
//! 1. Consensus. Participants are partitioned into agreement classes at
//!    [`PROB_EPS`]. A single class passes. Two or more classes fail, and the
//!    message names every class with its members, so a split is reported as a
//!    split rather than as "everyone except the statevector".
//! 2. Independent anchors. [`Anchor::Tableau`] (stabilizer and factored
//!    stabilizer) shares no arithmetic with the amplitude backends and decides
//!    Clifford cases. [`Anchor::Channel`] (density matrix) owns its own
//!    measurement, reset, and Kraus code and decides non-unitary semantics; its
//!    unitary evolution reuses the statevector kernels, so it anchors channel
//!    behavior only. When an anchor is present, its class is the correct one
//!    and the failure message says so.
//! 3. Closed-form goldens. Every family names a test in
//!    `tests/golden_small_circuits.rs` that pins its motif against hand-computed
//!    values ([`Family::golden_anchor`]). Non-Clifford unitary families have no
//!    independent anchor among the backends, so they run under
//!    [`AnchorPolicy::ConsensusOnly`], which requires a minimum number of
//!    compared participants instead.
//!
//! # Comparison regimes
//!
//! Backends consume randomness on their own schedule, so one seed does not put
//! two of them on the same trajectory. Each family declares how it is compared:
//!
//! - [`Regime::Deterministic`]: every collapse point is forced, so one run per
//!   participant is exact. Classical bits, probabilities, and (where the matrix
//!   says the backend exposes them) amplitudes must all agree, and the result
//!   must not move across [`DETERMINISM_SEEDS`].
//! - [`Regime::ExactMixture`]: the trajectory branches and an exact mixture
//!   oracle exists. Each participant is swept over [`BRANCH_SEEDS`] seeds and
//!   reduced to its set of distinct branches; the sets must agree across
//!   participants, and the branches must recombine into the oracle mixture at
//!   [`MIXTURE_EPS`] with non-negative weights summing to one. The oracle is the
//!   density matrix with `measure` lowered to the dephasing Kraus pair and
//!   `reset` left as the channel it already implements.
//! - [`Regime::SampledMixture`]: the trajectory branches and a conditional gate
//!   reads a random outcome, which the dephasing lowering cannot express. Branch
//!   sets must still agree exactly; the weights are compared as seed-averaged
//!   means at [`SAMPLED_MEAN_EPS`].
//!
//! # Skips
//!
//! Every (case, participant) pair resolves to [`Eligibility::Compare`] or
//! [`Eligibility::Skip`] with a named [`SkipReason`], for probabilities and for
//! amplitudes separately. Nothing is skipped by catching an error: a participant
//! the matrix declares eligible that fails to run is a failure. [`skip_ledger`]
//! reports which rules fired, and `tests/conformance_matrix.rs` pins that set.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
use prism_q::error::Result;
use prism_q::gates::Gate;
use prism_q::sim::{self, BackendKind, RunOutcome};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generator seed. Every case in the corpus derives from this value alone.
pub const CONFORMANCE_SEED: u64 = 42;

/// Probability agreement tolerance between two participants.
pub const PROB_EPS: f64 = 1e-9;

/// Amplitude agreement tolerance, after fixing the global phase.
pub const AMP_EPS: f64 = 1e-9;

/// Residual tolerance on the branch-mixture identity.
pub const MIXTURE_EPS: f64 = 1e-9;

/// Seeds swept per participant when the trajectory branches.
pub const BRANCH_SEEDS: u64 = 64;

/// Band on a seed-averaged probability at [`BRANCH_SEEDS`] samples. A fair coin
/// has a standard error of 0.0625 here, so the band admits two of them.
pub const SAMPLED_MEAN_EPS: f64 = 0.13;

/// Seeds a deterministic case is replayed over. A forced trajectory must not
/// move, which is what makes one run per participant a fair comparison. Wide
/// cases use the first two: the density-matrix participant stores `4^n`
/// amplitudes, and two seeds already catch a trajectory that moves.
pub const DETERMINISM_SEEDS: [u64; 3] = [CONFORMANCE_SEED, CONFORMANCE_SEED + 1, 7];

/// Bond dimension the harness gives the MPS participant.
pub const MPS_MAX_BOND: usize = 8;

/// Discarded weight an untruncated MPS run may still report. Cutting a bond the
/// state does not use drops singular values at the rounding floor; a genuine
/// truncation discards many orders of magnitude more.
pub const MPS_TRUNCATION_TOLERANCE: f64 = 1e-12;

/// Dense-output guard for the tensor-network participant. The backend's own cap
/// is derived from system memory (`PRISM_MAX_PROB_QUBITS`); the harness keeps a
/// fixed bound so the matrix reads the same on every machine.
pub const TN_MAX_OUTPUT_QUBITS: usize = 20;

/// Qubit guard for the density-matrix participant, which stores `4^n`
/// amplitudes. Well inside the backend's own memory-derived cap.
pub const DM_MAX_QUBITS: usize = 12;

/// T-count above which the exact stabilizer-rank expansion costs `2^t` branches
/// and the harness stops comparing it.
pub const SR_MAX_T_COUNT: usize = 8;

/// T-count above which the `Auto` stabilizer-rank route switches to sparsified,
/// statistical estimates. Mirrors `MAX_AUTO_T_COUNT_EXACT` in `sim/dispatch.rs`.
pub const SR_STATISTICAL_T_COUNT: usize = 18;

/// Minimum compared participants for a case with no independent anchor.
pub const MIN_CONSENSUS_PARTICIPANTS: usize = 5;

// ---- Corpus ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    CliffordUnitary,
    CliffordTUnitary,
    RotationUnitary,
    Separable,
    ForcedTrajectory,
    EntangledReset,
    MidCircuitMeasure,
    ConditionalBranch,
    Decomposable,
    WideEntangled,
}

/// How a family's trajectory is compared. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Deterministic,
    ExactMixture,
    SampledMixture,
}

/// Which independent implementation decides a case when participants disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// Stabilizer tableau. Shares no arithmetic with the amplitude backends.
    Tableau,
    /// Density-matrix channel code: measurement, reset, and Kraus application.
    Channel,
}

/// What the harness requires before it trusts a consensus class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorPolicy {
    Required(Anchor),
    ConsensusOnly,
}

impl Family {
    pub const ALL: [Family; 10] = [
        Family::CliffordUnitary,
        Family::CliffordTUnitary,
        Family::RotationUnitary,
        Family::Separable,
        Family::ForcedTrajectory,
        Family::EntangledReset,
        Family::MidCircuitMeasure,
        Family::ConditionalBranch,
        Family::Decomposable,
        Family::WideEntangled,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Family::CliffordUnitary => "clifford_unitary",
            Family::CliffordTUnitary => "clifford_t_unitary",
            Family::RotationUnitary => "rotation_unitary",
            Family::Separable => "separable",
            Family::ForcedTrajectory => "forced_trajectory",
            Family::EntangledReset => "entangled_reset",
            Family::MidCircuitMeasure => "mid_circuit_measure",
            Family::ConditionalBranch => "conditional_branch",
            Family::Decomposable => "decomposable",
            Family::WideEntangled => "wide_entangled",
        }
    }

    pub const fn regime(self) -> Regime {
        match self {
            Family::EntangledReset | Family::MidCircuitMeasure => Regime::ExactMixture,
            Family::ConditionalBranch => Regime::SampledMixture,
            _ => Regime::Deterministic,
        }
    }

    pub const fn anchor_policy(self) -> AnchorPolicy {
        match self {
            Family::CliffordUnitary => AnchorPolicy::Required(Anchor::Tableau),
            Family::ForcedTrajectory
            | Family::EntangledReset
            | Family::MidCircuitMeasure
            | Family::ConditionalBranch => AnchorPolicy::Required(Anchor::Channel),
            _ => AnchorPolicy::ConsensusOnly,
        }
    }

    /// Test in `tests/golden_small_circuits.rs` that pins this family's motif
    /// against hand-computed values.
    pub const fn golden_anchor(self) -> &'static str {
        match self {
            Family::CliffordUnitary | Family::Decomposable => {
                "stabilizer_bell_between_hadamard_layers_matches_closed_form"
            }
            Family::CliffordTUnitary => "t_tdg_cancel",
            Family::RotationUnitary | Family::WideEntangled => "rx_half_pi_creates_superposition",
            Family::Separable => "product_rotation_circuit",
            Family::ForcedTrajectory => "reset_from_one_preserves_a_spectator_superposition",
            Family::EntangledReset => "density_matrix_reset_of_an_entangled_partner_is_the_channel",
            Family::MidCircuitMeasure | Family::ConditionalBranch => {
                "density_matrix_dephasing_removes_coherence"
            }
        }
    }

    const fn case_count(self) -> usize {
        match self {
            Family::Decomposable => 3,
            Family::WideEntangled => 2,
            _ => 4,
        }
    }
}

/// Circuit properties the capability matrix reads.
///
/// Everything except `branching` is derived from the circuit, so a generator
/// change cannot leave a stale declaration behind. `branching` comes from the
/// family and is verified at runtime: a deterministic family that branches
/// fails [`assert_deterministic_case`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseProfile {
    pub num_qubits: usize,
    pub num_classical_bits: usize,
    pub max_component: usize,
    pub clifford_only: bool,
    pub clifford_plus_t: bool,
    pub t_count: usize,
    pub entangling: bool,
    pub has_measure: bool,
    pub has_reset: bool,
    pub has_conditional: bool,
    pub decomposes: bool,
    pub branching: bool,
}

impl CaseProfile {
    fn of(circuit: &Circuit, family: Family) -> Self {
        let mut has_measure = false;
        let mut has_reset = false;
        let mut has_conditional = false;
        for instr in &circuit.instructions {
            match instr {
                Instruction::Measure { .. } => has_measure = true,
                Instruction::Reset { .. } => has_reset = true,
                Instruction::Conditional { .. } => has_conditional = true,
                _ => {}
            }
        }
        let components = circuit.independent_subsystems();
        let max_component = components.iter().map(Vec::len).max().unwrap_or(0);
        Self {
            num_qubits: circuit.num_qubits,
            num_classical_bits: circuit.num_classical_bits,
            max_component,
            clifford_only: circuit.is_clifford_only(),
            clifford_plus_t: circuit.is_clifford_plus_t(),
            t_count: circuit.t_count(),
            entangling: circuit.has_entangling_gates(),
            has_measure,
            has_reset,
            has_conditional,
            decomposes: circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS
                && components.len() >= 2
                && max_component + 3 <= circuit.num_qubits,
            branching: family.regime() != Regime::Deterministic,
        }
    }

    pub fn unitary(&self) -> bool {
        !(self.has_measure || self.has_reset || self.has_conditional)
    }

    /// Bond dimension an exact MPS needs for the widest entangled block.
    pub fn required_bond_dim(&self) -> usize {
        1usize << (self.max_component / 2)
    }
}

/// Qubit floor for the subsystem-decomposition route. Mirrors
/// `MIN_DECOMPOSITION_QUBITS` in `sim/decomposed.rs`.
const MIN_DECOMPOSITION_QUBITS: usize = 8;

pub struct GeneratedCase {
    pub index: usize,
    pub family: Family,
    pub circuit: Circuit,
    pub profile: CaseProfile,
}

impl GeneratedCase {
    pub fn name(&self) -> String {
        format!("{}#{}", self.family.name(), self.index)
    }

    /// Everything needed to rebuild the case by hand: the generator seed, the
    /// index into the corpus, and the full instruction listing.
    pub fn context(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "case `{}` (generator seed {CONFORMANCE_SEED}, corpus index {}, regime {:?})",
            self.name(),
            self.index,
            self.family.regime()
        )
        .unwrap();
        writeln!(
            out,
            "  rebuild: common::conformance::generated_cases()[{}]",
            self.index
        )
        .unwrap();
        writeln!(
            out,
            "  qubits {}, classical bits {}",
            self.circuit.num_qubits, self.circuit.num_classical_bits
        )
        .unwrap();
        writeln!(out, "  {}", describe(&self.circuit)).unwrap();
        out
    }
}

/// One-line rendering of every instruction, in order.
pub fn describe(circuit: &Circuit) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(circuit.instructions.len());
    for instr in &circuit.instructions {
        parts.push(match instr {
            Instruction::Gate { gate, targets } => describe_gate(gate, targets),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => format!("measure q{qubit}->c{classical_bit}"),
            Instruction::Reset { qubit } => format!("reset q{qubit}"),
            Instruction::Barrier { qubits } => format!("barrier {}", describe_targets(qubits)),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => format!(
                "if({}) {}",
                describe_condition(condition),
                describe_gate(gate, targets)
            ),
        });
    }
    parts.join("; ")
}

fn describe_targets(targets: &[usize]) -> String {
    targets
        .iter()
        .map(|q| format!("q{q}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn describe_gate(gate: &Gate, targets: &[usize]) -> String {
    let name = match gate {
        Gate::Id => "id".to_string(),
        Gate::X => "x".to_string(),
        Gate::Y => "y".to_string(),
        Gate::Z => "z".to_string(),
        Gate::H => "h".to_string(),
        Gate::S => "s".to_string(),
        Gate::Sdg => "sdg".to_string(),
        Gate::T => "t".to_string(),
        Gate::Tdg => "tdg".to_string(),
        Gate::SX => "sx".to_string(),
        Gate::SXdg => "sxdg".to_string(),
        Gate::Rx(theta) => format!("rx({theta:.17})"),
        Gate::Ry(theta) => format!("ry({theta:.17})"),
        Gate::Rz(theta) => format!("rz({theta:.17})"),
        Gate::P(theta) => format!("p({theta:.17})"),
        Gate::Rzz(theta) => format!("rzz({theta:.17})"),
        Gate::Cx => "cx".to_string(),
        Gate::Cz => "cz".to_string(),
        Gate::Swap => "swap".to_string(),
        other => format!("{other:?}"),
    };
    format!("{name} {}", describe_targets(targets))
}

fn describe_condition(condition: &ClassicalCondition) -> String {
    match condition {
        ClassicalCondition::BitIsOne(bit) => format!("c{bit}==1"),
        ClassicalCondition::BitIsZero(bit) => format!("c{bit}==0"),
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => format!("c[{offset}..{}]=={value}", offset + size),
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => format!("c[{offset}..{}]!={value}", offset + size),
    }
}

// ---- Generator ----

struct CaseRng {
    rng: ChaCha8Rng,
}

const CLIFFORD_1Q: [Gate; 8] = [
    Gate::H,
    Gate::S,
    Gate::Sdg,
    Gate::X,
    Gate::Y,
    Gate::Z,
    Gate::SX,
    Gate::SXdg,
];

const CLIFFORD_2Q: [Gate; 3] = [Gate::Cx, Gate::Cz, Gate::Swap];

impl CaseRng {
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        self.rng.random_range(lo..=hi)
    }

    fn chance(&mut self, p: f64) -> bool {
        self.rng.random_bool(p)
    }

    fn clifford_1q(&mut self) -> Gate {
        CLIFFORD_1Q[self.rng.random_range(0..CLIFFORD_1Q.len())].clone()
    }

    fn clifford_2q(&mut self) -> Gate {
        CLIFFORD_2Q[self.rng.random_range(0..CLIFFORD_2Q.len())].clone()
    }

    /// Angles land away from multiples of `pi/2`, so a drawn rotation is never
    /// accidentally Clifford and no branch weight lands near zero.
    fn angle(&mut self) -> f64 {
        let quarter = self.rng.random_range(0..4) as f64;
        quarter * std::f64::consts::FRAC_PI_2 + 0.15 + self.rng.random::<f64>() * 1.25
    }

    fn rotation_1q(&mut self) -> Gate {
        let theta = self.angle();
        match self.rng.random_range(0..4) {
            0 => Gate::Rx(theta),
            1 => Gate::Ry(theta),
            2 => Gate::Rz(theta),
            _ => Gate::P(theta),
        }
    }

    fn clifford_layer(&mut self, circuit: &mut Circuit, qubits: &[usize]) {
        for &q in qubits {
            if self.chance(0.8) {
                let gate = self.clifford_1q();
                circuit.add_gate(gate, &[q]);
            }
        }
    }

    fn entangling_layer(&mut self, circuit: &mut Circuit, qubits: &[usize], offset: usize) {
        for pair in qubits[offset.min(qubits.len())..].chunks_exact(2) {
            if self.chance(0.75) {
                let gate = self.clifford_2q();
                circuit.add_gate(gate, &[pair[0], pair[1]]);
            }
        }
    }
}

/// The full corpus, reproducible from [`CONFORMANCE_SEED`] alone.
pub fn generated_cases() -> Vec<GeneratedCase> {
    let mut rng = CaseRng::new(CONFORMANCE_SEED);
    let mut cases = Vec::new();
    for family in Family::ALL {
        for _ in 0..family.case_count() {
            let circuit = build(&mut rng, family);
            let profile = CaseProfile::of(&circuit, family);
            cases.push(GeneratedCase {
                index: cases.len(),
                family,
                circuit,
                profile,
            });
        }
    }
    cases
}

pub fn cases_in(family: Family) -> Vec<GeneratedCase> {
    generated_cases()
        .into_iter()
        .filter(|case| case.family == family)
        .collect()
}

fn build(rng: &mut CaseRng, family: Family) -> Circuit {
    match family {
        Family::CliffordUnitary => build_clifford_unitary(rng),
        Family::CliffordTUnitary => build_clifford_t_unitary(rng),
        Family::RotationUnitary => build_rotation_unitary(rng),
        Family::Separable => build_separable(rng),
        Family::ForcedTrajectory => build_forced_trajectory(rng),
        Family::EntangledReset => build_entangled_reset(rng),
        Family::MidCircuitMeasure => build_mid_circuit_measure(rng),
        Family::ConditionalBranch => build_conditional_branch(rng),
        Family::Decomposable => build_decomposable(rng),
        Family::WideEntangled => build_wide_entangled(rng),
    }
}

fn build_clifford_unitary(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(2, 5);
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..rng.range(2, 4) {
        rng.clifford_layer(&mut circuit, &qubits);
        rng.entangling_layer(&mut circuit, &qubits, layer % 2);
    }
    if !circuit.has_entangling_gates() {
        circuit.add_gate(Gate::Cx, &[0, 1]);
    }
    circuit
}

fn build_clifford_t_unitary(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(3, 5);
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 0);
    let t_budget = rng.range(1, 5);
    let mut t_used = 0;
    for layer in 0..rng.range(2, 4) {
        rng.clifford_layer(&mut circuit, &qubits);
        for &q in &qubits {
            if t_used < t_budget && rng.chance(0.4) {
                let gate = if rng.chance(0.5) { Gate::T } else { Gate::Tdg };
                circuit.add_gate(gate, &[q]);
                t_used += 1;
            }
        }
        rng.entangling_layer(&mut circuit, &qubits, layer % 2);
    }
    if t_used == 0 {
        circuit.add_gate(Gate::T, &[0]);
    }
    if !circuit.has_entangling_gates() {
        circuit.add_gate(Gate::Cx, &[0, 1]);
    }
    circuit
}

fn build_rotation_unitary(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(2, 5);
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..rng.range(2, 3) {
        for &q in &qubits {
            let gate = if rng.chance(0.3) {
                rng.clifford_1q()
            } else {
                rng.rotation_1q()
            };
            circuit.add_gate(gate, &[q]);
        }
        for pair in qubits[(layer % 2).min(n)..].chunks_exact(2) {
            let gate = if rng.chance(0.5) {
                Gate::Rzz(rng.angle())
            } else {
                Gate::Cx
            };
            circuit.add_gate(gate, &[pair[0], pair[1]]);
        }
    }
    if !circuit.has_entangling_gates() {
        circuit.add_gate(Gate::Cx, &[0, 1]);
    }
    circuit
}

fn build_separable(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(2, 5);
    let mut circuit = Circuit::new(n, 0);
    for _ in 0..rng.range(2, 4) {
        for q in 0..n {
            let gate = if rng.chance(0.4) {
                rng.clifford_1q()
            } else {
                rng.rotation_1q()
            };
            circuit.add_gate(gate, &[q]);
        }
    }
    circuit
}

/// Measurement, reset, and conditioning at points where the outcome is forced.
///
/// The ancilla is prepared in a computational basis state and kept out of every
/// entangling gate until after its collapse, so one run is already exact. The
/// work register stays in superposition, which is what makes the case
/// discriminating: a backend that reinitializes the whole state instead of the
/// target qubit collapses the spectators too.
fn build_forced_trajectory(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(3, 5);
    let ancilla = n - 1;
    let work: Vec<usize> = (0..ancilla).collect();
    let cbits = if n >= 4 && rng.chance(0.5) { 2 } else { 1 };
    let mut circuit = Circuit::new(n, cbits);

    for layer in 0..rng.range(1, 3) {
        rng.clifford_layer(&mut circuit, &work);
        rng.entangling_layer(&mut circuit, &work, layer % 2);
    }
    if !circuit.has_entangling_gates() {
        circuit.add_gate(Gate::Cx, &[work[0], work[1]]);
    }

    let ancilla_one = rng.chance(0.5);
    if ancilla_one {
        circuit.add_gate(Gate::X, &[ancilla]);
    }
    circuit.add_measure(ancilla, 0);
    if cbits == 2 {
        circuit.add_measure(ancilla, 1);
    }
    if rng.chance(0.5) {
        circuit.add_reset(ancilla);
    }

    let condition = if cbits == 2 {
        ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 2,
            value: if ancilla_one { 3 } else { 0 },
        }
    } else if rng.chance(0.5) {
        ClassicalCondition::BitIsOne(0)
    } else {
        ClassicalCondition::BitIsZero(0)
    };
    let gate = rng.clifford_1q();
    circuit.instructions.push(Instruction::Conditional {
        condition,
        gate,
        targets: SmallVec::from_slice(&[work[0]]),
    });

    circuit.add_gate(Gate::Cx, &[ancilla, work[0]]);
    rng.clifford_layer(&mut circuit, &work);
    circuit
}

/// Reset of a qubit entangled with the rest of the register.
///
/// This is the shape that separates the reset channel from a renormalized
/// projection onto |0>: a projection also collapses the partners into the
/// |0>-outcome branch.
fn build_entangled_reset(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(2, 4);
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 0);

    circuit.add_gate(Gate::H, &[0]);
    for q in 1..n {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    for &q in &qubits {
        if rng.chance(0.4) {
            let gate = rng.clifford_1q();
            circuit.add_gate(gate, &[q]);
        }
    }

    let target = rng.range(1, n - 1);
    circuit.add_reset(target);

    rng.clifford_layer(&mut circuit, &qubits);
    if rng.chance(0.6) {
        circuit.add_gate(Gate::Cx, &[target, (target + 1) % n]);
    }
    circuit
}

/// Mid-circuit measurement with a genuinely random outcome, followed by further
/// evolution. No reset and no conditioning, so lowering each measurement to the
/// dephasing Kraus pair reproduces the exact unconditional mixture.
fn build_mid_circuit_measure(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(2, 4);
    let qubits: Vec<usize> = (0..n).collect();
    let cbits = if n >= 3 && rng.chance(0.5) { 2 } else { 1 };
    let mut circuit = Circuit::new(n, cbits);

    circuit.add_gate(Gate::H, &[0]);
    for q in 1..n {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    rng.clifford_layer(&mut circuit, &qubits);

    let first = rng.range(0, n - 1);
    circuit.add_measure(first, 0);
    rng.clifford_layer(&mut circuit, &qubits);
    if rng.chance(0.6) {
        circuit.add_gate(Gate::Cx, &[0, n - 1]);
    }
    if cbits == 2 {
        let second = rng.range(0, n - 1);
        circuit.add_measure(second, 1);
        rng.clifford_layer(&mut circuit, &qubits);
    }
    circuit
}

/// A conditional gate reading a random measurement outcome. The dephasing
/// lowering cannot express classical feedback, so this family is compared as
/// branch sets plus seed-averaged weights.
fn build_conditional_branch(rng: &mut CaseRng) -> Circuit {
    let n = rng.range(3, 4);
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 1);

    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    rng.clifford_layer(&mut circuit, &qubits);

    circuit.add_measure(0, 0);
    let gate = rng.clifford_1q();
    let condition = if rng.chance(0.5) {
        ClassicalCondition::BitIsOne(0)
    } else {
        ClassicalCondition::BitIsZero(0)
    };
    circuit.instructions.push(Instruction::Conditional {
        condition,
        gate,
        targets: SmallVec::from_slice(&[n - 1]),
    });
    rng.clifford_layer(&mut circuit, &qubits);
    circuit
}

/// Independent blocks with no gate crossing a block boundary, sized so the
/// simulator takes the subsystem-decomposition route.
fn build_decomposable(rng: &mut CaseRng) -> Circuit {
    let blocks: Vec<usize> = if rng.chance(0.5) {
        vec![4, 4]
    } else {
        vec![3, 3, 3]
    };
    let n: usize = blocks.iter().sum();
    let mut circuit = Circuit::new(n, 0);
    let mut base = 0;
    for width in blocks {
        let qubits: Vec<usize> = (base..base + width).collect();
        let rotations = rng.chance(0.5);
        for layer in 0..rng.range(2, 3) {
            for &q in &qubits {
                let gate = if rotations {
                    rng.rotation_1q()
                } else {
                    rng.clifford_1q()
                };
                circuit.add_gate(gate, &[q]);
            }
            for pair in qubits[layer % 2..].chunks_exact(2) {
                circuit.add_gate(Gate::Cx, &[pair[0], pair[1]]);
            }
        }
        circuit.add_gate(Gate::Cx, &[qubits[0], qubits[width - 1]]);
        base += width;
    }
    circuit
}

/// Eight qubits of dense non-Clifford entanglement. A dense state that wide
/// needs bond `2^4`, above [`MPS_MAX_BOND`], which is the case the truncation
/// skip rule exists for.
fn build_wide_entangled(rng: &mut CaseRng) -> Circuit {
    let n = 8;
    let qubits: Vec<usize> = (0..n).collect();
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..4 {
        for &q in &qubits {
            let gate = rng.rotation_1q();
            circuit.add_gate(gate, &[q]);
        }
        for pair in qubits[layer % 2..].chunks_exact(2) {
            circuit.add_gate(Gate::Cx, &[pair[0], pair[1]]);
        }
    }
    circuit
}

// ---- Capability matrix ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkipReason {
    /// Stabilizer family: the circuit is not Clifford.
    NonClifford,
    /// Product state: the circuit contains entangling gates.
    EntanglingGates,
    /// Explicit stabilizer rank rejects circuits with no T gates.
    StabilizerRankNoTGates,
    /// Stabilizer rank has no measurement, reset, or conditional semantics.
    StabilizerRankNonUnitary,
    /// The exact stabilizer-rank expansion costs `2^t` branches.
    StabilizerRankExpansionCost,
    /// Above `MAX_AUTO_T_COUNT_EXACT` the stabilizer-rank route sparsifies and
    /// returns statistical estimates instead of exact probabilities. Reachable
    /// only when the size-derived stabilizer-rank budget admits a T count that
    /// large, which needs far more qubits than a differential corpus can
    /// compare against a dense reference.
    StabilizerRankStatistical,
    /// A dense state on this block needs more bond dimension than the harness
    /// gives the MPS participant, so its output would be truncated.
    MpsTruncation,
    /// The tensor-network dense probability output is above the harness bound.
    TensorNetworkOutputLimit,
    /// The density matrix stores `4^n` amplitudes.
    DensityMatrixQubitCap,
    /// On a branching case the density matrix holds the reset mixture rather
    /// than one trajectory of it, so its output is not a branch of the
    /// pure-state participants. It serves as the mixture oracle instead.
    DensityMatrixMixesBranches,
    /// The circuit has no independent subsystems to decompose.
    NotDecomposable,
    /// Blocks of a decomposed run take independent seeds, so a branching
    /// trajectory is not comparable branch for branch.
    DecompositionPerBlockSeed,
    /// The participant is a route or a mixed state and exposes no amplitudes.
    NoAmplitudeExport,
    /// The factored stabilizer holds one tableau per independent block and has
    /// no joint statevector to export while more than one block is live.
    FactoredBlocksHaveNoJointState,
    /// Amplitudes are only defined once the trajectory is fixed.
    BranchingTrajectory,
}

impl SkipReason {
    pub const ALL: [SkipReason; 15] = [
        SkipReason::NonClifford,
        SkipReason::EntanglingGates,
        SkipReason::StabilizerRankNoTGates,
        SkipReason::StabilizerRankNonUnitary,
        SkipReason::StabilizerRankExpansionCost,
        SkipReason::StabilizerRankStatistical,
        SkipReason::MpsTruncation,
        SkipReason::TensorNetworkOutputLimit,
        SkipReason::DensityMatrixQubitCap,
        SkipReason::DensityMatrixMixesBranches,
        SkipReason::NotDecomposable,
        SkipReason::DecompositionPerBlockSeed,
        SkipReason::NoAmplitudeExport,
        SkipReason::FactoredBlocksHaveNoJointState,
        SkipReason::BranchingTrajectory,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            SkipReason::NonClifford => "non_clifford",
            SkipReason::EntanglingGates => "entangling_gates",
            SkipReason::StabilizerRankNoTGates => "stabilizer_rank_no_t_gates",
            SkipReason::StabilizerRankNonUnitary => "stabilizer_rank_non_unitary",
            SkipReason::StabilizerRankExpansionCost => "stabilizer_rank_expansion_cost",
            SkipReason::StabilizerRankStatistical => "stabilizer_rank_statistical",
            SkipReason::MpsTruncation => "mps_truncation",
            SkipReason::TensorNetworkOutputLimit => "tensor_network_output_limit",
            SkipReason::DensityMatrixQubitCap => "density_matrix_qubit_cap",
            SkipReason::DensityMatrixMixesBranches => "density_matrix_mixes_branches",
            SkipReason::NotDecomposable => "not_decomposable",
            SkipReason::DecompositionPerBlockSeed => "decomposition_per_block_seed",
            SkipReason::NoAmplitudeExport => "no_amplitude_export",
            SkipReason::FactoredBlocksHaveNoJointState => "factored_blocks_have_no_joint_state",
            SkipReason::BranchingTrajectory => "branching_trajectory",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eligibility {
    Compare,
    Skip(SkipReason),
}

impl Eligibility {
    pub fn is_compared(self) -> bool {
        matches!(self, Eligibility::Compare)
    }
}

enum Executor {
    /// Constructed through the `Backend` trait and driven by `sim::run_on`.
    Direct(fn(u64) -> Box<dyn Backend>),
    /// Constructed with a bond dimension and checked for truncation after the
    /// run, so the matrix's exactness claim is verified rather than asserted.
    Mps,
    /// Driven through `sim::simulate`, exercising dispatch and the fusion
    /// pipeline rather than a fixed backend instance.
    Route(fn(&Circuit, u64) -> Result<RunOutcome>),
}

pub struct Participant {
    pub name: &'static str,
    exec: Executor,
    rules: fn(&CaseProfile) -> Eligibility,
    export_rules: fn(&CaseProfile) -> Eligibility,
    pub anchor: Option<Anchor>,
}

impl Participant {
    pub fn eligibility(&self, profile: &CaseProfile) -> Eligibility {
        (self.rules)(profile)
    }

    /// Amplitude comparison narrows the probability rule: the trajectory has to
    /// be fixed and the participant has to expose a joint state.
    pub fn amplitude_eligibility(&self, profile: &CaseProfile) -> Eligibility {
        match self.eligibility(profile) {
            Eligibility::Skip(reason) => Eligibility::Skip(reason),
            Eligibility::Compare if profile.branching => {
                Eligibility::Skip(SkipReason::BranchingTrajectory)
            }
            Eligibility::Compare => (self.export_rules)(profile),
        }
    }
}

fn always(_profile: &CaseProfile) -> Eligibility {
    Eligibility::Compare
}

fn no_state_export(_profile: &CaseProfile) -> Eligibility {
    Eligibility::Skip(SkipReason::NoAmplitudeExport)
}

fn factored_stabilizer_export(profile: &CaseProfile) -> Eligibility {
    if profile.max_component < profile.num_qubits {
        return Eligibility::Skip(SkipReason::FactoredBlocksHaveNoJointState);
    }
    Eligibility::Compare
}

fn mps_rules(profile: &CaseProfile) -> Eligibility {
    if profile.required_bond_dim() > MPS_MAX_BOND {
        return Eligibility::Skip(SkipReason::MpsTruncation);
    }
    Eligibility::Compare
}

fn tensor_network_rules(profile: &CaseProfile) -> Eligibility {
    if profile.num_qubits > TN_MAX_OUTPUT_QUBITS {
        return Eligibility::Skip(SkipReason::TensorNetworkOutputLimit);
    }
    Eligibility::Compare
}

fn stabilizer_rules(profile: &CaseProfile) -> Eligibility {
    if !profile.clifford_only {
        return Eligibility::Skip(SkipReason::NonClifford);
    }
    Eligibility::Compare
}

fn product_rules(profile: &CaseProfile) -> Eligibility {
    if profile.entangling {
        return Eligibility::Skip(SkipReason::EntanglingGates);
    }
    Eligibility::Compare
}

fn density_matrix_rules(profile: &CaseProfile) -> Eligibility {
    if profile.num_qubits > DM_MAX_QUBITS {
        return Eligibility::Skip(SkipReason::DensityMatrixQubitCap);
    }
    if profile.branching && profile.has_reset {
        return Eligibility::Skip(SkipReason::DensityMatrixMixesBranches);
    }
    Eligibility::Compare
}

fn decomposition_rules(profile: &CaseProfile) -> Eligibility {
    if !profile.decomposes {
        return Eligibility::Skip(SkipReason::NotDecomposable);
    }
    if !profile.unitary() {
        return Eligibility::Skip(SkipReason::DecompositionPerBlockSeed);
    }
    Eligibility::Compare
}

fn stabilizer_rank_rules(profile: &CaseProfile) -> Eligibility {
    if !profile.unitary() {
        return Eligibility::Skip(SkipReason::StabilizerRankNonUnitary);
    }
    if !profile.clifford_plus_t || profile.t_count == 0 {
        return Eligibility::Skip(SkipReason::StabilizerRankNoTGates);
    }
    if profile.t_count > SR_STATISTICAL_T_COUNT {
        return Eligibility::Skip(SkipReason::StabilizerRankStatistical);
    }
    if profile.t_count > SR_MAX_T_COUNT {
        return Eligibility::Skip(SkipReason::StabilizerRankExpansionCost);
    }
    Eligibility::Compare
}

fn run_auto(circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    sim::simulate(circuit).seed(seed).run()
}

fn run_decomposition(circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    sim::simulate(circuit)
        .backend(BackendKind::Statevector)
        .seed(seed)
        .run()
}

fn run_stabilizer_rank(circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    sim::simulate(circuit)
        .backend(BackendKind::StabilizerRank)
        .seed(seed)
        .run()
}

pub fn participants() -> Vec<Participant> {
    vec![
        Participant {
            name: "statevector",
            exec: Executor::Direct(|s| Box::new(StatevectorBackend::new(s))),
            rules: always,
            export_rules: always,
            anchor: None,
        },
        Participant {
            name: "sparse",
            exec: Executor::Direct(|s| Box::new(SparseBackend::new(s))),
            rules: always,
            export_rules: always,
            anchor: None,
        },
        Participant {
            name: "mps",
            exec: Executor::Mps,
            rules: mps_rules,
            export_rules: always,
            anchor: None,
        },
        Participant {
            name: "tensor_network",
            exec: Executor::Direct(|s| Box::new(TensorNetworkBackend::new(s))),
            rules: tensor_network_rules,
            export_rules: always,
            anchor: None,
        },
        Participant {
            name: "factored",
            exec: Executor::Direct(|s| Box::new(FactoredBackend::new(s))),
            rules: always,
            export_rules: no_state_export,
            anchor: None,
        },
        Participant {
            name: "stabilizer",
            exec: Executor::Direct(|s| Box::new(StabilizerBackend::new(s))),
            rules: stabilizer_rules,
            export_rules: always,
            anchor: Some(Anchor::Tableau),
        },
        Participant {
            name: "factored_stabilizer",
            exec: Executor::Direct(|s| Box::new(FactoredStabilizerBackend::new(s))),
            rules: stabilizer_rules,
            export_rules: factored_stabilizer_export,
            anchor: Some(Anchor::Tableau),
        },
        Participant {
            name: "product",
            exec: Executor::Direct(|s| Box::new(ProductStateBackend::new(s))),
            rules: product_rules,
            export_rules: always,
            anchor: None,
        },
        Participant {
            name: "density_matrix",
            exec: Executor::Direct(|s| Box::new(DensityMatrixBackend::new(s))),
            rules: density_matrix_rules,
            export_rules: no_state_export,
            anchor: Some(Anchor::Channel),
        },
        Participant {
            name: "auto",
            exec: Executor::Route(run_auto),
            rules: always,
            export_rules: no_state_export,
            anchor: None,
        },
        Participant {
            name: "decomposition",
            exec: Executor::Route(run_decomposition),
            rules: decomposition_rules,
            export_rules: no_state_export,
            anchor: None,
        },
        Participant {
            name: "stabilizer_rank",
            exec: Executor::Route(run_stabilizer_rank),
            rules: stabilizer_rank_rules,
            export_rules: no_state_export,
            anchor: None,
        },
    ]
}

/// Which skip rules the corpus fires, over both the probability matrix and the
/// amplitude matrix, with counts.
pub fn skip_ledger() -> BTreeMap<SkipReason, usize> {
    let mut ledger: BTreeMap<SkipReason, usize> = BTreeMap::new();
    let participants = participants();
    for case in generated_cases() {
        for participant in &participants {
            if let Eligibility::Skip(reason) = participant.eligibility(&case.profile) {
                *ledger.entry(reason).or_default() += 1;
            }
            if let Eligibility::Skip(reason) = participant.amplitude_eligibility(&case.profile) {
                *ledger.entry(reason).or_default() += 1;
            }
        }
    }
    ledger
}

/// The full matrix as text: one row per case, listing compared participants and
/// every skip with its reason.
pub fn matrix_report() -> String {
    let participants = participants();
    let mut out = String::new();
    for case in generated_cases() {
        writeln!(out, "{}", case.name()).unwrap();
        for participant in &participants {
            let probs = match participant.eligibility(&case.profile) {
                Eligibility::Compare => "compare".to_string(),
                Eligibility::Skip(reason) => format!("skip:{}", reason.name()),
            };
            let amps = match participant.amplitude_eligibility(&case.profile) {
                Eligibility::Compare => "compare".to_string(),
                Eligibility::Skip(reason) => format!("skip:{}", reason.name()),
            };
            writeln!(
                out,
                "  {:<20} probs {:<34} amps {amps}",
                participant.name, probs
            )
            .unwrap();
        }
    }
    out
}

// ---- Execution ----

pub struct RunResult {
    pub classical_bits: Vec<bool>,
    pub probs: Vec<f64>,
    pub amplitudes: Option<Vec<Complex64>>,
}

fn run_participant(
    participant: &Participant,
    circuit: &Circuit,
    seed: u64,
    want_amplitudes: bool,
) -> Result<RunResult> {
    match &participant.exec {
        Executor::Direct(construct) => {
            let mut backend = construct(seed);
            sim::run_on(backend.as_mut(), circuit)?;
            let amplitudes = if want_amplitudes {
                Some(backend.export_statevector()?)
            } else {
                None
            };
            Ok(RunResult {
                classical_bits: backend.classical_results().to_vec(),
                probs: backend.probabilities()?,
                amplitudes,
            })
        }
        Executor::Mps => {
            let mut backend = MpsBackend::new(seed, MPS_MAX_BOND);
            backend.reset_truncation_tracking();
            sim::run_on(&mut backend, circuit)?;
            let discarded = backend.truncation_discarded();
            assert!(
                discarded < MPS_TRUNCATION_TOLERANCE,
                "mps discarded {discarded:.3e} of the state at bond {MPS_MAX_BOND}, but the \
                 capability matrix declared this case exact"
            );
            let amplitudes = if want_amplitudes {
                Some(backend.export_statevector()?)
            } else {
                None
            };
            Ok(RunResult {
                classical_bits: backend.classical_results().to_vec(),
                probs: backend.probabilities()?,
                amplitudes,
            })
        }
        Executor::Route(run) => {
            let outcome = run(circuit, seed)?;
            let probs = outcome
                .probabilities
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "{} returned no probabilities, but the capability matrix declared it \
                         eligible",
                        participant.name
                    )
                })
                .to_vec();
            Ok(RunResult {
                classical_bits: outcome.classical_bits,
                probs,
                amplitudes: None,
            })
        }
    }
}

fn run_or_fail(
    participant: &Participant,
    case: &GeneratedCase,
    seed: u64,
    want_amplitudes: bool,
) -> RunResult {
    run_participant(participant, &case.circuit, seed, want_amplitudes).unwrap_or_else(|err| {
        panic!(
            "{}participant `{}` was declared eligible but failed at seed {seed}: {err}",
            case.context(),
            participant.name
        )
    })
}

/// Exact unconditional mixture over the circuit's trajectories.
///
/// `reset` is already a channel on the density matrix. `measure` is lowered to
/// the dephasing Kraus pair `{|0><0|, |1><1|}`, which is the channel the
/// recorded outcome averages to. Conditioning on a recorded outcome has no
/// dephased equivalent, so this is only valid for circuits without
/// `Conditional`.
pub fn exact_mixture(circuit: &Circuit) -> Vec<f64> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let dephasing = [[[one, zero], [zero, zero]], [[zero, zero], [zero, one]]];

    let mut backend = DensityMatrixBackend::new(CONFORMANCE_SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instr in &circuit.instructions {
        match instr {
            Instruction::Measure { qubit, .. } => backend.apply_1q_kraus(*qubit, &dephasing),
            Instruction::Reset { qubit } => backend.reset(*qubit).unwrap(),
            other => backend.apply(other).unwrap(),
        }
    }
    backend.probabilities().unwrap()
}

// ---- Comparison ----

fn first_difference(a: &[f64], b: &[f64], eps: f64) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.is_nan() || y.is_nan() || (x - y).abs() >= eps)
}

fn agree(a: &[f64], b: &[f64], eps: f64) -> bool {
    first_difference(a, b, eps).is_none()
}

struct Class {
    members: Vec<&'static str>,
    probs: Vec<f64>,
    classical_bits: Vec<bool>,
    anchors: Vec<Anchor>,
}

fn partition(runs: &[(&Participant, RunResult)]) -> Vec<Class> {
    let mut classes: Vec<Class> = Vec::new();
    for (participant, result) in runs {
        match classes.iter_mut().find(|class| {
            class.classical_bits == result.classical_bits
                && agree(&class.probs, &result.probs, PROB_EPS)
        }) {
            Some(class) => {
                class.members.push(participant.name);
                if let Some(anchor) = participant.anchor {
                    class.anchors.push(anchor);
                }
            }
            None => classes.push(Class {
                members: vec![participant.name],
                probs: result.probs.clone(),
                classical_bits: result.classical_bits.clone(),
                anchors: participant.anchor.into_iter().collect(),
            }),
        }
    }
    classes
}

fn report_split(case: &GeneratedCase, classes: &[Class], seed: u64) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "{}participants split into {} disagreeing classes at seed {seed} (eps {PROB_EPS:.0e})",
        case.context(),
        classes.len()
    )
    .unwrap();
    for (i, class) in classes.iter().enumerate() {
        writeln!(
            out,
            "  class {i}: {} | bits {:?}",
            class.members.join(", "),
            class.classical_bits
        )
        .unwrap();
        for (j, other) in classes.iter().enumerate().skip(i + 1) {
            if let Some(idx) = first_difference(&class.probs, &other.probs, PROB_EPS) {
                writeln!(
                    out,
                    "    vs class {j} at prob[{idx}]: {:.15} vs {:.15}",
                    class.probs.get(idx).copied().unwrap_or(f64::NAN),
                    other.probs.get(idx).copied().unwrap_or(f64::NAN)
                )
                .unwrap();
            }
        }
    }
    let anchored: Vec<usize> = classes
        .iter()
        .enumerate()
        .filter(|(_, class)| !class.anchors.is_empty())
        .map(|(i, _)| i)
        .collect();
    match anchored.as_slice() {
        [i] => writeln!(
            out,
            "  verdict: class {i} holds the independent anchors {:?} and is the correct side; \
             every other class is the failure",
            classes[*i].anchors
        )
        .unwrap(),
        [] => writeln!(
            out,
            "  verdict: no independent anchor participated, so no class is presumed correct; \
             the closed-form anchor for this motif is `{}` in tests/golden_small_circuits.rs",
            case.family.golden_anchor()
        )
        .unwrap(),
        many => writeln!(
            out,
            "  verdict: independent anchors landed in more than one class ({many:?}), so the \
             anchors themselves disagree"
        )
        .unwrap(),
    }
    out
}

/// Anchors available to a case: the compared participants that carry one, plus
/// the density-matrix mixture oracle where the regime uses it.
fn available_anchors(case: &GeneratedCase, participants: &[Participant]) -> Vec<Anchor> {
    let mut anchors: Vec<Anchor> = participants
        .iter()
        .filter(|participant| participant.eligibility(&case.profile).is_compared())
        .filter_map(|participant| participant.anchor)
        .collect();
    if case.family.regime() == Regime::ExactMixture {
        anchors.push(Anchor::Channel);
    }
    anchors
}

fn assert_anchor_policy(case: &GeneratedCase, anchors: &[Anchor], compared: usize) {
    match case.family.anchor_policy() {
        AnchorPolicy::Required(anchor) => assert!(
            anchors.contains(&anchor),
            "{}requires the {anchor:?} anchor, but no compared participant and no oracle \
             provided it",
            case.context()
        ),
        AnchorPolicy::ConsensusOnly => assert!(
            compared >= MIN_CONSENSUS_PARTICIPANTS,
            "{}has no independent anchor and only {compared} compared participants \
             (minimum {MIN_CONSENSUS_PARTICIPANTS})",
            case.context()
        ),
    }
}

/// Participants the matrix compares for `case`, run once at `seed`.
fn compared_runs<'a>(
    case: &GeneratedCase,
    participants: &'a [Participant],
    seed: u64,
) -> Vec<(&'a Participant, RunResult)> {
    participants
        .iter()
        .filter(|participant| participant.eligibility(&case.profile).is_compared())
        .map(|participant| {
            let result = run_or_fail(participant, case, seed, false);
            (participant, result)
        })
        .collect()
}

/// How many of [`DETERMINISM_SEEDS`] a case replays over. Cost grows as `4^n` on
/// the density-matrix participant, so wide cases take the shorter sweep.
fn determinism_seeds(profile: &CaseProfile) -> &'static [u64] {
    if profile.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        &DETERMINISM_SEEDS[..2]
    } else {
        &DETERMINISM_SEEDS
    }
}

/// One run per participant, compared for identical classical bits and
/// probabilities, replayed across [`DETERMINISM_SEEDS`] to confirm that the
/// family's forced-trajectory claim holds.
pub fn assert_deterministic_case(case: &GeneratedCase, participants: &[Participant]) {
    assert_eq!(
        case.family.regime(),
        Regime::Deterministic,
        "{}",
        case.context()
    );
    let anchors = available_anchors(case, participants);
    let mut reference: Option<(Vec<f64>, Vec<bool>)> = None;
    for &seed in determinism_seeds(&case.profile) {
        let runs = compared_runs(case, participants, seed);
        assert_anchor_policy(case, &anchors, runs.len());
        let classes = partition(&runs);
        assert!(classes.len() == 1, "{}", report_split(case, &classes, seed));

        let class = &classes[0];
        match &reference {
            None => reference = Some((class.probs.clone(), class.classical_bits.clone())),
            Some((probs, bits)) => assert!(
                agree(probs, &class.probs, PROB_EPS) && bits == &class.classical_bits,
                "{}was declared a forced trajectory, but seed {seed} produced bits {:?} \
                 against {bits:?} at seed {}",
                case.context(),
                class.classical_bits,
                DETERMINISM_SEEDS[0]
            ),
        }
    }
}

/// Amplitudes from every participant the matrix says exposes them, compared
/// after fixing the global phase.
pub fn assert_amplitudes(case: &GeneratedCase, participants: &[Participant]) {
    let mut reference: Option<(&'static str, Vec<Complex64>)> = None;
    for participant in participants {
        if !participant
            .amplitude_eligibility(&case.profile)
            .is_compared()
        {
            continue;
        }
        let result = run_or_fail(participant, case, CONFORMANCE_SEED, true);
        let exported = result.amplitudes.unwrap_or_else(|| {
            panic!(
                "{}participant `{}` is declared to expose amplitudes but returned none",
                case.context(),
                participant.name
            )
        });
        let amplitudes = canonical_phase(&exported);
        match &reference {
            None => reference = Some((participant.name, amplitudes)),
            Some((name, expected)) => {
                assert_eq!(
                    amplitudes.len(),
                    expected.len(),
                    "{}`{}` exported {} amplitudes against {} from `{name}`",
                    case.context(),
                    participant.name,
                    amplitudes.len(),
                    expected.len()
                );
                for (i, (got, want)) in amplitudes.iter().zip(expected).enumerate() {
                    let diff = (*got - *want).norm();
                    assert!(
                        diff < AMP_EPS,
                        "{}amplitude[{i}]: `{}` gave {got}, `{name}` gave {want} \
                         (diff {diff:.2e}, eps {AMP_EPS:.0e})",
                        case.context(),
                        participant.name
                    );
                }
            }
        }
    }
}

/// Divide out the global phase of the first amplitude with appreciable weight.
/// Probabilities already agree to [`PROB_EPS`] before this runs, so every
/// participant selects the same pivot.
fn canonical_phase(amplitudes: &[Complex64]) -> Vec<Complex64> {
    const PIVOT_MIN: f64 = 1e-6;
    let Some(pivot) = amplitudes.iter().find(|a| a.norm() > PIVOT_MIN) else {
        return amplitudes.to_vec();
    };
    let phase = *pivot / pivot.norm();
    amplitudes.iter().map(|a| *a / phase).collect()
}

// ---- Branching regimes ----

pub struct Branch {
    pub classical_bits: Vec<bool>,
    pub probs: Vec<f64>,
    pub count: usize,
}

/// Distinct trajectories a participant produces over the seed sweep, with how
/// often each occurred.
fn collect_branches(participant: &Participant, case: &GeneratedCase, seeds: u64) -> Vec<Branch> {
    let mut branches: Vec<Branch> = Vec::new();
    for offset in 0..seeds {
        let result = run_or_fail(participant, case, CONFORMANCE_SEED + offset, false);
        match branches.iter_mut().find(|branch| {
            branch.classical_bits == result.classical_bits
                && agree(&branch.probs, &result.probs, PROB_EPS)
        }) {
            Some(branch) => branch.count += 1,
            None => branches.push(Branch {
                classical_bits: result.classical_bits,
                probs: result.probs,
                count: 1,
            }),
        }
    }
    branches.sort_by(|a, b| {
        a.classical_bits
            .cmp(&b.classical_bits)
            .then_with(|| a.probs.partial_cmp(&b.probs).unwrap())
    });
    branches
}

fn describe_branches(branches: &[Branch]) -> String {
    branches
        .iter()
        .map(|branch| {
            format!(
                "[bits {:?} count {} probs {}]",
                branch.classical_bits,
                branch.count,
                branch
                    .probs
                    .iter()
                    .map(|p| format!("{p:.6}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn branch_sets_match(a: &[Branch], b: &[Branch]) -> bool {
    a.len() == b.len()
        && a.iter().all(|left| {
            b.iter().any(|right| {
                left.classical_bits == right.classical_bits
                    && agree(&left.probs, &right.probs, PROB_EPS)
            })
        })
}

struct BranchClass {
    members: Vec<&'static str>,
    anchors: Vec<Anchor>,
    branches: Vec<Branch>,
}

fn partition_branch_sets(
    collected: Vec<(&'static str, Option<Anchor>, Vec<Branch>)>,
) -> Vec<BranchClass> {
    let mut classes: Vec<BranchClass> = Vec::new();
    for (name, anchor, branches) in collected {
        match classes
            .iter_mut()
            .find(|class| branch_sets_match(&class.branches, &branches))
        {
            Some(class) => {
                class.members.push(name);
                class.anchors.extend(anchor);
            }
            None => classes.push(BranchClass {
                members: vec![name],
                anchors: anchor.into_iter().collect(),
                branches,
            }),
        }
    }
    classes
}

/// Blame report for a branching split. When a mixture oracle is available, the
/// class whose branches recombine into it is the correct one; otherwise the
/// class holding an independent anchor is.
fn report_branch_split(
    case: &GeneratedCase,
    classes: &[BranchClass],
    oracle: Option<&[f64]>,
) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "{}participants split into {} disagreeing trajectory sets over {BRANCH_SEEDS} seeds",
        case.context(),
        classes.len()
    )
    .unwrap();
    let mut consistent: Vec<usize> = Vec::new();
    for (i, class) in classes.iter().enumerate() {
        writeln!(out, "  class {i}: {}", class.members.join(", ")).unwrap();
        writeln!(out, "    branches: {}", describe_branches(&class.branches)).unwrap();
        if let Some(oracle) = oracle {
            match mixture_residual(&distinct_probability_branches(&class.branches), oracle) {
                Ok((weights, residual)) if residual < MIXTURE_EPS => {
                    consistent.push(i);
                    writeln!(
                        out,
                        "    recombines into the density-matrix channel with weights {weights:?}"
                    )
                    .unwrap();
                }
                Ok((_, residual)) => writeln!(
                    out,
                    "    does not recombine into the density-matrix channel (residual {residual:.2e})"
                )
                .unwrap(),
                Err(reason) => writeln!(out, "    cannot be checked against the channel: {reason}")
                    .unwrap(),
            }
        }
    }
    if let Some(oracle) = oracle {
        writeln!(out, "  channel mixture: {oracle:?}").unwrap();
        match consistent.as_slice() {
            [i] => writeln!(
                out,
                "  verdict: class {i} is the reset channel `rho -> |0><0| (x) tr_q rho`; every \
                 other class is the failure"
            )
            .unwrap(),
            [] => writeln!(
                out,
                "  verdict: no class reproduces the channel, so the density-matrix oracle itself \
                 is suspect; check `{}` in tests/golden_small_circuits.rs first",
                case.family.golden_anchor()
            )
            .unwrap(),
            many => writeln!(
                out,
                "  verdict: {many:?} all reproduce the channel, so the split is in the recorded \
                 bits rather than the state"
            )
            .unwrap(),
        }
        return out;
    }
    let anchored: Vec<usize> = classes
        .iter()
        .enumerate()
        .filter(|(_, class)| !class.anchors.is_empty())
        .map(|(i, _)| i)
        .collect();
    match anchored.as_slice() {
        [i] => writeln!(
            out,
            "  verdict: class {i} holds the independent anchors {:?} and is the correct side",
            classes[*i].anchors
        )
        .unwrap(),
        _ => writeln!(
            out,
            "  verdict: no single class holds an independent anchor; the closed-form anchor for \
             this motif is `{}` in tests/golden_small_circuits.rs",
            case.family.golden_anchor()
        )
        .unwrap(),
    }
    out
}

fn collect_all_branches(case: &GeneratedCase, participants: &[Participant]) -> Vec<BranchClass> {
    let collected: Vec<(&'static str, Option<Anchor>, Vec<Branch>)> = participants
        .iter()
        .filter(|participant| participant.eligibility(&case.profile).is_compared())
        .map(|participant| {
            (
                participant.name,
                participant.anchor,
                collect_branches(participant, case, BRANCH_SEEDS),
            )
        })
        .collect();
    assert!(
        collected.len() >= 2,
        "{}only {} participant(s) were compared; a differential check needs at least two",
        case.context(),
        collected.len()
    );
    assert_anchor_policy(
        case,
        &available_anchors(case, participants),
        collected.len(),
    );
    partition_branch_sets(collected)
}

/// Branch sets must agree across participants, and the branches must recombine
/// into the exact mixture with non-negative weights summing to one.
///
/// Returns how many distinct probability vectors the sweep produced. A collapse
/// can leave the register in states the computational basis cannot tell apart,
/// so one is a legitimate result for a single case; callers check that the
/// family as a whole reaches more.
pub fn assert_exact_mixture_case(case: &GeneratedCase, participants: &[Participant]) -> usize {
    assert_eq!(
        case.family.regime(),
        Regime::ExactMixture,
        "{}",
        case.context()
    );
    assert!(
        !case.profile.has_conditional,
        "{}the dephasing lowering cannot express classical feedback",
        case.context()
    );

    let classes = collect_all_branches(case, participants);
    let oracle = exact_mixture(&case.circuit);
    assert!(
        classes.len() == 1,
        "{}",
        report_branch_split(case, &classes, Some(&oracle))
    );

    let branches = &classes[0].branches;
    let distinct = distinct_probability_branches(branches);
    let (weights, residual) =
        mixture_residual(&distinct, &oracle).unwrap_or_else(|reason| {
            panic!(
                "{}the observed branches cannot be matched to the density-matrix mixture:                  {reason}
  branches: {}
  mixture: {oracle:?}",
                case.context(),
                describe_branches(branches)
            )
        });
    assert!(
        residual < MIXTURE_EPS,
        "{}the branch mixture misses the density-matrix channel by {residual:.2e}          (eps {MIXTURE_EPS:.0e})
  weights {weights:?}
  branches: {}
  mixture: {oracle:?}",
        case.context(),
        describe_branches(branches)
    );
    distinct.len()
}

/// Branch sets must agree across participants, and the seed-averaged means must
/// agree within [`SAMPLED_MEAN_EPS`]. Used where classical feedback puts the
/// exact mixture out of reach.
pub fn assert_sampled_mixture_case(case: &GeneratedCase, participants: &[Participant]) {
    assert_eq!(
        case.family.regime(),
        Regime::SampledMixture,
        "{}",
        case.context()
    );
    let classes = collect_all_branches(case, participants);
    assert!(
        classes.len() == 1,
        "{}",
        report_branch_split(case, &classes, None)
    );

    let means: Vec<(&'static str, Vec<f64>)> = participants
        .iter()
        .filter(|participant| participant.eligibility(&case.profile).is_compared())
        .map(|participant| {
            (
                participant.name,
                branch_mean(&collect_branches(participant, case, BRANCH_SEEDS)),
            )
        })
        .collect();
    let (reference_name, reference_mean) = &means[0];
    for (name, mean) in &means[1..] {
        for (slot, (got, want)) in mean.iter().zip(reference_mean).enumerate() {
            assert!(
                (got - want).abs() < SAMPLED_MEAN_EPS,
                "{}seed-averaged probability at slot {slot} differs: `{name}` {got:.6} against                  `{reference_name}` {want:.6} (band {SAMPLED_MEAN_EPS})",
                case.context()
            );
        }
    }
}

/// Weights and worst-slot residual for `sum_b w_b * branches[b] = oracle`.
/// Weights outside `[0, 1]` or a sum away from one mean the observed branches
/// are not the channel's decomposition, which the residual alone would not
/// catch, so both are folded into the returned residual.
fn mixture_residual(
    branches: &[Vec<f64>],
    oracle: &[f64],
) -> std::result::Result<(Vec<f64>, f64), String> {
    let vectors: Vec<&[f64]> = branches.iter().map(Vec::as_slice).collect();
    let weights = solve_mixture_weights(&vectors, oracle)
        .ok_or_else(|| format!("the {} branches are linearly dependent", branches.len()))?;
    let total: f64 = weights.iter().sum();
    let mut residual = (total - 1.0).abs();
    for weight in &weights {
        residual = residual.max(-weight);
    }
    for (slot, want) in oracle.iter().enumerate() {
        let got: f64 = weights
            .iter()
            .zip(branches)
            .map(|(weight, probs)| weight * probs[slot])
            .sum();
        residual = residual.max((got - want).abs());
    }
    Ok((weights, residual))
}

/// Branch probability vectors with duplicates merged. Two trajectories that
/// differ only in their recorded bits carry the same state, and feeding both to
/// the weight solve would leave it rank deficient.
fn distinct_probability_branches(branches: &[Branch]) -> Vec<Vec<f64>> {
    let mut distinct: Vec<Vec<f64>> = Vec::new();
    for branch in branches {
        if !distinct
            .iter()
            .any(|probs| agree(probs, &branch.probs, PROB_EPS))
        {
            distinct.push(branch.probs.clone());
        }
    }
    distinct
}

fn branch_mean(branches: &[Branch]) -> Vec<f64> {
    let len = branches[0].probs.len();
    let total: usize = branches.iter().map(|branch| branch.count).sum();
    let mut mean = vec![0.0; len];
    for branch in branches {
        for (slot, p) in branch.probs.iter().enumerate() {
            mean[slot] += p * branch.count as f64;
        }
    }
    for value in &mut mean {
        *value /= total as f64;
    }
    mean
}

/// Least-squares weights `w` solving `sum_b w_b * vectors[b] = target`, via the
/// normal equations with partial pivoting. Returns `None` when the branch
/// vectors are linearly dependent, which leaves the decomposition
/// undetermined.
fn solve_mixture_weights(vectors: &[&[f64]], target: &[f64]) -> Option<Vec<f64>> {
    let k = vectors.len();
    let mut system = vec![vec![0.0f64; k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            system[i][j] = dot(vectors[i], vectors[j]);
        }
        system[i][k] = dot(vectors[i], target);
    }

    for col in 0..k {
        let pivot = (col..k).max_by(|&a, &b| {
            system[a][col]
                .abs()
                .partial_cmp(&system[b][col].abs())
                .unwrap()
        })?;
        if system[pivot][col].abs() < 1e-12 {
            return None;
        }
        system.swap(col, pivot);
        for row in col + 1..k {
            let factor = system[row][col] / system[col][col];
            let (upper, lower) = system.split_at_mut(row);
            for (entry, pivot_entry) in lower[0].iter_mut().zip(&upper[col]).skip(col) {
                *entry -= factor * pivot_entry;
            }
        }
    }

    let mut weights = vec![0.0f64; k];
    for row in (0..k).rev() {
        let mut acc = system[row][k];
        for col in row + 1..k {
            acc -= system[row][col] * weights[col];
        }
        weights[row] = acc / system[row][row];
    }
    Some(weights)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
