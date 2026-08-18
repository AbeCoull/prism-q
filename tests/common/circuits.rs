#![allow(dead_code)]

use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec, guarded};
use prism_q::circuits as builtins;
use prism_q::gates::Gate;

use super::SEED;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Sparse,
    Mps,
    TensorNetwork,
    Factored,
    Stabilizer,
    Product,
}

impl BackendKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Mps => "mps",
            Self::TensorNetwork => "tensor_network",
            Self::Factored => "factored",
            Self::Stabilizer => "stabilizer",
            Self::Product => "product",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSupport {
    Supported,
    Rejected,
}

impl BackendSupport {
    pub const fn is_supported(self) -> bool {
        matches!(self, BackendSupport::Supported)
    }
}

/// The circuit properties that decide which backends a case may run on.
///
/// Every field here has a reader in [`CircuitCase::support`]. A property that
/// only documents the case belongs in the case name or a comment, not in this
/// struct, where an unread field reads as a rule that is being enforced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitCapabilities {
    pub requires_non_clifford: bool,
    pub safe_for_mps: bool,
    pub safe_for_tensor_network: bool,
    pub product_separable: bool,
}

impl CircuitCapabilities {
    pub const fn new() -> Self {
        Self {
            requires_non_clifford: false,
            safe_for_mps: true,
            safe_for_tensor_network: true,
            product_separable: false,
        }
    }

    pub const fn requires_non_clifford(mut self) -> Self {
        self.requires_non_clifford = true;
        self
    }

    pub const fn unsafe_for_mps(mut self) -> Self {
        self.safe_for_mps = false;
        self
    }

    pub const fn unsafe_for_tensor_network(mut self) -> Self {
        self.safe_for_tensor_network = false;
        self
    }

    pub const fn product_separable(mut self) -> Self {
        self.product_separable = true;
        self
    }
}

#[derive(Clone, Copy)]
pub struct CircuitCase {
    pub name: &'static str,
    pub build: fn() -> Circuit,
    pub capabilities: CircuitCapabilities,
}

impl CircuitCase {
    pub const fn new(
        name: &'static str,
        build: fn() -> Circuit,
        capabilities: CircuitCapabilities,
    ) -> Self {
        Self {
            name,
            build,
            capabilities,
        }
    }

    pub fn circuit(self) -> Circuit {
        (self.build)()
    }

    pub const fn support(self, backend: BackendKind) -> BackendSupport {
        let supported = BackendSupport::Supported;
        let rejected = BackendSupport::Rejected;
        match backend {
            BackendKind::Sparse | BackendKind::Factored => supported,
            BackendKind::Mps if self.capabilities.safe_for_mps => supported,
            BackendKind::TensorNetwork if self.capabilities.safe_for_tensor_network => supported,
            BackendKind::Stabilizer if !self.capabilities.requires_non_clifford => supported,
            BackendKind::Product if self.capabilities.product_separable => supported,
            _ => rejected,
        }
    }
}

pub fn find_case<I>(cases: I, name: &str) -> CircuitCase
where
    I: IntoIterator<Item = CircuitCase>,
{
    cases
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing circuit case {name}"))
}

pub fn bell() -> Circuit {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit
}

pub fn ghz_3() -> Circuit {
    builtins::ghz_circuit(3)
}

pub fn ghz_5() -> Circuit {
    builtins::ghz_circuit(5)
}

pub fn qft_4() -> Circuit {
    builtins::qft_circuit(4)
}

pub fn qft_8() -> Circuit {
    builtins::qft_circuit(8)
}

pub fn random_4() -> Circuit {
    builtins::random_circuit(4, 10, SEED)
}

pub fn random_8() -> Circuit {
    builtins::random_circuit(8, 10, SEED)
}

pub fn hea_4() -> Circuit {
    builtins::hardware_efficient_ansatz(4, 3, SEED)
}

pub fn ghz_4() -> Circuit {
    builtins::ghz_circuit(4)
}

pub fn qaoa_4() -> Circuit {
    builtins::qaoa_circuit(4, 2, SEED)
}

pub fn qaoa_4_l3() -> Circuit {
    builtins::qaoa_circuit(4, 3, SEED)
}

pub fn qpe_4() -> Circuit {
    builtins::phase_estimation_circuit(4)
}

pub fn qpe_8() -> Circuit {
    builtins::phase_estimation_circuit(8)
}

pub fn cz_chain_8() -> Circuit {
    builtins::cz_chain_circuit(8, 5, SEED)
}

pub fn w_state_4() -> Circuit {
    builtins::w_state_circuit(4)
}

pub fn single_qubit_rotations() -> Circuit {
    builtins::single_qubit_rotation_circuit(6, 5, SEED)
}

pub fn single_qubit_rotations_4q() -> Circuit {
    builtins::single_qubit_rotation_circuit(4, 5, SEED)
}

pub fn single_qubit_rotations_8q() -> Circuit {
    builtins::single_qubit_rotation_circuit(8, 10, SEED)
}

pub fn single_qubit_rotations_12q() -> Circuit {
    builtins::single_qubit_rotation_circuit(12, 10, SEED)
}

pub fn single_qubit_rotations_16q() -> Circuit {
    builtins::single_qubit_rotation_circuit(16, 5, SEED)
}

pub fn clifford_random_small() -> Circuit {
    builtins::clifford_heavy_circuit(6, 8, SEED)
}

pub fn sparse_basis_permutation() -> Circuit {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[2]);
    circuit.add_gate(Gate::Swap, &[0, 3]);
    circuit.add_gate(Gate::Cx, &[2, 1]);
    circuit
}

pub fn deterministic_measurement() -> Circuit {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit
}

pub fn reset_from_one() -> Circuit {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);
    circuit
}

/// Reset a qubit holding |1> while another qubit is in superposition. The
/// reset outcome is forced, so the case stays deterministic, but the spectator
/// is what makes it discriminating: an implementation that reinitializes the
/// whole register instead of the reset qubit collapses qubit 0 as well.
/// One-qubit `reset_from_one` cannot see that.
pub fn reset_from_one_with_spectator() -> Circuit {
    let mut circuit = Circuit::new(2, 1);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(1);
    circuit.add_measure(1, 0);
    circuit
}

pub fn superposition_measurement() -> Circuit {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit
}

pub fn measurement_reset_conditional() -> Circuit {
    let mut circuit = Circuit::new(2, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_reset(0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    });
    circuit
}

pub fn product_separable_cases() -> [CircuitCase; 4] {
    [
        CircuitCase::new(
            "single_qubit_rotations_4q",
            single_qubit_rotations_4q,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .product_separable(),
        ),
        CircuitCase::new(
            "single_qubit_rotations_8q",
            single_qubit_rotations_8q,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .product_separable(),
        ),
        CircuitCase::new(
            "single_qubit_rotations_12q",
            single_qubit_rotations_12q,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .product_separable(),
        ),
        CircuitCase::new(
            "single_qubit_rotations_16q",
            single_qubit_rotations_16q,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .product_separable(),
        ),
    ]
}

pub fn exact_small_cases() -> [CircuitCase; 18] {
    [
        CircuitCase::new("bell", bell, CircuitCapabilities::new()),
        CircuitCase::new("ghz_3", ghz_3, CircuitCapabilities::new()),
        CircuitCase::new("ghz_4", ghz_4, CircuitCapabilities::new()),
        CircuitCase::new("ghz_5", ghz_5, CircuitCapabilities::new()),
        CircuitCase::new(
            "qft_4",
            qft_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "qft_8",
            qft_8,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "random_4",
            random_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "random_8",
            random_8,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "hea_4",
            hea_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "qaoa_4",
            qaoa_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "qaoa_4_l3",
            qaoa_4_l3,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "qpe_4",
            qpe_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "qpe_8",
            qpe_8,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "cz_chain_8",
            cz_chain_8,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "w_state_4",
            w_state_4,
            CircuitCapabilities::new().requires_non_clifford(),
        ),
        CircuitCase::new(
            "single_qubit_rotations",
            single_qubit_rotations,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .product_separable(),
        ),
        CircuitCase::new(
            "clifford_random_small",
            clifford_random_small,
            CircuitCapabilities::new(),
        ),
        CircuitCase::new(
            "sparse_basis_permutation",
            sparse_basis_permutation,
            CircuitCapabilities::new(),
        ),
    ]
}

// The region body holds a gate, a measure, and a reset, so a backend that
// handled only the guarded-gate case cannot pass it.
pub fn measurement_reset_region() -> Circuit {
    let mut circuit = Circuit::new(3, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(
        guarded(
            ClassicalCondition::BitIsOne(0),
            vec![
                Instruction::Gate {
                    gate: Gate::X,
                    targets: SmallVec::from_slice(&[1]),
                },
                Instruction::Reset { qubit: 0 },
                Instruction::Measure {
                    qubit: 1,
                    classical_bit: 1,
                },
                Instruction::Gate {
                    gate: Gate::Cx,
                    targets: SmallVec::from_slice(&[1, 2]),
                },
            ],
        )
        .expect("body is not empty"),
    );
    circuit
}

// The shape `else` and `switch` lower to: sibling guards on a condition and its
// negation, each re-reading the classical bits after the previous body ran, and
// each body measuring. The condition is a parity, so every backend's shared
// `ClassicalCondition::evaluate` carries that variant here too.
pub fn parity_sibling_regions() -> Circuit {
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let parity = ClassicalCondition::Parity {
        bits: vec![0, 1].into(),
        expected: true,
    };
    circuit.instructions.push(
        guarded(
            parity.clone(),
            vec![
                Instruction::Gate {
                    gate: Gate::X,
                    targets: SmallVec::from_slice(&[2]),
                },
                Instruction::Measure {
                    qubit: 2,
                    classical_bit: 2,
                },
            ],
        )
        .expect("body is not empty"),
    );
    circuit.instructions.push(
        guarded(
            parity.negate(),
            vec![
                Instruction::Gate {
                    gate: Gate::X,
                    targets: SmallVec::from_slice(&[3]),
                },
                Instruction::Gate {
                    gate: Gate::Cx,
                    targets: SmallVec::from_slice(&[3, 2]),
                },
            ],
        )
        .expect("body is not empty"),
    );
    circuit
}

pub fn measurement_cases() -> [CircuitCase; 6] {
    [
        CircuitCase::new(
            "deterministic_measurement",
            deterministic_measurement,
            CircuitCapabilities::new().product_separable(),
        ),
        CircuitCase::new(
            "reset_from_one",
            reset_from_one,
            CircuitCapabilities::new().product_separable(),
        ),
        CircuitCase::new(
            "reset_from_one_with_spectator",
            reset_from_one_with_spectator,
            CircuitCapabilities::new().product_separable(),
        ),
        CircuitCase::new(
            "measurement_reset_conditional",
            measurement_reset_conditional,
            CircuitCapabilities::new().product_separable(),
        ),
        CircuitCase::new(
            "measurement_reset_region",
            measurement_reset_region,
            CircuitCapabilities::new(),
        ),
        CircuitCase::new(
            "parity_sibling_regions",
            parity_sibling_regions,
            CircuitCapabilities::new(),
        ),
    ]
}

pub fn random_measurement_cases() -> [CircuitCase; 1] {
    [CircuitCase::new(
        "superposition_measurement",
        superposition_measurement,
        CircuitCapabilities::new().product_separable(),
    )]
}

/// One case per qubit count from 9 to 19, straddling every fusion threshold.
/// The shape changes with size because the passes that switch on at each
/// threshold need a circuit that exercises them: random and ansatz circuits
/// below the diagonal-batch thresholds, QFT above them.
pub fn fusion_threshold_cases() -> [CircuitCase; 11] {
    type NamedBuilder = (&'static str, fn() -> Circuit);
    const BUILDERS: [NamedBuilder; 11] = [
        ("fusion_threshold_9", || {
            builtins::random_circuit(9, 10, SEED)
        }),
        ("fusion_threshold_10", || {
            builtins::random_circuit(10, 10, SEED)
        }),
        ("fusion_threshold_11", || {
            builtins::hardware_efficient_ansatz(11, 3, SEED)
        }),
        ("fusion_threshold_12", || {
            builtins::hardware_efficient_ansatz(12, 3, SEED)
        }),
        ("fusion_threshold_13", || {
            builtins::random_circuit(13, 10, SEED)
        }),
        ("fusion_threshold_14", || {
            builtins::random_circuit(14, 10, SEED)
        }),
        ("fusion_threshold_15", || builtins::qft_circuit(15)),
        ("fusion_threshold_16", || builtins::qft_circuit(16)),
        ("fusion_threshold_17", || builtins::qft_circuit(17)),
        ("fusion_threshold_18", || builtins::qft_circuit(18)),
        ("fusion_threshold_19", || builtins::qft_circuit(19)),
    ];
    BUILDERS.map(|(name, build)| {
        CircuitCase::new(
            name,
            build,
            CircuitCapabilities::new().requires_non_clifford(),
        )
    })
}
