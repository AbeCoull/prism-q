#![allow(dead_code)]

use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendExpectation {
    pub support: BackendSupport,
    pub max_qubits: Option<usize>,
    pub tolerance_override: Option<f64>,
    pub export_required: bool,
}

impl BackendExpectation {
    pub const fn supported() -> Self {
        Self {
            support: BackendSupport::Supported,
            max_qubits: None,
            tolerance_override: None,
            export_required: false,
        }
    }

    pub const fn rejected() -> Self {
        Self {
            support: BackendSupport::Rejected,
            max_qubits: None,
            tolerance_override: None,
            export_required: false,
        }
    }

    pub const fn is_supported(self) -> bool {
        matches!(self.support, BackendSupport::Supported)
    }

    pub const fn with_max_qubits(mut self, max_qubits: usize) -> Self {
        self.max_qubits = Some(max_qubits);
        self
    }

    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance_override = Some(tolerance);
        self
    }

    pub const fn requiring_export(mut self) -> Self {
        self.export_required = true;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitCapabilities {
    pub exact_probabilities: bool,
    pub clifford_only: bool,
    pub requires_measurement: bool,
    pub requires_reset: bool,
    pub requires_non_clifford: bool,
    pub requires_qft_block_expansion: bool,
    pub safe_for_mps: bool,
    pub safe_for_tensor_network: bool,
    pub product_separable: bool,
}

impl CircuitCapabilities {
    pub const fn new() -> Self {
        Self {
            exact_probabilities: true,
            clifford_only: false,
            requires_measurement: false,
            requires_reset: false,
            requires_non_clifford: false,
            requires_qft_block_expansion: false,
            safe_for_mps: true,
            safe_for_tensor_network: true,
            product_separable: false,
        }
    }

    pub const fn clifford_only(mut self) -> Self {
        self.clifford_only = true;
        self
    }

    pub const fn requires_measurement(mut self) -> Self {
        self.requires_measurement = true;
        self
    }

    pub const fn requires_reset(mut self) -> Self {
        self.requires_reset = true;
        self
    }

    pub const fn requires_non_clifford(mut self) -> Self {
        self.requires_non_clifford = true;
        self
    }

    pub const fn requires_qft_block_expansion(mut self) -> Self {
        self.requires_qft_block_expansion = true;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendExpectations {
    pub sparse: BackendExpectation,
    pub mps: BackendExpectation,
    pub tensor_network: BackendExpectation,
    pub factored: BackendExpectation,
    pub stabilizer: BackendExpectation,
    pub product: BackendExpectation,
}

impl BackendExpectations {
    pub const fn from_capabilities(capabilities: CircuitCapabilities) -> Self {
        let supported = BackendExpectation::supported();
        let rejected = BackendExpectation::rejected();
        Self {
            sparse: supported,
            mps: if capabilities.safe_for_mps {
                supported
            } else {
                rejected
            },
            tensor_network: if capabilities.safe_for_tensor_network {
                supported
            } else {
                rejected
            },
            factored: supported,
            stabilizer: if capabilities.requires_non_clifford {
                rejected
            } else {
                supported
            },
            product: if capabilities.product_separable {
                supported
            } else {
                rejected
            },
        }
    }

    pub const fn for_backend(self, backend: BackendKind) -> BackendExpectation {
        match backend {
            BackendKind::Sparse => self.sparse,
            BackendKind::Mps => self.mps,
            BackendKind::TensorNetwork => self.tensor_network,
            BackendKind::Factored => self.factored,
            BackendKind::Stabilizer => self.stabilizer,
            BackendKind::Product => self.product,
        }
    }

    pub const fn with_sparse(mut self, expectation: BackendExpectation) -> Self {
        self.sparse = expectation;
        self
    }

    pub const fn with_mps(mut self, expectation: BackendExpectation) -> Self {
        self.mps = expectation;
        self
    }

    pub const fn with_tensor_network(mut self, expectation: BackendExpectation) -> Self {
        self.tensor_network = expectation;
        self
    }

    pub const fn with_factored(mut self, expectation: BackendExpectation) -> Self {
        self.factored = expectation;
        self
    }

    pub const fn with_stabilizer(mut self, expectation: BackendExpectation) -> Self {
        self.stabilizer = expectation;
        self
    }

    pub const fn with_product(mut self, expectation: BackendExpectation) -> Self {
        self.product = expectation;
        self
    }
}

#[derive(Clone, Copy)]
pub struct CircuitCase {
    pub name: &'static str,
    pub build: fn() -> Circuit,
    pub capabilities: CircuitCapabilities,
    pub expectations: BackendExpectations,
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
            expectations: BackendExpectations::from_capabilities(capabilities),
        }
    }

    pub const fn with_expectations(mut self, expectations: BackendExpectations) -> Self {
        self.expectations = expectations;
        self
    }

    pub fn circuit(self) -> Circuit {
        (self.build)()
    }

    pub const fn expectation(self, backend: BackendKind) -> BackendExpectation {
        self.expectations.for_backend(backend)
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
        CircuitCase::new("bell", bell, CircuitCapabilities::new().clifford_only()),
        CircuitCase::new("ghz_3", ghz_3, CircuitCapabilities::new().clifford_only()),
        CircuitCase::new("ghz_4", ghz_4, CircuitCapabilities::new().clifford_only()),
        CircuitCase::new("ghz_5", ghz_5, CircuitCapabilities::new().clifford_only()),
        CircuitCase::new(
            "qft_4",
            qft_4,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .requires_qft_block_expansion(),
        ),
        CircuitCase::new(
            "qft_8",
            qft_8,
            CircuitCapabilities::new()
                .requires_non_clifford()
                .requires_qft_block_expansion(),
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
            CircuitCapabilities::new().clifford_only(),
        ),
        CircuitCase::new(
            "sparse_basis_permutation",
            sparse_basis_permutation,
            CircuitCapabilities::new().clifford_only(),
        ),
    ]
}

pub fn measurement_cases() -> [CircuitCase; 3] {
    [
        CircuitCase::new(
            "deterministic_measurement",
            deterministic_measurement,
            CircuitCapabilities::new()
                .clifford_only()
                .requires_measurement()
                .product_separable(),
        ),
        CircuitCase::new(
            "reset_from_one",
            reset_from_one,
            CircuitCapabilities::new()
                .clifford_only()
                .requires_measurement()
                .requires_reset()
                .product_separable(),
        ),
        CircuitCase::new(
            "measurement_reset_conditional",
            measurement_reset_conditional,
            CircuitCapabilities::new()
                .clifford_only()
                .requires_measurement()
                .requires_reset()
                .product_separable(),
        ),
    ]
}

pub fn random_measurement_cases() -> [CircuitCase; 1] {
    [CircuitCase::new(
        "superposition_measurement",
        superposition_measurement,
        CircuitCapabilities::new()
            .clifford_only()
            .requires_measurement()
            .product_separable(),
    )]
}

pub fn fusion_threshold_cases() -> Vec<CircuitCase> {
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        .into_iter()
        .map(|n| {
            let name = match n {
                9 => "fusion_threshold_9",
                10 => "fusion_threshold_10",
                11 => "fusion_threshold_11",
                12 => "fusion_threshold_12",
                13 => "fusion_threshold_13",
                14 => "fusion_threshold_14",
                15 => "fusion_threshold_15",
                16 => "fusion_threshold_16",
                17 => "fusion_threshold_17",
                18 => "fusion_threshold_18",
                19 => "fusion_threshold_19",
                _ => unreachable!(),
            };
            let build = match n {
                9 => fusion_threshold_9,
                10 => fusion_threshold_10,
                11 => fusion_threshold_11,
                12 => fusion_threshold_12,
                13 => fusion_threshold_13,
                14 => fusion_threshold_14,
                15 => fusion_threshold_15,
                16 => fusion_threshold_16,
                17 => fusion_threshold_17,
                18 => fusion_threshold_18,
                19 => fusion_threshold_19,
                _ => unreachable!(),
            };
            let capabilities = if n >= 15 {
                CircuitCapabilities::new()
                    .requires_non_clifford()
                    .requires_qft_block_expansion()
            } else {
                CircuitCapabilities::new().requires_non_clifford()
            };
            CircuitCase::new(name, build, capabilities)
        })
        .collect()
}

fn fusion_threshold_9() -> Circuit {
    builtins::random_circuit(9, 10, SEED)
}

fn fusion_threshold_10() -> Circuit {
    builtins::random_circuit(10, 10, SEED)
}

fn fusion_threshold_11() -> Circuit {
    builtins::hardware_efficient_ansatz(11, 3, SEED)
}

fn fusion_threshold_12() -> Circuit {
    builtins::hardware_efficient_ansatz(12, 3, SEED)
}

fn fusion_threshold_13() -> Circuit {
    builtins::random_circuit(13, 10, SEED)
}

fn fusion_threshold_14() -> Circuit {
    builtins::random_circuit(14, 10, SEED)
}

fn fusion_threshold_15() -> Circuit {
    builtins::qft_circuit(15)
}

fn fusion_threshold_16() -> Circuit {
    builtins::qft_circuit(16)
}

fn fusion_threshold_17() -> Circuit {
    builtins::qft_circuit(17)
}

fn fusion_threshold_18() -> Circuit {
    builtins::qft_circuit(18)
}

fn fusion_threshold_19() -> Circuit {
    builtins::qft_circuit(19)
}
