//! Cross-backend conformance for the `Backend::reset` contract.
//!
//! `reset` is the channel `rho -> |0><0| (x) tr_q rho`, not a projection onto
//! `|0>`. The two agree whenever the reset qubit is unentangled, so a separable
//! fixture cannot tell them apart. The cases below add an entangled partner,
//! where projection leaks the `|0>`-outcome branch into the rest of the
//! register, and a reset from `|1>`, where a projection has no branch to
//! renormalize.
//!
//! The oracle is the density-matrix backend, which holds the mixture and
//! applies the channel directly. It is deliberately not the statevector
//! backend: the sv-reference matrices in `tests/backend_matrix.rs` and
//! `tests/measurement_matrix.rs` cannot pin this contract, because the
//! statevector is one of the implementations under test. Pure-state backends
//! carry one trajectory per run, so a case whose branches differ is compared as
//! an average over seeds.

mod common;

use common::{SEED, assert_probs_close};
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim;

const TRIALS: usize = 2_000;
/// Band on a seed-averaged probability at `TRIALS` samples. The tightest
/// entry is a fair coin, whose standard error here is 0.011.
const AVERAGED_EPS: f64 = 0.05;
const EXACT_EPS: f64 = 1e-9;

struct BackendEntry {
    name: &'static str,
    construct: fn(u64) -> Box<dyn Backend>,
    /// Whether the representation can hold an entangled two-qubit state.
    entangling: bool,
}

const BACKENDS: &[BackendEntry] = &[
    BackendEntry {
        name: "statevector",
        construct: |s| Box::new(StatevectorBackend::new(s)),
        entangling: true,
    },
    BackendEntry {
        name: "sparse",
        construct: |s| Box::new(SparseBackend::new(s)),
        entangling: true,
    },
    BackendEntry {
        name: "mps",
        construct: |s| Box::new(MpsBackend::new(s, 64)),
        entangling: true,
    },
    BackendEntry {
        name: "factored",
        construct: |s| Box::new(FactoredBackend::new(s)),
        entangling: true,
    },
    BackendEntry {
        name: "tensornetwork",
        construct: |s| Box::new(TensorNetworkBackend::new(s)),
        entangling: true,
    },
    BackendEntry {
        name: "stabilizer",
        construct: |s| Box::new(StabilizerBackend::new(s)),
        entangling: true,
    },
    BackendEntry {
        name: "product",
        construct: |s| Box::new(ProductStateBackend::new(s)),
        entangling: false,
    },
    BackendEntry {
        name: "density_matrix",
        construct: |s| Box::new(DensityMatrixBackend::new(s)),
        entangling: true,
    },
];

/// How a case's branches behave, which decides how it is compared.
enum Branches {
    /// Every branch lands on the same state, so one run is already exact.
    Collapse,
    /// The branches differ, so only the seed-averaged run is the channel. The
    /// density-matrix backend still matches exactly, holding the mixture.
    Differ,
}

struct ResetCase {
    name: &'static str,
    build: fn() -> Circuit,
    expected: [f64; 4],
    branches: Branches,
    /// Whether the fixture entangles, which excludes product-state backends.
    entangles: bool,
}

/// Bell pair, then reset the partner. The channel traces qubit 1 out and
/// leaves qubit 0 maximally mixed; a projection onto |0> would also collapse
/// qubit 0 and put all the weight on |00>.
fn entangled_reset() -> Circuit {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_reset(1);
    circuit
}

/// Reset a qubit holding |1>. A projection onto |0> has no surviving branch,
/// and the fallback for that case used to reinitialize the whole register,
/// destroying qubit 0's superposition.
fn reset_from_one() -> Circuit {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(1);
    circuit
}

/// Reset an unentangled qubit in superposition. Both branches land on |0>
/// without touching qubit 0, so projection and channel agree here.
fn separable_superposition_reset() -> Circuit {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_reset(1);
    circuit
}

const CASES: &[ResetCase] = &[
    ResetCase {
        name: "entangled_partner",
        build: entangled_reset,
        expected: [0.5, 0.5, 0.0, 0.0],
        branches: Branches::Differ,
        entangles: true,
    },
    ResetCase {
        name: "from_one",
        build: reset_from_one,
        expected: [0.5, 0.5, 0.0, 0.0],
        branches: Branches::Collapse,
        entangles: false,
    },
    ResetCase {
        name: "separable_superposition",
        build: separable_superposition_reset,
        expected: [0.5, 0.5, 0.0, 0.0],
        branches: Branches::Collapse,
        entangles: false,
    },
];

fn probabilities(backend: &mut dyn Backend, circuit: &Circuit) -> Vec<f64> {
    sim::run_on(backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

/// Mean probability vector over `TRIALS` independently seeded runs. Each run
/// is one trajectory of the channel; the mean converges to the channel.
fn averaged_probabilities(entry: &BackendEntry, circuit: &Circuit) -> Vec<f64> {
    let mut totals = vec![0.0f64; 1 << circuit.num_qubits];
    for trial in 0..TRIALS {
        let mut backend = (entry.construct)(SEED + trial as u64);
        for (slot, p) in probabilities(backend.as_mut(), circuit).iter().enumerate() {
            totals[slot] += p;
        }
    }
    totals.iter().map(|t| t / TRIALS as f64).collect()
}

fn find_case(name: &str) -> &'static ResetCase {
    CASES
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("unknown reset case `{name}`"))
}

fn assert_reset_case(case: &ResetCase) {
    let circuit = (case.build)();

    // The oracle is exact on every case: it holds the mixture directly.
    let mut oracle = DensityMatrixBackend::new(SEED);
    assert_probs_close(
        &probabilities(&mut oracle, &circuit),
        &case.expected,
        1e-12,
        &format!("{}/density_matrix oracle", case.name),
    );

    for entry in BACKENDS {
        if case.entangles && !entry.entangling {
            continue;
        }
        let label = format!("{}/{}", case.name, entry.name);
        match case.branches {
            Branches::Collapse => {
                // Forced or convergent branches, so each seed is exact.
                for trial in 0..8u64 {
                    let mut backend = (entry.construct)(SEED + trial);
                    assert_probs_close(
                        &probabilities(backend.as_mut(), &circuit),
                        &case.expected,
                        EXACT_EPS,
                        &format!("{label} seed {trial}"),
                    );
                }
            }
            Branches::Differ if entry.name == "density_matrix" => {
                let mut backend = (entry.construct)(SEED);
                assert_probs_close(
                    &probabilities(backend.as_mut(), &circuit),
                    &case.expected,
                    1e-12,
                    &label,
                );
            }
            Branches::Differ => {
                assert_probs_close(
                    &averaged_probabilities(entry, &circuit),
                    &case.expected,
                    AVERAGED_EPS,
                    &format!("{label} seed-averaged"),
                );
            }
        }
    }
}

#[test]
fn reset_of_an_entangled_partner_leaves_it_mixed() {
    assert_reset_case(find_case("entangled_partner"));
}

#[test]
fn reset_from_one_preserves_the_rest_of_the_register() {
    assert_reset_case(find_case("from_one"));
}

#[test]
fn reset_of_a_separable_superposition_is_deterministic() {
    assert_reset_case(find_case("separable_superposition"));
}
