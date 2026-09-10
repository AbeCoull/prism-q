//! A reduced density matrix on `k` qubits holds `4^k` entries, the bytes of a
//! `2k`-qubit statevector, and is priced against the dense export cap on
//! every backend that answers. Isolated in its own test binary: it overrides
//! `PRISM_MAX_EXPORT_QUBITS`, which is cached per process.

mod common;

use common::SEED;
use prism_q::PrismError;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits::ghz_circuit;
use prism_q::gates::Gate;
use prism_q::sim;

fn assert_answers_at_the_cap_and_declines_past_it(backend: &mut dyn Backend, circuit: &Circuit) {
    sim::run_on(backend, circuit).unwrap();
    let name = backend.name();
    let rho = backend.reduced_density_matrix(&[0, 3]).unwrap();
    assert_eq!(rho.len(), 16, "{name}");
    match backend.reduced_density_matrix(&[0, 3, 1]).unwrap_err() {
        PrismError::BackendUnsupported { backend, operation } => {
            assert_eq!(backend, name);
            assert!(
                operation.starts_with(
                    "reduced density matrix on 3 qubits, which is the size of a statevector \
                     for 6 qubits"
                ),
                "{name}: {operation}"
            );
        }
        other => panic!("{name}: unexpected error {other:?}"),
    }
}

#[test]
fn a_matrix_declines_when_its_entries_price_past_the_export_cap() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_EXPORT_QUBITS", "4") };

    // Two qubits price as a 4-qubit statevector, at the cap; three price as
    // six, past it.
    let ghz = ghz_circuit(5);
    assert_answers_at_the_cap_and_declines_past_it(&mut StatevectorBackend::new(SEED), &ghz);
    assert_answers_at_the_cap_and_declines_past_it(&mut SparseBackend::new(SEED), &ghz);
    assert_answers_at_the_cap_and_declines_past_it(&mut FactoredBackend::new(SEED), &ghz);
    assert_answers_at_the_cap_and_declines_past_it(&mut DensityMatrixBackend::new(SEED), &ghz);

    let mut rotations = Circuit::new(5, 0);
    for q in 0..5 {
        rotations.add_gate(Gate::Ry(0.3 * q as f64 + 0.1), &[q]);
    }
    assert_answers_at_the_cap_and_declines_past_it(&mut ProductStateBackend::new(SEED), &rotations);
}
