//! Dense export cap on the queries priced against it: reduced density matrices
//! and Schmidt spectra. One binary because each pins `PRISM_MAX_EXPORT_QUBITS`
//! to 4, and the cap is cached per process.

mod common;

use common::{SEED, caps};
use prism_q::PrismError;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits::ghz_circuit;
use prism_q::gates::Gate;
use prism_q::sim;

fn small_export_cap() {
    caps::set_once(&[("PRISM_MAX_EXPORT_QUBITS", "4")]);
}

fn assert_answers_at_the_cap_and_declines_past_it(backend: &mut dyn Backend, circuit: &Circuit) {
    sim::run_on(backend, circuit).unwrap();
    let name = backend.name();
    let rho = backend.reduced_density_matrix(&[0, 3]).unwrap();
    assert_eq!(rho.len(), 16, "{name}");
    match backend.reduced_density_matrix(&[0, 3, 1]).unwrap_err() {
        PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, name);
            assert!(
                reason.starts_with(
                    "reduced density matrix on 3 qubits, which is the size of a statevector \
                     for 6 qubits"
                ),
                "{name}: {reason}"
            );
        }
        other => panic!("{name}: unexpected error {other:?}"),
    }
}

// A reduced density matrix on `k` qubits holds `4^k` entries, the bytes of a
// `2k`-qubit statevector, and is priced against the dense export cap on every
// backend that answers.
#[test]
fn a_matrix_declines_when_its_entries_price_past_the_export_cap() {
    small_export_cap();

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

fn assert_ghz_spectrum(values: &[f64]) {
    let half = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(values.len(), 2, "{values:?}");
    assert!((values[0] - half).abs() < 1e-12 && (values[1] - half).abs() < 1e-12);
}

fn assert_declines_past_the_cap(backend: &mut dyn Backend, subsystem: &[usize], route: &str) {
    let name = backend.name();
    match backend.schmidt_values(subsystem).unwrap_err() {
        PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, name);
            assert!(reason.starts_with(route), "{reason}");
        }
        other => panic!("unexpected error {other:?}"),
    }
}

// A Schmidt spectrum is priced by what it allocates: the dense route holds a
// gathered copy of the state and the thin factor, two vectors of the state's
// length, and the reduced density matrix the MPS takes for a cut that is not
// contiguous in chain order holds `4^side` entries.
#[test]
fn a_spectrum_declines_when_its_transient_prices_past_the_export_cap() {
    small_export_cap();

    // Three qubits price as a 4-qubit statevector, at the cap; four price as
    // five, past it, whichever side is named.
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &ghz_circuit(3)).unwrap();
    assert_ghz_spectrum(&sv.schmidt_values(&[0]).unwrap());
    assert_ghz_spectrum(&sv.schmidt_values(&[1, 2]).unwrap());
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &ghz_circuit(4)).unwrap();
    assert_declines_past_the_cap(
        &mut sv,
        &[0],
        "Schmidt values across a cut, whose gathered copy and thin factor together are the \
         size of a statevector for 5 qubits",
    );

    // Two qubits on the smaller side price as a 4-qubit statevector, at the
    // cap; three price as six, past it.
    let mut mps = MpsBackend::new(SEED, 64);
    sim::run_on(&mut mps, &ghz_circuit(8)).unwrap();
    assert_ghz_spectrum(&mps.schmidt_values(&[0, 2]).unwrap());
    assert_declines_past_the_cap(
        &mut mps,
        &[0, 2, 4],
        "Schmidt values across a cut that is not contiguous in chain order with 3 qubits on \
         the smaller side",
    );
    // A run of sites at either end is one SVD at the cut, whatever its width.
    assert_ghz_spectrum(&mps.schmidt_values(&[0, 1, 2, 3, 4]).unwrap());
}
