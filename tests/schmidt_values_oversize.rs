//! A Schmidt spectrum is priced by what it allocates: the dense route holds a
//! gathered copy of the state and the thin factor, two vectors of the state's
//! length, and the reduced density matrix the MPS takes for a cut that is not
//! contiguous in chain order holds `4^side` entries. Both are checked against
//! the dense export cap. Isolated in its own test binary: it overrides
//! `PRISM_MAX_EXPORT_QUBITS`, which is cached per process.

mod common;

use common::SEED;
use prism_q::PrismError;
use prism_q::backend::Backend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuits::ghz_circuit;
use prism_q::sim;

fn assert_ghz_spectrum(values: &[f64]) {
    let half = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(values.len(), 2, "{values:?}");
    assert!((values[0] - half).abs() < 1e-12 && (values[1] - half).abs() < 1e-12);
}

fn assert_declines_past_the_cap(backend: &mut dyn Backend, subsystem: &[usize], route: &str) {
    let name = backend.name();
    match backend.schmidt_values(subsystem).unwrap_err() {
        PrismError::BackendUnsupported { backend, operation } => {
            assert_eq!(backend, name);
            assert!(operation.starts_with(route), "{operation}");
        }
        other => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn a_spectrum_declines_when_its_transient_prices_past_the_export_cap() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_EXPORT_QUBITS", "4") };

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
