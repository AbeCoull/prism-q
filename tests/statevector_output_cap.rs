//! Dense output caps on the statevector backend. Isolated in its own test
//! binary: it overrides `PRISM_MAX_PROB_QUBITS` and `PRISM_MAX_EXPORT_QUBITS`,
//! which the cap helpers cache per process. Every rejection is decided before
//! the output allocates, so no test builds an oversize vector.

mod common;

use common::{SEED, caps};
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuits;
use prism_q::{BackendKind, PrismError, simulate};

const CAP: usize = 4;

fn small_caps() {
    caps::set_once(&[
        ("PRISM_MAX_PROB_QUBITS", "4"),
        ("PRISM_MAX_EXPORT_QUBITS", "4"),
    ]);
}

fn incompatible_reason(err: PrismError) -> String {
    caps::incompatible_reason(err, "statevector")
}

// The dispatch layer reads `BackendUnsupported` from a probability query as
// "no dense terminal" and reports `None`; the ceiling must not travel that way.
#[test]
fn explicit_run_one_qubit_above_the_probability_cap_names_it() {
    small_caps();
    let n = CAP + 1;
    let err = simulate(&circuits::ghz_circuit(n))
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap_err();
    let reason = incompatible_reason(err);
    assert!(
        reason.contains(&format!("{n} qubits"))
            && reason.contains(&format!("cap of {CAP}"))
            && reason.contains("PRISM_MAX_PROB_QUBITS"),
        "{reason}"
    );

    let at_cap = simulate(&circuits::ghz_circuit(CAP))
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap();
    assert!(at_cap.probabilities.is_some());
}

#[test]
fn export_one_qubit_above_the_export_cap_names_it() {
    small_caps();
    let n = CAP + 1;
    let circuit = circuits::ghz_circuit(n);
    let mut sv = StatevectorBackend::new(SEED);
    sv.init(n, 0).unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let reason = incompatible_reason(sv.export_statevector().unwrap_err());
    assert!(
        reason.contains("statevector export")
            && reason.contains(&format!("{n} qubits"))
            && reason.contains(&format!("cap of {CAP}"))
            && reason.contains("PRISM_MAX_EXPORT_QUBITS"),
        "{reason}"
    );

    let mut at_cap = StatevectorBackend::new(SEED);
    at_cap.init(CAP, 0).unwrap();
    assert_eq!(at_cap.export_statevector().unwrap().len(), 1 << CAP);
}
