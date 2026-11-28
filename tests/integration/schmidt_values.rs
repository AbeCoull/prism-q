//! Declines and argument checks for `Backend::schmidt_values` and
//! `Backend::entanglement_entropy`, and agreement of the `Simulate` terminal
//! over them with the statevector. The values themselves are pinned in
//! `golden_small_circuits.rs`, and the MPS routes in its own unit tests.

use crate::common;

use common::circuits::{CircuitCase, exact_small_cases, product_separable_cases};
use common::{FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SV_EPS};
use prism_q::PrismError;
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
use prism_q::circuit::Circuit;
use prism_q::circuits::{brickwork_circuit, ghz_circuit};
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::{BackendKind, simulate};

fn assert_declines(backend: &mut dyn Backend, operation: &str) {
    sim::run_on(backend, &ghz_circuit(4)).unwrap();
    let expected = PrismError::BackendUnsupported {
        backend: backend.name().to_string(),
        operation: operation.to_string(),
    };
    assert_eq!(backend.schmidt_values(&[0, 1]).unwrap_err(), expected);
    assert_eq!(backend.entanglement_entropy(&[0, 1]).unwrap_err(), expected);
}

// A stabilizer cut of rank r has 2^r equal weights, so both tableau backends
// read the whole spectrum off that rank: a GHZ cut is one ebit, two values of
// 1 / sqrt 2, and `ln 2`.
#[test]
fn the_tableau_backends_read_the_flat_spectrum_from_the_rank() {
    let backends: [Box<dyn Backend>; 2] = [
        Box::new(StabilizerBackend::new(SEED)),
        Box::new(FactoredStabilizerBackend::new(SEED)),
    ];
    let half = std::f64::consts::FRAC_1_SQRT_2;
    for mut backend in backends {
        sim::run_on(backend.as_mut(), &ghz_circuit(4)).unwrap();
        let name = backend.name();
        let values = backend.schmidt_values(&[0, 1]).unwrap();
        assert_eq!(values.len(), 2, "{name}: {values:?}");
        for value in &values {
            assert!((value - half).abs() < 1e-12, "{name}: {values:?}");
        }
        let entropy = backend.entanglement_entropy(&[0, 1]).unwrap();
        assert!(
            (entropy - std::f64::consts::LN_2).abs() < 1e-12,
            "{name}: entropy {entropy}"
        );
    }
}

#[test]
fn density_matrix_declines_schmidt_values_of_a_mixed_state() {
    assert_declines(
        &mut DensityMatrixBackend::new(SEED),
        "Schmidt values of a mixed state",
    );
}

#[test]
fn sparse_and_tensor_network_schmidt_values_match_statevector() {
    let circuit = brickwork_circuit(6, 3, SEED);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();

    let mut sparse = SparseBackend::new(SEED);
    sim::run_on(&mut sparse, &circuit).unwrap();

    let mut tn = TensorNetworkBackend::new(SEED);
    sim::run_on(&mut tn, &circuit).unwrap();

    for subsystem in [vec![0usize, 1], vec![0, 2, 4], vec![1, 3, 4, 5]] {
        let want_s = sv.schmidt_values(&subsystem).unwrap();
        let want_h = sv.entanglement_entropy(&subsystem).unwrap();
        for (name, backend) in [
            ("sparse", &mut sparse as &mut dyn Backend),
            ("tensornetwork", &mut tn as &mut dyn Backend),
        ] {
            let got_s = backend.schmidt_values(&subsystem).unwrap();
            assert_eq!(got_s.len(), want_s.len(), "{name} {subsystem:?}");
            for (g, w) in got_s.iter().zip(&want_s) {
                assert!((g - w).abs() < 1e-6, "{name} {subsystem:?}: {g} vs {w}");
            }
            let got_h = backend.entanglement_entropy(&subsystem).unwrap();
            assert!(
                (got_h - want_h).abs() < 1e-10,
                "{name} {subsystem:?} entropy: {got_h} vs {want_h}"
            );
        }
    }
}

#[cfg(feature = "distributed")]
#[test]
fn distributed_statevector_schmidt_values_match_statevector() {
    use prism_q::distributed::DistributedContext;

    let circuit = brickwork_circuit(6, 3, SEED);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    let want_h = sv.entanglement_entropy(&[0, 2, 4]).unwrap();

    let ctx = DistributedContext::serial();
    let res = simulate(&circuit)
        .backend(BackendKind::StatevectorDistributed { context: ctx })
        .seed(SEED)
        .entanglement_entropy(&[0, 2, 4])
        .unwrap();
    assert!(
        (res.entropy - want_h).abs() < 1e-10,
        "{} vs {want_h}",
        res.entropy
    );
}

// The same four rejections on every backend that answers, on a circuit all
// four can run.
#[test]
fn a_subsystem_is_a_proper_non_empty_set_of_distinct_qubits() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::Ry(0.5 * q as f64 + 0.2), &[q]);
    }
    let backends: [Box<dyn Backend>; 4] = [
        Box::new(StatevectorBackend::new(SEED)),
        Box::new(MpsBackend::new(SEED, 64)),
        Box::new(ProductStateBackend::new(SEED)),
        Box::new(FactoredBackend::new(SEED)),
    ];
    for mut backend in backends {
        sim::run_on(backend.as_mut(), &circuit).unwrap();
        let name = backend.name();
        assert!(
            matches!(
                backend.schmidt_values(&[]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took an empty subsystem"
        );
        assert!(
            matches!(
                backend.entanglement_entropy(&[0, 1, 2, 3]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took the whole register"
        );
        assert!(
            matches!(
                backend.schmidt_values(&[1, 1]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took a repeated qubit"
        );
        assert_eq!(
            backend.schmidt_values(&[0, 4]).unwrap_err(),
            PrismError::InvalidQubit {
                index: 4,
                register_size: 4
            },
            "{name} took an out-of-range qubit"
        );
        assert!(
            backend.entanglement_entropy(&[3, 0]).is_ok(),
            "{name} rejected a valid cut"
        );
    }
}

/// The terminal's entropy for one explicit backend kind, with the spectrum it
/// carries checked for the ordering and normalization the contract states.
fn terminal_entropy(kind: BackendKind, circuit: &Circuit, subsystem: &[usize]) -> f64 {
    let result = simulate(circuit)
        .backend(kind)
        .seed(SEED)
        .entanglement_entropy(subsystem)
        .unwrap();
    assert_eq!(
        result.subsystem, subsystem,
        "the terminal echoed a different subsystem"
    );
    let values = result
        .schmidt_values
        .as_deref()
        .expect("a backend answering from a spectrum returns it");
    assert!(
        values.windows(2).all(|pair| pair[0] >= pair[1]),
        "Schmidt values out of order: {values:?}"
    );
    let total: f64 = values.iter().map(|s| s * s).sum();
    assert!((total - 1.0).abs() < 1e-12, "squares sum to {total}");
    result.entropy
}

/// The terminal on each backend it can route to, against the statevector
/// kernel, at every cut of every case the backend accepts.
fn assert_terminal_matches_statevector(rows: &[(BackendKind, f64)], cases: &[CircuitCase]) {
    for case in cases {
        let circuit = case.circuit();
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        for cut in 1..circuit.num_qubits {
            let subsystem: Vec<usize> = (0..cut).collect();
            let expected = sv.entanglement_entropy(&subsystem).unwrap();
            for (kind, eps) in rows {
                let entropy = terminal_entropy(kind.clone(), &circuit, &subsystem);
                assert!(
                    (entropy - expected).abs() < *eps,
                    "{kind:?} {} at cut {cut}: {entropy} against {expected}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn the_terminal_matches_the_statevector_on_the_small_corpus() {
    assert_terminal_matches_statevector(
        &[
            (BackendKind::Statevector, SV_EPS),
            (BackendKind::Mps { max_bond_dim: 64 }, MPS_EPS),
            (BackendKind::Factored, FACTORED_EPS),
        ],
        &exact_small_cases(),
    );
}

#[test]
fn the_terminal_matches_the_statevector_on_the_separable_corpus() {
    assert_terminal_matches_statevector(
        &[(BackendKind::ProductState, PRODUCT_EPS)],
        &product_separable_cases(),
    );
}

// A backend with no spectrum and no entropy of its own is named by the
// terminal, not by the route that selected it.
#[test]
fn the_terminal_names_the_backend_that_declines() {
    let circuit = ghz_circuit(4);
    assert_eq!(
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .seed(SEED)
            .entanglement_entropy(&[0, 1])
            .unwrap_err(),
        PrismError::BackendUnsupported {
            backend: "density_matrix".to_string(),
            operation: "Schmidt values of a mixed state".to_string(),
        }
    );
}

// A bond cap of 2 truncates a brick-wall state, so the chain is unnormalized
// until it is read. The spectrum is still normalized on the way out and the
// result says the route discarded weight.
#[test]
fn the_terminal_normalizes_a_truncating_chain_and_says_so() {
    let circuit = brickwork_circuit(8, 6, SEED);
    let result = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 2 })
        .seed(SEED)
        .entanglement_entropy(&[0, 1, 2, 3])
        .unwrap();
    let values = result.schmidt_values.as_deref().unwrap();
    let total: f64 = values.iter().map(|s| s * s).sum();
    assert!((total - 1.0).abs() < 1e-12, "squares sum to {total}");
    assert!(
        !result.metadata.is_exact(),
        "a chain truncated at bond 2 reported {:?}",
        result.metadata.exactness
    );
}

// Non-contiguous 4-qubit subsystem on an 8-qubit entangled state builds a 16x16
// reduced density matrix (`dim * dim = 256`), exercising `faer_hermitian_eigenvalues`
// under `parallel`, while contiguous prefix/suffix cuts exercise `svd_singular_values_with_workspace`.
#[test]
fn mps_schmidt_values_match_statevector_on_large_density_and_bond_cuts() {
    let circuit = brickwork_circuit(8, 6, SEED);

    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();

    let mut mps = MpsBackend::new(SEED, 128);
    sim::run_on(&mut mps, &circuit).unwrap();

    for subsystem in [
        vec![0usize, 1, 2],
        vec![5, 6, 7],
        vec![0, 2, 4, 6],
        vec![1, 3, 5, 7],
    ] {
        let expected = sv.schmidt_values(&subsystem).unwrap();
        let actual = mps.schmidt_values(&subsystem).unwrap();
        assert_eq!(
            actual.len(),
            expected.len(),
            "cut {subsystem:?}: len {} vs {}",
            actual.len(),
            expected.len()
        );
        for (k, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < MPS_EPS,
                "cut {subsystem:?} s[{k}]: {got} vs {want}"
            );
        }
        let s_mps = mps.entanglement_entropy(&subsystem).unwrap();
        let s_sv = sv.entanglement_entropy(&subsystem).unwrap();
        assert!(
            (s_mps - s_sv).abs() < MPS_EPS,
            "cut {subsystem:?} entropy: {s_mps} vs {s_sv}"
        );
    }
}

#[test]
fn mps_overlap_aligns_permuted_layouts_past_export_cap() {
    use prism_q::Limits;

    let n = 40;
    let mut c1 = Circuit::new(n, 0);
    let mut c2 = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c1.add_gate(Gate::H, &[q]);
        c2.add_gate(Gate::H, &[q]);
    }
    // Non-adjacent CX leaves virtual SWAPs in different orders on c1 vs c2 while
    // producing the exact same state (`Cx(5, 0)` twice cancels on c2, leaving
    // qubits 0 and 5 virtually swapped relative to c1).
    c1.add_gate(Gate::Cx, &[0, 5]);
    c1.add_gate(Gate::Cx, &[10, 17]);
    c2.add_gate(Gate::Cx, &[10, 17]);
    c2.add_gate(Gate::Cx, &[0, 5]);
    c2.add_gate(Gate::Cx, &[2, 9]);
    c2.add_gate(Gate::Cx, &[2, 9]);

    let limits = Limits::new().max_export_qubits(8);
    let res = simulate(&c1)
        .backend(BackendKind::Mps { max_bond_dim: 32 })
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c2)
                .backend(BackendKind::Mps { max_bond_dim: 32 })
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    assert!(
        (res.fidelity - 1.0).abs() < 1e-10,
        "40q permuted MPS overlap: {}",
        res.fidelity
    );

    // A non-trivial overlap (`0.25`) across different layouts: SWAP(0, 5) maps
    // `|+>|0>` to `|0>|+>`, whose squared overlap with `(|00> + |11>)/sqrt(2)` is 1/4.
    let mut c3 = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c3.add_gate(Gate::H, &[q]);
    }
    c3.add_gate(Gate::Cx, &[5, 0]);
    c3.add_gate(Gate::Cx, &[0, 5]);
    c3.add_gate(Gate::Cx, &[5, 0]);
    c3.add_gate(Gate::Cx, &[10, 17]);
    let res2 = simulate(&c1)
        .backend(BackendKind::Mps { max_bond_dim: 32 })
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c3)
                .backend(BackendKind::Mps { max_bond_dim: 32 })
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    assert!(
        (res2.fidelity - 0.25).abs() < 1e-10,
        "40q permuted MPS overlap (0.25): {}",
        res2.fidelity
    );
}

#[test]
fn factored_stabilizer_overlap_works_at_500_qubits() {
    use prism_q::Limits;

    let n = 500;
    let mut c1 = Circuit::new(n, 0);
    let mut c2 = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c1.add_gate(Gate::H, &[q]);
        c1.add_gate(Gate::Cx, &[q, q + 1]);
    }
    // Shift pairing on c2 so clusters straddle c1's clusters into components of size 4.
    for q in (0..n).step_by(4) {
        c2.add_gate(Gate::H, &[q]);
        c2.add_gate(Gate::Cx, &[q, q + 2]);
        c2.add_gate(Gate::H, &[q + 1]);
        c2.add_gate(Gate::Cx, &[q + 1, q + 3]);
    }

    let limits = Limits::new().max_export_qubits(8);
    let ff = simulate(&c1)
        .backend(BackendKind::FactoredStabilizer)
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c2)
                .backend(BackendKind::FactoredStabilizer)
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    let ss = simulate(&c1)
        .backend(BackendKind::Stabilizer)
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c2)
                .backend(BackendKind::Stabilizer)
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    let fs = simulate(&c1)
        .backend(BackendKind::FactoredStabilizer)
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c2)
                .backend(BackendKind::Stabilizer)
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    assert_eq!(ff.fidelity, ss.fidelity);
    assert_eq!(fs.fidelity, ss.fidelity);

    let self_ov = simulate(&c1)
        .backend(BackendKind::FactoredStabilizer)
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c1)
                .backend(BackendKind::FactoredStabilizer)
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();
    assert_eq!(self_ov.fidelity, 1.0);
}

#[test]
fn factored_overlap_works_past_export_cap_on_independent_blocks() {
    use prism_q::Limits;

    let n = 24;
    let mut c1 = Circuit::new(n, 0);
    let mut c2 = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c1.add_gate(Gate::Rx(0.3 + 0.01 * q as f64), &[q]);
        c1.add_gate(Gate::Cx, &[q, q + 1]);
        c2.add_gate(Gate::Rx(0.3 + 0.01 * q as f64), &[q]);
        c2.add_gate(Gate::Cx, &[q, q + 1]);
    }
    // Cross-pair only on qubits (0, 2) and (1, 3) to test union connected components.
    c2.add_gate(Gate::Rz(0.4), &[0]);
    c2.add_gate(Gate::Cx, &[1, 2]);

    let limits = Limits::new().max_export_qubits(6);
    let res = simulate(&c1)
        .backend(BackendKind::Factored)
        .limits(limits)
        .seed(SEED)
        .overlap(
            simulate(&c2)
                .backend(BackendKind::Factored)
                .limits(limits)
                .seed(SEED),
        )
        .unwrap();

    // Check against a 4-qubit statevector reference on the first 4 qubits (rest match identically).
    let mut r1 = Circuit::new(4, 0);
    let mut r2 = Circuit::new(4, 0);
    for q in (0..4).step_by(2) {
        r1.add_gate(Gate::Rx(0.3 + 0.01 * q as f64), &[q]);
        r1.add_gate(Gate::Cx, &[q, q + 1]);
        r2.add_gate(Gate::Rx(0.3 + 0.01 * q as f64), &[q]);
        r2.add_gate(Gate::Cx, &[q, q + 1]);
    }
    r2.add_gate(Gate::Rz(0.4), &[0]);
    r2.add_gate(Gate::Cx, &[1, 2]);
    let ref_ov = simulate(&r1)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .overlap(simulate(&r2).backend(BackendKind::Statevector).seed(SEED))
        .unwrap();
    assert!(
        (res.fidelity - ref_ov.fidelity).abs() < 1e-12,
        "factored overlap {} vs ref {}",
        res.fidelity,
        ref_ov.fidelity
    );
}

#[test]
fn mps_overlap_cross_layout_exact_with_tight_bond_and_min_fidelity() {
    use prism_q::Limits;

    // Two 4q Bell-pair states with crossing layouts: each is exact at bond 2,
    // while aligning one layout onto the other temporarily crosses bond 4.
    let mut a = Circuit::new(4, 0);
    a.add_gate(Gate::H, &[0]);
    a.add_gate(Gate::Cx, &[0, 1]);
    a.add_gate(Gate::H, &[2]);
    a.add_gate(Gate::Cx, &[2, 3]);

    let mut b = Circuit::new(4, 0);
    b.add_gate(Gate::H, &[0]);
    b.add_gate(Gate::Cx, &[0, 2]);
    b.add_gate(Gate::H, &[1]);
    b.add_gate(Gate::Cx, &[1, 3]);

    let limits = Limits::new().min_fidelity_estimate(1.0);
    let mps2 = BackendKind::Mps { max_bond_dim: 2 };
    let ab = simulate(&a)
        .backend(mps2.clone())
        .limits(limits)
        .seed(SEED)
        .overlap(simulate(&b).backend(mps2.clone()).limits(limits).seed(SEED))
        .unwrap();
    let ba = simulate(&b)
        .backend(mps2.clone())
        .limits(limits)
        .seed(SEED)
        .overlap(simulate(&a).backend(mps2).limits(limits).seed(SEED))
        .unwrap();
    let sv = simulate(&a)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .overlap(simulate(&b).backend(BackendKind::Statevector).seed(SEED))
        .unwrap();
    assert!((ab.fidelity - sv.fidelity).abs() < 1e-12);
    assert!((ba.fidelity - sv.fidelity).abs() < 1e-12);
    assert!((ab.fidelity - ba.fidelity).abs() < 1e-12);
}
