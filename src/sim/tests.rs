use super::dispatch::min_clifford_prefix_gates;
use super::*;
use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::SparseBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::circuit::smallvec;
use crate::gates::Gate;

fn make_clifford_circuit() -> Circuit {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);
    c
}

// The probability-route planner is the single source of the auto routing
// precedence; every entry point (run, shots, counts via the terminal
// candidate) consults it. Pin the route for representative circuits so a
// precedence change is a deliberate edit here, not silent drift.
#[test]
fn probability_route_precedence_is_pinned() {
    let ghz = make_clifford_circuit();
    assert!(matches!(
        plan_probability_route(&BackendKind::Auto, &ghz),
        ProbabilityRoute::Direct { .. }
    ));

    let mut clifford_t = Circuit::new(16, 0);
    clifford_t.add_gate(Gate::H, &[0]);
    for i in 0..15 {
        clifford_t.add_gate(Gate::Cx, &[i, i + 1]);
    }
    clifford_t.add_gate(Gate::T, &[1]);
    assert!(matches!(
        plan_probability_route(&BackendKind::Auto, &clifford_t),
        ProbabilityRoute::StabilizerRank { t_count: 1 }
    ));

    let mut general = Circuit::new(4, 0);
    general.add_gate(Gate::H, &[0]);
    general.add_gate(Gate::Rx(0.3), &[1]);
    general.add_gate(Gate::Cx, &[0, 1]);
    assert!(matches!(
        plan_probability_route(&BackendKind::Auto, &general),
        ProbabilityRoute::Direct { .. }
    ));

    let candidate_is_direct = |c: &Circuit| {
        matches!(
            plan_probability_route(&BackendKind::Auto, c),
            ProbabilityRoute::Direct { .. }
        )
    };
    for circuit in [&ghz, &clifford_t, &general] {
        if auto_terminal_statevector_candidate(circuit) {
            assert!(candidate_is_direct(circuit));
        }
    }
}

// The decomposition bypass in `try_native_terminal_backend` is scoped to
// the product state. Pin both halves: a product circuit reaches the native
// sampler even though its route is decomposed, and a decomposable circuit
// on any other backend keeps the block split it had before.
#[test]
fn only_the_product_state_takes_the_native_sampler_past_decomposition() {
    let mut product = Circuit::new(12, 0);
    for q in 0..12 {
        product.add_gate(Gate::Ry(0.3 + 0.1 * q as f64), &[q]);
    }
    assert!(matches!(
        plan_probability_route(&BackendKind::Auto, &product),
        ProbabilityRoute::Decomposed(_)
    ));
    for kind in [BackendKind::Auto, BackendKind::ProductState] {
        let backend = try_native_terminal_backend(&kind, &product, 42).unwrap();
        assert_eq!(
            backend.map(|b| b.name()),
            Some("productstate"),
            "{kind:?} did not reach the product sampler"
        );
    }

    let mut blocks = Circuit::new(10, 0);
    for pair in 0..5 {
        blocks.add_gate(Gate::Ry(0.2 + 0.1 * pair as f64), &[2 * pair]);
        blocks.add_gate(Gate::Cx, &[2 * pair, 2 * pair + 1]);
    }
    assert!(matches!(
        plan_probability_route(&BackendKind::Sparse, &blocks),
        ProbabilityRoute::Decomposed(_)
    ));
    for kind in [
        BackendKind::Sparse,
        BackendKind::Factored,
        BackendKind::Mps { max_bond_dim: 32 },
    ] {
        assert!(
            try_native_terminal_backend(&kind, &blocks, 42)
                .unwrap()
                .is_none(),
            "{kind:?} left the decomposed route"
        );
    }
}

fn make_product_circuit() -> Circuit {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(1.0), &[1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Y, &[3]);
    c
}

fn make_general_circuit() -> Circuit {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c
}

#[derive(Debug, Clone, Copy)]
enum ProbabilityFailure {
    Unsupported,
    Invalid,
}

struct ProbabilityFailureBackend {
    failure: ProbabilityFailure,
    classical_bits: Vec<bool>,
    num_qubits: usize,
}

impl ProbabilityFailureBackend {
    fn new(failure: ProbabilityFailure) -> Self {
        Self {
            failure,
            classical_bits: Vec::new(),
            num_qubits: 0,
        }
    }
}

impl Backend for ProbabilityFailureBackend {
    fn name(&self) -> &'static str {
        "probability_failure"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.classical_bits = vec![false; num_classical_bits];
        Ok(())
    }

    fn apply(&mut self, _instruction: &Instruction) -> Result<()> {
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        match self.failure {
            ProbabilityFailure::Unsupported => Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: "probabilities".to_string(),
            }),
            ProbabilityFailure::Invalid => Err(PrismError::InvalidParameter {
                message: "probability extraction failed".to_string(),
            }),
        }
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

fn pauli_marginal_errors(circuit: &Circuit) -> Vec<PrismError> {
    [
        BackendKind::StochasticPauli { num_samples: 100 },
        BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        },
    ]
    .into_iter()
    .map(|backend| {
        simulate(circuit)
            .backend(backend)
            .seed(42)
            .marginals()
            .unwrap_err()
    })
    .collect()
}

#[test]
fn test_circuit_is_clifford_only() {
    assert!(make_clifford_circuit().is_clifford_only());
    assert!(!make_general_circuit().is_clifford_only());
    assert!(!make_product_circuit().is_clifford_only());
}

#[test]
fn test_circuit_has_entangling_gates() {
    assert!(make_clifford_circuit().has_entangling_gates());
    assert!(make_general_circuit().has_entangling_gates());
    assert!(!make_product_circuit().has_entangling_gates());
}

#[test]
fn test_auto_selects_product() {
    let circuit = make_product_circuit();
    let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
    assert_eq!(backend.name(), "productstate");
}

#[test]
fn test_auto_selects_stabilizer() {
    let circuit = make_clifford_circuit();
    let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
    assert_eq!(backend.name(), "stabilizer");
}

#[test]
fn test_auto_selects_statevector() {
    let circuit = make_general_circuit();
    let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
    assert_eq!(backend.name(), "statevector");
}

#[test]
fn test_run_with_auto_matches_explicit() {
    let circuit = make_general_circuit();
    let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
    let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_run_with_explicit_backends() {
    let circuit = make_clifford_circuit();
    assert!(run_with(BackendKind::Statevector, &circuit, 42).is_ok());
    assert!(run_with(BackendKind::Stabilizer, &circuit, 42).is_ok());
    assert!(run_with(BackendKind::Sparse, &circuit, 42).is_ok());
    assert!(run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).is_ok());
}

#[test]
fn test_run_auto_clifford_probs_match_statevector() {
    let circuit = make_clifford_circuit();
    let auto_result = run(&circuit, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let sv_result = run_on(&mut sv, &circuit).unwrap();
    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_run_qasm() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];";
    let result = run_qasm(qasm, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert!((probs[0] - 0.5).abs() < 1e-10);
    assert!((probs[3] - 0.5).abs() < 1e-10);
}

#[test]
fn test_empty_circuit_is_clifford_and_no_entangling() {
    let c = Circuit::new(2, 0);
    assert!(c.is_clifford_only());
    assert!(!c.has_entangling_gates());
}

#[test]
fn test_validate_stabilizer_rejects_non_clifford() {
    let circuit = make_general_circuit(); // has T gate
    let result = run_with(BackendKind::Stabilizer, &circuit, 42);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_validate_product_rejects_entangling() {
    let circuit = make_clifford_circuit(); // has CX
    let result = run_with(BackendKind::ProductState, &circuit, 42);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_validate_passes_for_compatible() {
    let clifford = make_clifford_circuit();
    assert!(run_with(BackendKind::Stabilizer, &clifford, 42).is_ok());

    let product = make_product_circuit();
    assert!(run_with(BackendKind::ProductState, &product, 42).is_ok());
}

#[test]
fn test_auto_moderate_qubit_count_uses_statevector() {
    let mut circuit = Circuit::new(20, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
    assert_eq!(backend.name(), "statevector");
}

#[test]
fn test_auto_selects_factored_with_partial_independence() {
    let mut circuit = Circuit::new(10, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let backend = resolve_backend(&BackendKind::Auto, &circuit, true).build(42);
    assert_eq!(backend.name(), "factored");
}

#[test]
fn test_auto_ignores_partial_independence_when_no_entangling() {
    let circuit = make_product_circuit();
    let backend = resolve_backend(&BackendKind::Auto, &circuit, true).build(42);
    assert_eq!(backend.name(), "productstate");
}

#[test]
fn test_classical_only_skips_probabilities() {
    let qasm =
        "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];";
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_with_internal(
        BackendKind::Statevector,
        &circuit,
        42,
        SimOptions::classical_only(),
    )
    .unwrap();
    assert!(result.probabilities.is_none());
    assert_eq!(result.classical_bits.len(), 1);
}

#[test]
fn test_default_options_include_probabilities() {
    let circuit = make_general_circuit();
    let result = run_with_internal(
        BackendKind::Statevector,
        &circuit,
        42,
        SimOptions::default(),
    )
    .unwrap();
    assert!(result.probabilities.is_some());
}

#[test]
fn test_run_on_always_computes_probabilities() {
    let circuit = make_clifford_circuit();
    let mut backend = StatevectorBackend::new(42);
    let result = run_on(&mut backend, &circuit).unwrap();
    assert!(result.probabilities.is_some());
    let probs = result.probabilities.unwrap().to_vec();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_temporal_clifford_matches_statevector() {
    let mut c = Circuit::new(10, 0);

    // Clifford prefix: 22 gates (above min_clifford_prefix_gates(10)=20)
    for i in 0..10 {
        c.add_gate(Gate::H, &[i]);
    }
    for i in 0..9 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Sdg, &[3]);
    c.add_gate(Gate::SX, &[7]);

    // Non-Clifford tail
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Rx(0.7), &[1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Rz(1.2), &[2]);

    let (prefix, _tail) = c.clifford_prefix_split().unwrap();
    assert!(prefix.gate_count() >= min_clifford_prefix_gates(c.num_qubits));

    let auto_result = run(&c, 42).unwrap();

    // Pure statevector reference (no temporal decomposition)
    let mut sv = StatevectorBackend::new(42);
    let sv_result = run_on(&mut sv, &c).unwrap();

    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    assert_eq!(auto_probs.len(), sv_probs.len());
    for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
        assert!(
            (a - s).abs() < 1e-10,
            "temporal decomp mismatch: auto={a}, sv={s}"
        );
    }
}

#[test]
fn test_temporal_clifford_complex_circuit_matches_sv() {
    // 3q circuit: prefix too short for temporal (min_clifford_prefix_gates(3)=16)
    // but auto must still match statevector.
    let mut c = Circuit::new(3, 0);

    // Clifford prefix with i-phase generators
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Y, &[1]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::SXdg, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[0, 2]);
    c.add_gate(Gate::S, &[1]);

    // Non-Clifford tail
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Ry(0.3), &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);

    let auto_result = run(&c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let sv_result = run_on(&mut sv, &c).unwrap();

    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
        assert!(
            (a - s).abs() < 1e-10,
            "complex temporal mismatch: auto={a}, sv={s}"
        );
    }
}

#[test]
fn test_temporal_clifford_skipped_when_prefix_too_short() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]); // 2 Clifford gates < min_clifford_prefix_gates(2)=16

    let auto_result = run(&c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let sv_result = run_on(&mut sv, &c).unwrap();

    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
        assert!((a - s).abs() < 1e-10);
    }
}

#[test]
fn test_decomposed_random_blocks_matches_monolithic() {
    let circuit = crate::circuits::independent_random_blocks(10, 2, 5, 0xDEAD_BEEF);
    let decomposed = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let monolithic = run_on(&mut sv, &circuit).unwrap();
    let d_probs = decomposed.probabilities.unwrap().to_vec();
    let m_probs = monolithic.probabilities.unwrap().to_vec();
    assert_eq!(d_probs.len(), m_probs.len());
    for (d, m) in d_probs.iter().zip(m_probs.iter()) {
        assert!(
            (d - m).abs() < 1e-10,
            "mismatch: decomposed={d}, monolithic={m}"
        );
    }
}

#[test]
fn test_per_block_clifford_dispatch() {
    let mut c = Circuit::new(6, 0);

    // Block A (Clifford): GHZ state on qubits 0,1,2
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);

    // Block B (non-Clifford): qubits 3,4,5
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Cx, &[3, 4]);
    c.add_gate(Gate::Rx(0.7), &[5]);
    c.add_gate(Gate::Cx, &[4, 5]);

    let components = c.independent_subsystems();
    assert_eq!(components.len(), 2);

    let (sub_a, _, _) = c.extract_subcircuit(&components[0]);
    assert!(sub_a.is_clifford_only());
    let backend_a = resolve_backend(&BackendKind::Auto, &sub_a, false).build(42);
    assert_eq!(backend_a.name(), "stabilizer");

    let (sub_b, _, _) = c.extract_subcircuit(&components[1]);
    assert!(!sub_b.is_clifford_only());
    let backend_b = resolve_backend(&BackendKind::Auto, &sub_b, false).build(43);
    assert_eq!(backend_b.name(), "statevector");

    // End-to-end: auto (decomposed) must match monolithic statevector
    let auto_result = run(&c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let mono_result = run_on(&mut sv, &c).unwrap();
    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let mono_probs = mono_result.probabilities.unwrap().to_vec();
    assert_eq!(auto_probs.len(), mono_probs.len());
    for (a, m) in auto_probs.iter().zip(mono_probs.iter()) {
        assert!((a - m).abs() < 1e-10, "prob mismatch: auto={a}, mono={m}");
    }
}

#[test]
fn test_decomposed_bell_pairs_matches_monolithic() {
    let circuit = crate::circuits::independent_bell_pairs(10);
    let decomposed = run(&circuit, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
    let monolithic = run_on(&mut sv, &circuit).unwrap();
    let d_probs = decomposed.probabilities.unwrap().to_vec();
    let m_probs = monolithic.probabilities.unwrap().to_vec();
    assert_eq!(d_probs.len(), m_probs.len());
    for (d, m) in d_probs.iter().zip(m_probs.iter()) {
        assert!(
            (d - m).abs() < 1e-10,
            "mismatch: decomposed={d}, monolithic={m}"
        );
    }
}

#[test]
fn test_measurement_normalization_statevector() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "statevector post-measurement probs sum to {sum}, expected 1.0"
    );
}

#[test]
fn test_measurement_normalization_mps() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "MPS post-measurement probs sum to {sum}, expected 1.0"
    );
}

#[test]
fn test_measurement_normalization_sparse() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_with(BackendKind::Sparse, &circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "sparse post-measurement probs sum to {sum}, expected 1.0"
    );
}

#[test]
fn test_conditional_gate_execution() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0]) x q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
    // q[0] measured as 1, so if-gate fires, q[1] flipped to |1⟩
    // Final state: |11⟩ = index 3
    let probs = result.probabilities.unwrap().to_vec();
    assert!(
        probs[3] > 0.99,
        "conditional gate should flip q[1]: probs={probs:?}"
    );
    assert!(result.classical_bits[0]);
}

fn make_bell_with_measure() -> Circuit {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    crate::circuit::openqasm::parse(qasm).unwrap()
}

#[test]
fn test_shots_deterministic() {
    let circuit = make_bell_with_measure();
    let a = run_shots(&circuit, 10, 42).unwrap();
    let b = run_shots(&circuit, 10, 42).unwrap();
    assert_eq!(a.shots, b.shots);
}

#[test]
fn test_shots_distribution_convergence() {
    let circuit = make_bell_with_measure();
    let result = run_shots(&circuit, 10000, 42).unwrap();
    let counts = result.counts();
    let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
    let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
    let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
    let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
    assert!(
        (4500..=5500).contains(&n_00),
        "|00> count {n_00} outside [4500, 5500]"
    );
    assert!(
        (4500..=5500).contains(&n_11),
        "|11> count {n_11} outside [4500, 5500]"
    );
    assert_eq!(n_01, 0, "|01> should never appear in Bell state");
    assert_eq!(n_10, 0, "|10> should never appear in Bell state");
}

#[test]
fn test_shots_single_valid_outcome() {
    let circuit = make_bell_with_measure();
    let shots_result = run_shots(&circuit, 1, 42).unwrap();
    let shot = &shots_result.shots[0];
    assert_eq!(shot[0], shot[1], "Bell state: both bits must agree");
}

#[test]
fn test_shots_all_zero() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        c[0] = measure q[0];
        c[1] = measure q[1];
        c[2] = measure q[2];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_shots(&circuit, 100, 42).unwrap();
    for (i, shot) in result.shots.iter().enumerate() {
        assert!(
            shot.iter().all(|&b| !b),
            "shot {i} should be all-zero: {shot:?}"
        );
    }
}

#[test]
fn test_shots_mid_circuit_measurement() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0]) x q[1];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_shots(&circuit, 100, 42).unwrap();
    for (i, shot) in result.shots.iter().enumerate() {
        assert!(shot[0], "shot {i}: q[0] should always be 1");
        assert!(shot[1], "shot {i}: q[1] should always be 1 (conditional)");
    }
}

#[test]
fn test_shots_counts_sum() {
    let circuit = make_bell_with_measure();
    let result = run_shots(&circuit, 500, 42).unwrap();
    let counts = result.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 500);
}

#[test]
fn test_run_counts_factored_stabilizer() {
    let circuit = make_bell_with_measure();
    let counts = run_counts_with(BackendKind::FactoredStabilizer, &circuit, 128, 42)
        .unwrap()
        .0;
    let total: u64 = counts.values().sum();
    let bell_total = counts.get(&vec![0u64]).copied().unwrap_or(0)
        + counts.get(&vec![3u64]).copied().unwrap_or(0);

    assert_eq!(total, 128);
    assert_eq!(bell_total, 128);
}

fn assert_unit_norm(state: &[num_complex::Complex64], label: &str) {
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "{label}: norm = {norm}, expected 1.0"
    );
}

#[test]
fn test_export_norm_statevector_bell() {
    let circuit = make_clifford_circuit();
    let mut backend = StatevectorBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/bell");
}

#[test]
fn test_export_norm_statevector_parametric() {
    let circuit = crate::circuits::hardware_efficient_ansatz(6, 3, 42);
    let mut backend = StatevectorBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/hea_6q");
}

#[test]
fn test_export_norm_stabilizer() {
    let circuit = make_clifford_circuit();
    let mut backend = StabilizerBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "stabilizer");
}

#[test]
fn test_export_norm_sparse() {
    let circuit = make_general_circuit();
    let mut backend = SparseBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "sparse");
}

#[test]
fn test_export_norm_mps() {
    let circuit = make_general_circuit();
    let mut backend = MpsBackend::new(64, 42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "mps");
}

#[test]
fn test_export_norm_product_state() {
    let circuit = make_product_circuit();
    let mut backend = ProductStateBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "productstate");
}

#[test]
fn test_export_norm_tensor_network() {
    let circuit = make_general_circuit();
    let mut backend = TensorNetworkBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    assert_unit_norm(&backend.export_statevector().unwrap(), "tensornetwork");
}

#[test]
fn test_export_norm_after_measurement() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        bit[1] c;
        h q[0];
        cx q[0], q[1];
        h q[2];
        c[0] = measure q[0];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    for backend_kind in [
        BackendKind::Statevector,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 64 },
    ] {
        let label = format!("{backend_kind:?}/post-measure");
        let mut backend = resolve_backend(&backend_kind, &circuit, false).build(42);
        run_on(backend.as_mut(), &circuit).unwrap();
        let state = backend.export_statevector().unwrap();
        assert_unit_norm(&state, &label);
    }
}

#[test]
fn test_export_norm_qft() {
    let circuit = crate::circuits::qft_circuit(8);
    for (kind, label) in [
        (BackendKind::Statevector, "statevector/qft8"),
        (BackendKind::Sparse, "sparse/qft8"),
        (BackendKind::Mps { max_bond_dim: 128 }, "mps/qft8"),
        (BackendKind::TensorNetwork, "tn/qft8"),
    ] {
        let mut backend = resolve_backend(&kind, &circuit, false).build(42);
        run_on(backend.as_mut(), &circuit).unwrap();
        let state = backend.export_statevector().unwrap();
        assert_unit_norm(&state, label);
    }
}

// `make_general_circuit` entangles `{0, 1}` and leaves qubit 2 alone, so the
// export runs with two live sub-states rather than one.
#[test]
fn test_export_factored_tensors_multiple_blocks() {
    let circuit = make_general_circuit();
    let mut backend = crate::backend::factored::FactoredBackend::new(42);
    run_on(&mut backend, &circuit).unwrap();
    let state = backend.export_statevector().unwrap();
    assert_unit_norm(&state, "factored multi-block export");

    let mut sv = StatevectorBackend::new(42);
    run_on(&mut sv, &circuit).unwrap();
    let expected = sv.export_statevector().unwrap();
    assert_eq!(state.len(), expected.len());
    for (i, (a, e)) in state.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).norm() < 1e-12,
            "factored export[{i}]: expected {e}, got {a}"
        );
    }
}

#[test]
fn test_shots_random_convergence() {
    let circuit = make_bell_with_measure();
    let result = run_shots(&circuit, 10000, rand::random()).unwrap();
    let counts = result.counts();
    let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
    let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
    let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
    let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
    // p=0.5, n=10000 → σ=50. Bounds at ±10σ: failure prob < 10^-23.
    assert!(
        (4500..=5500).contains(&n_00),
        "|00> count {n_00} outside [4500, 5500]"
    );
    assert!(
        (4500..=5500).contains(&n_11),
        "|11> count {n_11} outside [4500, 5500]"
    );
    assert_eq!(n_01, 0, "|01> should never appear in Bell state");
    assert_eq!(n_10, 0, "|10> should never appear in Bell state");
}

#[test]
fn test_has_terminal_measurements_only() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    assert!(c.has_terminal_measurements_only());

    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(circuit.has_terminal_measurements_only());

    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        c[0] = measure q[0];
        h q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(!circuit.has_terminal_measurements_only());

    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0]) x q[1];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(!circuit.has_terminal_measurements_only());

    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        h q[0];
        c[0] = measure q[0];
        x q[0];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(!circuit.has_terminal_measurements_only());
}

#[test]
fn test_measurement_map() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        c[2] = measure q[0];
        c[0] = measure q[2];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let map = circuit.measurement_map();
    assert_eq!(map, vec![(0, 2), (2, 0), (1, 1)]);
}

#[test]
fn test_fast_path_deterministic_x() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        x q[0];
        c[0] = measure q[0];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(circuit.has_terminal_measurements_only());
    let result = run_shots(&circuit, 100, 42).unwrap();
    for (i, shot) in result.shots.iter().enumerate() {
        assert!(shot[0], "shot {i}: X|0> should always measure 1");
    }
}

#[test]
fn test_fast_path_preserves_classical_bit_index() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[3] c;
        x q[0];
        c[2] = measure q[0];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    let result = run_shots(&circuit, 16, 42).unwrap();
    assert_eq!(result.num_classical_bits, 3);
    for shot in &result.shots {
        assert_eq!(shot, &vec![false, false, true]);
    }
}

#[test]
fn test_terminal_statevector_sampling_matches_probability_path() {
    let mut c = Circuit::new(5, 5);
    for q in 0..5 {
        c.add_gate(Gate::Ry(0.17 + q as f64 * 0.11), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Rz(0.41), &[3]);
    c.add_gate(Gate::Rx(0.23), &[4]);
    c.add_measure(3, 1);
    c.add_measure(0, 4);

    let stripped = c.without_measurements();
    let reference = run_with_internal(
        BackendKind::Statevector,
        &stripped,
        42,
        SimOptions::default(),
    )
    .unwrap();
    let probs = reference.probabilities.unwrap();
    let expected = shots::sample_shots(&probs, &c.measurement_map(), c.num_classical_bits, 256, 42);

    let actual = run_shots_with(BackendKind::Statevector, &c, 256, 42).unwrap();
    assert_eq!(actual.shots, expected);
}

#[test]
fn test_terminal_statevector_counts_match_probability_path_all_measured() {
    let mut c = Circuit::new(4, 4);
    c.add_gate(Gate::Ry(0.31), &[0]);
    c.add_gate(Gate::Ry(0.47), &[1]);
    c.add_gate(Gate::Cx, &[0, 2]);
    c.add_gate(Gate::Rx(0.19), &[3]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.add_measure(2, 2);
    c.add_measure(3, 3);

    let stripped = c.without_measurements();
    let reference = run_with_internal(
        BackendKind::Statevector,
        &stripped,
        7,
        SimOptions::default(),
    )
    .unwrap();
    let probs = reference.probabilities.unwrap();
    let expected_shots =
        shots::sample_shots(&probs, &c.measurement_map(), c.num_classical_bits, 512, 7);
    let expected = ShotsResult::from_shots(expected_shots, c.num_classical_bits).counts();

    let actual = run_counts_with(BackendKind::Statevector, &c, 512, 7)
        .unwrap()
        .0;
    assert_eq!(actual, expected);
}

#[test]
fn test_terminal_statevector_duplicate_classical_bit_uses_last_measurement() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_measure(1, 0);

    let shots = run_shots_with(BackendKind::Statevector, &c, 16, 42).unwrap();
    for shot in &shots.shots {
        assert_eq!(shot, &vec![false]);
    }

    let counts = run_counts_with(BackendKind::Statevector, &c, 16, 42)
        .unwrap()
        .0;
    assert_eq!(counts.get(&vec![0]), Some(&16));
}

#[test]
fn test_terminal_statevector_counts_wide_classical_register() {
    let mut c = Circuit::new(2, 72);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 70);

    let counts = run_counts_with(BackendKind::Statevector, &c, 10, 11)
        .unwrap()
        .0;
    let mut expected = vec![0u64; 2];
    expected[1] = 1u64 << 6;
    assert_eq!(counts.get(&expected), Some(&10));
}

#[test]
fn test_terminal_statevector_subset_counts_sum_to_shots() {
    let mut c = Circuit::new(5, 5);
    for q in 0..5 {
        c.add_gate(Gate::Ry(0.21 + q as f64 * 0.07), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[3, 4]);
    c.add_measure(1, 4);
    c.add_measure(4, 0);

    let counts = run_counts_with(BackendKind::Statevector, &c, 1024, 42)
        .unwrap()
        .0;
    assert_eq!(counts.values().sum::<u64>(), 1024);
    assert!(counts.keys().all(|key| key[0] & !0b1_0001 == 0));
}

#[test]
fn test_fast_path_no_measurements() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    let result = run_shots(&c, 50, 42).unwrap();
    for shot in &result.shots {
        assert_eq!(shot.len(), 2);
        assert!(!shot[0] && !shot[1], "no measurements → all-false");
    }
}

#[test]
fn test_shots_cached_fusion_matches_uncached() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        x q[1];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(!circuit.has_terminal_measurements_only());

    let cached = run_shots_with(BackendKind::Statevector, &circuit, 20, 42).unwrap();
    for i in 0..20 {
        let seed_i = 42u64.wrapping_add(i as u64);
        let single = run_with_internal(
            BackendKind::Statevector,
            &circuit,
            seed_i,
            SimOptions::default(),
        )
        .unwrap();
        assert_eq!(cached.shots[i], single.classical_bits, "shot {i} mismatch");
    }
}

#[test]
fn test_shots_decomposed_cached() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[8] q;
        bit[8] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        x q[1];
        c[1] = measure q[1];
        h q[4];
        cx q[4], q[5];
        c[4] = measure q[4];
        x q[5];
        c[5] = measure q[5];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
    assert!(!circuit.has_terminal_measurements_only());
    let comps = circuit.independent_subsystems();
    assert!(comps.len() > 1, "circuit should decompose");

    let result = run_shots_with(BackendKind::Statevector, &circuit, 10, 42).unwrap();
    assert_eq!(result.shots.len(), 10);
    for shot in &result.shots {
        assert_eq!(shot.len(), 8);
    }
}

#[test]
fn test_shots_temporal_clifford_fallback() {
    let mut c = Circuit::new(4, 4);
    for i in 0..4 {
        c.add_gate(Gate::H, &[i]);
    }
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    c.add_gate(Gate::T, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_measure(1, 1);

    let result = run_shots_with(BackendKind::Auto, &c, 10, 42).unwrap();
    assert_eq!(result.shots.len(), 10);
    for shot in &result.shots {
        assert_eq!(shot.len(), 4);
    }
}

#[test]
fn test_stabilizer_rank_dispatch() {
    let circuit = make_general_circuit();
    let result = run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert_eq!(probs.len(), 8);
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);

    let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (i, (sr, sv)) in probs.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr - sv).abs() < 1e-10,
            "prob[{i}]: stab_rank={sr}, statevector={sv}"
        );
    }
}

#[test]
fn test_stabilizer_rank_rejects_no_t() {
    let circuit = make_clifford_circuit();
    let result = run_with(BackendKind::StabilizerRank, &circuit, 42);
    assert!(result.is_err());
}

#[test]
fn test_auto_clifford_plus_t_probabilities() {
    let circuit = make_general_circuit();
    assert!(circuit.is_clifford_plus_t());
    assert!(circuit.has_t_gates());

    let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
    let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();

    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (a - s).abs() < 1e-10,
            "prob[{i}]: auto={a}, statevector={s}"
        );
    }
}

#[test]
fn test_auto_clifford_plus_t_shots() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    let result = run_shots_with(BackendKind::Auto, &c, 100, 42).unwrap();
    assert_eq!(result.shots.len(), 100);
    for shot in &result.shots {
        assert_eq!(shot.len(), 2);
    }
}

#[test]
fn test_decomposed_mixed_clifford_and_t() {
    // Two independent subsystems: q0-q1 (Clifford+T), q2-q3 (Clifford-only)
    // Under decomposition, q2-q3 should route to Stabilizer, q0-q1 to StabilizerRank
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);

    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 2);

    let auto_result = run_with(BackendKind::Auto, &c, 42).unwrap();
    let sv_result = run_with(BackendKind::Statevector, &c, 42).unwrap();

    let auto_probs = auto_result.probabilities.unwrap().to_vec();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();
    for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (a - s).abs() < 1e-10,
            "prob[{i}]: auto={a}, statevector={s}"
        );
    }
}

#[test]
fn test_run_shots_with_noise_clifford_uses_compiled() {
    let n = 10;
    let mut circuit = crate::circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
    assert_eq!(result.shots.len(), 100);
    assert!(result.shots[0].len() == n);
}

#[test]
fn test_run_shots_with_noise_statevector_brute() {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let result = run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 50, 42).unwrap();
    assert_eq!(result.shots.len(), 50);
    assert_eq!(result.shots[0].len(), 3);
}

#[test]
fn test_run_shots_with_noise_auto_non_clifford() {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
    assert_eq!(result.shots.len(), 100);
}

#[cfg(feature = "gpu")]
#[test]
fn test_run_shots_with_stabilizer_gpu_falls_back_for_reset_circuits() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);

    let cpu = run_shots_with(BackendKind::Stabilizer, &circuit, 8, 42).unwrap();
    let gpu = run_shots_with(
        BackendKind::StabilizerGpu {
            context: crate::gpu::GpuContext::stub_for_tests(),
        },
        &circuit,
        8,
        42,
    )
    .unwrap();

    assert_eq!(gpu.shots, cpu.shots);
}

#[cfg(feature = "gpu")]
#[test]
fn test_run_shots_with_stabilizer_gpu_falls_back_for_conditionals() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: crate::circuit::smallvec![1],
    });
    circuit.add_measure(1, 1);

    let cpu = run_shots_with(BackendKind::Stabilizer, &circuit, 256, 42).unwrap();
    let gpu = run_shots_with(
        BackendKind::StabilizerGpu {
            context: crate::gpu::GpuContext::stub_for_tests(),
        },
        &circuit,
        256,
        42,
    )
    .unwrap();

    assert_eq!(gpu.shots, cpu.shots);
}

#[cfg(feature = "gpu")]
#[test]
fn test_run_shots_with_noise_stabilizer_gpu_matches_stabilizer() {
    let n = 8;
    let mut circuit = crate::circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let cpu = run_shots_with_noise(BackendKind::Stabilizer, &circuit, &noise, 128, 42).unwrap();
    let gpu = run_shots_with_noise(
        BackendKind::StabilizerGpu {
            context: crate::gpu::GpuContext::stub_for_tests(),
        },
        &circuit,
        &noise,
        128,
        42,
    )
    .unwrap();

    assert_eq!(gpu.shots, cpu.shots);
}

#[test]
fn test_run_marginals_bell_pair() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let m = run_marginals(&c, 42).unwrap();
    assert_eq!(m.len(), 2);
    assert!((m[0].0 - 0.5).abs() < 1e-10);
    assert!((m[0].1 - 0.5).abs() < 1e-10);
    assert!((m[1].0 - 0.5).abs() < 1e-10);
    assert!((m[1].1 - 0.5).abs() < 1e-10);
}

#[test]
fn test_run_marginals_x_gate() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    let m = run_marginals(&c, 42).unwrap();
    assert!((m[0].0 - 0.0).abs() < 1e-10);
    assert!((m[0].1 - 1.0).abs() < 1e-10);
    assert!((m[1].0 - 1.0).abs() < 1e-10);
    assert!((m[1].1 - 0.0).abs() < 1e-10);
}

#[test]
fn test_run_handles_backend_probability_failures() {
    let circuit = Circuit::new(1, 0);

    let mut unsupported = ProbabilityFailureBackend::new(ProbabilityFailure::Unsupported);
    let result = run_on(&mut unsupported, &circuit).unwrap();
    assert!(result.classical_bits.is_empty());
    assert!(result.probabilities.is_none());

    let mut invalid = ProbabilityFailureBackend::new(ProbabilityFailure::Invalid);
    let err = run_on(&mut invalid, &circuit).unwrap_err();

    assert!(matches!(err, PrismError::InvalidParameter { .. }));
}

#[test]
fn test_run_marginals_rejects_missing_probability_output() {
    let circuit = Circuit::new(65, 0);
    let run_result = simulate(&circuit)
        .backend(BackendKind::FactoredStabilizer)
        .seed(42)
        .run()
        .unwrap();
    assert!(run_result.probabilities.is_none());

    let err = simulate(&circuit)
        .backend(BackendKind::FactoredStabilizer)
        .seed(42)
        .marginals()
        .unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_run_marginals_clifford_t_spd_path() {
    let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
    let m_spd = run_marginals(&c, 42).unwrap();
    assert_eq!(m_spd.len(), 14);
    for (p0, p1) in &m_spd {
        assert!(*p0 >= 0.0 && *p0 <= 1.0);
        assert!((p0 + p1 - 1.0).abs() < 1e-10);
    }

    let m_sv = run_marginals_with(BackendKind::Statevector, &c, 42).unwrap();
    for i in 0..14 {
        assert!(
            (m_spd[i].0 - m_sv[i].0).abs() < 1e-6,
            "qubit {i}: SPD p0={} vs SV p0={}",
            m_spd[i].0,
            m_sv[i].0
        );
    }
}

// ---- Dispatch validation ----

#[test]
fn test_simulate_builder_run_matches_run() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(0.31), &[2]);

    let expected = run(&c, 42).unwrap();
    let actual = simulate(&c).seed(42).run().unwrap();

    assert_eq!(actual.classical_bits, expected.classical_bits);
    assert_eq!(
        actual.probabilities.unwrap().to_vec(),
        expected.probabilities.unwrap().to_vec()
    );
}

#[test]
fn test_simulate_builder_sample_counts_matches_run_counts() {
    let mut c = Circuit::new(4, 4);
    c.add_gate(Gate::Ry(0.25), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rx(0.17), &[2]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.add_measure(2, 2);
    c.add_measure(3, 3);

    let expected = run_counts(&c, 256, 42).unwrap();
    let actual = simulate(&c).seed(42).sample_counts(256).unwrap();

    assert_eq!(actual.num_classical_bits, c.num_classical_bits);
    assert_eq!(actual.counts, expected);
}

#[test]
fn test_simulate_builder_marginals_matches_run_marginals() {
    let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
    let expected = run_marginals(&c, 42).unwrap();
    let actual = simulate(&c).seed(42).marginals().unwrap();

    assert_eq!(actual.marginals.len(), expected.len());
    for (a, b) in actual.marginals.iter().zip(expected.iter()) {
        assert!((a.0 - b.0).abs() < 1e-12);
        assert!((a.1 - b.1).abs() < 1e-12);
    }
}

#[test]
fn test_validate_factored_stabilizer_rejects_non_clifford() {
    let circuit = make_general_circuit();
    assert!(matches!(
        run_with(BackendKind::FactoredStabilizer, &circuit, 42).unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_validate_stabilizer_rank_rejects_no_t_gates() {
    let circuit = make_clifford_circuit();
    assert!(matches!(
        run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_validate_factored_stabilizer_accepts_clifford() {
    assert!(
        run_with(
            BackendKind::FactoredStabilizer,
            &make_clifford_circuit(),
            42
        )
        .is_ok()
    );
}

// ---- Pauli backend error paths ----

#[test]
fn test_pauli_backends_reject_mid_circuit_measurements() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        h q[0];
        c[0] = measure q[0];
        cx q[0], q[1];
        c[1] = measure q[1];
    "#;
    let circuit = crate::circuit::openqasm::parse(qasm).unwrap();

    assert!(matches!(
        run_shots_with(
            BackendKind::StochasticPauli { num_samples: 100 },
            &circuit,
            10,
            42
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
    assert!(matches!(
        run_shots_with(
            BackendKind::DeterministicPauli {
                epsilon: 1e-3,
                max_terms: 1000
            },
            &circuit,
            10,
            42,
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

// ---- Noisy simulation error paths ----

#[test]
fn test_pauli_backends_reject_generic_run() {
    let c = crate::circuits::clifford_t_circuit(4, 2, 0.1, 42);

    assert!(matches!(
        simulate(&c)
            .backend(BackendKind::StochasticPauli { num_samples: 100 })
            .seed(42)
            .run()
            .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
    assert!(matches!(
        simulate(&c)
            .backend(BackendKind::DeterministicPauli {
                epsilon: 0.0,
                max_terms: 0
            })
            .seed(42)
            .run()
            .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_pauli_backends_return_marginals_through_builder() {
    let c = crate::circuits::clifford_t_circuit(4, 2, 0.1, 42);

    let spp = simulate(&c)
        .backend(BackendKind::StochasticPauli { num_samples: 1_000 })
        .seed(42)
        .marginals()
        .unwrap();
    let spd = simulate(&c)
        .backend(BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        })
        .seed(42)
        .marginals()
        .unwrap();

    assert_eq!(spp.marginals.len(), c.num_qubits);
    assert_eq!(spd.marginals.len(), c.num_qubits);
    assert!(
        spp.marginals
            .iter()
            .chain(spd.marginals.iter())
            .all(|(p0, p1)| *p0 >= 0.0 && *p0 <= 1.0 && (p0 + p1 - 1.0).abs() < 1e-10)
    );
}

#[test]
fn test_pauli_marginals_reject_gates_off_the_z_axis() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(0.25), &[0]);

    for err in pauli_marginal_errors(&c) {
        assert!(
            matches!(err, PrismError::BackendUnsupported { .. }),
            "{err:?}"
        );
    }

    let mut supported = Circuit::new(1, 0);
    supported.add_gate(Gate::H, &[0]);
    supported.add_gate(Gate::Rz(0.25), &[0]);
    assert_eq!(
        simulate(&supported)
            .backend(BackendKind::DeterministicPauli {
                epsilon: 0.0,
                max_terms: 0,
            })
            .seed(42)
            .marginals()
            .unwrap()
            .marginals
            .len(),
        1
    );
}

#[test]
fn test_pauli_marginals_reject_measurements_resets_and_conditionals() {
    let mut measured = Circuit::new(1, 1);
    measured.add_gate(Gate::H, &[0]);
    measured.add_gate(Gate::T, &[0]);
    measured.add_measure(0, 0);

    let mut reset = Circuit::new(1, 0);
    reset.add_gate(Gate::T, &[0]);
    reset.add_reset(0);

    let mut conditional = Circuit::new(2, 1);
    conditional.add_measure(0, 0);
    conditional.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: Gate::T,
        targets: smallvec![1],
    });

    for circuit in [&measured, &reset, &conditional] {
        for err in pauli_marginal_errors(circuit) {
            assert!(
                matches!(err, PrismError::IncompatibleBackend { .. }),
                "{err:?}"
            );
        }
    }
}

#[test]
fn test_noise_rejects_stabilizer_rank() {
    let circuit = make_general_circuit();
    let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(matches!(
        run_shots_with_noise(BackendKind::StabilizerRank, &circuit, &nm, 10, 42).unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_noise_rejects_pauli_backends() {
    let circuit = make_general_circuit();
    let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);

    assert!(matches!(
        run_shots_with_noise(
            BackendKind::StochasticPauli { num_samples: 100 },
            &circuit,
            &nm,
            10,
            42,
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
    assert!(matches!(
        run_shots_with_noise(
            BackendKind::DeterministicPauli {
                epsilon: 1e-3,
                max_terms: 1000
            },
            &circuit,
            &nm,
            10,
            42,
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_noise_stabilizer_rejects_non_pauli_noise() {
    let circuit = make_clifford_circuit();
    let nm = noise::NoiseModel {
        after_gate: {
            let mut ag = vec![Vec::new(); circuit.instructions.len()];
            ag[0].push(noise::NoiseEvent {
                channel: noise::NoiseChannel::AmplitudeDamping { gamma: 0.1 },
                qubits: smallvec![0],
            });
            ag
        },
        readout: vec![None; circuit.num_qubits],
    };
    let err = run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err();
    match err {
        crate::error::PrismError::IncompatibleBackend { reason, .. } => {
            assert!(reason.contains("Statevector"));
            assert!(reason.contains("Sparse"));
            assert!(reason.contains("Factored"));
        }
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

#[test]
fn test_noise_auto_general_noise_avoids_stabilizer_dispatch() {
    let circuit = make_clifford_circuit();
    let nm = noise::NoiseModel::with_amplitude_damping(&circuit, 0.1);
    let result = run_shots_with_noise(BackendKind::Auto, &circuit, &nm, 16, 42);
    assert!(result.is_ok());
}

#[cfg(feature = "gpu")]
#[test]
fn test_noise_stabilizer_gpu_rejects_non_pauli_noise() {
    let circuit = make_clifford_circuit();
    let nm = noise::NoiseModel {
        after_gate: {
            let mut ag = vec![Vec::new(); circuit.instructions.len()];
            ag[0].push(noise::NoiseEvent {
                channel: noise::NoiseChannel::AmplitudeDamping { gamma: 0.1 },
                qubits: smallvec![0],
            });
            ag
        },
        readout: vec![None; circuit.num_qubits],
    };
    assert!(matches!(
        run_shots_with_noise(
            BackendKind::StabilizerGpu {
                context: crate::gpu::GpuContext::stub_for_tests(),
            },
            &circuit,
            &nm,
            10,
            42,
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn test_noise_stabilizer_rejects_non_clifford() {
    let circuit = make_general_circuit();
    let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(matches!(
        run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[cfg(feature = "gpu")]
#[test]
fn test_noise_stabilizer_gpu_rejects_non_clifford() {
    let circuit = make_general_circuit();
    let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(matches!(
        run_shots_with_noise(
            BackendKind::StabilizerGpu {
                context: crate::gpu::GpuContext::stub_for_tests(),
            },
            &circuit,
            &nm,
            10,
            42,
        )
        .unwrap_err(),
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

// ---- Backend smoke tests ----

fn assert_probs_match(kind: BackendKind, circuit: &Circuit, expected: &[f64], tol: f64) {
    let label = format!("{kind:?}");
    let result = run_with(kind, circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert_eq!(probs.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, b)) in probs.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < tol,
            "{label}: prob[{i}] = {a}, expected {b}"
        );
    }
}

#[test]
fn test_smoke_all_backends_clifford() {
    let circuit = make_clifford_circuit();
    let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();

    for kind in [
        BackendKind::Stabilizer,
        BackendKind::FactoredStabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 64 },
        BackendKind::TensorNetwork,
        BackendKind::Factored,
    ] {
        assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
    }
}

#[test]
fn test_smoke_all_backends_general() {
    let circuit = make_general_circuit();
    let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();

    for kind in [
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 64 },
        BackendKind::TensorNetwork,
        BackendKind::Factored,
    ] {
        assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
    }
}

#[test]
fn test_smoke_product_state() {
    let circuit = make_product_circuit();
    let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();
    assert_probs_match(BackendKind::ProductState, &circuit, &sv_probs, 1e-8);
}

#[test]
fn test_smoke_stabilizer_rank() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::T, &[2]);

    let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();
    assert_probs_match(BackendKind::StabilizerRank, &circuit, &sv_probs, 1e-6);
}
