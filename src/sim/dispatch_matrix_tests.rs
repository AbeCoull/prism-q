use super::*;
use crate::gates::Gate;

fn product(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    c
}

fn clifford(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

fn dense(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

/// One qubit past the statevector memory cap, or `None` when memory
/// detection failed and the cap is disabled (no width is oversize then).
fn oversize_qubits() -> Option<usize> {
    let cap = ResolvedLimits::process_default().statevector_qubits.value;
    if cap >= usize::BITS as usize {
        eprintln!("SKIP: statevector qubit cap disabled on this host; skipping oversize checks");
        return None;
    }
    Some(cap + 1)
}

fn oversize_sparse(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::T, &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

fn oversize_dense(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

fn resolved(kind: &BackendKind, circuit: &Circuit, hpi: bool) -> BackendPlan {
    match resolve(kind, circuit, hpi, &ResolvedLimits::process_default()) {
        ExecutionPlan::Backend(plan) => plan,
        _ => panic!("expected a backend plan"),
    }
}

fn assert_cpu_family(kind: &BackendKind, circuit: &Circuit, hpi: bool, family: Family) {
    let plan = resolved(kind, circuit, hpi);
    assert_eq!(plan.family(), family, "kind {kind:?}");
    assert!(
        matches!(plan.accel(), Accel::Cpu),
        "kind {kind:?} family {family:?} must resolve to the host"
    );
}

fn auto_kinds() -> Vec<BackendKind> {
    #[cfg(feature = "gpu")]
    {
        vec![
            BackendKind::Auto,
            BackendKind::AutoGpu {
                context: GpuContext::stub_for_tests(),
            },
        ]
    }
    #[cfg(not(feature = "gpu"))]
    {
        vec![BackendKind::Auto]
    }
}

// Auto and AutoGpu make identical family choices across every circuit
// shape class; on the stub context every choice resolves to the host
// (small circuits by crossover, large ones by the fail-closed VRAM gate).
#[test]
fn auto_family_matrix() {
    let oversize = oversize_qubits();
    for kind in auto_kinds() {
        assert_cpu_family(&kind, &product(6), false, Family::ProductState);
        assert_cpu_family(&kind, &clifford(6), false, Family::Stabilizer);
        assert_cpu_family(&kind, &clifford(16), false, Family::Stabilizer);
        assert_cpu_family(&kind, &dense(8), false, Family::Statevector);
        assert_cpu_family(&kind, &dense(16), false, Family::Statevector);
        assert_cpu_family(&kind, &dense(8), true, Family::Factored);
        if let Some(n) = oversize {
            assert_cpu_family(&kind, &oversize_sparse(n), false, Family::Sparse);
            assert_cpu_family(&kind, &oversize_dense(n), false, Family::Mps);
            let w_state = crate::circuits::w_state_circuit(n);
            assert_cpu_family(&kind, &w_state, false, Family::Sparse);
        }
    }
}

#[test]
fn sparse_support_probe_bounds_every_gate_not_the_final_state() {
    // A W state ends on 64 basis states but holds 126 midway through its last
    // controlled rotation.
    let w_state = crate::circuits::w_state_circuit(64);
    assert!(sparse_support_fits(
        &w_state,
        126,
        &ResolvedLimits::process_default()
    ));
    assert!(!sparse_support_fits(
        &w_state,
        125,
        &ResolvedLimits::process_default()
    ));
    assert!(!sparse_support_fits(
        &oversize_dense(40),
        AUTO_SPARSE_MAX_SUPPORT,
        &ResolvedLimits::process_default()
    ));

    let mut measured = Circuit::new(8, 1);
    measured.instructions = crate::circuits::w_state_circuit(8).instructions;
    measured.add_measure(0, 0);
    assert!(sparse_support_fits(
        &measured,
        AUTO_SPARSE_MAX_SUPPORT,
        &ResolvedLimits::process_default()
    ));
    measured.add_gate(Gate::H, &[1]);
    assert!(!sparse_support_fits(
        &measured,
        AUTO_SPARSE_MAX_SUPPORT,
        &ResolvedLimits::process_default()
    ));
}

#[test]
fn auto_oversize_mps_uses_auto_bond_dim() {
    let Some(n) = oversize_qubits() else { return };
    for kind in auto_kinds() {
        let plan = resolved(&kind, &oversize_dense(n), false);
        let auto_bond = ResolvedLimits::process_default().auto_mps_bond_dim.value;
        assert!(matches!(
            plan,
            BackendPlan::Mps { max_bond_dim } if max_bond_dim == auto_bond
        ));
    }
}

// Explicit CPU kinds map 1:1 onto their family regardless of circuit
// shape, always on the host.
#[test]
fn explicit_cpu_kind_matrix() {
    let circuit = dense(6);
    let cases = [
        (BackendKind::Statevector, Family::Statevector),
        (BackendKind::Stabilizer, Family::Stabilizer),
        (BackendKind::Sparse, Family::Sparse),
        (BackendKind::ProductState, Family::ProductState),
        (BackendKind::TensorNetwork, Family::TensorNetwork),
        (BackendKind::Factored, Family::Factored),
        (BackendKind::FactoredStabilizer, Family::FactoredStabilizer),
    ];
    for (kind, family) in cases {
        assert_cpu_family(&kind, &circuit, false, family);
    }

    let plan = resolved(&BackendKind::Mps { max_bond_dim: 77 }, &circuit, false);
    assert!(matches!(plan, BackendPlan::Mps { max_bond_dim: 77 }));
}

#[test]
fn non_backend_kinds_resolve_to_their_engines() {
    let circuit = dense(6);
    assert!(matches!(
        resolve(
            &BackendKind::StabilizerRank,
            &circuit,
            false,
            &ResolvedLimits::process_default()
        ),
        ExecutionPlan::StabilizerRank
    ));
    assert!(matches!(
        resolve(
            &BackendKind::StochasticPauli { num_samples: 9 },
            &circuit,
            false,
            &ResolvedLimits::process_default()
        ),
        ExecutionPlan::StochasticPauli { num_samples: 9 }
    ));
    assert!(matches!(
        resolve(
            &BackendKind::DeterministicPauli {
                truncation: SpdTruncation::Threshold {
                    epsilon: 0.5,
                    max_terms: 3
                }
            },
            &circuit,
            false, &ResolvedLimits::process_default()),
        ExecutionPlan::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: e,
                max_terms: 3
            }
        } if e == 0.5
    ));
}

#[test]
fn density_matrix_is_explicit_only() {
    // Auto never routes to the density-matrix family, whatever the shape.
    for circuit in [product(6), clifford(6), dense(6)] {
        for hpi in [false, true] {
            assert_ne!(
                resolved(&BackendKind::Auto, &circuit, hpi).family(),
                Family::DensityMatrix
            );
        }
    }
    // Explicit selection maps 1:1 onto the density-matrix plan, and the
    // backend it builds accepts fused payloads on every terminal.
    let plan = resolved(&BackendKind::DensityMatrix, &dense(6), false);
    assert_eq!(plan.family(), Family::DensityMatrix);
    assert!(
        plan.build(42, &ResolvedLimits::process_default())
            .supports_fused_gates()
    );
    assert!(matches!(plan.accel(), Accel::Cpu));
}

#[test]
fn density_matrix_rejects_over_cap() {
    let cap = ResolvedLimits::process_default()
        .density_matrix_qubits
        .value;
    if cap >= usize::BITS as usize {
        eprintln!("SKIP: density-matrix cap disabled on this host");
        return;
    }
    let circuit = dense(cap + 1);
    let err = validate_explicit_backend(
        &BackendKind::DensityMatrix,
        &circuit,
        &ResolvedLimits::process_default(),
    )
    .unwrap_err();
    assert!(matches!(err, PrismError::ResourceLimit(_)));
    // At or below the cap the same shape validates.
    assert!(
        validate_explicit_backend(
            &BackendKind::DensityMatrix,
            &dense(cap),
            &ResolvedLimits::process_default()
        )
        .is_ok()
    );
}

// Explicit GPU kinds resolve hard device execution at their crossover
// (no VRAM gate) and host execution below it; the family choice never
// changes with the target.
#[cfg(feature = "gpu")]
#[test]
fn explicit_gpu_kind_matrix() {
    let sv_kind = BackendKind::StatevectorGpu {
        context: GpuContext::stub_for_tests(),
    };
    let large = dense(crate::gpu::min_qubits());
    let plan = resolved(&sv_kind, &large, false);
    assert_eq!(plan.family(), Family::Statevector);
    assert!(matches!(plan.accel(), Accel::Gpu { soft: false, .. }));
    assert_cpu_family(
        &sv_kind,
        &dense(crate::gpu::min_qubits() - 1),
        false,
        Family::Statevector,
    );

    let stab_kind = BackendKind::StabilizerGpu {
        context: GpuContext::stub_for_tests(),
    };
    let huge = Circuit::new(crate::gpu::stabilizer_min_qubits(), 0);
    let plan = resolved(&stab_kind, &huge, false);
    assert_eq!(plan.family(), Family::Stabilizer);
    assert!(matches!(plan.accel(), Accel::Gpu { soft: false, .. }));
    assert_cpu_family(&stab_kind, &clifford(8), false, Family::Stabilizer);
}

// Three CX on one pair form a block of rank at most 4, so the middle cut
// reads bonds 2, 4, 4; a CX on a neighbouring pair in between ends the block
// and the next CX doubles the bond from where it stands.
#[test]
fn mps_apply_cost_merges_a_pair_block_and_breaks_it_on_a_neighbour() {
    let overhead = MPS_GATE_CUT_OVERHEAD;
    let mut block = Circuit::new(8, 0);
    for _ in 0..3 {
        block.add_gate(Gate::Cx, &[3, 4]);
        block.add_gate(Gate::Ry(0.4), &[4]);
    }
    let (cost, block_cuts) = mps_apply_cost(&block, 256);
    assert_eq!(cost, 3.0 * overhead + 8.0 + 64.0 + 64.0);
    assert_eq!(block_cuts, 1);

    let mut broken = Circuit::new(8, 0);
    broken.add_gate(Gate::Cx, &[3, 4]);
    broken.add_gate(Gate::Cx, &[3, 4]);
    broken.add_gate(Gate::Cx, &[2, 3]);
    broken.add_gate(Gate::Cx, &[3, 4]);
    let (cost, block_cuts) = mps_apply_cost(&broken, 256);
    assert_eq!(cost, 4.0 * overhead + 8.0 + 64.0 + 8.0 + 512.0);
    assert_eq!(block_cuts, 3);
}

fn rcs_brickwork(n: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..depth {
        for q in 0..n {
            circuit.add_gate(Gate::Rz(0.3 + q as f64 * 0.01), &[q]);
            circuit.add_gate(Gate::Ry(1.1), &[q]);
            circuit.add_gate(Gate::Rz(0.7), &[q]);
        }
        for i in (layer % 2..n - 1).step_by(2) {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
    }
    circuit
}

fn su4_brickwork(n: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..depth {
        for i in (layer % 2..n - 1).step_by(2) {
            for _ in 0..3 {
                for q in [i, i + 1] {
                    circuit.add_gate(Gate::Ry(0.9), &[q]);
                    circuit.add_gate(Gate::Rz(0.4), &[q]);
                }
                circuit.add_gate(Gate::Cx, &[i, i + 1]);
            }
        }
    }
    circuit
}

// The cross-simulator comparison's rcs observables: four Z and four adjacent ZZ.
fn rcs_fixture_observables(n: usize) -> Vec<Vec<crate::PauliTerm>> {
    let (singles, pairs): ([usize; 4], [usize; 4]) = match n {
        32 => ([2, 14, 20, 23], [2, 5, 16, 19]),
        64 => ([5, 28, 41, 47], [5, 12, 33, 41]),
        128 => ([11, 56, 83, 97], [11, 25, 66, 86]),
        256 => ([22, 112, 166, 196], [23, 50, 134, 175]),
        _ => unreachable!("no fixture at {n} qubits"),
    };
    let singles = singles.map(|q| vec![crate::PauliTerm::z(q)]);
    let pairs = pairs.map(|q| vec![crate::PauliTerm::z(q), crate::PauliTerm::z(q + 1)]);
    singles.into_iter().chain(pairs).collect()
}

// Above the cap, eight local observables on an rcs brickwork contract faster
// than the MPS run from 128 qubits at four or eight layers. Four layers pass on
// the block count, 24 per observable from 97 qubits; eight layers at 64 qubits
// have cones too wide for the block count, as does a two-layer ladder at 256
// qubits. su4_brick_d4/64 under eight two-local terms sits under both floors.
#[test]
fn above_cap_tensor_route_takes_rcs_brickworks_from_128_qubits() {
    let limits = ResolvedLimits::resolve(Some(&crate::Limits::new().max_statevector_qubits(24)));
    let takes_tensor = |circuit: &Circuit, observables: &[Vec<crate::PauliTerm>]| {
        crate::sim::route::decide_tensor_over_mps(circuit, observables, &limits)
            .map(|planned| planned.expect("observables are valid"))
            .is_some()
    };
    for n in [32, 64, 128, 256] {
        let observables = rcs_fixture_observables(n);
        assert_eq!(
            takes_tensor(&rcs_brickwork(n, 4), &observables),
            n >= 128,
            "rcs d4 n{n}"
        );
        assert_eq!(
            takes_tensor(&rcs_brickwork(n, 8), &observables),
            n >= 128,
            "rcs d8 n{n}"
        );
    }
    for n in [96, 97] {
        let observables: Vec<Vec<crate::PauliTerm>> = (0..8)
            .map(|k| vec![crate::PauliTerm::z(k * n / 8 + 5)])
            .collect();
        assert_eq!(
            takes_tensor(&rcs_brickwork(n, 4), &observables),
            n == 97,
            "rcs d4 n{n}"
        );
    }
    let ladder_observables: Vec<Vec<crate::PauliTerm>> = (0..8)
        .map(|k| vec![crate::PauliTerm::z(k * 32 + 16)])
        .collect();
    assert!(!takes_tensor(
        &crate::circuits::hardware_efficient_ansatz(256, 2, 42),
        &ladder_observables
    ));
    let two_local: Vec<Vec<crate::PauliTerm>> = (0..8)
        .map(|k| {
            let a = k * 8;
            vec![crate::PauliTerm::z(a), crate::PauliTerm::z((a + 22) % 64)]
        })
        .collect();
    assert!(!takes_tensor(&su4_brickwork(64, 4), &two_local));
}
