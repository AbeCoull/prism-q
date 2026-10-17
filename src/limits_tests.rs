use super::*;

const GIB: u64 = 1 << 30;

fn resolve(
    budget: Option<u64>,
    env: Limits,
    default: Limits,
    request: Option<Limits>,
) -> ResolvedLimits {
    let mut layers = Layers::base(budget, &env, None);
    layers.apply(&default, LimitSource::ProcessDefault);
    if let Some(request) = request {
        layers.apply(&request, LimitSource::Request);
    }
    layers.finish()
}

#[test]
fn dense_budget_counts_fit_elements() {
    assert_eq!(dense_qubits_for_budget(8, 8), 0);
    assert_eq!(dense_qubits_for_budget(16, 8), 1);
    assert_eq!(dense_qubits_for_budget(31, 8), 1);
    assert_eq!(dense_qubits_for_budget(32, 8), 2);
    assert_eq!(dense_qubits_for_budget(7, 8), 0);
}

#[test]
fn detection_sets_every_cap_from_half_of_memory() {
    let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), None);
    assert_eq!(caps.statevector_qubits.value, 30);
    assert_eq!(caps.statevector_qubits.source, LimitSource::Detected);
    assert_eq!(caps.density_matrix_qubits.value, 15);
    assert_eq!(caps.probability_qubits.value, 31);
    assert_eq!(caps.sparse_qubits.value, 28);
    assert_eq!(caps.tn_slices.value, DEFAULT_TN_SLICES);
    assert_eq!(caps.auto_mps_bond_dim.value, DEFAULT_AUTO_MPS_BOND_DIM);
}

#[test]
fn failed_detection_disables_the_memory_caps() {
    let caps = resolve(None, Limits::new(), Limits::new(), None);
    assert_eq!(caps.statevector_qubits.value, usize::MAX);
    assert_eq!(caps.stabilizer_cluster_qubits.value, usize::MAX);
    assert_eq!(caps.sparse_entries(), usize::MAX);
}

#[test]
fn each_layer_overrides_the_one_below() {
    let env = Limits::new()
        .max_statevector_qubits(24)
        .max_export_qubits(20);
    let default = Limits::new().max_statevector_qubits(22);
    let request = Limits::new().max_statevector_qubits(18);

    let caps = resolve(Some(16 * GIB), env, Limits::new(), None);
    assert_eq!(
        caps.statevector_qubits,
        Limit {
            value: 24,
            source: LimitSource::Env
        }
    );
    let caps = resolve(Some(16 * GIB), env, default, None);
    assert_eq!(
        caps.statevector_qubits,
        Limit {
            value: 22,
            source: LimitSource::ProcessDefault
        }
    );
    let caps = resolve(Some(16 * GIB), env, default, Some(request));
    assert_eq!(
        caps.statevector_qubits,
        Limit {
            value: 18,
            source: LimitSource::Request
        }
    );
    assert_eq!(
        caps.export_qubits,
        Limit {
            value: 20,
            source: LimitSource::Env
        },
        "a cap no higher layer sets keeps the lower layer's value"
    );
    let wider = Limits::new().max_statevector_qubits(28);
    let caps = resolve(Some(16 * GIB), env, default, Some(wider));
    assert_eq!(
        caps.statevector_qubits.value, 28,
        "a request may raise a cap the process default lowered"
    );
}

#[test]
fn a_budget_replaces_lower_per_cap_values() {
    let env = Limits::new()
        .max_statevector_qubits(34)
        .max_density_matrix_qubits(10)
        .max_tn_slices(8);
    let tenant = Limits::new().memory_budget_bytes(4 * GIB);
    let caps = resolve(Some(16 * GIB), env, Limits::new(), Some(tenant));
    assert_eq!(
        caps.statevector_qubits,
        Limit {
            value: 28,
            source: LimitSource::Request
        },
        "a 4 GiB budget must not inherit PRISM_MAX_SV_QUBITS=34"
    );
    assert_eq!(caps.density_matrix_qubits.value, 14);
    assert_eq!(caps.density_matrix_qubits.source, LimitSource::Request);
    assert_eq!(
        caps.tn_slices.value, 8,
        "the slice budget is not a memory cap"
    );

    let both = tenant.max_export_qubits(12);
    let caps = resolve(Some(16 * GIB), env, Limits::new(), Some(both));
    assert_eq!(
        caps.export_qubits.value, 12,
        "the budget's own layer still applies its per-cap values"
    );

    let caps = resolve(Some(16 * GIB), env, tenant, None);
    assert_eq!(caps.statevector_qubits.source, LimitSource::ProcessDefault);
    assert_eq!(caps.statevector_qubits.value, 28);
}

#[test]
fn the_density_matrix_cap_stays_within_half_the_statevector_cap() {
    let caps = resolve(
        Some(16 * GIB),
        Limits::new().max_density_matrix_qubits(40),
        Limits::new(),
        None,
    );
    assert_eq!(
        caps.density_matrix_qubits,
        Limit {
            value: 15,
            source: LimitSource::Detected
        }
    );
    let request = Limits::new().max_statevector_qubits(12);
    let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
    assert_eq!(
        caps.density_matrix_qubits,
        Limit {
            value: 6,
            source: LimitSource::Request
        }
    );
    let request = Limits::new().max_density_matrix_qubits(4);
    let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
    assert_eq!(caps.density_matrix_qubits.value, 4);
}

#[test]
fn the_probability_knob_sets_both_probability_caps() {
    let request = Limits::new().max_probability_qubits(9);
    let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
    assert_eq!(caps.probability_qubits.value, 9);
    assert_eq!(caps.tensor_probability_qubits.value, 9);
}

#[test]
fn zero_counts_read_as_one() {
    let request = Limits::new().max_tn_slices(0).auto_mps_bond_dim(0);
    let caps = resolve(Some(GIB), Limits::new(), Limits::new(), Some(request));
    assert_eq!(caps.tn_slices.value, 1);
    assert_eq!(caps.auto_mps_bond_dim.value, 1);
}

#[test]
fn a_request_error_names_the_setter_and_no_variable() {
    let err = resource_limit_error(
        "statevector",
        "the circuit",
        ResourceKind::Qubits,
        20,
        18,
        LimitKnob::StatevectorQubits,
        LimitSource::Request,
    );
    let text = err.to_string();
    assert!(text.contains("set by this request's limits"), "{text}");
    assert!(text.contains("Limits::max_statevector_qubits"), "{text}");
    assert!(!text.contains("PRISM_MAX_SV_QUBITS"), "{text}");
    let PrismError::ResourceLimit(data) = err else {
        panic!("expected ResourceLimit");
    };
    assert_eq!(data.env_var, None);
    assert_eq!(data.knob, Some(LimitKnob::StatevectorQubits));
}

#[test]
fn a_detected_error_names_the_setter_and_the_variable() {
    let text = resource_limit_error(
        "statevector",
        "the circuit",
        ResourceKind::Qubits,
        40,
        30,
        LimitKnob::StatevectorQubits,
        LimitSource::Detected,
    )
    .to_string();
    assert!(text.contains("on this machine"), "{text}");
    assert!(text.contains("Limits::max_statevector_qubits"), "{text}");
    assert!(text.contains("PRISM_MAX_SV_QUBITS"), "{text}");
}
