use prism_q::qec::observable_reroute::{
    min_cone_z_representative, xor_z_support, ObservableRerouteResult,
};
use prism_q::{run_spd_observable, run_spd_observable_light_cone, Circuit, Gate, PauliTerm};
use std::collections::HashSet;

fn z_terms(support: &[usize]) -> Vec<PauliTerm> {
    let mut out: Vec<_> = support.iter().copied().map(PauliTerm::z).collect();
    out.sort_by_key(|t| t.qubit);
    out
}

fn reroute_fixture(
    circuit: &Circuit,
    observable: &[usize],
    stabilizers: &[Vec<usize>],
) -> ObservableRerouteResult {
    min_cone_z_representative(circuit, observable, stabilizers)
        .expect("reroute fixture should stay within production search limits")
}

fn reroute_localized_t_circuit(depth: usize) -> Circuit {
    let mut c = Circuit::new(4, 0);
    for _ in 0..depth {
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::S, &[0]);
    }
    c
}

fn reroute_uniform_t_circuit(depth: usize) -> Circuit {
    let mut c = Circuit::new(4, 0);
    for _ in 0..depth {
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[3]);
    }
    c
}

fn strip_qubit(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

fn strip_support(rows: usize, cols: usize, col: usize) -> Vec<usize> {
    (0..rows).map(|row| strip_qubit(row, col, cols)).collect()
}

fn strip_chain_stabilizers(rows: usize, cols: usize) -> Vec<Vec<usize>> {
    (0..cols.saturating_sub(1))
        .map(|col| {
            let mut support = strip_support(rows, cols, col);
            support.extend(strip_support(rows, cols, col + 1));
            support.sort_unstable();
            support
        })
        .collect()
}

fn surface_site_t_circuit(
    rows: usize,
    cols: usize,
    depth: usize,
    t_sites: &[(usize, usize)],
) -> Circuit {
    let mut c = Circuit::new(rows * cols, 0);
    let t_sites: HashSet<(usize, usize)> = t_sites.iter().copied().collect();
    for _ in 0..depth {
        for col in 0..cols {
            for row in 0..rows {
                let q = strip_qubit(row, col, cols);
                c.add_gate(Gate::S, &[q]);
                if t_sites.contains(&(row, col)) {
                    c.add_gate(Gate::T, &[q]);
                }
            }
            for row in 0..rows.saturating_sub(1) {
                c.add_gate(
                    Gate::Cz,
                    &[strip_qubit(row, col, cols), strip_qubit(row + 1, col, cols)],
                );
            }
        }
    }
    c
}

fn column_sites(rows: usize, col: usize) -> Vec<(usize, usize)> {
    (0..rows).map(|row| (row, col)).collect()
}

fn coset_patchy_sites(rows: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    out.extend(column_sites(rows, 0));
    out.extend(
        [0usize, 2, 4]
            .into_iter()
            .filter(|&row| row < rows)
            .map(|row| (row, 1)),
    );
    out.extend(
        [1usize, 3]
            .into_iter()
            .filter(|&row| row < rows)
            .map(|row| (row, 2)),
    );
    out
}

fn coset_residual_sites(rows: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    out.extend(column_sites(rows, 0));
    out.extend(column_sites(rows, 1));
    out.extend(
        [(1usize, 2), (3usize, 3)]
            .into_iter()
            .filter(|&(row, _)| row < rows),
    );
    out
}

#[test]
fn rerouting_invariance_reduces_localized_t_cone() {
    let circuit = reroute_localized_t_circuit(8);
    let route = reroute_fixture(&circuit, &[0], &[vec![0, 3]]);
    assert_eq!(route.rerouted_support, vec![3]);
    assert!(route.rerouted.t_in_cone < route.original.t_in_cone);

    let original_result = run_spd_observable_light_cone(&circuit, &z_terms(&[0]), 0.0, 0).unwrap();
    let rerouted_result =
        run_spd_observable_light_cone(&circuit, &z_terms(&route.rerouted_support), 0.0, 0).unwrap();
    assert!((original_result.mean - rerouted_result.mean).abs() < 1e-10);
}

#[test]
fn rerouting_noop_when_all_representatives_hit_t_sites() {
    let circuit = reroute_uniform_t_circuit(8);
    let route = reroute_fixture(&circuit, &[0], &[vec![0, 3]]);
    assert_eq!(route.rerouted_support, route.original_support);
    assert_eq!(route.rerouted.t_in_cone, route.original.t_in_cone);
    assert_eq!(route.rerouted.gates_in_cone, route.original.gates_in_cone);
}

#[test]
fn overlapping_coset_patchy_rerouting_requires_three_stabilizers() {
    let rows = 5usize;
    let cols = 4usize;
    let circuit = surface_site_t_circuit(rows, cols, 6, &coset_patchy_sites(rows));
    let observable = strip_support(rows, cols, 0);
    let stabilizers = strip_chain_stabilizers(rows, cols);
    let route = reroute_fixture(&circuit, &observable, &stabilizers);

    assert_eq!(route.rerouted_support, strip_support(rows, cols, 3));
    assert_eq!(route.rerouted.t_in_cone, 0);
    assert!(route.rerouted.t_in_cone < route.original.t_in_cone);
    for mask in 0..(1usize << stabilizers.len()) {
        if mask.count_ones() >= 3 {
            continue;
        }
        let mut candidate = observable.clone();
        for (idx, stabilizer) in stabilizers.iter().enumerate() {
            if (mask >> idx) & 1 == 1 {
                candidate = xor_z_support(&candidate, stabilizer);
            }
        }
        assert_ne!(candidate, route.rerouted_support);
    }

    let original_result =
        run_spd_observable_light_cone(&circuit, &z_terms(&observable), 0.0, 0).unwrap();
    let rerouted_result =
        run_spd_observable_light_cone(&circuit, &z_terms(&route.rerouted_support), 0.0, 0).unwrap();
    assert!((original_result.mean - rerouted_result.mean).abs() < 1e-10);
}

#[test]
fn overlapping_coset_adversarial_keeps_residual_t() {
    let rows = 5usize;
    let cols = 4usize;
    let circuit = surface_site_t_circuit(rows, cols, 6, &coset_residual_sites(rows));
    let observable = strip_support(rows, cols, 0);
    let stabilizers = strip_chain_stabilizers(rows, cols);
    let route = reroute_fixture(&circuit, &observable, &stabilizers);

    assert_eq!(route.rerouted_support, strip_support(rows, cols, 2));
    assert!(route.rerouted.t_in_cone > 0);
    assert!(route.rerouted.t_in_cone < route.original.t_in_cone);
}

#[test]
fn unrestricted_light_cone_and_rerouted_light_cone_agree() {
    let circuit = reroute_localized_t_circuit(6);
    let route = reroute_fixture(&circuit, &[0], &[vec![0, 3]]);
    let original_terms = z_terms(&[0]);
    let rerouted_terms = z_terms(&route.rerouted_support);

    let unrestricted = run_spd_observable(&circuit, &original_terms, 0.0, 0).unwrap();
    let light_cone = run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap();
    let rerouted_light_cone =
        run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap();

    assert!((unrestricted.mean - light_cone.mean).abs() < 1e-10);
    assert!((unrestricted.mean - rerouted_light_cone.mean).abs() < 1e-10);
}
