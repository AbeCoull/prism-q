//! T-gate strategy benchmark harness.
//!
//! Run with:
//!   cargo bench --bench qec_t_strategies --features parallel
//!
//! Groups:
//! - `qec_t_strategies/<strategy>/<fixture>`: wall-clock to sample the
//!   shot budget for that fixture under that strategy.
//! - `qec_t_strategies_scaling`: production T-count scaling for Auto, SPD,
//!   and CAMPS.
//! - `spd_light_cone`: production SPD cone-skip scaling.
//!
//! Reference and internal sweeps are opt-in benchmarks. Enable them with
//! `--features parallel,bench-internal`.

#[cfg(feature = "bench-internal")]
use criterion::{black_box, measurement::WallTime, BenchmarkGroup};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "bench-internal")]
use prism_q::qec::cut_selection::{cut_score, min_fill_treewidth_proxy, InteractionGraph};
#[cfg(feature = "bench-internal")]
use prism_q::qec::observable_reroute::{cone_telemetry, min_cone_z_representative, xor_z_support};
use prism_q::{
    run_qec_program_with_strategy, run_spd_observable, run_spd_observable_light_cone, Circuit,
    Gate, PauliTerm, QecOptions, QecProgram, QecRecordRef, QecTStrategy,
};
#[cfg(feature = "bench-internal")]
use std::collections::HashSet;
use std::time::Duration;

const SEED: u64 = 0xDEAD_BEEF;

fn options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(2048),
        keep_measurements: false,
    }
}

fn one_qubit_clifford_t_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(1, options(shots));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

fn two_qubit_single_t_per_path(shots: usize) -> QecProgram {
    let mut p = QecProgram::with_options(2, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    p.push_gate(Gate::T, &[0]).unwrap();
    p.push_gate(Gate::H, &[0]).unwrap();
    p.push_gate(Gate::H, &[1]).unwrap();
    p.push_gate(Gate::T, &[1]).unwrap();
    p.push_gate(Gate::H, &[1]).unwrap();
    let m0 = p.measure_z(0).unwrap();
    let m1 = p.measure_z(1).unwrap();
    p.observable_include(0, &[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();
    p
}

fn cx_t_zz_program(shots: usize) -> QecProgram {
    let mut p = QecProgram::with_options(2, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    p.push_gate(Gate::Cx, &[0, 1]).unwrap();
    p.push_gate(Gate::T, &[0]).unwrap();
    let m0 = p.measure_z(0).unwrap();
    let m1 = p.measure_z(1).unwrap();
    p.observable_include(0, &[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();
    p
}

fn multi_t_single_path_program(shots: usize) -> QecProgram {
    t_chain_program(shots, 2)
}

/// Single-qubit `(H*T)^t*H` rotation. T-count `t` parametrizes the
/// scaling fixture used to compare strategies as the non-Clifford
/// budget grows. The logical observable is Z on the final state.
fn t_chain_program(shots: usize, t_count: usize) -> QecProgram {
    let mut p = QecProgram::with_options(1, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    for _ in 0..t_count {
        p.push_gate(Gate::T, &[0]).unwrap();
        p.push_gate(Gate::H, &[0]).unwrap();
    }
    let m0 = p.measure_z(0).unwrap();
    p.observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    p
}

type FixtureBuilder = fn(usize) -> QecProgram;

struct FixtureSpec {
    label: &'static str,
    builder: FixtureBuilder,
    skipped: &'static [QecTStrategy],
}

fn fixtures() -> Vec<FixtureSpec> {
    vec![
        FixtureSpec {
            label: "h_t_h_1q",
            builder: one_qubit_clifford_t_program,
            skipped: &[],
        },
        FixtureSpec {
            label: "two_qubit_single_t",
            builder: two_qubit_single_t_per_path,
            skipped: &[],
        },
        FixtureSpec {
            label: "cx_t_zz",
            builder: cx_t_zz_program,
            skipped: &[],
        },
        FixtureSpec {
            label: "h_t_t_h_1q",
            builder: multi_t_single_path_program,
            skipped: &[],
        },
    ]
}

fn strategies() -> Vec<QecTStrategy> {
    #[cfg(feature = "bench-internal")]
    {
        vec![
            QecTStrategy::Auto,
            QecTStrategy::Spd,
            QecTStrategy::Camps,
            QecTStrategy::Reference,
        ]
    }
    #[cfg(not(feature = "bench-internal"))]
    {
        vec![QecTStrategy::Auto, QecTStrategy::Spd, QecTStrategy::Camps]
    }
}

fn bench_qec_t_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_t_strategies");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_secs(3));

    let shots = 1_000usize;
    for fixture in fixtures() {
        for strategy in strategies() {
            if fixture.skipped.contains(&strategy) {
                continue;
            }
            let program = (fixture.builder)(shots);
            let id = BenchmarkId::new(strategy.label(), fixture.label);
            group.bench_function(id, |b| {
                b.iter(|| {
                    run_qec_program_with_strategy(&program, strategy)
                        .expect("strategy must succeed on benchmark fixture")
                })
            });
        }
    }
    group.finish();
}

/// T-scaling sweep: single-qubit `(H*T)^t*H` over several T counts.
fn bench_qec_t_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_t_strategies_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_secs(3));

    let shots = 1_000usize;
    let t_counts = [1usize, 2, 4, 8, 12, 16, 20, 24, 32];
    let strategies = strategies();

    for &t in &t_counts {
        let program = t_chain_program(shots, t);
        for &strategy in &strategies {
            let id = BenchmarkId::new(strategy.label(), format!("t{t}"));
            group.bench_function(id, |b| {
                b.iter(|| {
                    run_qec_program_with_strategy(&program, strategy)
                        .expect("strategy must succeed on scaling fixture")
                })
            });
        }
    }
    group.finish();
}

fn local_observable_with_disjoint_t_block(t_outside: usize) -> Circuit {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    for _ in 0..t_outside {
        c.add_gate(Gate::H, &[3]);
        c.add_gate(Gate::T, &[3]);
        c.add_gate(Gate::Cx, &[3, 4]);
        c.add_gate(Gate::T, &[4]);
        c.add_gate(Gate::Cx, &[4, 5]);
        c.add_gate(Gate::T, &[5]);
    }
    c
}

#[cfg(feature = "bench-internal")]
fn telemetry_label(
    base: &str,
    circuit: &Circuit,
    observable: &[PauliTerm],
    peak_terms: usize,
    total_discarded: f64,
) -> String {
    let t = cone_telemetry(circuit, observable);
    format!(
        "{base}_gt{}_gc{}_tt{}_tc{}_k{}_discarded{:.3e}",
        t.gates_total, t.gates_in_cone, t.t_total, t.t_in_cone, peak_terms, total_discarded
    )
}

#[cfg(feature = "bench-internal")]
fn local_observable_with_local_t_block(depth: usize) -> Circuit {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for _ in 0..depth {
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cx, &[1, 0]);
    }
    c.add_gate(Gate::H, &[0]);
    c
}

#[cfg(feature = "bench-internal")]
fn local_observable_with_uniform_t(depth: usize) -> Circuit {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::H, &[0]);
    for _ in 0..depth {
        for q in 0..6 {
            c.add_gate(Gate::T, &[q]);
        }
        for q in 0..5 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c.add_gate(Gate::H, &[0]);
    c
}

#[cfg(feature = "bench-internal")]
fn strip_qubit(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

#[cfg(feature = "bench-internal")]
fn strip_support(rows: usize, cols: usize, col: usize) -> Vec<usize> {
    (0..rows).map(|row| strip_qubit(row, col, cols)).collect()
}

#[cfg(feature = "bench-internal")]
fn strip_swap_stabilizers(rows: usize, cols: usize, base_col: usize) -> Vec<Vec<usize>> {
    let base = strip_support(rows, cols, base_col);
    (0..cols)
        .filter(|&col| col != base_col)
        .map(|col| {
            let mut support = base.clone();
            support.extend(strip_support(rows, cols, col));
            support.sort_unstable();
            support
        })
        .collect()
}

#[cfg(feature = "bench-internal")]
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

#[cfg(feature = "bench-internal")]
fn surface_strip_t_circuit(rows: usize, cols: usize, depth: usize, t_cols: &[usize]) -> Circuit {
    let mut c = Circuit::new(rows * cols, 0);
    let t_cols: HashSet<usize> = t_cols.iter().copied().collect();
    for _ in 0..depth {
        for col in 0..cols {
            for row in 0..rows {
                let q = strip_qubit(row, col, cols);
                c.add_gate(Gate::S, &[q]);
                if t_cols.contains(&col) {
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

#[cfg(feature = "bench-internal")]
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

#[cfg(feature = "bench-internal")]
fn stabilizer_preserving_branch_circuit(rows: usize, depth: usize) -> Circuit {
    let cols = 2usize;
    let mut c = Circuit::new(rows * cols, 0);
    for _ in 0..depth {
        for row in 0..rows {
            let hot = strip_qubit(row, 0, cols);
            let cold = strip_qubit(row, 1, cols);
            c.add_gate(Gate::Cx, &[hot, cold]);
            c.add_gate(Gate::H, &[hot]);
            c.add_gate(Gate::T, &[hot]);
            c.add_gate(Gate::H, &[hot]);
            c.add_gate(Gate::Cx, &[hot, cold]);
        }
    }
    c
}

#[cfg(feature = "bench-internal")]
fn column_sites(rows: usize, col: usize) -> Vec<(usize, usize)> {
    (0..rows).map(|row| (row, col)).collect()
}

#[cfg(feature = "bench-internal")]
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

#[cfg(feature = "bench-internal")]
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

#[cfg(feature = "bench-internal")]
fn z_terms(support: &[usize]) -> Vec<PauliTerm> {
    let mut out: Vec<_> = support.iter().copied().map(PauliTerm::z).collect();
    out.sort_by_key(|t| t.qubit);
    out
}

#[cfg(feature = "bench-internal")]
fn support_label(support: &[usize]) -> String {
    support
        .iter()
        .map(|q| q.to_string())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(feature = "bench-internal")]
fn reroute_localized_t_circuit(depth: usize) -> Circuit {
    let mut c = Circuit::new(4, 0);
    for _ in 0..depth {
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::S, &[0]);
    }
    c
}

#[cfg(feature = "bench-internal")]
fn reroute_uniform_t_circuit(depth: usize) -> Circuit {
    let mut c = Circuit::new(4, 0);
    for _ in 0..depth {
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[3]);
    }
    c
}

#[cfg(feature = "bench-internal")]
fn path_graph_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n.saturating_sub(1) {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

#[cfg(feature = "bench-internal")]
fn grid_graph_circuit(side: usize) -> Circuit {
    let mut c = Circuit::new(side * side, 0);
    for r in 0..side {
        for col in 0..side {
            let q = r * side + col;
            if col + 1 < side {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
            if r + 1 < side {
                c.add_gate(Gate::Cx, &[q, q + side]);
            }
        }
    }
    c
}

#[cfg(feature = "bench-internal")]
fn complete_graph_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for a in 0..n {
        for b in (a + 1)..n {
            c.add_gate(Gate::Cz, &[a, b]);
        }
    }
    c
}

#[cfg(feature = "bench-internal")]
fn bench_reroute_coset_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    circuit: Circuit,
    observable: Vec<usize>,
    stabilizers: Vec<Vec<usize>>,
) {
    let route = min_cone_z_representative(&circuit, &observable, &stabilizers).unwrap();
    let original_terms = z_terms(&route.original_support);
    let rerouted_terms = z_terms(&route.rerouted_support);
    let unrestricted = run_spd_observable(&circuit, &original_terms, 0.0, 0).unwrap();
    let original = run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap();
    let rerouted = run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap();
    assert!((unrestricted.mean - original.mean).abs() < 1e-10);
    assert!((unrestricted.mean - rerouted.mean).abs() < 1e-10);
    let label = format!(
        "{name}_full_k{}_orig_tc{}_gc{}_k{}_discarded{:.3e}_reroute_tc{}_gc{}_k{}_discarded{:.3e}_support{}",
        unrestricted.peak_terms,
        route.original.t_in_cone,
        route.original.gates_in_cone,
        original.peak_terms,
        original.total_discarded,
        route.rerouted.t_in_cone,
        route.rerouted.gates_in_cone,
        rerouted.peak_terms,
        rerouted.total_discarded,
        support_label(&route.rerouted_support)
    );
    group.bench_function(
        BenchmarkId::new("reroute_coset_unrestricted_spd", label.clone()),
        |b| b.iter(|| run_spd_observable(&circuit, &original_terms, 0.0, 0).unwrap()),
    );
    group.bench_function(
        BenchmarkId::new("reroute_coset_original_lc_spd", label.clone()),
        |b| b.iter(|| run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap()),
    );
    group.bench_function(
        BenchmarkId::new("reroute_coset_rerouted_lc_spd", label),
        |b| b.iter(|| run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap()),
    );
}

#[cfg(feature = "bench-internal")]
fn bench_reroute_branch_preserving_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    rows: usize,
    depth: usize,
) {
    let cols = 2usize;
    let name = format!("reroute_branch_preserving_r{rows}_d{depth}");
    let circuit = stabilizer_preserving_branch_circuit(rows, depth);
    let observable = strip_support(rows, cols, 0);
    let alternate_support = strip_support(rows, cols, 1);
    let stabilizers = strip_swap_stabilizers(rows, cols, 0);
    let route = min_cone_z_representative(&circuit, &observable, &stabilizers).unwrap();
    let original_terms = z_terms(&observable);
    let alternate_terms = z_terms(&alternate_support);
    let stabilizer_terms = z_terms(&xor_z_support(&observable, &alternate_support));
    let original = run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap();
    let alternate = run_spd_observable_light_cone(&circuit, &alternate_terms, 0.0, 0).unwrap();
    let stabilizer = run_spd_observable(&circuit, &stabilizer_terms, 0.0, 0).unwrap();
    let original_unrestricted = run_spd_observable(&circuit, &original_terms, 0.0, 0).unwrap();
    let alternate_telemetry = cone_telemetry(&circuit, &alternate_terms);

    assert!((original_unrestricted.mean - original.mean).abs() < 1e-10);
    assert!((original.mean - alternate.mean).abs() < 1e-10);
    assert!((stabilizer.mean - 1.0).abs() < 1e-10);
    assert_eq!(route.rerouted_support, observable);
    assert_eq!(route.original.t_in_cone, alternate_telemetry.t_in_cone);
    assert!(original.peak_terms > 1);
    assert!(alternate.peak_terms > 1);

    let label = format!(
        "{name}_orig_tc{}_gc{}_k{}_alt_tc{}_gc{}_k{}_stab_k{}_support{}",
        route.original.t_in_cone,
        route.original.gates_in_cone,
        original.peak_terms,
        alternate_telemetry.t_in_cone,
        alternate_telemetry.gates_in_cone,
        alternate.peak_terms,
        stabilizer.peak_terms,
        support_label(&alternate_support)
    );
    group.bench_function(
        BenchmarkId::new("reroute_branch_unrestricted_spd", label.clone()),
        |b| b.iter(|| run_spd_observable(&circuit, &original_terms, 0.0, 0).unwrap()),
    );
    group.bench_function(
        BenchmarkId::new("reroute_branch_original_lc_spd", label.clone()),
        |b| b.iter(|| run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap()),
    );
    group.bench_function(
        BenchmarkId::new("reroute_branch_alternate_lc_spd", label),
        |b| b.iter(|| run_spd_observable_light_cone(&circuit, &alternate_terms, 0.0, 0).unwrap()),
    );
}

fn bench_spd_light_cone(c: &mut Criterion) {
    let mut group = c.benchmark_group("spd_light_cone");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500));

    for &t_outside in &[2usize, 4, 8, 16] {
        let circuit = local_observable_with_disjoint_t_block(t_outside);
        let obs = [PauliTerm::z(0)];

        group.bench_with_input(BenchmarkId::new("unrestricted", t_outside), &(), |b, _| {
            b.iter(|| run_spd_observable(&circuit, &obs, 0.0, 0).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("light_cone", t_outside), &(), |b, _| {
            b.iter(|| run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap())
        });
    }

    group.finish();
}

#[cfg(feature = "bench-internal")]
fn bench_qec_internal_sweeps(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_internal_sweeps");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    let obs0 = [PauliTerm::z(0)];

    for (name, circuit) in [
        ("spd_disjoint", local_observable_with_disjoint_t_block(8)),
        ("spd_localized", local_observable_with_local_t_block(6)),
        ("spd_uniform", local_observable_with_uniform_t(3)),
    ] {
        let full = run_spd_observable(&circuit, &obs0, 0.0, 0).unwrap();
        let cone = run_spd_observable_light_cone(&circuit, &obs0, 0.0, 0).unwrap();
        assert!((full.mean - cone.mean).abs() < 1e-10);
        let unrestricted_label = telemetry_label(
            &format!("{name}_spd"),
            &circuit,
            &obs0,
            full.peak_terms,
            full.total_discarded,
        );
        let cone_label = telemetry_label(
            &format!("{name}_lc"),
            &circuit,
            &obs0,
            cone.peak_terms,
            cone.total_discarded,
        );
        group.bench_function(
            BenchmarkId::new("spd_unrestricted", unrestricted_label),
            |b| b.iter(|| run_spd_observable(&circuit, &obs0, 0.0, 0).unwrap()),
        );
        group.bench_function(BenchmarkId::new("spd_light_cone", cone_label), |b| {
            b.iter(|| run_spd_observable_light_cone(&circuit, &obs0, 0.0, 0).unwrap())
        });
    }

    for (name, circuit) in [
        ("reroute_localized", reroute_localized_t_circuit(16)),
        ("reroute_uniform_noop", reroute_uniform_t_circuit(8)),
    ] {
        let route = min_cone_z_representative(&circuit, &[0], &[vec![0, 3]]).unwrap();
        let original_terms = z_terms(&route.original_support);
        let rerouted_terms = z_terms(&route.rerouted_support);
        let original = run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap();
        let rerouted = run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap();
        assert!((original.mean - rerouted.mean).abs() < 1e-10);
        let label = format!(
            "{name}_orig_tc{}_gc{}_k{}_discarded{:.3e}_reroute_tc{}_gc{}_k{}_discarded{:.3e}_support{}",
            route.original.t_in_cone,
            route.original.gates_in_cone,
            original.peak_terms,
            original.total_discarded,
            route.rerouted.t_in_cone,
            route.rerouted.gates_in_cone,
            rerouted.peak_terms,
            rerouted.total_discarded,
            support_label(&route.rerouted_support)
        );
        group.bench_function(
            BenchmarkId::new("reroute_original_light_cone_spd", label.clone()),
            |b| {
                b.iter(|| run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap())
            },
        );
        group.bench_function(
            BenchmarkId::new("reroute_rerouted_light_cone_spd", label),
            |b| {
                b.iter(|| run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap())
            },
        );
    }

    for (name, circuit, observable, stabilizers) in [
        (
            "strip_localized",
            surface_strip_t_circuit(5, 3, 12, &[0]),
            strip_support(5, 3, 0),
            strip_swap_stabilizers(5, 3, 0),
        ),
        (
            "strip_uniform_noop",
            surface_strip_t_circuit(5, 3, 8, &[0, 1, 2]),
            strip_support(5, 3, 0),
            strip_swap_stabilizers(5, 3, 0),
        ),
    ] {
        let route = min_cone_z_representative(&circuit, &observable, &stabilizers).unwrap();
        let original_terms = z_terms(&route.original_support);
        let rerouted_terms = z_terms(&route.rerouted_support);
        let original = run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap();
        let rerouted = run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap();
        assert!((original.mean - rerouted.mean).abs() < 1e-10);
        let label = format!(
            "{name}_orig_tc{}_gc{}_k{}_discarded{:.3e}_reroute_tc{}_gc{}_k{}_discarded{:.3e}_support{}",
            route.original.t_in_cone,
            route.original.gates_in_cone,
            original.peak_terms,
            original.total_discarded,
            route.rerouted.t_in_cone,
            route.rerouted.gates_in_cone,
            rerouted.peak_terms,
            rerouted.total_discarded,
            support_label(&route.rerouted_support)
        );
        group.bench_function(
            BenchmarkId::new("strip_original_lc_spd", label.clone()),
            |b| {
                b.iter(|| run_spd_observable_light_cone(&circuit, &original_terms, 0.0, 0).unwrap())
            },
        );
        group.bench_function(BenchmarkId::new("strip_rerouted_lc_spd", label), |b| {
            b.iter(|| run_spd_observable_light_cone(&circuit, &rerouted_terms, 0.0, 0).unwrap())
        });
    }

    bench_reroute_coset_case(
        &mut group,
        "coset_localized_multihop",
        surface_strip_t_circuit(5, 4, 10, &[0, 1]),
        strip_support(5, 4, 0),
        strip_chain_stabilizers(5, 4),
    );
    bench_reroute_coset_case(
        &mut group,
        "coset_patchy_threehop",
        surface_site_t_circuit(5, 4, 10, &coset_patchy_sites(5)),
        strip_support(5, 4, 0),
        strip_chain_stabilizers(5, 4),
    );
    bench_reroute_coset_case(
        &mut group,
        "coset_uniform_noop",
        surface_strip_t_circuit(5, 4, 6, &[0, 1, 2, 3]),
        strip_support(5, 4, 0),
        strip_chain_stabilizers(5, 4),
    );
    bench_reroute_coset_case(
        &mut group,
        "coset_residual",
        surface_site_t_circuit(5, 4, 10, &coset_residual_sites(5)),
        strip_support(5, 4, 0),
        strip_chain_stabilizers(5, 4),
    );
    bench_reroute_branch_preserving_case(&mut group, 3, 3);

    for (name, circuit, cut) in [
        ("cut_path", path_graph_circuit(16), vec![8usize]),
        ("cut_grid", grid_graph_circuit(4), vec![5usize, 6, 9, 10]),
        ("cut_complete", complete_graph_circuit(8), vec![0usize, 1]),
    ] {
        let graph = InteractionGraph::from_circuit(&circuit);
        let excluded: HashSet<usize> = cut.into_iter().collect();
        let width = min_fill_treewidth_proxy(&graph, &HashSet::new());
        let cut_width = min_fill_treewidth_proxy(&graph, &excluded);
        let score = cut_score(&graph, &excluded, 1.0);
        let label = format!(
            "{name}_edges{}_w{}_cutw{}_score{score:.3}",
            graph.num_edges(),
            width,
            cut_width
        );
        group.bench_function(BenchmarkId::new("cut_width_proxy", label), |b| {
            b.iter(|| {
                let graph = black_box(&graph);
                let excluded = black_box(&excluded);
                (
                    min_fill_treewidth_proxy(graph, &HashSet::new()),
                    min_fill_treewidth_proxy(graph, excluded),
                    cut_score(graph, excluded, 1.0),
                )
            })
        });
    }

    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
criterion_group!(
    qec_t_strategy_benches,
    bench_qec_t_strategies,
    bench_qec_t_scaling,
    bench_spd_light_cone
);

#[cfg(feature = "bench-internal")]
criterion_group!(
    qec_t_strategy_benches,
    bench_qec_t_strategies,
    bench_qec_t_scaling,
    bench_spd_light_cone,
    bench_qec_internal_sweeps
);
criterion_main!(qec_t_strategy_benches);
