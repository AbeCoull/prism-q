//! Register census behind the `auto/clifford_t_price` rows, printed for each fixture.
//!
//! `price` prints, per terminal fixture, the register's width, rotation work, reduction
//! passes and parts in the product gauge, the route `Auto` takes and whether pricing
//! the rule by width would need a tableau dry run; per mid-circuit fixture, the branch
//! count and the inner products one shot costs. `seeds` prints the split over 20 seeds
//! per fixture shape (the bench seed plus 0 to 19): circuits each option admits at a
//! dense cap, the widest part, the cheapest cut of it in two along sets that excess
//! rotations link, and median log2 work at 1024 shots with a dense part at 40 passes.
//!
//! ```text
//! cargo run --release --features "parallel bench-internal" --example clifford_t_census -- [price|seeds]
//! ```

use prism_q::circuit::Circuit;
use prism_q::sim::{self, ResolvedBackend, RouteRule};
use prism_q::{BackendKind, circuits};

#[allow(dead_code)]
#[path = "../benches/common/fixtures.rs"]
mod bench_fixtures;

const SEED: u64 = 0xDEAD_BEEF;

fn price_layers(n: usize, t: usize) -> Circuit {
    bench_fixtures::random_clifford_t_layers(n, 10, t, SEED)
}

fn price_scrambled(n: usize, t: usize) -> Circuit {
    let mut circuit = bench_fixtures::scrambled_clifford_t_circuit(n, t, SEED);
    circuit.measure_all();
    circuit
}

fn price_distinct() -> Circuit {
    bench_fixtures::distinct_t_layers(17, 10, 16, SEED)
}

fn price_onelayer() -> Circuit {
    let mut circuit = bench_fixtures::random_t_insertion_circuit(20, 80, SEED);
    circuit.measure_all();
    circuit
}

fn shot_plan(kind: BackendKind, circuit: &Circuit, shots: usize) -> (ResolvedBackend, RouteRule) {
    let plan = sim::simulate(circuit)
        .backend(kind)
        .seed(42)
        .plan(sim::Query::Shots(shots))
        .unwrap();
    (plan.backend, plan.rule)
}

fn auto_mid_rows() -> Vec<(String, Circuit, usize)> {
    let mut rows = Vec::new();
    for n in [22, 24, 26] {
        let circuit = bench_fixtures::clifford_t_layers_mid_circuit(n, 10, 8, SEED);
        rows.push((format!("layers_{n}q_t8"), circuit, 64));
    }
    for n in [26, 32] {
        let circuit = bench_fixtures::stabilizer_rank_mid_circuit_shot_circuit(n, 6, SEED);
        rows.push((format!("mid_{n}q_t6_16shots"), circuit, 16));
    }
    for n in [24, 32] {
        for t in [6, 8] {
            let circuit = bench_fixtures::guarded_s_mid_circuit(n, t, SEED);
            rows.push((format!("guarded_s_{n}q_t{t}"), circuit, 4));
        }
    }
    rows
}

fn main() {
    let which = std::env::args().nth(1);
    match which.as_deref() {
        None => {
            price_census();
            println!();
            split_seed_census();
        }
        Some("price") => price_census(),
        Some("seeds") => split_seed_census(),
        Some(other) => {
            eprintln!("unknown census {other:?}; expected price or seeds");
            std::process::exit(2);
        }
    }
}

fn price_census() {
    use prism_q::{Gate, Instruction};

    const REGISTER_PASS_STATEVECTOR_AMPLITUDES: f64 = 25.0;
    const MAX_REGISTER_QUBITS: usize = 20;

    let mut terminal = Vec::new();
    for (n, t) in [
        (20, 40),
        (22, 24),
        (24, 24),
        (26, 10),
        (26, 20),
        (26, 24),
        (28, 24),
        (30, 24),
        (32, 24),
        (26, 40),
        (30, 40),
        (32, 40),
    ] {
        terminal.push((format!("layers_{n}q_t{t}"), price_layers(n, t)));
    }
    terminal.push((
        "local_layers_64q_t40".into(),
        bench_fixtures::local_clifford_t_layers(64, 10, 40, SEED),
    ));
    for n in [16, 20, 24] {
        terminal.push((
            format!("clifford_t_shots/{n}q_t10"),
            bench_fixtures::random_clifford_t_layers(n, 10, 10, SEED),
        ));
    }
    for (n, t) in [
        (20, 20),
        (22, 20),
        (24, 20),
        (26, 16),
        (26, 18),
        (26, 20),
        (32, 20),
    ] {
        terminal.push((format!("scrambled_{n}q_t{t}"), price_scrambled(n, t)));
    }
    terminal.push(("onelayer_20q_t80".into(), price_onelayer()));
    terminal.push(("distinct_17q_t16".into(), price_distinct()));
    let mut counts_row = circuits::clifford_t_circuit(40, 20, 0.05, SEED);
    counts_row.measure_all();
    terminal.push(("clifford_register_counts/40q_d20_t5pct".into(), counts_row));

    println!(
        "{:<40} {:>3} {:>3} {:>3} {:>4} {:>8} {:>6} {:>3} {:>7} {:>5} {:>4} {:>5} {:>5} {:>6}  auto route",
        "fixture",
        "n",
        "t",
        "k",
        "e",
        "R/2^k",
        "passes",
        "r",
        "dry_run",
        "parts",
        "kmax",
        "dense",
        "split",
        "pairs"
    );
    let mut split_lines = Vec::new();
    for (label, circuit) in &terminal {
        let census = prism_q::clifford_register_census(circuit).unwrap();
        let parts = census
            .split
            .as_ref()
            .expect("census fixtures hold at most 64 anchors");
        let (n, t, k) = (circuit.num_qubits, census.t_count, census.width);
        let m = circuit.measurement_map().len();
        let instructions = circuit.instructions.len() as f64;
        let w_alt: f64 = circuit
            .independent_subsystems()
            .iter()
            .map(|block| instructions * 2f64.powi(block.len() as i32))
            .sum();
        let w_reg = |k: usize, rotation_amps: f64| {
            let reduction = 3.0 * (k.min(m) + 1) as f64 * 2f64.powi(k as i32);
            REGISTER_PASS_STATEVECTOR_AMPLITUDES * (rotation_amps + reduction)
        };
        let kb = t.min(n);
        let bound_admits = kb <= MAX_REGISTER_QUBITS
            && w_reg(kb, (t - kb + 2) as f64 * 2f64.powi(kb as i32)) < w_alt;
        let k_stop = (0..=kb.min(MAX_REGISTER_QUBITS))
            .rev()
            .find(|&k| w_reg(k, 2.0 * 2f64.powi(k as i32)) < w_alt);
        let dry_run = !bound_admits && k_stop.is_some();
        let (backend, rule) = shot_plan(BackendKind::Auto, circuit, 1024);
        let show = |v: Option<usize>| v.map_or("-".to_string(), |v| v.to_string());
        // log2 of the rotation work `(e + 2) 2^k`, for the one register and summed over
        // its parts in the product gauge.
        let dense = ((t - k + 2) as f64).log2() + k as f64;
        let split = parts
            .parts
            .iter()
            .map(|p| (p.excess + 2) as f64 * 2f64.powi(p.width as i32))
            .sum::<f64>()
            .log2();
        let kmax = parts.parts.first().map_or(0, |p| p.width);
        println!(
            "{:<40} {:>3} {:>3} {:>3} {:>4} {:>8.2} {:>6} {:>3} {:>7} {:>5} {:>4} {:>5.1} {:>5.1} {:>6}  {backend:?}/{rule:?}",
            label,
            n,
            t,
            k,
            t as i64 - k as i64,
            census.rotation_work as f64 / 2f64.powi(k as i32),
            show(census.reduction_passes),
            show(census.register_rows),
            dry_run,
            parts.parts.len(),
            kmax,
            dense,
            split,
            parts.row_pair_checks,
        );
        let widest: Vec<String> = parts
            .parts
            .iter()
            .filter(|p| p.width > 1 || p.excess > 0)
            .take(6)
            .map(|p| format!("{}w{}e{}r", p.width, p.excess, p.rows))
            .collect();
        let singles = parts
            .parts
            .iter()
            .filter(|p| p.width == 1 && p.excess == 0)
            .count();
        // The widest part cut in two when that is cheaper at 1024 shots, a dense part at
        // 40 passes: sides, row pairs across, and log2 work for the whole register.
        let dense_part = |w: usize| 40.0 * 2f64.powi(w as i32);
        let split_work: f64 = parts.parts.iter().map(|p| dense_part(p.width)).sum();
        let cut = match parts.cut {
            Some(cut) if cut.work < dense_part(kmax) => format!(
                "cut {}+{} E {} work {:.1}",
                cut.sides.0,
                cut.sides.1,
                cut.pairs,
                (split_work - dense_part(kmax) + cut.work).log2()
            ),
            _ => format!("no cut, work {:.1}", split_work.log2()),
        };
        split_lines.push(format!(
            "{label:<40} {} + {singles} single | {cut} | canonical {:?}",
            widest.join(" "),
            parts.canonical_widths
        ));
    }
    println!();
    println!(
        "{:<40} parts past one qubit (width, excess, table bits)",
        "fixture"
    );
    for line in &split_lines {
        println!("{line}");
    }

    let mut mid = Vec::new();
    for (t, shots) in [(6, 16), (8, 4)] {
        for n in [26, 32] {
            let circuit = bench_fixtures::stabilizer_rank_mid_circuit_shot_circuit(n, t, SEED);
            mid.push((
                format!("rank_mid/mid_{n}q_t{t}_{shots}shots"),
                circuit,
                shots,
            ));
        }
    }
    for n in [22, 24, 26] {
        let circuit = bench_fixtures::clifford_t_layers_mid_circuit(n, 10, 8, SEED);
        mid.push((format!("statevector_mid/layers_{n}q_t8"), circuit, 64));
    }
    for (label, circuit, shots) in auto_mid_rows() {
        mid.push((format!("auto_mid/{label}"), circuit, shots));
    }
    println!();
    println!(
        "{:<40} {:>3} {:>3} {:>6} {:>6} {:>12}  auto route",
        "mid fixture", "n", "t", "B", "events", "inner_prods"
    );
    for (label, circuit, shots) in &mid {
        let t = circuit
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: Gate::T | Gate::Tdg,
                        ..
                    }
                )
            })
            .count();
        let events = circuit
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Measure { .. } | Instruction::Reset { .. }
                )
            })
            .count();
        let branches = 1u64 << t;
        // Each event takes the norm of both projections, `B (B + 1) / 2` overlaps each
        // while no branch projects to zero.
        let inner_products = events as u64 * branches * (branches + 1);
        let (backend, rule) = shot_plan(BackendKind::Auto, circuit, *shots);
        println!(
            "{:<40} {:>3} {:>3} {:>6} {:>6} {:>12}  {backend:?}/{rule:?}",
            label, circuit.num_qubits, t, branches, events, inner_products
        );
    }
}

fn split_seed_census() {
    type Make = Box<dyn Fn(u64) -> Circuit>;

    const DENSE: f64 = 40.0;
    const SEEDS: u64 = 20;

    let scrambled = |n, t| -> Make {
        Box::new(move |seed| {
            let mut circuit = bench_fixtures::scrambled_clifford_t_circuit(n, t, seed);
            circuit.measure_all();
            circuit
        })
    };
    let layers = |n, t| -> Make {
        Box::new(move |seed| bench_fixtures::random_clifford_t_layers(n, 10, t, seed))
    };
    let mut shapes: Vec<(String, Make)> = Vec::new();
    for (n, t) in [
        (20, 40),
        (22, 24),
        (24, 24),
        (26, 10),
        (26, 20),
        (26, 24),
        (28, 24),
        (30, 24),
        (32, 24),
        (26, 40),
        (30, 40),
        (32, 40),
        (40, 48),
        (48, 64),
    ] {
        shapes.push((format!("layers_{n}q_t{t}"), layers(n, t)));
    }
    shapes.push((
        "local_layers_64q_t40".into(),
        Box::new(|seed| bench_fixtures::local_clifford_t_layers(64, 10, 40, seed)),
    ));
    for n in [16, 20, 24] {
        shapes.push((
            format!("clifford_t_shots/{n}q_t10"),
            Box::new(move |seed| bench_fixtures::random_clifford_t_layers(n, 10, 10, seed)),
        ));
    }
    for (n, t) in [
        (20, 20),
        (22, 20),
        (24, 20),
        (26, 16),
        (26, 18),
        (26, 20),
        (32, 20),
        (32, 24),
        (32, 28),
        (32, 32),
    ] {
        shapes.push((format!("scrambled_{n}q_t{t}"), scrambled(n, t)));
    }
    shapes.push((
        "onelayer_20q_t80".into(),
        Box::new(|seed| {
            let mut circuit = bench_fixtures::random_t_insertion_circuit(20, 80, seed);
            circuit.measure_all();
            circuit
        }),
    ));
    shapes.push((
        "distinct_17q_t16".into(),
        Box::new(|seed| bench_fixtures::distinct_t_layers(17, 10, 16, seed)),
    ));
    shapes.push((
        "counts/40q_d20_t5pct".into(),
        Box::new(|seed| {
            let mut circuit = circuits::clifford_t_circuit(40, 20, 0.05, seed);
            circuit.measure_all();
            circuit
        }),
    ));
    for (n, depth, t) in [(48, 20, 48), (56, 20, 56), (56, 40, 56), (64, 30, 64)] {
        shapes.push((
            format!("local_layers_{n}q_d{depth}_t{t}"),
            Box::new(move |seed| bench_fixtures::local_clifford_t_layers(n, depth, t, seed)),
        ));
    }

    struct Row {
        k: usize,
        w2: usize,
        canon: usize,
        agrees: bool,
        side: usize,
        pairs: usize,
        work: [f64; 3],
        readout: [f64; 2],
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
    };
    let span = |v: &[usize]| {
        let mut sorted = v.to_vec();
        sorted.sort_unstable();
        let med = (sorted[(v.len() - 1) / 2] + sorted[v.len() / 2]) as f64 / 2.0;
        format!("{}..{} med {med}", sorted[0], sorted[v.len() - 1])
    };
    let mut forty_t = [0usize; 5];
    for (name, make) in &shapes {
        let mut rows = Vec::new();
        let mut declined = 0;
        for i in 0..SEEDS {
            let circuit = make(SEED.wrapping_add(i));
            let census = prism_q::clifford_register_census(&circuit).unwrap();
            let k = census.width;
            let Some(split) = census.split else {
                assert!(k > 64, "{name} seed {i}: split declined at width {k}");
                declined += 1;
                continue;
            };
            assert!(k <= 64, "{name} seed {i}: split past 64 anchors");
            assert_eq!(
                split.parts.iter().map(|p| p.width).sum::<usize>(),
                k,
                "{name} seed {i}: part widths"
            );
            assert_eq!(
                split.canonical_widths.iter().sum::<usize>(),
                k,
                "{name} seed {i}: canonical part widths"
            );
            let w2 = split.parts.first().map_or(0, |p| p.width);
            let canon = split.canonical_widths.first().copied().unwrap_or(0);
            if !split.canonical_agrees {
                let fixed: Vec<usize> = split.parts.iter().map(|p| p.width).collect();
                println!(
                    "{name} seed {i}: fixed point {fixed:?}, canonical {:?}",
                    split.canonical_widths
                );
            }
            let dense = |w: usize| DENSE * 2f64.powi(w as i32);
            let split_work: f64 = split.parts.iter().map(|p| dense(p.width)).sum();
            let (side, pairs, cut_work) = match split.cut {
                Some(cut) if cut.work < dense(w2) => {
                    (cut.sides.0, cut.pairs, split_work - dense(w2) + cut.work)
                }
                _ => (w2, 0, split_work),
            };
            let readout_unsplit = 3.0 * (k + 1) as f64 * 2f64.powi(k as i32);
            let readout_split: f64 = split.parts.iter().map(|p| p.readout_work()).sum();
            rows.push(Row {
                k,
                w2,
                canon,
                agrees: split.canonical_agrees,
                side,
                pairs,
                work: [dense(k).log2(), split_work.log2(), cut_work.log2()],
                readout: [readout_unsplit.log2(), readout_split.log2()],
            });
        }
        if rows.is_empty() {
            println!("{name:<28} split declined on {declined} of {SEEDS}, past 64 anchors");
            continue;
        }
        let col = |f: fn(&Row) -> usize| rows.iter().map(f).collect::<Vec<_>>();
        let admits = |f: fn(&Row) -> usize, cap| rows.iter().filter(|r| f(r) <= cap).count();
        let counts = [
            admits(|r| r.k, 20),
            admits(|r| r.k, 24),
            admits(|r| r.w2, 20),
            admits(|r| r.w2, 24),
            admits(|r| r.side, 20),
        ];
        if name.starts_with("layers_") && name.ends_with("q_t40") && name != "layers_20q_t40" {
            for (total, c) in forty_t.iter_mut().zip(counts) {
                *total += c;
            }
        }
        let work = |j: usize| median(rows.iter().map(|r| r.work[j]).collect());
        let work_max = |j: usize| rows.iter().map(|r| r.work[j]).fold(f64::MIN, f64::max);
        let readout = |j: usize| median(rows.iter().map(|r| r.readout[j]).collect());
        println!(
            "{name:<28} k {} | w2 {} | cut side {} E {} | admits: today {} cap24 {} split20 {} \
             split24 {} cut20 {} | work med {:.1} {:.1} {:.1} max {:.1} {:.1} {:.1} | readout \
             med {:.1} {:.1} | canonical: same parts {} widest {} | past 64 anchors {declined}",
            span(&col(|r| r.k)),
            span(&col(|r| r.w2)),
            span(&col(|r| r.side)),
            span(&col(|r| r.pairs)),
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            work(0),
            work(1),
            work(2),
            work_max(0),
            work_max(1),
            work_max(2),
            readout(0),
            readout(1),
            rows.iter().filter(|r| r.agrees).count(),
            span(&col(|r| r.canon)),
        );
    }
    println!(
        "layers_{{26,30,32}}q_t40, {} circuits: today {} cap24 {} split20 {} split24 {} cut20 {}",
        3 * SEEDS,
        forty_t[0],
        forty_t[1],
        forty_t[2],
        forty_t[3],
        forty_t[4]
    );
}
