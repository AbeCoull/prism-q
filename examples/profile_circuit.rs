//! Per-gate-type hotspot profiler for PRISM-Q circuits.
//!
//! Runs a circuit with per-instruction timing and reports a breakdown showing
//! where wall-clock time is spent. Much more actionable than flamegraphs for
//! understanding gate kernel performance.
//!
//! ```text
//! cargo run --release --features parallel --example profile_circuit -- qft 16
//! cargo run --release --features parallel --example profile_circuit -- hea 20 10
//! ```

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use std::collections::BTreeMap;
use std::time::Instant;

const SEED: u64 = 0xDEAD_BEEF;

fn build_circuit(name: &str, n: usize) -> Circuit {
    match name {
        "qft" => circuits::qft_circuit(n),
        "random" => circuits::random_circuit(n, 10, SEED),
        "hea" => circuits::hardware_efficient_ansatz(n, 5, SEED),
        "qpe" => circuits::phase_estimation_circuit(n),
        "clifford" => circuits::clifford_heavy_circuit(n, 10, SEED),
        "ghz" => circuits::ghz_circuit(n),
        "deep" => circuits::random_circuit(n, 50, SEED),
        "qaoa" => circuits::qaoa_circuit(n, 3, SEED),
        other => {
            eprintln!("Unknown circuit: {other}");
            eprintln!("Available: qft, random, hea, qpe, clifford, ghz, deep, qaoa");
            std::process::exit(1);
        }
    }
}

// ---- Profiling data ----

struct IterProfile {
    fusion_ns: u64,
    init_ns: u64,
    gates: BTreeMap<String, (u64, usize)>,
    probs_ns: u64,
}

fn profile_one(circuit: &Circuit) -> IterProfile {
    let mut backend = StatevectorBackend::new(SEED);
    let mut gates: BTreeMap<String, (u64, usize)> = BTreeMap::new();

    let start = Instant::now();
    let fused = fuse_circuit(circuit, backend.supports_fused_gates());
    let fusion_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    backend
        .init(fused.num_qubits, fused.num_classical_bits)
        .unwrap();
    let init_ns = start.elapsed().as_nanos() as u64;

    for instruction in &fused.instructions {
        let name = match instruction {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => gate.name(),
            Instruction::Measure { .. } => "measure",
            Instruction::Barrier { .. } => continue,
        };

        let start = Instant::now();
        backend.apply(instruction).unwrap();
        let elapsed = start.elapsed().as_nanos() as u64;

        let entry = gates.entry(name.to_string()).or_insert((0, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    }

    let start = Instant::now();
    let _ = backend.probabilities();
    let probs_ns = start.elapsed().as_nanos() as u64;

    IterProfile {
        fusion_ns,
        init_ns,
        gates,
        probs_ns,
    }
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn format_duration(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1e9)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2} us", ns as f64 / 1e3)
    } else {
        format!("{} ns", ns)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: profile_circuit <circuit> <n_qubits> [iterations]");
        eprintln!("  circuit: qft | random | hea | qpe | clifford | ghz | deep | qaoa");
        eprintln!("  iterations: default 5");
        std::process::exit(1);
    }

    let circuit_name = &args[1];
    let n_qubits: usize = args[2]
        .parse()
        .expect("n_qubits must be a positive integer");
    let iterations: usize = args
        .get(3)
        .map(|s| s.parse().expect("iterations must be a positive integer"))
        .unwrap_or(5);

    let circuit = build_circuit(circuit_name, n_qubits);

    let fused_preview = fuse_circuit(&circuit, true);

    let pre_gates = circuit.instructions.len();
    let post_gates = fused_preview.instructions.len();

    eprintln!(
        "Circuit: {} {}q -- {} instructions (post-fusion: {})",
        circuit_name, n_qubits, pre_gates, post_gates
    );
    eprintln!("Running {} warmup + {} timed iterations...", 1, iterations);
    eprintln!();

    // Warmup
    let _ = profile_one(&circuit);

    // Timed iterations
    let profiles: Vec<IterProfile> = (0..iterations).map(|_| profile_one(&circuit)).collect();

    // Collect all gate names across iterations
    let mut all_gate_names: Vec<String> = profiles
        .iter()
        .flat_map(|p| p.gates.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    all_gate_names.sort();

    // Compute medians
    let mut fusion_vals: Vec<u64> = profiles.iter().map(|p| p.fusion_ns).collect();
    let mut init_vals: Vec<u64> = profiles.iter().map(|p| p.init_ns).collect();
    let mut probs_vals: Vec<u64> = profiles.iter().map(|p| p.probs_ns).collect();

    let med_fusion = median(&mut fusion_vals);
    let med_init = median(&mut init_vals);
    let med_probs = median(&mut probs_vals);

    struct GateRow {
        name: String,
        median_ns: u64,
        count: usize,
    }

    let mut gate_rows: Vec<GateRow> = Vec::new();
    for name in &all_gate_names {
        let mut time_vals: Vec<u64> = profiles
            .iter()
            .map(|p| p.gates.get(name).map_or(0, |g| g.0))
            .collect();
        let count = profiles
            .iter()
            .filter_map(|p| p.gates.get(name))
            .next()
            .map_or(0, |g| g.1);
        gate_rows.push(GateRow {
            name: name.clone(),
            median_ns: median(&mut time_vals),
            count,
        });
    }

    // Sort by time descending (hotspots first)
    gate_rows.sort_by(|a, b| b.median_ns.cmp(&a.median_ns));

    let total_gates_ns: u64 = gate_rows.iter().map(|r| r.median_ns).sum();
    let grand_total = med_fusion + med_init + total_gates_ns + med_probs;

    // ---- Print report ----

    let pct = |ns: u64| -> f64 {
        if grand_total == 0 {
            0.0
        } else {
            ns as f64 / grand_total as f64 * 100.0
        }
    };

    eprintln!(
        "=== PRISM-Q Hotspot Profile: {} {}q (median of {}) ===",
        circuit_name, n_qubits, iterations
    );
    eprintln!();
    eprintln!(
        "  {:<20} {:>12} {:>7} {:>7} {:>12}",
        "Phase", "Total", "%", "Count", "Per-call"
    );
    eprintln!("  {}", "-".repeat(62));

    // Overhead phases
    eprintln!(
        "  {:<20} {:>12} {:>6.1}%",
        "fusion",
        format_duration(med_fusion),
        pct(med_fusion)
    );
    eprintln!(
        "  {:<20} {:>12} {:>6.1}%",
        "init",
        format_duration(med_init),
        pct(med_init)
    );
    eprintln!("  {}", "-".repeat(62));

    // Gate breakdown
    let total_count: usize = gate_rows.iter().map(|r| r.count).sum();
    for row in &gate_rows {
        let per_call = if row.count > 0 {
            row.median_ns / row.count as u64
        } else {
            0
        };
        eprintln!(
            "  {:<20} {:>12} {:>6.1}% {:>7} {:>12}",
            row.name,
            format_duration(row.median_ns),
            pct(row.median_ns),
            row.count,
            format_duration(per_call)
        );
    }

    eprintln!("  {}", "-".repeat(62));
    eprintln!(
        "  {:<20} {:>12} {:>6.1}%",
        "probabilities",
        format_duration(med_probs),
        pct(med_probs)
    );
    eprintln!("  {}", "-".repeat(62));
    eprintln!(
        "  {:<20} {:>12} {:>6.1}% {:>7}",
        "TOTAL",
        format_duration(grand_total),
        100.0,
        total_count
    );

    // Top hotspots summary
    eprintln!();
    eprintln!("  Top hotspots:");
    let all_phases: Vec<(&str, u64)> = std::iter::once(("fusion", med_fusion))
        .chain(std::iter::once(("init", med_init)))
        .chain(gate_rows.iter().map(|r| (r.name.as_str(), r.median_ns)))
        .chain(std::iter::once(("probabilities", med_probs)))
        .collect();

    let mut sorted_phases = all_phases;
    sorted_phases.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (name, ns)) in sorted_phases.iter().take(3).enumerate() {
        eprintln!(
            "    {}. {:<18} {:>6.1}%  ({})",
            i + 1,
            name,
            pct(*ns),
            format_duration(*ns)
        );
    }
    eprintln!();

    // JSON output on stdout for programmatic consumption
    print!("{{\"circuit\":\"{circuit_name}\",\"qubits\":{n_qubits},\"iterations\":{iterations}");
    print!(",\"total_ns\":{grand_total}");
    print!(",\"fusion_ns\":{med_fusion},\"init_ns\":{med_init},\"probabilities_ns\":{med_probs}");
    print!(",\"gates\":{{");
    for (i, row) in gate_rows.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(
            "\"{}\":{{\"total_ns\":{},\"count\":{},\"per_call_ns\":{}}}",
            row.name,
            row.median_ns,
            row.count,
            if row.count > 0 {
                row.median_ns / row.count as u64
            } else {
                0
            }
        );
    }
    println!("}}}}");
}
