//! Stabilizer backend scaling benchmark.
//!
//! Times gate application, measurement, and (where feasible) probability
//! extraction across a range of qubit counts to verify O(n^2) per-gate scaling.
//!
//! ```text
//! cargo run --release --example bench_stabilizer
//! cargo run --release --example bench_stabilizer -- --qubits 50,100,500,1000 --iters 3
//! ```

use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::Backend;
use prism_q::circuit::Instruction;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::Circuit;
use std::time::Instant;

const SEED: u64 = 0xDEAD_BEEF;

struct BenchResult {
    qubits: usize,
    circuit_name: String,
    gate_count: usize,
    init_us: f64,
    gates_us: f64,
    measure_us: f64,
    probs_us: Option<f64>,
    total_us: f64,
    per_gate_ns: f64,
}

fn bench_circuit(circuit: &Circuit, name: &str, iterations: usize) -> BenchResult {
    let n = circuit.num_qubits;
    let mut timings = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut backend = StabilizerBackend::new(SEED);

        let t0 = Instant::now();
        backend.init(n, circuit.num_classical_bits).unwrap();
        let init_ns = t0.elapsed().as_nanos() as u64;

        let t1 = Instant::now();
        let mut gate_count = 0usize;
        let mut measure_ns = 0u64;
        for instr in &circuit.instructions {
            match instr {
                Instruction::Measure { .. } => {
                    let tm = Instant::now();
                    backend.apply(instr).unwrap();
                    measure_ns += tm.elapsed().as_nanos() as u64;
                }
                Instruction::Barrier { .. } => {}
                _ => {
                    backend.apply(instr).unwrap();
                    gate_count += 1;
                }
            }
        }
        let gates_ns = t1.elapsed().as_nanos() as u64 - measure_ns;

        let t2 = Instant::now();
        let probs_ns = if n <= 20 {
            let _ = backend.probabilities().unwrap();
            Some(t2.elapsed().as_nanos() as u64)
        } else {
            None
        };

        timings.push((init_ns, gates_ns, measure_ns, probs_ns, gate_count));
    }

    timings.sort_by_key(|t| t.1);
    let med = &timings[timings.len() / 2];

    let init_us = med.0 as f64 / 1e3;
    let gates_us = med.1 as f64 / 1e3;
    let measure_us = med.2 as f64 / 1e3;
    let probs_us = med.3.map(|ns| ns as f64 / 1e3);
    let gate_count = med.4;
    let total_ns = med.0 + med.1 + med.2 + med.3.unwrap_or(0);
    let per_gate_ns = if gate_count > 0 {
        med.1 as f64 / gate_count as f64
    } else {
        0.0
    };

    BenchResult {
        qubits: n,
        circuit_name: name.to_string(),
        gate_count,
        init_us,
        gates_us,
        measure_us,
        probs_us,
        total_us: total_ns as f64 / 1e3,
        per_gate_ns,
    }
}

fn build_clifford_circuit(n: usize, depth: usize) -> Circuit {
    circuits::clifford_heavy_circuit(n, depth, SEED)
}

fn build_ghz_circuit(n: usize) -> Circuit {
    circuits::ghz_circuit(n)
}

fn build_measure_all(base: &Circuit) -> Circuit {
    let n = base.num_qubits;
    let mut c = Circuit::new(n, n);
    for instr in &base.instructions {
        c.instructions.push(instr.clone());
    }
    for q in 0..n {
        c.instructions.push(Instruction::Measure {
            qubit: q,
            classical_bit: q,
        });
    }
    c
}

fn build_single_gate_circuit(n: usize, gate: Gate, name: &str, count: usize) -> (Circuit, String) {
    let mut c = Circuit::new(n, 0);
    match gate.num_qubits() {
        1 => {
            for _ in 0..count {
                for q in 0..n {
                    c.add_gate(gate.clone(), &[q]);
                }
            }
        }
        2 => {
            for _ in 0..count {
                for q in 0..n - 1 {
                    c.add_gate(gate.clone(), &[q, q + 1]);
                }
            }
        }
        _ => unreachable!(),
    }
    (c, format!("{}x{}", name, count))
}

fn fmt_duration(us: f64) -> String {
    if us >= 1_000_000.0 {
        format!("{:>8.2} s ", us / 1e6)
    } else if us >= 1000.0 {
        format!("{:>8.2} ms", us / 1e3)
    } else {
        format!("{:>8.2} us", us)
    }
}

fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:>7.1} ms", ns / 1e6)
    } else if ns >= 1000.0 {
        format!("{:>7.1} us", ns / 1e3)
    } else {
        format!("{:>7.0} ns", ns)
    }
}

fn print_header() {
    eprintln!(
        "{:<22} {:>6} {:>7} {:>11} {:>11} {:>11} {:>11} {:>11} {:>10}",
        "Circuit", "Qubits", "Gates", "Init", "Gates", "Measure", "Probs", "Total", "Per-gate"
    );
    eprintln!("{}", "-".repeat(105));
}

fn print_row(r: &BenchResult) {
    let probs_str = match r.probs_us {
        Some(us) => fmt_duration(us),
        None => "     n/a   ".to_string(),
    };
    eprintln!(
        "{:<22} {:>6} {:>7} {:>11} {:>11} {:>11} {:>11} {:>11} {:>10}",
        r.circuit_name,
        r.qubits,
        r.gate_count,
        fmt_duration(r.init_us),
        fmt_duration(r.gates_us),
        fmt_duration(r.measure_us),
        probs_str,
        fmt_duration(r.total_us),
        fmt_ns(r.per_gate_ns),
    );
}

fn parse_qubit_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|x| x.trim().parse().expect("invalid qubit count"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut qubit_counts = vec![10, 20, 50, 100, 200, 500, 1000];
    let mut iterations = 5;
    let mut depth = 10;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--qubits" | "-q" => {
                i += 1;
                qubit_counts = parse_qubit_list(&args[i]);
            }
            "--iters" | "-i" => {
                i += 1;
                iterations = args[i].parse().expect("invalid iteration count");
            }
            "--depth" | "-d" => {
                i += 1;
                depth = args[i].parse().expect("invalid depth");
            }
            "--help" | "-h" => {
                eprintln!("Usage: bench_stabilizer [OPTIONS]");
                eprintln!("  --qubits, -q  Comma-separated qubit counts (default: 10,20,50,100,200,500,1000)");
                eprintln!("  --iters,  -i  Iterations per benchmark (default: 5)");
                eprintln!("  --depth,  -d  Circuit depth for random Clifford (default: 10)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("=== PRISM-Q Stabilizer Scaling Benchmark ===");
    eprintln!("Iterations: {iterations}, Depth: {depth}, Seed: 0x{SEED:X}");
    eprintln!();

    // --- Section 1: GHZ scaling ---
    eprintln!("--- GHZ State Preparation (H + CX chain) ---");
    eprintln!();
    print_header();
    for &n in &qubit_counts {
        let circ = build_measure_all(&build_ghz_circuit(n));
        let r = bench_circuit(&circ, "ghz", iterations);
        print_row(&r);
    }
    eprintln!();

    // --- Section 2: Random Clifford scaling ---
    eprintln!("--- Random Clifford depth-{depth} (H/S/X/Y/Z + brick CX) ---");
    eprintln!();
    print_header();
    for &n in &qubit_counts {
        let circ = build_measure_all(&build_clifford_circuit(n, depth));
        let r = bench_circuit(&circ, &format!("clifford_d{depth}"), iterations);
        print_row(&r);
    }
    eprintln!();

    // --- Section 3: Per-gate-type scaling ---
    eprintln!("--- Per-Gate Scaling (1000 layers each) ---");
    eprintln!();
    print_header();
    let gates: Vec<(Gate, &str)> = vec![
        (Gate::H, "H"),
        (Gate::S, "S"),
        (Gate::X, "X"),
        (Gate::Z, "Z"),
        (Gate::Cx, "CX"),
        (Gate::Cz, "CZ"),
        (Gate::Swap, "SWAP"),
    ];
    let per_gate_qubits: Vec<usize> = qubit_counts.iter().copied().filter(|&n| n <= 500).collect();
    for (gate, name) in &gates {
        for &n in &per_gate_qubits {
            let layers = if n <= 100 { 1000 } else { 100 };
            let (circ, label) = build_single_gate_circuit(n, gate.clone(), name, layers);
            let r = bench_circuit(&circ, &label, iterations);
            print_row(&r);
        }
        eprintln!();
    }

    // --- Section 4: Measurement scaling ---
    eprintln!("--- Measurement-Only (GHZ state, measure all qubits) ---");
    eprintln!();
    print_header();
    for &n in &qubit_counts {
        let ghz = build_ghz_circuit(n);
        let mut c = Circuit::new(n, n);
        for instr in &ghz.instructions {
            c.instructions.push(instr.clone());
        }
        for q in 0..n {
            c.add_measure(q, q);
        }
        let r = bench_circuit(&c, "ghz+measure_all", iterations);
        print_row(&r);
    }
    eprintln!();

    // --- Section 5: Scaling summary ---
    eprintln!("--- Scaling Summary (clifford_d{depth}, per-gate cost) ---");
    eprintln!();
    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>12}",
        "Qubits", "Per-gate", "Total", "Expected O(n^2)"
    );
    eprintln!("{}", "-".repeat(48));
    let mut prev_per_gate: Option<(usize, f64)> = None;
    for &n in &qubit_counts {
        let circ = build_clifford_circuit(n, depth);
        let r = bench_circuit(&circ, "tmp", iterations);
        let ratio = if let Some((pn, pns)) = prev_per_gate {
            let n_ratio = (n as f64 / pn as f64).powi(2);
            let actual_ratio = r.per_gate_ns / pns;
            format!("{:.2}x (expect {:.2}x)", actual_ratio, n_ratio)
        } else {
            "baseline".to_string()
        };
        eprintln!(
            "{:>6}  {:>10}  {:>10}  {}",
            n,
            fmt_ns(r.per_gate_ns),
            fmt_duration(r.gates_us),
            ratio
        );
        prev_per_gate = Some((n, r.per_gate_ns));
    }
    eprintln!();
}
