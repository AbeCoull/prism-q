//! PRISM-Q side of the cross-simulator comparison harness in `comparison/`.
//!
//! `export` writes one of the benchmark suite's circuit families as a gate list
//! that every comparator replays natively, so each simulator runs the same
//! gates in the same order. `time` and `probabilities` read that list back from
//! stdin. The timed region covers circuit execution and materialization of the
//! dense probability vector, the same region the other adapters time; building
//! the circuit from the list is excluded, matching the exclusion of transpiling
//! on the other side.
//!
//! Gate list format, one gate per line after a `qubits N` header, qubit 0 the
//! least significant bit of the probability index:
//!
//! ```text
//! qubits 3
//! h 0
//! cx 0 1
//! ry 2 0.4
//! rz 2 1.1
//! cp 0 2 0.7853981633974483
//! swap 0 2
//! ```
//!
//! ```text
//! cargo run --release --features parallel --example compare_runner -- export qft 16 > qft16.txt
//! cargo run --release --features parallel --example compare_runner -- time 10 < qft16.txt
//! cargo run --release --features parallel --example compare_runner -- probabilities out.f64 < qft16.txt
//! ```

use prism_q::circuit::{Circuit, Instruction, expand_qft_blocks};
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim::{self, BackendKind, Probabilities};
use std::io::{Read, Write};
use std::time::Instant;

const CIRCUIT_SEED: u64 = 0xDEAD_BEEF;
const SIM_SEED: u64 = 42;
const HEA_LAYERS: usize = 5;

fn read_stdin() -> String {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("failed to read the gate list from stdin");
    text
}

fn fail(code: i32, message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}

fn build_family(family: &str, n: usize) -> Circuit {
    match family {
        "ghz" => circuits::ghz_circuit(n),
        "qft" => circuits::qft_circuit(n),
        "hea" => circuits::hardware_efficient_ansatz(n, HEA_LAYERS, CIRCUIT_SEED),
        "qv" => circuits::quantum_volume_circuit(n, n, CIRCUIT_SEED),
        other => fail(2, format!("unknown family: {other} (ghz, qft, hea, qv)")),
    }
}

/// The controlled phase angle of a `Gate::Cu` whose matrix is `diag(1, e^{i theta})`.
fn cphase_angle(matrix: &[[num_complex::Complex64; 2]; 2]) -> Option<f64> {
    let [[a, b], [c, d]] = matrix;
    let tol = 1e-12;
    if (a.re - 1.0).abs() > tol || a.im.abs() > tol || b.norm() > tol || c.norm() > tol {
        return None;
    }
    if (d.norm() - 1.0).abs() > tol {
        return None;
    }
    Some(d.arg())
}

fn export(family: &str, n: usize) {
    let circuit = build_family(family, n);
    let circuit = expand_qft_blocks(&circuit);
    let mut out = String::new();
    out.push_str(&format!("qubits {}\n", circuit.num_qubits));
    for inst in &circuit.instructions {
        let Instruction::Gate { gate, targets, .. } = inst else {
            fail(
                2,
                format!("{family} contains a non-gate instruction: {inst:?}"),
            );
        };
        let line = match (gate, targets.as_slice()) {
            (Gate::H, [q]) => format!("h {q}"),
            (Gate::Cx, [c, t]) => format!("cx {c} {t}"),
            (Gate::Swap, [a, b]) => format!("swap {a} {b}"),
            (Gate::Ry(theta), [q]) => format!("ry {q} {theta:?}"),
            (Gate::Rz(theta), [q]) => format!("rz {q} {theta:?}"),
            (Gate::Cu(matrix), [c, t]) => match cphase_angle(matrix) {
                Some(theta) => format!("cp {c} {t} {theta:?}"),
                None => fail(
                    2,
                    format!("{family} contains a controlled unitary that is not a phase"),
                ),
            },
            (gate, targets) => fail(
                2,
                format!(
                    "{family} emits {gate:?} on {targets:?}, which the shared gate list does not carry"
                ),
            ),
        };
        out.push_str(&line);
        out.push('\n');
    }
    std::io::stdout()
        .write_all(out.as_bytes())
        .expect("failed to write the gate list");
}

fn parse_gate_list(text: &str) -> Circuit {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().unwrap_or_else(|| fail(3, "empty gate list"));
    let n: usize = match header.split_whitespace().collect::<Vec<_>>().as_slice() {
        ["qubits", n] => n.parse().unwrap_or_else(|_| fail(3, "bad qubit count")),
        _ => fail(3, format!("expected `qubits N`, got `{header}`")),
    };
    let mut circuit = Circuit::new(n, 0);
    let q = |s: &str| -> usize {
        s.parse()
            .unwrap_or_else(|_| fail(3, format!("bad qubit index `{s}`")))
    };
    let a = |s: &str| -> f64 {
        s.parse()
            .unwrap_or_else(|_| fail(3, format!("bad angle `{s}`")))
    };
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["h", t] => circuit.add_gate(Gate::H, &[q(t)]),
            ["cx", c, t] => circuit.add_gate(Gate::Cx, &[q(c), q(t)]),
            ["swap", x, y] => circuit.add_gate(Gate::Swap, &[q(x), q(y)]),
            ["ry", t, theta] => circuit.add_gate(Gate::Ry(a(theta)), &[q(t)]),
            ["rz", t, theta] => circuit.add_gate(Gate::Rz(a(theta)), &[q(t)]),
            ["cp", c, t, theta] => circuit.add_gate(Gate::cphase(a(theta)), &[q(c), q(t)]),
            _ => fail(3, format!("unsupported gate line `{line}`")),
        }
    }
    circuit
}

/// Runs under [`BackendKind::Auto`] and materializes the full 2^n probability
/// vector, matching the work the other adapters are timed on. Auto is the
/// label a user gets by default, and the fusion pass it runs sits inside the
/// timed region, as each comparator's own optimization does.
fn dense_probabilities(circuit: &Circuit) -> Vec<f64> {
    let outcome = sim::simulate(circuit)
        .backend(BackendKind::Auto)
        .seed(SIM_SEED)
        .run()
        .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
    match outcome.probabilities {
        Some(Probabilities::Dense(probs)) => probs,
        Some(factored) => factored.to_vec(),
        None => fail(
            4,
            "simulation error: backend exposed no probability distribution",
        ),
    }
}

fn rayon_threads() -> String {
    match std::env::var("RAYON_NUM_THREADS") {
        Ok(value) if !value.is_empty() => value,
        _ => std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
    }
}

fn print_times(circuit: &Circuit, times_ms: &[f64]) {
    let mut sorted = times_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print!(
        "{{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"prismq\",\"version\":\"{}\",\
         \"num_qubits\":{},\"num_operations\":{},\"backend\":\"auto\",\"threads\":\"{}\",\
         \"median_ms\":{:.4},\"min_ms\":{:.4},\"times_ms\":[",
        env!("CARGO_PKG_VERSION"),
        circuit.num_qubits,
        circuit.instructions.len(),
        rayon_threads(),
        sorted[sorted.len() / 2],
        sorted[0],
    );
    for (i, t) in times_ms.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{t:.4}");
    }
    println!("]}}");
}

fn run_time(iterations: usize) {
    let circuit = parse_gate_list(&read_stdin());
    std::hint::black_box(dense_probabilities(&circuit));
    let mut times_ms: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let probs = dense_probabilities(&circuit);
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(probs);
    }
    print_times(&circuit, &times_ms);
}

fn run_probabilities(path: &str) {
    let circuit = parse_gate_list(&read_stdin());
    let probs = dense_probabilities(&circuit);
    let mut bytes: Vec<u8> = Vec::with_capacity(probs.len() * 8);
    for p in &probs {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &bytes).expect("failed to write the probability vector");
    println!(
        "{{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"prismq\",\"num_qubits\":{},\
         \"num_operations\":{},\"length\":{},\"dtype\":\"<f8\",\"path\":{:?}}}",
        circuit.num_qubits,
        circuit.instructions.len(),
        probs.len(),
        path
    );
}

fn usage() -> ! {
    fail(
        1,
        "Usage: compare_runner export <ghz|qft|hea|qv> <qubits>\n\
         \x20      compare_runner time <iterations>        (gate list on stdin)\n\
         \x20      compare_runner probabilities <out_path> (gate list on stdin)\n\
         \x20      compare_runner version",
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("version") => println!(
            "{{\"simulator\":\"prismq\",\"version\":\"{}\"}}",
            env!("CARGO_PKG_VERSION")
        ),
        Some("export") if args.len() == 4 => {
            let n: usize = args[3].parse().unwrap_or_else(|_| usage());
            export(&args[2], n);
        }
        Some("time") if args.len() == 3 => {
            let iterations: usize = args[2].parse().unwrap_or_else(|_| usage());
            if iterations == 0 {
                usage();
            }
            run_time(iterations);
        }
        Some("probabilities") if args.len() == 3 => run_probabilities(&args[2]),
        _ => usage(),
    }
}
