//! PRISM-Q side of the cross-simulator comparison harness in `comparison/`.
//!
//! `export` writes one of the benchmark suite's circuit families as a gate list
//! that every comparator replays natively, so each simulator runs the same
//! gates in the same order. `time`, `probabilities` and `exec` read a gate list
//! back from stdin. Building the circuit from the list is excluded from every
//! timed region, matching the exclusion of transpiling on the other side.
//!
//! Gate list format, one instruction per line after a `qubits N` header and an
//! optional `clbits M` line, qubit 0 the least significant bit of the
//! probability index:
//!
//! ```text
//! qubits 3
//! clbits 2
//! h 0
//! cx 0 1
//! ry 2 0.4
//! cp 0 2 0.7853981633974483
//! measure 1 0
//! if 0 1 x 2
//! reset 1
//! measure 2 1
//! ```
//!
//! `if c v <gate>` applies the gate when classical bit `c` reads `v`. The
//! single-qubit gates are `h x z s sdg ry rz`, the two-qubit gates `cx cz swap cp`.
//!
//! ```text
//! cargo run --release --features parallel --example compare_runner -- export qft 16 > qft16.txt
//! cargo run --release --features parallel --example compare_runner -- time 10 < qft16.txt
//! cargo run --release --features parallel --example compare_runner -- probabilities out.f64 < qft16.txt
//! cargo run --release --features parallel --example compare_runner -- \
//!     exec --mode counts --shots 1000 --iterations 5 --out counts.txt --backend auto < circuit.txt
//! ```

use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, expand_qft_blocks};
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim::{self, BackendKind, Probabilities, Query};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::time::Instant;

const CIRCUIT_SEED: u64 = 0xDEAD_BEEF;
const SIM_SEED: u64 = 42;
const HEA_LAYERS: usize = 5;
const PLAN_REPS: usize = 21;

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

fn build_family(family: &str, n: usize, seed: u64) -> Circuit {
    match family {
        "ghz" => circuits::ghz_circuit(n),
        "qft" => circuits::qft_circuit(n),
        "hea" => circuits::hardware_efficient_ansatz(n, HEA_LAYERS, seed),
        "qv" => circuits::quantum_volume_circuit(n, n, seed),
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

fn export(family: &str, n: usize, seed: u64) {
    let circuit = build_family(family, n, seed);
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

fn parse_index(s: &str) -> usize {
    s.parse()
        .unwrap_or_else(|_| fail(3, format!("bad index `{s}`")))
}

fn parse_angle(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| fail(3, format!("bad angle `{s}`")))
}

fn parse_gate(parts: &[&str]) -> Option<(Gate, Vec<usize>)> {
    let q = parse_index;
    Some(match parts {
        ["h", t] => (Gate::H, vec![q(t)]),
        ["x", t] => (Gate::X, vec![q(t)]),
        ["z", t] => (Gate::Z, vec![q(t)]),
        ["s", t] => (Gate::S, vec![q(t)]),
        ["sdg", t] => (Gate::Sdg, vec![q(t)]),
        ["ry", t, theta] => (Gate::Ry(parse_angle(theta)), vec![q(t)]),
        ["rz", t, theta] => (Gate::Rz(parse_angle(theta)), vec![q(t)]),
        ["cx", c, t] => (Gate::Cx, vec![q(c), q(t)]),
        ["cz", c, t] => (Gate::Cz, vec![q(c), q(t)]),
        ["swap", x, y] => (Gate::Swap, vec![q(x), q(y)]),
        ["cp", c, t, theta] => (Gate::cphase(parse_angle(theta)), vec![q(c), q(t)]),
        _ => return None,
    })
}

fn parse_gate_list(text: &str) -> Circuit {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
    let header = lines.next().unwrap_or_else(|| fail(3, "empty gate list"));
    let n: usize = match header.split_whitespace().collect::<Vec<_>>().as_slice() {
        ["qubits", n] => n.parse().unwrap_or_else(|_| fail(3, "bad qubit count")),
        _ => fail(3, format!("expected `qubits N`, got `{header}`")),
    };
    let mut clbits = 0;
    if let Some(line) = lines.peek()
        && let ["clbits", m] = line.split_whitespace().collect::<Vec<_>>().as_slice()
    {
        clbits = parse_index(m);
        lines.next();
    }
    let mut circuit = Circuit::new(n, clbits);
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["measure", qubit, bit] => circuit.add_measure(parse_index(qubit), parse_index(bit)),
            ["reset", qubit] => circuit.add_reset(parse_index(qubit)),
            ["if", bit, value, rest @ ..] => {
                let Some((gate, targets)) = parse_gate(rest) else {
                    fail(3, format!("unsupported conditional `{line}`"));
                };
                let bit = parse_index(bit);
                let condition = match *value {
                    "1" => ClassicalCondition::BitIsOne(bit),
                    "0" => ClassicalCondition::BitIsZero(bit),
                    _ => fail(3, format!("conditional value must be 0 or 1 in `{line}`")),
                };
                circuit.instructions.push(Instruction::Conditional {
                    condition,
                    gate,
                    targets: targets.into_iter().collect(),
                });
            }
            gate => match parse_gate(gate) {
                Some((gate, targets)) => circuit.add_gate(gate, &targets),
                None => fail(3, format!("unsupported gate line `{line}`")),
            },
        }
    }
    circuit
}

fn parse_backend(name: &str) -> BackendKind {
    match name.split_once(':') {
        Some(("mps", bond)) => BackendKind::Mps {
            max_bond_dim: parse_index(bond),
        },
        _ => match name {
            "auto" => BackendKind::Auto,
            "statevector" => BackendKind::Statevector,
            "stabilizer" => BackendKind::Stabilizer,
            "factored-stabilizer" => BackendKind::FactoredStabilizer,
            "sparse" => BackendKind::Sparse,
            "product" => BackendKind::ProductState,
            "factored" => BackendKind::Factored,
            "tensor-network" => BackendKind::TensorNetwork,
            "stabilizer-rank" => BackendKind::StabilizerRank,
            other => fail(2, format!("unknown backend `{other}`")),
        },
    }
}

/// Run on `kind` and materialize the full 2^n probability vector, the work the
/// other adapters are timed on. The fusion pass sits inside the timed region,
/// as each comparator's own optimization does.
fn dense_probabilities_on(circuit: &Circuit, kind: BackendKind) -> (Vec<f64>, sim::RunMetadata) {
    let outcome = sim::simulate(circuit)
        .backend(kind)
        .seed(SIM_SEED)
        .run()
        .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
    let probs = match outcome.probabilities {
        Some(Probabilities::Dense(probs)) => probs,
        Some(factored) => factored.to_vec(),
        None => fail(
            4,
            "simulation error: backend exposed no probability distribution",
        ),
    };
    (probs, outcome.metadata)
}

fn dense_probabilities(circuit: &Circuit) -> Vec<f64> {
    dense_probabilities_on(circuit, BackendKind::Auto).0
}

fn sample_counts_on(
    circuit: &Circuit,
    kind: BackendKind,
    shots: usize,
) -> (HashMap<Vec<u64>, u64>, sim::RunMetadata) {
    let result = sim::simulate(circuit)
        .backend(kind)
        .seed(SIM_SEED)
        .sample_counts(shots)
        .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
    (result.counts, result.metadata)
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
    print!("{}", join_ms(times_ms));
    println!("]}}");
}

fn join_ms(values: &[f64]) -> String {
    values
        .iter()
        .map(|t| format!("{t:.4}"))
        .collect::<Vec<_>>()
        .join(",")
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

fn probability_bytes(probs: &[f64]) -> Vec<u8> {
    let mut bytes: Vec<u8> = Vec::with_capacity(probs.len() * 8);
    for p in probs {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    bytes
}

fn run_probabilities(path: &str) {
    let circuit = parse_gate_list(&read_stdin());
    let probs = dense_probabilities(&circuit);
    std::fs::write(path, probability_bytes(&probs))
        .expect("failed to write the probability vector");
    println!(
        "{{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"prismq\",\"num_qubits\":{},\
         \"num_operations\":{},\"length\":{},\"dtype\":\"<f8\",\"path\":{:?}}}",
        circuit.num_qubits,
        circuit.instructions.len(),
        probs.len(),
        path
    );
}

/// One line per outcome, `w0[,w1...] count`, words in hex with classical bit
/// `j` at bit `j % 64` of word `j / 64`, sorted so equal histograms are equal text.
fn counts_text(counts: &HashMap<Vec<u64>, u64>) -> String {
    let mut entries: Vec<(&Vec<u64>, &u64)> = counts.iter().collect();
    entries.sort();
    let mut out = String::new();
    for (key, count) in entries {
        let words: Vec<String> = key.iter().map(|w| format!("{w:x}")).collect();
        out.push_str(&words.join(","));
        out.push(' ');
        out.push_str(&count.to_string());
        out.push('\n');
    }
    out
}

fn fnv1a(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

struct ExecArgs {
    mode: String,
    shots: usize,
    iterations: usize,
    out: Option<String>,
    backend: String,
    budget_ms: f64,
    min_iterations: usize,
}

fn parse_exec_args(args: &[String]) -> ExecArgs {
    let mut parsed = ExecArgs {
        mode: "probs".to_string(),
        shots: 0,
        iterations: 5,
        out: None,
        backend: "auto".to_string(),
        budget_ms: f64::INFINITY,
        min_iterations: 1,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .unwrap_or_else(|| fail(1, format!("{flag} needs a value")));
        match flag.as_str() {
            "--mode" => parsed.mode = value.clone(),
            "--shots" => parsed.shots = parse_index(value),
            "--iterations" => parsed.iterations = parse_index(value),
            "--out" => parsed.out = Some(value.clone()),
            "--backend" => parsed.backend = value.clone(),
            "--budget-ms" => parsed.budget_ms = parse_angle(value),
            "--min-iterations" => parsed.min_iterations = parse_index(value),
            other => fail(1, format!("unknown exec flag {other}")),
        }
    }
    parsed
}

fn json_str(s: &str) -> String {
    format!("{s:?}")
}

/// Time one simulator call end to end: plan inspection, a first (cold) call
/// whose output is written for validation, then up to `--iterations` warm
/// calls, stopping once the warm calls have spent `--budget-ms` and at least
/// `--min-iterations` have run.
fn run_exec(args: &[String]) {
    let args = parse_exec_args(args);
    let text = read_stdin();
    let build_start = Instant::now();
    let circuit = parse_gate_list(&text);
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let kind = parse_backend(&args.backend);

    let query = match args.mode.as_str() {
        "probs" => Query::Run,
        "counts" => Query::Counts(args.shots),
        other => fail(1, format!("unknown mode {other} (probs, counts)")),
    };
    let mut plan_us: Vec<f64> = Vec::with_capacity(PLAN_REPS);
    let mut plan_desc = String::new();
    for _ in 0..PLAN_REPS {
        let start = Instant::now();
        let plan = sim::simulate(&circuit)
            .backend(kind.clone())
            .seed(SIM_SEED)
            .plan(query);
        plan_us.push(start.elapsed().as_secs_f64() * 1e6);
        plan_desc = match plan {
            Ok(plan) => format!("{:?} via {:?}", plan.backend, plan.rule),
            Err(err) => format!("plan error: {err}"),
        };
    }
    plan_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let execute = |kind: BackendKind| -> (Vec<u8>, sim::RunMetadata) {
        match args.mode.as_str() {
            "probs" => {
                let (probs, meta) = dense_probabilities_on(&circuit, kind);
                (probability_bytes(&probs), meta)
            }
            _ => {
                let (counts, meta) = sample_counts_on(&circuit, kind, args.shots);
                (counts_text(&counts).into_bytes(), meta)
            }
        }
    };

    let first_start = Instant::now();
    let (first_output, metadata) = execute(kind.clone());
    let first_ms = first_start.elapsed().as_secs_f64() * 1000.0;
    let mut hashes = vec![fnv1a(&first_output)];
    if let Some(path) = &args.out {
        std::fs::write(path, &first_output).expect("failed to write the output");
    }
    drop(first_output);

    let mut times_ms: Vec<f64> = Vec::with_capacity(args.iterations);
    for i in 0..args.iterations {
        if i >= args.min_iterations && times_ms.iter().sum::<f64>() >= args.budget_ms {
            break;
        }
        let start = Instant::now();
        let output = match args.mode.as_str() {
            "probs" => {
                let (probs, _) = dense_probabilities_on(&circuit, kind.clone());
                times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                probability_bytes(&probs)
            }
            _ => {
                let (counts, _) = sample_counts_on(&circuit, kind.clone(), args.shots);
                times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                counts_text(&counts).into_bytes()
            }
        };
        hashes.push(fnv1a(&output));
    }

    let hash_list = hashes
        .iter()
        .map(|h| json_str(h))
        .collect::<Vec<_>>()
        .join(",");
    let bond = metadata
        .bond
        .as_ref()
        .map(|b| json_str(&format!("{b:?}")))
        .unwrap_or_else(|| "null".to_string());
    println!(
        "{{\"schema\":\"prismq-compare-runner/3\",\"simulator\":\"prismq\",\"version\":\"{}\",\
         \"num_qubits\":{},\"num_operations\":{},\"requested_backend\":{},\"threads\":{},\
         \"build_ms\":{build_ms:.4},\"plan_median_us\":{:.3},\"plan\":{},\
         \"resolved_backend\":{},\"route\":{},\"engine\":{},\"exactness\":{},\"bond\":{bond},\
         \"first_ms\":{first_ms:.4},\"times_ms\":[{}],\"output_hashes\":[{hash_list}]}}",
        env!("CARGO_PKG_VERSION"),
        circuit.num_qubits,
        circuit.instructions.len(),
        json_str(&args.backend),
        json_str(&rayon_threads()),
        plan_us[plan_us.len() / 2],
        json_str(&plan_desc),
        json_str(&format!("{:?}", metadata.backend)),
        json_str(&format!("{:?}", metadata.reason)),
        json_str(&format!("{:?}", metadata.engine)),
        json_str(&format!("{:?}", metadata.exactness)),
        join_ms(&times_ms),
    );
}

/// Instruction counts before and after the fusion pass the statevector runs,
/// by gate kind: the number of passes over the state each circuit costs.
fn run_fusion() {
    let circuit = parse_gate_list(&read_stdin());
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for inst in &fused.instructions {
        let name = match inst {
            Instruction::Gate { gate, .. } => {
                let text = format!("{gate:?}");
                text.split(['(', ' ', '{']).next().unwrap_or("").to_string()
            }
            other => {
                let text = format!("{other:?}");
                text.split([' ', '{']).next().unwrap_or("").to_string()
            }
        };
        *kinds.entry(name).or_default() += 1;
    }
    let kinds = kinds
        .iter()
        .map(|(k, v)| format!("{}:{v}", json_str(k)))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"num_qubits\":{},\"input_instructions\":{},\"fused_instructions\":{},\"kinds\":{{{kinds}}}}}",
        circuit.num_qubits,
        circuit.instructions.len(),
        fused.instructions.len()
    );
}

fn usage() -> ! {
    fail(
        1,
        "Usage: compare_runner export <ghz|qft|hea|qv> <qubits> [seed]\n\
         \x20      compare_runner time <iterations>        (gate list on stdin)\n\
         \x20      compare_runner probabilities <out_path> (gate list on stdin)\n\
         \x20      compare_runner exec --mode probs|counts [--shots S] [--iterations K]\n\
         \x20                          [--out PATH] [--backend NAME] [--budget-ms B]\n\
         \x20                          [--min-iterations M]   (gate list on stdin)\n\
         \x20      compare_runner fusion                   (gate list on stdin)\n\
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
        Some("export") if args.len() == 4 || args.len() == 5 => {
            let n: usize = args[3].parse().unwrap_or_else(|_| usage());
            let seed = match args.get(4) {
                Some(s) => {
                    let s = s.trim_start_matches("0x").replace('_', "");
                    u64::from_str_radix(&s, 16).unwrap_or_else(|_| usage())
                }
                None => CIRCUIT_SEED,
            };
            export(&args[2], n, seed);
        }
        Some("time") if args.len() == 3 => {
            let iterations: usize = args[2].parse().unwrap_or_else(|_| usage());
            if iterations == 0 {
                usage();
            }
            run_time(iterations);
        }
        Some("probabilities") if args.len() == 3 => run_probabilities(&args[2]),
        Some("exec") => run_exec(&args[2..]),
        Some("fusion") => run_fusion(),
        _ => usage(),
    }
}
