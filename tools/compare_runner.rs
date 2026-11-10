//! PRISM-Q side of the cross-simulator comparison harness in `comparison/`.
//!
//! `export` writes one of the benchmark suite's circuit families as a gate list
//! that every comparator replays natively, so each simulator runs the same
//! gates in the same order. `time`, `probabilities` and `exec` read a gate list
//! back from stdin. Building the circuit from the list is excluded from the
//! timed region here and in the qip and Python adapters, which build their
//! circuit objects first. Spinoza records its gate list on the circuit that owns
//! the state and drains it on execute, and QuEST applies each gate as it is
//! called, so their timed regions include the per-gate calls that record or
//! apply the circuit; parsing the list stays outside every timed region.
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
use prism_q::{PackedShots, PauliTerm, ShotLayout, compile_measurements};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::time::Instant;

const CIRCUIT_SEED: u64 = 0xDEAD_BEEF;
const SIM_SEED: u64 = 42;
const HEA_LAYERS: usize = 5;
const PLAN_REPS: usize = 21;
const STATS_CHUNK_BYTES: usize = 1 << 26;

fn read_stdin() -> String {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .unwrap_or_else(|err| fail(3, format!("failed to read the gate list from stdin: {err}")));
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
        .unwrap_or_else(|err| fail(5, format!("failed to write the gate list: {err}")));
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

fn check_qubits(targets: &[usize], n: usize, line: &str) {
    if let Some(q) = targets.iter().find(|&&q| q >= n) {
        fail(
            3,
            format!("qubit {q} out of range for {n} qubits in `{line}`"),
        );
    }
    if targets.len() == 2 && targets[0] == targets[1] {
        fail(3, format!("repeated qubit in `{line}`"));
    }
}

fn check_clbit(bit: usize, clbits: usize, line: &str) {
    if bit >= clbits {
        fail(
            3,
            format!("classical bit {bit} out of range for {clbits} clbits in `{line}`"),
        );
    }
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
            ["measure", qubit, bit] => {
                let (qubit, bit) = (parse_index(qubit), parse_index(bit));
                check_qubits(&[qubit], n, line);
                check_clbit(bit, clbits, line);
                circuit.add_measure(qubit, bit);
            }
            ["reset", qubit] => {
                let qubit = parse_index(qubit);
                check_qubits(&[qubit], n, line);
                circuit.add_reset(qubit);
            }
            ["if", bit, value, rest @ ..] => {
                let Some((gate, targets)) = parse_gate(rest) else {
                    fail(3, format!("unsupported conditional `{line}`"));
                };
                check_qubits(&targets, n, line);
                let bit = parse_index(bit);
                check_clbit(bit, clbits, line);
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
                Some((gate, targets)) => {
                    check_qubits(&targets, n, line);
                    circuit.add_gate(gate, &targets);
                }
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

fn pool_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
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
        pool_threads(),
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

/// Write `probs` as little-endian f64 through a bounded buffer, so a 2^30
/// vector is not duplicated in memory.
fn write_probabilities(path: &str, probs: &[f64]) {
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
        for chunk in probs.chunks(1 << 13) {
            buf.clear();
            for p in chunk {
                buf.extend_from_slice(&p.to_le_bytes());
            }
            file.write_all(&buf)?;
        }
        file.flush()
    };
    write().unwrap_or_else(|err| fail(5, format!("failed to write `{path}`: {err}")));
}

fn run_probabilities(path: &str) {
    let circuit = parse_gate_list(&read_stdin());
    let probs = dense_probabilities(&circuit);
    write_probabilities(path, &probs);
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

/// Multiplicative word hash over the f64 bit patterns in place. Four lanes keep
/// the multiply chain from serializing a 2^n pass; each step is a bijection of
/// its word, so a single changed probability always changes the hash.
fn hash_probs(probs: &[f64]) -> String {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let step = |h: u64, w: u64| (h.rotate_left(5) ^ w).wrapping_mul(K);
    let mut lanes = [0u64, 1, 2, 3];
    let (chunks, tail) = probs.as_chunks::<4>();
    for chunk in chunks {
        for (lane, p) in lanes.iter_mut().zip(chunk) {
            *lane = step(*lane, p.to_bits());
        }
    }
    let hash = lanes
        .into_iter()
        .chain(tail.iter().map(|p| p.to_bits()))
        .fold(probs.len() as u64, step);
    format!("{hash:016x}")
}

enum Output {
    Probs(Vec<f64>),
    Text(String),
}

impl Output {
    fn hash(&self) -> String {
        match self {
            Output::Probs(probs) => hash_probs(probs),
            Output::Text(text) => fnv1a(text.as_bytes()),
        }
    }

    fn write(&self, path: &str) {
        match self {
            Output::Probs(probs) => write_probabilities(path, probs),
            Output::Text(text) => std::fs::write(path, text)
                .unwrap_or_else(|err| fail(5, format!("failed to write `{path}`: {err}"))),
        }
    }
}

/// Keys from a `--keys` file, one per line: `Z5` or `Z5Z6` for `expval`, `b5` or
/// `b5^b6` for `stats`.
fn read_keys(path: Option<&str>) -> Vec<String> {
    let path = path.unwrap_or_else(|| fail(1, "this mode needs --keys PATH"));
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| fail(3, format!("failed to read `{path}`: {err}")));
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

fn z_observable(key: &str, n: usize) -> Vec<PauliTerm> {
    let qubits: Vec<usize> = key.split('Z').skip(1).map(parse_index).collect();
    if !key.starts_with('Z') || qubits.is_empty() || qubits.iter().any(|&q| q >= n) {
        fail(3, format!("bad observable key `{key}` for {n} qubits"));
    }
    qubits.into_iter().map(PauliTerm::z).collect()
}

fn values_text(keys: &[String], values: &[f64]) -> String {
    keys.iter()
        .zip(values)
        .map(|(k, v)| format!("{k} {v:?}\n"))
        .collect()
}

/// Measurement record indices a statistic key reads: one for `bA`, two for `bA^bB`.
fn stat_records(key: &str, record_of_bit: &HashMap<usize, usize>) -> Vec<usize> {
    key.split('^')
        .map(|part| {
            let bit = part
                .strip_prefix('b')
                .map(parse_index)
                .unwrap_or_else(|| fail(3, format!("bad statistic key `{key}`")));
            *record_of_bit
                .get(&bit)
                .unwrap_or_else(|| fail(3, format!("classical bit {bit} is never measured")))
        })
        .collect()
}

/// Add each statistic's count of ones (or of odd parities) over one shot chunk.
fn reduce_stats(packed: &PackedShots, records: &[Vec<usize>], totals: &mut [u64]) {
    let owned;
    let packed = if packed.layout() == ShotLayout::MeasMajor {
        packed
    } else {
        owned = packed.to_meas_major();
        &owned
    };
    for (total, recs) in totals.iter_mut().zip(records) {
        let ones: u64 = match recs.as_slice() {
            [m] => packed
                .meas_words(*m)
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum(),
            [a, b] => packed
                .meas_words(*a)
                .iter()
                .zip(packed.meas_words(*b))
                .map(|(x, y)| u64::from((x ^ y).count_ones()))
                .sum(),
            _ => unreachable!("a statistic reads one or two bits"),
        };
        *total += ones;
    }
}

/// What a run reports about the route it took, as display strings.
struct RunInfo {
    backend: String,
    route: String,
    engine: String,
    exactness: String,
    bond: Option<String>,
}

impl RunInfo {
    fn from_metadata(meta: &sim::RunMetadata) -> Self {
        Self {
            backend: format!("{:?}", meta.backend),
            route: format!("{:?}", meta.reason),
            engine: format!("{:?}", meta.engine),
            exactness: format!("{:?}", meta.exactness),
            bond: meta.bond.as_ref().map(|b| format!("{b:?}")),
        }
    }
}

struct ExecArgs {
    mode: String,
    shots: usize,
    iterations: usize,
    out: Option<String>,
    backend: String,
    budget_ms: f64,
    min_iterations: usize,
    keys: Option<String>,
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
        keys: None,
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
            "--keys" => parsed.keys = Some(value.clone()),
            other => fail(1, format!("unknown exec flag {other}")),
        }
    }
    parsed
}

fn json_str(s: &str) -> String {
    format!("{s:?}")
}

/// Time one simulator call end to end: a first (cold) call whose output is
/// written for validation, plan inspection, then up to `--iterations` warm
/// calls, stopping once the warm calls have spent `--budget-ms` and at least
/// `--min-iterations` have run.
///
/// `expval` times one `expectation_values` call over the `--keys` observables.
/// `stats` times `compile_measurements` plus `sample_bulk_packed` into packed
/// buffers of at most 64 MiB; reducing each buffer to the `--keys` counts
/// happens outside the timer.
fn run_exec(args: &[String]) {
    let args = parse_exec_args(args);
    let text = read_stdin();
    let build_start = Instant::now();
    let circuit = parse_gate_list(&text);
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let kind = parse_backend(&args.backend);

    let keys = match args.mode.as_str() {
        "expval" | "stats" => read_keys(args.keys.as_deref()),
        "probs" | "counts" => Vec::new(),
        other => fail(
            1,
            format!("unknown mode {other} (probs, counts, expval, stats)"),
        ),
    };
    let observables: Vec<Vec<PauliTerm>> = if args.mode == "expval" {
        keys.iter()
            .map(|k| z_observable(k, circuit.num_qubits))
            .collect()
    } else {
        Vec::new()
    };
    let stat_rows: Vec<Vec<usize>> = if args.mode == "stats" {
        if !matches!(kind, BackendKind::Auto) {
            fail(
                1,
                "stats mode samples through compile_measurements; pass --backend auto",
            );
        }
        let record_of_bit: HashMap<usize, usize> = circuit
            .measurement_map()
            .into_iter()
            .enumerate()
            .map(|(record, (_, bit))| (bit, record))
            .collect();
        keys.iter()
            .map(|k| stat_records(k, &record_of_bit))
            .collect()
    } else {
        Vec::new()
    };
    let query = match args.mode.as_str() {
        "probs" => Some(Query::Run),
        "counts" => Some(Query::Counts(args.shots)),
        "expval" => Some(Query::Expectation(&observables)),
        _ => None,
    };

    let execute = |kind: BackendKind| -> (Output, RunInfo, f64) {
        match args.mode.as_str() {
            "probs" => {
                let start = Instant::now();
                let (probs, meta) = dense_probabilities_on(&circuit, kind);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                (Output::Probs(probs), RunInfo::from_metadata(&meta), ms)
            }
            "counts" => {
                let start = Instant::now();
                let (counts, meta) = sample_counts_on(&circuit, kind, args.shots);
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                (
                    Output::Text(counts_text(&counts)),
                    RunInfo::from_metadata(&meta),
                    ms,
                )
            }
            "expval" => {
                let start = Instant::now();
                let result = sim::simulate(&circuit)
                    .backend(kind)
                    .seed(SIM_SEED)
                    .expectation_values_reported(&observables)
                    .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                (
                    Output::Text(values_text(&keys, &result.values)),
                    RunInfo::from_metadata(&result.metadata),
                    ms,
                )
            }
            _ => {
                let measurements = circuit.measurement_map().len().max(1);
                let chunk = (STATS_CHUNK_BYTES * 8 / measurements / 64).max(1) * 64;
                let mut totals = vec![0u64; stat_rows.len()];
                let start = Instant::now();
                let mut sampler = compile_measurements(&circuit, SIM_SEED)
                    .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
                let mut ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut left = args.shots;
                while left > 0 {
                    let take = left.min(chunk);
                    let start = Instant::now();
                    let packed = sampler
                        .try_sample_bulk_packed(take)
                        .unwrap_or_else(|err| fail(4, format!("simulation error: {err}")));
                    ms += start.elapsed().as_secs_f64() * 1000.0;
                    reduce_stats(&packed, &stat_rows, &mut totals);
                    left -= take;
                }
                let mut out = format!("shots {}\n", args.shots);
                for (key, total) in keys.iter().zip(&totals) {
                    out.push_str(&format!("{key} {total}\n"));
                }
                let info = RunInfo {
                    backend: "CompiledStabilizer".to_string(),
                    route: "compile_measurements".to_string(),
                    engine: "CompiledSampler".to_string(),
                    exactness: "Exact".to_string(),
                    bond: None,
                };
                (Output::Text(out), info, ms)
            }
        }
    };

    let (first_output, info, first_ms) = execute(kind.clone());
    let mut hashes = vec![first_output.hash()];
    if let Some(path) = &args.out {
        first_output.write(path);
    }
    drop(first_output);

    let mut plan_us: Vec<f64> = Vec::with_capacity(PLAN_REPS);
    let mut plan_desc = "compile_measurements, no dispatch".to_string();
    if let Some(query) = query {
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
    }
    plan_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let plan_median = plan_us
        .get(plan_us.len() / 2)
        .map(|us| format!("{us:.3}"))
        .unwrap_or_else(|| "null".to_string());

    let mut times_ms: Vec<f64> = Vec::with_capacity(args.iterations);
    for i in 0..args.iterations {
        if i >= args.min_iterations && times_ms.iter().sum::<f64>() >= args.budget_ms {
            break;
        }
        let (output, _, ms) = execute(kind.clone());
        times_ms.push(ms);
        hashes.push(output.hash());
    }

    let hash_list = hashes
        .iter()
        .map(|h| json_str(h))
        .collect::<Vec<_>>()
        .join(",");
    let bond = info
        .bond
        .as_deref()
        .map(json_str)
        .unwrap_or_else(|| "null".to_string());
    println!(
        "{{\"schema\":\"prismq-compare-runner/3\",\"simulator\":\"prismq\",\"version\":\"{}\",\
         \"num_qubits\":{},\"num_operations\":{},\"requested_backend\":{},\"threads\":{},\
         \"build_ms\":{build_ms:.4},\"plan_median_us\":{plan_median},\"plan\":{},\
         \"resolved_backend\":{},\"route\":{},\"engine\":{},\"exactness\":{},\"bond\":{bond},\
         \"first_ms\":{first_ms:.4},\"times_ms\":[{}],\"output_hashes\":[{hash_list}]}}",
        env!("CARGO_PKG_VERSION"),
        circuit.num_qubits,
        circuit.instructions.len(),
        json_str(&args.backend),
        json_str(&pool_threads().to_string()),
        json_str(&plan_desc),
        json_str(&info.backend),
        json_str(&info.route),
        json_str(&info.engine),
        json_str(&info.exactness),
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
         \x20      compare_runner exec --mode probs|counts|expval|stats [--shots S]\n\
         \x20                          [--iterations K] [--out PATH] [--backend NAME]\n\
         \x20                          [--budget-ms B] [--min-iterations M]\n\
         \x20                          [--keys PATH]          (gate list on stdin)\n\
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
