//! PRISM-Q benchmark suite: time the library and record the results for the Performance page.
//!
//! Builds a fixed circuit suite from `prism_q::circuits` (the same generators the
//! library ships), times each circuit with auto dispatch, and rewrites
//! `docs/performance/suite.json`, which `comparison/matrix.py report` folds into the
//! page. Circuits come from `circuits.rs` directly, so the published numbers track the
//! library's own circuit definitions with no duplicated suite.
//!
//! The suite is pushed toward this machine's limits. Dense families (QFT, HEA,
//! QV) are bounded by the statevector memory cap; the Clifford family (GHZ)
//! routes to the stabilizer backend under auto dispatch and runs far past that
//! ceiling. Sizes that exceed available memory raise an error from `run()` and
//! are recorded under `skipped` with the error rather than aborting the suite.
//!
//! Methodology: wall-clock simulation time only. Circuit construction happens
//! once per circuit outside the timed region. The timed call is the full
//! user-facing `simulate().run()`, which includes the fusion pass and
//! probability extraction when the resolved backend offers it (the stabilizer
//! rows time the tableau alone). Each row records the backend that ran, whether
//! probabilities were produced, and its warm sample count. One untimed warmup
//! precedes the samples, which continue until `BUDGET_MS` is spent and at least
//! `MIN_SAMPLES` have run; a row whose warmup took longer than
//! `SLOW_ROW_MS / MIN_SAMPLES` takes fewer, down to one.
//!
//! Usage: cargo run --release --features parallel --example bench_suite

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use prism_q::circuit::Circuit;
use prism_q::{BackendKind, Result, circuits, simulate};

const SEED: u64 = 0xDEAD_BEEF;
const HEA_LAYERS: usize = 5;

// Per-family qubit ladders. Dense families stop at the safe statevector ceiling
// for this machine; GHZ (Clifford) continues into the thousands on the
// stabilizer backend. QV uses square depth, so it is compute-bound and tops out
// earlier than the fixed-depth dense families.
const FAMILIES: [(&str, &[usize]); 4] = [
    ("ghz", &[24, 28, 256, 1024, 4096]),
    ("qft", &[16, 20, 24, 26, 28]),
    ("hea", &[16, 20, 24, 26, 28]),
    ("qv", &[16, 20, 24]),
];

const BUDGET_MS: f64 = 2000.0;
const MIN_SAMPLES: usize = 5;
const MAX_SAMPLES: usize = 50;
const SLOW_ROW_MS: f64 = 30_000.0;

struct Row {
    family: &'static str,
    qubits: usize,
    median_ms: f64,
    samples: usize,
    backend: String,
    probabilities: bool,
}

struct Skipped {
    family: &'static str,
    qubits: usize,
    reason: String,
}

struct Sample {
    ms: f64,
    backend: String,
    probabilities: bool,
}

fn build(family: &str, n: usize) -> Circuit {
    match family {
        "ghz" => circuits::ghz_circuit(n),
        "qft" => circuits::qft_circuit(n),
        "hea" => circuits::hardware_efficient_ansatz(n, HEA_LAYERS, SEED),
        "qv" => circuits::quantum_volume_circuit(n, n, SEED),
        other => panic!("unknown family: {other}"),
    }
}

fn label(family: &str, n: usize) -> String {
    match family {
        "hea" => format!("hea_{n}q_l{HEA_LAYERS}"),
        _ => format!("{family}_{n}q"),
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

fn time_run(circuit: &Circuit) -> Result<Sample> {
    let start = Instant::now();
    let outcome = simulate(circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .run()?;
    std::hint::black_box(&outcome);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Sample {
        ms,
        backend: format!("{:?}", outcome.metadata.backend),
        probabilities: outcome.probabilities.is_some(),
    })
}

/// Warm up once, then sample until the budget and the minimum count are met.
fn measure(circuit: &Circuit) -> Result<(Sample, Vec<f64>)> {
    let warmup = time_run(circuit)?;
    let min_samples = ((SLOW_ROW_MS / warmup.ms.max(1e-3)) as usize).clamp(1, MIN_SAMPLES);
    let mut samples = Vec::new();
    while samples.len() < MAX_SAMPLES
        && (samples.len() < min_samples || samples.iter().sum::<f64>() < BUDGET_MS)
    {
        samples.push(time_run(circuit)?.ms);
    }
    Ok((warmup, samples))
}

/// UTC date as `YYYY-MM-DD` from a Unix timestamp (Howard Hinnant's civil_from_days).
fn utc_date(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", y + i64::from(m <= 2), m, d)
}

fn cpu_model() -> String {
    #[cfg(target_os = "windows")]
    if let Ok(v) = std::env::var("PROCESSOR_IDENTIFIER") {
        return v;
    }
    #[cfg(target_os = "linux")]
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if let Some(rest) = line.strip_prefix("model name")
                && let Some(idx) = rest.find(':')
            {
                return rest[idx + 1..].trim().to_string();
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
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

fn json_str(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render(rows: &[Row], skipped: &[Skipped], threads: usize) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"version\": {},", json_str(env!("CARGO_PKG_VERSION")));
    let _ = writeln!(s, "  \"date\": {},", json_str(&utc_date(now)));
    let _ = writeln!(s, "  \"cpu\": {},", json_str(cpu_model().trim()));
    let _ = writeln!(s, "  \"logical_cores\": {threads},");
    let _ = writeln!(s, "  \"rayon_threads\": {},", pool_threads());
    let _ = writeln!(s, "  \"rustc\": {},", json_str(&rustc_version()));
    s.push_str("  \"rows\": [\n");
    for (i, row) in rows.iter().enumerate() {
        let sep = if i + 1 < rows.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{\"family\": {}, \"qubits\": {}, \"ms\": {:.4}, \"samples\": {}, \
             \"backend\": {}, \"probabilities\": {}}}{sep}",
            json_str(row.family),
            row.qubits,
            row.median_ms,
            row.samples,
            json_str(&row.backend),
            row.probabilities
        );
    }
    s.push_str("  ],\n  \"skipped\": [\n");
    for (i, row) in skipped.iter().enumerate() {
        let sep = if i + 1 < skipped.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{\"family\": {}, \"qubits\": {}, \"reason\": {}}}{sep}",
            json_str(row.family),
            row.qubits,
            json_str(&row.reason)
        );
    }
    s.push_str("  ]\n}\n");
    s
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut rows: Vec<Row> = Vec::new();
    let mut skipped: Vec<Skipped> = Vec::new();

    for (family, sizes) in FAMILIES {
        for &n in sizes {
            let id = label(family, n);
            let circuit = build(family, n);
            match measure(&circuit) {
                Ok((warmup, samples)) => {
                    let count = samples.len();
                    let med = median(samples);
                    println!(
                        "{id:<16} {:<12} {med:>11.3} ms ({count} runs)",
                        warmup.backend
                    );
                    rows.push(Row {
                        family,
                        qubits: n,
                        median_ms: med,
                        samples: count,
                        backend: warmup.backend,
                        probabilities: warmup.probabilities,
                    });
                }
                Err(e) => {
                    eprintln!("{id:<16} skipped ({e})");
                    skipped.push(Skipped {
                        family,
                        qubits: n,
                        reason: e.to_string().replace(['\n', '\r'], " "),
                    });
                }
            }
        }
    }

    let out_path = root.join("docs/performance/suite.json");
    fs::write(&out_path, render(&rows, &skipped, threads)).expect("write suite.json");
    eprintln!("wrote {}", out_path.display());
}
