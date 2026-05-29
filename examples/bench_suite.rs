//! PRISM-Q benchmark suite: time the library and publish the results page.
//!
//! Builds a fixed circuit suite from `prism_q::circuits` (the same generators the
//! library ships), times each circuit with auto dispatch, and rewrites the
//! published report at `docs/benchmarks.md`. Circuits come from `circuits.rs`
//! directly, so the published numbers track the library's own circuit
//! definitions with no duplicated suite.
//!
//! The suite is pushed toward this machine's limits. Dense families (QFT, HEA,
//! QV) are bounded by the statevector memory cap; the Clifford family (GHZ)
//! routes to the stabilizer backend under auto dispatch and runs far past that
//! ceiling. Sizes that exceed available memory raise an error from `run()` and
//! are skipped rather than aborting the suite.
//!
//! Methodology: wall-clock simulation time only. Circuit construction happens
//! once per circuit outside the timed region. The timed call is the full
//! user-facing `simulate().run()`, which includes the fusion pass and
//! probability extraction.
//!
//! Usage: cargo run --release --features parallel --example bench_suite

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use prism_q::circuit::Circuit;
use prism_q::{circuits, simulate, BackendKind, Result};

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

struct Row {
    family: &'static str,
    qubits: usize,
    median_ms: f64,
}

fn runs_for(qubits: usize) -> (usize, usize) {
    match qubits {
        0..=18 => (2, 7),
        19..=22 => (1, 5),
        23..=25 => (1, 3),
        _ => (0, 1),
    }
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
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

fn time_run(circuit: &Circuit) -> Result<f64> {
    let start = Instant::now();
    let outcome = simulate(circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .run()?;
    std::hint::black_box(&outcome);
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.3} s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{ms:.2} ms")
    } else {
        format!("{:.1} us", ms * 1000.0)
    }
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
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(idx) = rest.find(':') {
                    return rest[idx + 1..].trim().to_string();
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn render(rows: &[Row], threads: usize) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let mut s = String::new();
    let _ = writeln!(s, "# Benchmarks\n");
    let _ = writeln!(
        s,
        "Wall-clock simulation time for PRISM-Q on a fixed circuit suite built from \
         the `prism_q::circuits` generators, pushed toward this machine's limits. \
         Every number is reproducible with the command at the bottom of this page.\n"
    );
    let _ = writeln!(
        s,
        "<!-- Generated by `cargo run --example bench_suite`. Do not edit by hand. -->\n"
    );
    let _ = writeln!(s, "## Setup\n");
    let _ = writeln!(s, "- Date: {}", utc_date(now));
    let _ = writeln!(s, "- CPU: {}", cpu_model());
    let _ = writeln!(s, "- Threads available: {threads}");
    let _ = writeln!(s, "- PRISM-Q version: {}\n", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(s, "## Methodology\n");
    let _ = writeln!(
        s,
        "- Metric: median wall-clock over repeated runs after warmup, lower is better."
    );
    let _ = writeln!(
        s,
        "- Timed region is simulation only. Circuit construction happens once per \
         circuit outside the timer."
    );
    let _ = writeln!(
        s,
        "- Timings use the full `simulate().run()` path, including the fusion pass \
         and probability extraction."
    );
    let _ = writeln!(
        s,
        "- `auto` lets backend dispatch choose a specialized backend per circuit. \
         The dense families (QFT, HEA, QV) are bounded by the statevector memory \
         cap; GHZ is Clifford and runs into the thousands of qubits on the \
         stabilizer backend. QV uses square depth, so its gate count grows with \
         the qubit count and it is compute-bound earlier than the fixed-depth \
         families.\n"
    );

    for (family, _) in FAMILIES {
        let _ = writeln!(s, "## {}\n", family.to_uppercase());
        let _ = writeln!(s, "| Qubits | auto |");
        let _ = writeln!(s, "|---|---|");
        for row in rows.iter().filter(|r| r.family == family) {
            let _ = writeln!(s, "| {} | {} |", row.qubits, fmt_ms(row.median_ms));
        }
        s.push('\n');
    }

    let _ = writeln!(s, "## Reproducing\n");
    let _ = writeln!(s, "```bash");
    let _ = writeln!(
        s,
        "cargo run --release --features parallel --example bench_suite"
    );
    let _ = writeln!(s, "```\n");
    let _ = writeln!(
        s,
        "Times PRISM-Q on every circuit in the suite and rewrites this page."
    );
    s
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut rows: Vec<Row> = Vec::new();

    for (family, sizes) in FAMILIES {
        for &n in sizes {
            let id = label(family, n);
            let circuit = build(family, n);
            let (warmup, runs) = runs_for(n);

            let mut samples = Vec::with_capacity(runs);
            let mut failure = None;
            for _ in 0..(warmup + runs) {
                match time_run(&circuit) {
                    Ok(ms) => samples.push(ms),
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }

            if let Some(e) = failure {
                eprintln!("{id:<16} skipped ({e})");
                continue;
            }

            let timed = samples.split_off(warmup);
            let med = median(timed);
            println!("{id:<16} auto {med:>11.3} ms ({runs} runs)");
            rows.push(Row {
                family,
                qubits: n,
                median_ms: med,
            });
        }
    }

    let out_path = root.join("docs/benchmarks.md");
    fs::write(&out_path, render(&rows, threads)).expect("write benchmarks.md");
    eprintln!("wrote {}", out_path.display());
}
