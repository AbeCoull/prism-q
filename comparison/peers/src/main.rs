//! Rust comparators for the cross-simulator comparison harness: Spinoza and
//! RustQIP (`qip`), driven through the same gate-list protocol as
//! `examples/compare_runner.rs`.
//!
//! ```text
//! peers <spinoza|qip> time <iterations> [--threads N]   (gate list on stdin)
//! peers <spinoza|qip> probabilities <out_path>          (gate list on stdin)
//! peers <spinoza|qip> version
//! ```
//!
//! The timed region covers state allocation, circuit execution, and the
//! probability read-out, matching the other adapters. Both simulators index the
//! state with qubit 0 as the least significant bit, which `probe_order` checks
//! at startup rather than assumes.

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::time::Instant;

use qip::builder::{LocalBuilder, Qudit};
use qip::prelude::*;
use spinoza::circuit::{QuantumCircuit, QuantumRegister};
use spinoza::config::Config;
use spinoza::core::CONFIG;

#[derive(Clone, Copy, Debug)]
enum Op {
    H(usize),
    Cx(usize, usize),
    Swap(usize, usize),
    Ry(usize, f64),
    Rz(usize, f64),
    Cp(usize, usize, f64),
}

struct Program {
    num_qubits: usize,
    ops: Vec<Op>,
}

fn fail(code: i32, message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}

fn parse_gate_list(text: &str) -> Program {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().unwrap_or_else(|| fail(3, "empty gate list"));
    let num_qubits: usize = match header.split_whitespace().collect::<Vec<_>>().as_slice() {
        ["qubits", n] => n.parse().unwrap_or_else(|_| fail(3, "bad qubit count")),
        _ => fail(3, format!("expected `qubits N`, got `{header}`")),
    };
    let q = |s: &str| -> usize {
        s.parse()
            .unwrap_or_else(|_| fail(3, format!("bad qubit index `{s}`")))
    };
    let a = |s: &str| -> f64 {
        s.parse()
            .unwrap_or_else(|_| fail(3, format!("bad angle `{s}`")))
    };
    let mut ops = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        ops.push(match parts.as_slice() {
            ["h", t] => Op::H(q(t)),
            ["cx", c, t] => Op::Cx(q(c), q(t)),
            ["swap", x, y] => Op::Swap(q(x), q(y)),
            ["ry", t, theta] => Op::Ry(q(t), a(theta)),
            ["rz", t, theta] => Op::Rz(q(t), a(theta)),
            ["cp", c, t, theta] => Op::Cp(q(c), q(t), a(theta)),
            _ => fail(3, format!("unsupported gate line `{line}`")),
        });
    }
    Program { num_qubits, ops }
}

fn run_spinoza(program: &Program) -> Vec<f64> {
    let mut qr = QuantumRegister::new(program.num_qubits);
    let mut circuit = QuantumCircuit::new(&mut [&mut qr]);
    for op in &program.ops {
        match *op {
            Op::H(t) => circuit.h(t),
            Op::Cx(c, t) => circuit.cx(c, t),
            Op::Swap(x, y) => circuit.swap(x, y),
            Op::Ry(t, theta) => circuit.ry(theta, t),
            Op::Rz(t, theta) => circuit.rz(theta, t),
            Op::Cp(c, t, theta) => circuit.cp(theta, c, t),
        }
    }
    circuit.execute();
    let state = circuit.get_statevector();
    state
        .reals
        .iter()
        .zip(&state.imags)
        .map(|(re, im)| re * re + im * im)
        .collect()
}

fn run_qip(program: &Program) -> Vec<f64> {
    let mut b = LocalBuilder::<f64>::default();
    let register = b.register(NonZeroUsize::new(program.num_qubits).expect("qubits > 0"));
    let mut qubits: Vec<Option<Qudit>> = b
        .split_all_register(register)
        .into_iter()
        .map(Some)
        .collect();
    let take = |qubits: &mut Vec<Option<Qudit>>, i: usize| qubits[i].take().expect("qubit in use");
    for op in &program.ops {
        match *op {
            Op::H(t) => {
                let r = take(&mut qubits, t);
                qubits[t] = Some(b.h(r));
            }
            Op::Cx(c, t) => {
                let (rc, rt) = (take(&mut qubits, c), take(&mut qubits, t));
                let (rc, rt) = b
                    .cnot(rc, rt)
                    .unwrap_or_else(|e| fail(4, format!("qip: {e:?}")));
                qubits[c] = Some(rc);
                qubits[t] = Some(rt);
            }
            Op::Swap(x, y) => {
                let (rx, ry) = (take(&mut qubits, x), take(&mut qubits, y));
                let (rx, ry) = b
                    .swap(rx, ry)
                    .unwrap_or_else(|e| fail(4, format!("qip: {e:?}")));
                qubits[x] = Some(rx);
                qubits[y] = Some(ry);
            }
            Op::Ry(t, theta) => {
                let r = take(&mut qubits, t);
                let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
                let m = vec![
                    Complex::new(c, 0.0),
                    Complex::new(-s, 0.0),
                    Complex::new(s, 0.0),
                    Complex::new(c, 0.0),
                ];
                qubits[t] = Some(
                    b.apply_vec_matrix(r, m)
                        .unwrap_or_else(|e| fail(4, format!("qip: {e:?}"))),
                );
            }
            Op::Rz(t, theta) => {
                let r = take(&mut qubits, t);
                let m = vec![
                    Complex::from_polar(1.0, -theta / 2.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::from_polar(1.0, theta / 2.0),
                ];
                qubits[t] = Some(
                    b.apply_vec_matrix(r, m)
                        .unwrap_or_else(|e| fail(4, format!("qip: {e:?}"))),
                );
            }
            Op::Cp(c, t, theta) => {
                // qip leaves conditioned matrix application unimplemented, so the
                // controlled phase is the textbook Rz and CX form, equal up to a
                // global phase: Rz(t/2) on both, CX, Rz(-t/2) on the target, CX.
                let (rc, rt) = (take(&mut qubits, c), take(&mut qubits, t));
                let rc = b.rz(rc, theta / 2.0);
                let rt = b.rz(rt, theta / 2.0);
                let (rc, rt) = b
                    .cnot(rc, rt)
                    .unwrap_or_else(|e| fail(4, format!("qip: {e:?}")));
                let rt = b.rz(rt, -theta / 2.0);
                let (rc, rt) = b
                    .cnot(rc, rt)
                    .unwrap_or_else(|e| fail(4, format!("qip: {e:?}")));
                qubits[c] = Some(rc);
                qubits[t] = Some(rt);
            }
        }
    }
    let (state, _) = b.calculate_state();
    state.iter().map(|amp| amp.norm_sqr()).collect()
}

/// Whether a simulator indexes the state with qubit 0 as the least significant
/// bit. An X on qubit 0 of a two-qubit register must light index 1; index 2
/// means the opposite order and the output is reindexed.
fn probe_order(run: fn(&Program) -> Vec<f64>) -> bool {
    let probe = Program {
        num_qubits: 2,
        ops: vec![Op::H(0), Op::H(0), Op::Ry(0, std::f64::consts::PI)],
    };
    let probs = run(&probe);
    if (probs[1] - 1.0).abs() < 1e-9 {
        true
    } else if (probs[2] - 1.0).abs() < 1e-9 {
        false
    } else {
        fail(
            4,
            format!("order probe produced an unexpected state: {probs:?}"),
        )
    }
}

fn reindex(probs: Vec<f64>, num_qubits: usize) -> Vec<f64> {
    let mut out = vec![0.0; probs.len()];
    for (i, p) in probs.into_iter().enumerate() {
        let mut j = 0usize;
        for bit in 0..num_qubits {
            if i >> bit & 1 == 1 {
                j |= 1 << (num_qubits - 1 - bit);
            }
        }
        out[j] = p;
    }
    out
}

struct Simulator {
    name: &'static str,
    version: &'static str,
    run: fn(&Program) -> Vec<f64>,
    lsb_first: bool,
}

fn simulator(name: &str, threads: usize) -> Simulator {
    match name {
        "spinoza" => {
            let config = Config {
                threads: threads as u32,
                qubits: 0,
                print: false,
            };
            if CONFIG.set(config).is_err() {
                fail(2, "spinoza config already set");
            }
            Simulator {
                name: "spinoza",
                version: "0.5.1 (git f900971)",
                run: run_spinoza,
                lsb_first: probe_order(run_spinoza),
            }
        }
        "qip" => Simulator {
            name: "qip",
            version: "1.5.0",
            run: run_qip,
            lsb_first: probe_order(run_qip),
        },
        other => fail(2, format!("unknown simulator: {other} (spinoza, qip)")),
    }
}

fn probabilities(sim: &Simulator, program: &Program) -> Vec<f64> {
    let probs = (sim.run)(program);
    if sim.lsb_first {
        probs
    } else {
        reindex(probs, program.num_qubits)
    }
}

fn read_stdin() -> String {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("failed to read the gate list from stdin");
    text
}

fn usage() -> ! {
    fail(
        1,
        "Usage: peers <spinoza|qip> time <iterations> [--threads N]  (gate list on stdin)\n\
         \x20      peers <spinoza|qip> probabilities <out_path>         (gate list on stdin)\n\
         \x20      peers <spinoza|qip> version",
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }
    let threads = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    // The qip comparator runs on the global Rayon pool, which reads RAYON_NUM_THREADS.
    let sim = simulator(&args[1], threads);
    match args[2].as_str() {
        "version" => println!(
            "{{\"simulator\":\"{}\",\"version\":\"{}\"}}",
            sim.name, sim.version
        ),
        "time" => {
            let iterations: usize = args
                .get(3)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| usage());
            let program = parse_gate_list(&read_stdin());
            std::hint::black_box(probabilities(&sim, &program));
            let mut times_ms = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let probs = probabilities(&sim, &program);
                times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(probs);
            }
            let mut sorted = times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            print!(
                "{{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"{}\",\"version\":\"{}\",\
                 \"num_qubits\":{},\"num_operations\":{},\"threads\":\"{}\",\"median_ms\":{:.4},\
                 \"min_ms\":{:.4},\"times_ms\":[",
                sim.name,
                sim.version,
                program.num_qubits,
                program.ops.len(),
                threads,
                sorted[sorted.len() / 2],
                sorted[0]
            );
            for (i, t) in times_ms.iter().enumerate() {
                if i > 0 {
                    print!(",");
                }
                print!("{t:.4}");
            }
            println!("]}}");
        }
        "probabilities" => {
            let path = args.get(3).unwrap_or_else(|| usage());
            let program = parse_gate_list(&read_stdin());
            let probs = probabilities(&sim, &program);
            let mut bytes = Vec::with_capacity(probs.len() * 8);
            for p in &probs {
                bytes.extend_from_slice(&p.to_le_bytes());
            }
            std::fs::write(path, &bytes).expect("failed to write the probability vector");
            let mut out = std::io::stdout();
            writeln!(
                out,
                "{{\"schema\":\"prismq-compare-runner/2\",\"simulator\":\"{}\",\"num_qubits\":{},\
                 \"num_operations\":{},\"length\":{},\"dtype\":\"<f8\",\"path\":{:?}}}",
                sim.name,
                program.num_qubits,
                program.ops.len(),
                probs.len(),
                path
            )
            .expect("stdout");
        }
        _ => usage(),
    }
}
