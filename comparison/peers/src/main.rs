//! Rust comparators for the cross-simulator comparison harness: Spinoza and
//! RustQIP (`qip`), driven through the same gate-list protocol as
//! `tools/compare_runner.rs`.
//!
//! ```text
//! peers <spinoza|qip> time <iterations> [--threads N]   (gate list on stdin)
//! peers <spinoza|qip> probabilities <out_path>          (gate list on stdin)
//! peers <spinoza|qip> version
//! ```
//!
//! The timed region covers state allocation, circuit execution, and the
//! probability read-out. qip builds its circuit before the timer starts, as
//! PRISM-Q does; Spinoza records gates on the circuit that owns the state and
//! drains them on execute, so its gate calls stay inside the timer. The qubit
//! order of each simulator is checked by `probe_order` at startup rather than
//! assumed, and a reversed order is undone in the parallel read-out pass.

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::time::Instant;

use qip::builder::{LocalBuilder, Qudit};
use qip::prelude::*;
use rayon::prelude::*;
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
    if num_qubits == 0 {
        fail(3, "the register needs at least one qubit");
    }
    let mut ops = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let op = match parts.as_slice() {
            ["h", t] => Op::H(q(t)),
            ["cx", c, t] => Op::Cx(q(c), q(t)),
            ["swap", x, y] => Op::Swap(q(x), q(y)),
            ["ry", t, theta] => Op::Ry(q(t), a(theta)),
            ["rz", t, theta] => Op::Rz(q(t), a(theta)),
            ["cp", c, t, theta] => Op::Cp(q(c), q(t), a(theta)),
            _ => fail(3, format!("unsupported gate line `{line}`")),
        };
        let (first, second) = match op {
            Op::H(t) | Op::Ry(t, _) | Op::Rz(t, _) => (t, None),
            Op::Cx(x, y) | Op::Swap(x, y) | Op::Cp(x, y, _) => (x, Some(y)),
        };
        if let Some(bad) = std::iter::once(first)
            .chain(second)
            .find(|&i| i >= num_qubits)
        {
            fail(
                3,
                format!("qubit {bad} out of range for {num_qubits} qubits in `{line}`"),
            );
        }
        if second == Some(first) {
            fail(3, format!("repeated qubit in `{line}`"));
        }
        ops.push(op);
    }
    Program { num_qubits, ops }
}

/// Squared magnitudes for indices `0..len` in parallel. When the simulator puts
/// qubit 0 at the most significant bit, index `i` reads amplitude
/// `reverse(i)`, so the reorder costs no extra pass.
fn read_out(
    len: usize,
    num_qubits: usize,
    lsb_first: bool,
    norm: impl Fn(usize) -> f64 + Sync,
) -> Vec<f64> {
    if lsb_first {
        (0..len).into_par_iter().map(|i| norm(i)).collect()
    } else {
        let shift = usize::BITS as usize - num_qubits;
        (0..len)
            .into_par_iter()
            .map(|i| norm(i.reverse_bits() >> shift))
            .collect()
    }
}

fn run_spinoza(program: &Program, lsb_first: bool) -> Vec<f64> {
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
    read_out(state.reals.len(), program.num_qubits, lsb_first, |j| {
        state.reals[j] * state.reals[j] + state.imags[j] * state.imags[j]
    })
}

fn build_qip(program: &Program) -> LocalBuilder<f64> {
    let mut b = LocalBuilder::<f64>::default();
    let num_qubits = NonZeroUsize::new(program.num_qubits)
        .unwrap_or_else(|| fail(3, "the register needs at least one qubit"));
    let register = b.register(num_qubits);
    let mut qubits: Vec<Option<Qudit>> = b
        .split_all_register(register)
        .into_iter()
        .map(Some)
        .collect();
    let take = |qubits: &mut Vec<Option<Qudit>>, i: usize| {
        qubits[i]
            .take()
            .unwrap_or_else(|| fail(3, format!("qubit {i} used twice in one gate")))
    };
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
                // controlled phase is one diagonal 4x4 on the merged pair; the
                // diagonal is symmetric, so the pair's bit order does not matter.
                let pair = b.merge_two_registers(take(&mut qubits, c), take(&mut qubits, t));
                let one = Complex::new(1.0, 0.0);
                let zero = Complex::new(0.0, 0.0);
                let mut m = vec![zero; 16];
                m[0] = one;
                m[5] = one;
                m[10] = one;
                m[15] = Complex::from_polar(1.0, theta);
                let pair = b
                    .apply_vec_matrix(pair, m)
                    .unwrap_or_else(|e| fail(4, format!("qip: {e:?}")));
                let (rt, rc) = b.split_first_qubit(pair);
                qubits[c] = Some(rc);
                qubits[t] = Some(rt.unwrap_or_else(|| fail(4, "qip: merged pair lost a qubit")));
            }
        }
    }
    b
}

fn run_qip(builder: &mut LocalBuilder<f64>, num_qubits: usize, lsb_first: bool) -> Vec<f64> {
    let (state, _) = builder.calculate_state();
    read_out(state.len(), num_qubits, lsb_first, |j| state[j].norm_sqr())
}

#[derive(Clone, Copy)]
enum Kind {
    Spinoza,
    Qip,
}

/// A circuit ready to run: qip's builder holds the whole circuit and replays it
/// from a fresh state on every call, while Spinoza has nothing to build ahead.
enum Prepared<'a> {
    Spinoza(&'a Program),
    Qip(LocalBuilder<f64>, usize),
}

fn prepare(kind: Kind, program: &Program) -> Prepared<'_> {
    match kind {
        Kind::Spinoza => Prepared::Spinoza(program),
        Kind::Qip => Prepared::Qip(build_qip(program), program.num_qubits),
    }
}

fn run(prepared: &mut Prepared, lsb_first: bool) -> Vec<f64> {
    match prepared {
        Prepared::Spinoza(program) => run_spinoza(program, lsb_first),
        Prepared::Qip(builder, num_qubits) => run_qip(builder, *num_qubits, lsb_first),
    }
}

/// Whether a simulator indexes the state with qubit 0 as the least significant
/// bit. An X on qubit 0 of a two-qubit register must light index 1; index 2
/// means the opposite order and the output is reindexed.
fn probe_order(kind: Kind) -> bool {
    let probe = Program {
        num_qubits: 2,
        ops: vec![Op::H(0), Op::H(0), Op::Ry(0, std::f64::consts::PI)],
    };
    let probs = run(&mut prepare(kind, &probe), true);
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

struct Simulator {
    name: &'static str,
    version: &'static str,
    kind: Kind,
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
                version: env!("SPINOZA_VERSION"),
                kind: Kind::Spinoza,
                lsb_first: probe_order(Kind::Spinoza),
            }
        }
        "qip" => Simulator {
            name: "qip",
            version: env!("QIP_VERSION"),
            kind: Kind::Qip,
            lsb_first: probe_order(Kind::Qip),
        },
        other => fail(2, format!("unknown simulator: {other} (spinoza, qip)")),
    }
}

fn read_stdin() -> String {
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .unwrap_or_else(|err| fail(3, format!("failed to read the gate list from stdin: {err}")));
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
    // qip and the read-out run on the global Rayon pool; sizing it here makes
    // `--threads` govern them as it governs Spinoza's own config.
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| fail(2, format!("rayon pool: {err}")));
    let sim = simulator(&args[1], threads);
    match args[2].as_str() {
        "version" => println!(
            "{{\"simulator\":\"{}\",\"version\":\"{}\",\"build\":\"rustflags {}\"}}",
            sim.name,
            sim.version,
            env!("PEERS_RUSTFLAGS")
        ),
        "time" => {
            let iterations: usize = args
                .get(3)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| usage());
            let program = parse_gate_list(&read_stdin());
            let mut prepared = prepare(sim.kind, &program);
            std::hint::black_box(run(&mut prepared, sim.lsb_first));
            let mut times_ms = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let probs = run(&mut prepared, sim.lsb_first);
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
            let probs = run(&mut prepare(sim.kind, &program), sim.lsb_first);
            let mut bytes = Vec::with_capacity(probs.len() * 8);
            for p in &probs {
                bytes.extend_from_slice(&p.to_le_bytes());
            }
            std::fs::write(path, &bytes).unwrap_or_else(|err| {
                fail(5, format!("failed to write the probability vector: {err}"))
            });
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
            .unwrap_or_else(|err| fail(5, format!("failed to write to stdout: {err}")));
        }
        _ => usage(),
    }
}
