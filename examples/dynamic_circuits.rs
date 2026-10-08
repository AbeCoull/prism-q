//! Teleport a qubit with mid-circuit measurement and feed-forward from OpenQASM 3.

use prism_q::circuit::openqasm;
use prism_q::{bitstring, simulate};

fn main() {
    let theta = 1.2_f64;
    let teleport = openqasm::parse(&format!(
        r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        ry({theta}) q[0];
        h q[1];
        cx q[1], q[2];
        cx q[0], q[1];
        h q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
        if (c[1]) x q[2];
        if (c[0]) z q[2];
        c[2] = measure q[2];
        "#
    ))
    .expect("failed to parse QASM");
    let shots = simulate(&teleport).seed(42).shots(100_000).unwrap();
    let ones = shots.shots.iter().filter(|bits| bits[2]).count();
    println!(
        "P(1) on q[2]: {:.4}, expected {:.4}",
        ones as f64 / shots.shots.len() as f64,
        (theta / 2.0).sin().powi(2)
    );

    let branch = openqasm::parse(
        r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[3] c;
        h q[0];
        c[0] = measure q[0];
        reset q[0];
        if (c[0]) {
          x q[1];
        } else {
          h q[1];
        }
        c[1] = measure q[1];
        c[2] = measure q[0];
        "#,
    )
    .expect("failed to parse QASM");
    let counts = simulate(&branch).seed(42).sample_counts(1000).unwrap();
    let mut rows: Vec<(String, u64)> = counts
        .counts
        .iter()
        .map(|(key, &count)| (bitstring(key, counts.num_classical_bits), count))
        .collect();
    rows.sort();
    for (bits, count) in rows {
        println!("{bits}: {count}");
    }
}
