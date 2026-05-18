use prism_q::circuit::openqasm;
use prism_q::simulate;

fn main() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;

    let circuit = openqasm::parse(qasm).expect("failed to parse QASM");

    // Deterministic: 1024 shots with fixed seed
    let result = simulate(&circuit)
        .seed(42)
        .shots(1024)
        .expect("shots failed");
    println!("Deterministic (seed=42), 1024 shots:");
    print!("{result}");

    // Random seed: non-deterministic sampling
    let result = simulate(&circuit)
        .seed(rand::random())
        .shots(1024)
        .expect("shots failed");
    println!("Random seed, 1024 shots:");
    print!("{result}");
}
