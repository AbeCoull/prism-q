use super::PauliVec;
use super::propagation::{batch_propagate_backward, propagate_backward};
use crate::gates::Gate;
use num_complex::Complex64;

type Mat = Vec<Vec<Complex64>>;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn id_mat(n: usize) -> Mat {
    let d = 1 << n;
    (0..d)
        .map(|i| {
            (0..d)
                .map(|j| if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) })
                .collect()
        })
        .collect()
}

fn matmul(a: &Mat, b: &Mat) -> Mat {
    let d = a.len();
    let mut r = vec![vec![c(0.0, 0.0); d]; d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i][k];
            if aik.norm() < 1e-14 {
                continue;
            }
            for j in 0..d {
                r[i][j] += aik * b[k][j];
            }
        }
    }
    r
}

fn dagger(m: &Mat) -> Mat {
    let d = m.len();
    let mut r = vec![vec![c(0.0, 0.0); d]; d];
    for i in 0..d {
        for j in 0..d {
            r[i][j] = m[j][i].conj();
        }
    }
    r
}

fn kron(a: &Mat, b: &Mat) -> Mat {
    let da = a.len();
    let db = b.len();
    let d = da * db;
    let mut r = vec![vec![c(0.0, 0.0); d]; d];
    for ia in 0..da {
        for ja in 0..da {
            for ib in 0..db {
                for jb in 0..db {
                    r[ia * db + ib][ja * db + jb] = a[ia][ja] * b[ib][jb];
                }
            }
        }
    }
    r
}

fn approx_eq(a: &Mat, b: &Mat) -> bool {
    let d = a.len();
    for i in 0..d {
        for j in 0..d {
            if (a[i][j] - b[i][j]).norm() > 1e-9 {
                return false;
            }
        }
    }
    true
}

fn approx_eq_scalar(a: &Mat, b: &Mat, s: Complex64) -> bool {
    let d = a.len();
    for i in 0..d {
        for j in 0..d {
            if (a[i][j] - s * b[i][j]).norm() > 1e-9 {
                return false;
            }
        }
    }
    true
}

fn pauli_mat(p: u8) -> Mat {
    match p {
        0 => vec![
            vec![c(1.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(1.0, 0.0)],
        ],
        1 => vec![
            vec![c(0.0, 0.0), c(1.0, 0.0)],
            vec![c(1.0, 0.0), c(0.0, 0.0)],
        ],
        2 => vec![
            vec![c(0.0, 0.0), c(0.0, -1.0)],
            vec![c(0.0, 1.0), c(0.0, 0.0)],
        ],
        3 => vec![
            vec![c(1.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(-1.0, 0.0)],
        ],
        _ => unreachable!(),
    }
}

fn bits_to_pauli(x: u8, z: u8) -> u8 {
    match (x, z) {
        (0, 0) => 0,
        (1, 0) => 1,
        (1, 1) => 2,
        (0, 1) => 3,
        _ => unreachable!(),
    }
}

fn pauli_to_bits(p: u8) -> (u8, u8) {
    match p {
        0 => (0, 0),
        1 => (1, 0),
        2 => (1, 1),
        3 => (0, 1),
        _ => unreachable!(),
    }
}

fn make_pauli_n(n: usize, bits: &[(u8, u8)]) -> Mat {
    let mut m = pauli_mat(bits_to_pauli(bits[0].0, bits[0].1));
    for b in bits.iter().take(n).skip(1) {
        m = kron(&pauli_mat(bits_to_pauli(b.0, b.1)), &m);
    }
    m
}

fn gate_mat_1q(g: &Gate) -> Mat {
    let s = 1.0 / 2.0f64.sqrt();
    match g {
        Gate::H => vec![vec![c(s, 0.0), c(s, 0.0)], vec![c(s, 0.0), c(-s, 0.0)]],
        Gate::S => vec![
            vec![c(1.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(0.0, 1.0)],
        ],
        Gate::Sdg => vec![
            vec![c(1.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(0.0, -1.0)],
        ],
        Gate::X => pauli_mat(1),
        Gate::Y => pauli_mat(2),
        Gate::Z => pauli_mat(3),
        Gate::SX => vec![
            vec![c(0.5, 0.5), c(0.5, -0.5)],
            vec![c(0.5, -0.5), c(0.5, 0.5)],
        ],
        Gate::SXdg => vec![
            vec![c(0.5, -0.5), c(0.5, 0.5)],
            vec![c(0.5, 0.5), c(0.5, -0.5)],
        ],
        _ => panic!("not 1q clifford"),
    }
}

fn gate_mat_2q(g: &Gate) -> Mat {
    match g {
        Gate::Cx => {
            // q[0]=ctrl, q[1]=tgt. Project as compute basis index = (q1 q0)_2.
            // Build CX over (q1,q0): control = q0 = LSB. Standard CNOT matrix below
            // assumes ctrl=MSB; we transpose order.
            // Easier: emit directly via tensor with q0 LSB.
            let mut m = vec![vec![c(0.0, 0.0); 4]; 4];
            for q0 in 0..2usize {
                for q1 in 0..2usize {
                    let i = (q1 << 1) | q0;
                    let q1_out = q1 ^ q0;
                    let j = (q1_out << 1) | q0;
                    m[j][i] = c(1.0, 0.0);
                }
            }
            m
        }
        Gate::Cz => {
            // q0=q1=1 → -1. With q0 LSB, basis 0b11 = index 3.
            let mut m = id_mat(2);
            m[3][3] = c(-1.0, 0.0);
            m
        }
        Gate::Swap => {
            // swap q0 and q1.
            let mut m = vec![vec![c(0.0, 0.0); 4]; 4];
            for q0 in 0..2usize {
                for q1 in 0..2usize {
                    let i = (q1 << 1) | q0;
                    let j = (q0 << 1) | q1;
                    m[j][i] = c(1.0, 0.0);
                }
            }
            m
        }
        _ => panic!("not 2q clifford"),
    }
}

fn truth_table_1q(u: &Mat, bits: (u8, u8)) -> ((u8, u8), bool) {
    let p_in = make_pauli_n(1, &[bits]);
    let ud = dagger(u);
    let result = matmul(&ud, &matmul(&p_in, u));
    for p in 0..4u8 {
        let b = pauli_to_bits(p);
        let p_out = make_pauli_n(1, &[b]);
        if approx_eq(&result, &p_out) {
            return (b, false);
        }
        if approx_eq_scalar(&result, &p_out, c(-1.0, 0.0)) {
            return (b, true);
        }
    }
    panic!("no signed Pauli matched for input {bits:?}");
}

fn truth_table_2q(u: &Mat, bits: [(u8, u8); 2]) -> ([(u8, u8); 2], bool) {
    let p_in = make_pauli_n(2, &bits);
    let ud = dagger(u);
    let result = matmul(&ud, &matmul(&p_in, u));
    for p0 in 0..4u8 {
        for p1 in 0..4u8 {
            let b = [pauli_to_bits(p0), pauli_to_bits(p1)];
            let p_out = make_pauli_n(2, &b);
            if approx_eq(&result, &p_out) {
                return (b, false);
            }
            if approx_eq_scalar(&result, &p_out, c(-1.0, 0.0)) {
                return (b, true);
            }
        }
    }
    panic!("no signed Pauli matched for input {bits:?}");
}

fn run_batch_1q(gate: &Gate, x_in: u8, z_in: u8) -> (u8, u8, bool) {
    let mut x = vec![vec![x_in as u64]];
    let mut z = vec![vec![z_in as u64]];
    let mut sign = vec![0u64];
    batch_propagate_backward(&mut x, &mut z, &mut sign, gate, &[0], 1);
    ((x[0][0] & 1) as u8, (z[0][0] & 1) as u8, (sign[0] & 1) != 0)
}

fn run_batch_2q(gate: &Gate, bits: [(u8, u8); 2]) -> ([(u8, u8); 2], bool) {
    let mut x = vec![vec![bits[0].0 as u64], vec![bits[1].0 as u64]];
    let mut z = vec![vec![bits[0].1 as u64], vec![bits[1].1 as u64]];
    let mut sign = vec![0u64];
    batch_propagate_backward(&mut x, &mut z, &mut sign, gate, &[0, 1], 1);
    (
        [
            ((x[0][0] & 1) as u8, (z[0][0] & 1) as u8),
            ((x[1][0] & 1) as u8, (z[1][0] & 1) as u8),
        ],
        (sign[0] & 1) != 0,
    )
}

#[test]
fn one_qubit_clifford_signs() {
    let mut failures: Vec<String> = Vec::new();
    for (g, name) in [
        (Gate::H, "H"),
        (Gate::S, "S"),
        (Gate::Sdg, "Sdg"),
        (Gate::X, "X"),
        (Gate::Y, "Y"),
        (Gate::Z, "Z"),
        (Gate::SX, "SX"),
        (Gate::SXdg, "SXdg"),
    ] {
        let u = gate_mat_1q(&g);
        for x in 0..2u8 {
            for z in 0..2u8 {
                let (tb, ts) = truth_table_1q(&u, (x, z));
                let (gx, gz, gs) = run_batch_1q(&g, x, z);
                if (tb.0, tb.1, ts) != (gx, gz, gs) {
                    failures.push(format!(
                        "{name} P=(x={x},z={z}): truth=(x'={},z'={},sign={}), got=(x'={gx},z'={gz},sign={gs})",
                        tb.0, tb.1, ts
                    ));
                }

                let mut pv = PauliVec {
                    x: vec![x as u64],
                    z: vec![z as u64],
                };
                propagate_backward(&mut pv, &g, &[0]);
                if ((pv.x[0] & 1) as u8, (pv.z[0] & 1) as u8) != (gx, gz) {
                    failures.push(format!(
                        "scalar {name} bits disagree with batched on (x={x},z={z})"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "sign-table mismatches:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn two_qubit_clifford_signs() {
    let mut failures: Vec<String> = Vec::new();
    for (g, name) in [(Gate::Cx, "Cx"), (Gate::Cz, "Cz"), (Gate::Swap, "Swap")] {
        let u = gate_mat_2q(&g);
        for code in 0..16u8 {
            let b0 = pauli_to_bits(code & 0b11);
            let b1 = pauli_to_bits((code >> 2) & 0b11);
            let (tb, ts) = truth_table_2q(&u, [b0, b1]);
            let (gb, gs) = run_batch_2q(&g, [b0, b1]);
            if (tb, ts) != (gb, gs) {
                failures.push(format!(
                    "{name} P=({:?},{:?}): truth=({:?},{:?},sign={}), got=({:?},{:?},sign={})",
                    b0, b1, tb[0], tb[1], ts, gb[0], gb[1], gs
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "sign-table mismatches:\n  {}",
        failures.join("\n  ")
    );
}
