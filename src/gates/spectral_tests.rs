use super::*;

use crate::gates::Gate;

const EPS: f64 = 1e-9;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Column-major copy of a row-major square matrix.
fn columns(rows: &[Vec<Complex64>]) -> Vec<Complex64> {
    let dim = rows.len();
    let mut flat = vec![ZERO; dim * dim];
    for (row, entries) in rows.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            flat[column * dim + row] = *entry;
        }
    }
    flat
}

/// `V diag(values) V*` from a column-major `V`, back in column-major order.
fn rebuild(values: &[Complex64], vectors: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![ZERO; dim * dim];
    for column in 0..dim {
        for row in 0..dim {
            out[column * dim + row] = (0..dim)
                .map(|k| vectors[k * dim + row] * values[k] * vectors[k * dim + column].conj())
                .sum();
        }
    }
    out
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: wrong length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).norm() < EPS,
            "{label}: entry {index} is {got} against {want}"
        );
    }
}

fn assert_orthonormal(vectors: &[Complex64], dim: usize, label: &str) {
    for left in 0..dim {
        for right in 0..dim {
            let overlap: Complex64 = (0..dim)
                .map(|row| vectors[left * dim + row].conj() * vectors[right * dim + row])
                .sum();
            let expected = f64::from(u8::from(left == right));
            assert!(
                (overlap - expected).norm() < EPS,
                "{label}: columns {left} and {right} overlap by {overlap}"
            );
        }
    }
}

// The spectrum is read back from the factor rather than from a second
// implementation, so a wrong factor cannot agree with a wrong spectrum.
#[test]
fn hermitian_eigen_reconstructs_its_matrix() {
    let cases: Vec<Vec<Vec<Complex64>>> = vec![
        // Eigenvalues equal in magnitude and opposite in sign, which a plain
        // singular value decomposition cannot separate.
        vec![
            vec![c(0.0, 0.0), c(1.0, 0.0)],
            vec![c(1.0, 0.0), c(0.0, 0.0)],
        ],
        vec![
            vec![c(1.0, 0.0), c(2.0, -1.0)],
            vec![c(2.0, 1.0), c(-3.0, 0.0)],
        ],
        // Entirely negative, which the shift has to clear.
        vec![
            vec![c(-2.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(-5.0, 0.0)],
        ],
        // Degenerate, where any orthonormal basis of the eigenspace serves.
        vec![
            vec![c(4.0, 0.0), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(4.0, 0.0)],
        ],
        vec![
            vec![c(1.0, 0.0), c(0.0, 0.5), c(0.0, 0.0), c(0.2, 0.0)],
            vec![c(0.0, -0.5), c(-2.0, 0.0), c(0.3, 0.1), c(0.0, 0.0)],
            vec![c(0.0, 0.0), c(0.3, -0.1), c(0.0, 0.0), c(0.7, 0.0)],
            vec![c(0.2, 0.0), c(0.0, 0.0), c(0.7, 0.0), c(3.0, 0.0)],
        ],
    ];
    for rows in &cases {
        let dim = rows.len();
        let flat = columns(rows);
        let (values, vectors) = hermitian_eigen(&flat, dim);
        assert_eq!(values.len(), dim);
        assert!(
            values.windows(2).all(|pair| pair[0] <= pair[1] + EPS),
            "eigenvalues are not ascending: {values:?}"
        );
        assert_orthonormal(&vectors, dim, "hermitian");
        let complex: Vec<Complex64> = values.iter().map(|value| c(*value, 0.0)).collect();
        assert_close(
            &rebuild(&complex, &vectors, dim),
            &flat,
            "hermitian rebuild",
        );
    }
}

// A unitary's eigenvalues are separated through two commuting Hermitian parts,
// so the cases that discriminate are the ones where the first part alone
// cannot: a conjugate pair sharing `cos(theta)`, and a scalar block.
#[test]
fn unitary_eigen_reconstructs_its_matrix() {
    let mut cases: Vec<Vec<Complex64>> = Vec::new();
    for gate in [
        Gate::X,
        Gate::H,
        Gate::S,
        Gate::T,
        Gate::Rx(0.7),
        Gate::Ry(2.9),
        Gate::Id,
    ] {
        let m = gate.matrix_2x2();
        cases.push(columns(&[m[0].to_vec(), m[1].to_vec()]));
    }
    for gate in [
        Gate::Cx,
        Gate::Cz,
        Gate::Swap,
        Gate::Rzz(0.9),
        Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())),
    ] {
        let m = gate.matrix_4x4();
        cases.push(columns(
            &m.iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
        ));
    }
    // `Rz(theta) tensor I` carries each eigenvalue twice, so every cluster is
    // degenerate in both parts.
    let phase = Complex64::from_polar(1.0, 0.4);
    cases.push(columns(&[
        vec![phase.conj(), ZERO, ZERO, ZERO],
        vec![ZERO, phase.conj(), ZERO, ZERO],
        vec![ZERO, ZERO, phase, ZERO],
        vec![ZERO, ZERO, ZERO, phase],
    ]));

    for flat in &cases {
        let dim = (flat.len() as f64).sqrt() as usize;
        let (values, vectors) = unitary_eigen(flat, dim);
        assert_eq!(values.len(), dim);
        for value in &values {
            assert!((value.norm() - 1.0).abs() < EPS, "{value} is not a phase");
        }
        assert_orthonormal(&vectors, dim, "unitary");
        assert_close(&rebuild(&values, &vectors, dim), flat, "unitary rebuild");
    }
}
