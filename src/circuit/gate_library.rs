//! Standard gate names, their decompositions and the matrices they build.

use num_complex::Complex64;

use crate::circuit::openqasm::{Dialect, unsupported};
use crate::circuit::{Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

pub(crate) fn resolve_gate(
    name: &str,
    params: &[f64],
    dialect: Dialect,
    line_num: usize,
) -> Result<Gate> {
    match name {
        "id" | "i" => Ok(Gate::Id),
        "x" => Ok(Gate::X),
        "y" => Ok(Gate::Y),
        "z" => Ok(Gate::Z),
        "h" => Ok(Gate::H),
        "s" => Ok(Gate::S),
        "sdg" | "si" => Ok(Gate::Sdg),
        "t" => Ok(Gate::T),
        "tdg" | "ti" => Ok(Gate::Tdg),
        "rx" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Rx(params[0]))
        }
        "ry" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Ry(params[0]))
        }
        "rz" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Rz(params[0]))
        }
        "p" | "phase" | "phaseshift" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::P(params[0]))
        }
        "r" | "prx" => {
            expect_param_count(name, params, 2, line_num)?;
            Ok(Gate::Fused(Box::new(r_matrix(params[0], params[1]))))
        }
        "sx" | "v" => Ok(Gate::SX),
        "sxdg" | "vi" => Ok(Gate::SXdg),
        "u0" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Id)
        }
        "cp" | "cphase" | "cphaseshift" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::cphase(params[0]))
        }
        "cu1" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::cphase(params[0]))
        }
        "cx" | "CX" | "cnot" => Ok(Gate::Cx),
        "cy" => Ok(Gate::cu(Gate::Y.matrix_2x2())),
        "cs" => Ok(Gate::cu(Gate::S.matrix_2x2())),
        "csdg" => Ok(Gate::cu(Gate::Sdg.matrix_2x2())),
        "ch" => Ok(Gate::cu(Gate::H.matrix_2x2())),
        "cu" => {
            expect_param_count(name, params, 4, line_num)?;
            Ok(Gate::cu(cu_target_matrix(
                params[0], params[1], params[2], params[3],
            )))
        }
        "cu3" => {
            expect_param_count(name, params, 3, line_num)?;
            Ok(Gate::cu(u_matrix(params[0], params[1], params[2])))
        }
        "crx" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::cu(Gate::Rx(params[0]).matrix_2x2()))
        }
        "cry" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::cu(Gate::Ry(params[0]).matrix_2x2()))
        }
        "crz" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::cu(Gate::Rz(params[0]).matrix_2x2()))
        }
        "csx" | "cv" => Ok(Gate::cu(Gate::SX.matrix_2x2())),
        "cz" => Ok(Gate::Cz),
        "swap" => Ok(Gate::Swap),
        "ccx" | "toffoli" | "ccnot" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 2)),
        "ccz" => Ok(Gate::mcu(Gate::Z.matrix_2x2(), 2)),
        "c3x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 3)),
        "c3sqrtx" => Ok(Gate::mcu(Gate::SX.matrix_2x2(), 3)),
        "c4x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 4)),
        "xx_plus_yy" => {
            expect_param_count(name, params, 2, line_num)?;
            Ok(Gate::Fused2q(Box::new(xx_plus_yy_matrix(
                params[0], params[1],
            ))))
        }
        "xx_minus_yy" => {
            expect_param_count(name, params, 2, line_num)?;
            Ok(Gate::Fused2q(Box::new(xx_minus_yy_matrix(
                params[0], params[1],
            ))))
        }
        "gpi" => {
            expect_param_count(name, params, 1, line_num)?;
            let phi = dialect.native_turns(params[0]);
            Ok(Gate::Fused(Box::new(gpi_matrix(phi))))
        }
        "gpi2" => {
            expect_param_count(name, params, 1, line_num)?;
            let phi = dialect.native_turns(params[0]);
            Ok(Gate::Fused(Box::new(gpi2_matrix(phi))))
        }
        "ms" => {
            if !(params.len() == 2 || params.len() == 3) {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "`{name}` at line {line_num} requires 2 or 3 parameter(s), got {}",
                        params.len()
                    ),
                });
            }
            // The omitted third angle is a quarter turn under either
            // dialect, so the default is already in native units.
            let theta = match params.get(2) {
                Some(value) => dialect.native_turns(*value),
                None => 0.25,
            };
            Ok(Gate::Fused2q(Box::new(ms_matrix(
                dialect.native_turns(params[0]),
                dialect.native_turns(params[1]),
                theta,
            ))))
        }
        "ecr" => Ok(Gate::Fused2q(Box::new(ecr_matrix()))),
        "xy" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Fused2q(Box::new(xy_matrix(params[0]))))
        }
        "pswap" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Fused2q(Box::new(pswap_matrix(params[0]))))
        }
        "cphaseshift00" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Fused2q(Box::new(cphaseshift_matrix(0, params[0]))))
        }
        "cphaseshift01" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Fused2q(Box::new(cphaseshift_matrix(1, params[0]))))
        }
        "cphaseshift10" => {
            expect_param_count(name, params, 1, line_num)?;
            Ok(Gate::Fused2q(Box::new(cphaseshift_matrix(2, params[0]))))
        }
        "syc" => Ok(Gate::Fused2q(Box::new(syc_matrix()))),
        "sqrt_iswap" => Ok(Gate::Fused2q(Box::new(sqrt_iswap_matrix(1.0)))),
        "sqrt_iswap_inv" => Ok(Gate::Fused2q(Box::new(sqrt_iswap_matrix(-1.0)))),
        _ => Err(unsupported(line_num, name)),
    }
}

/// Decomposition-body shorthand for a gate instruction.
pub(crate) fn ig(gate: Gate, targets: &[usize]) -> Instruction {
    Instruction::Gate {
        gate,
        targets: SmallVec::from_slice(targets),
    }
}

/// Lower the gates the parser expands itself, or `Ok(None)` for any other
/// name, which falls through to `resolve_gate`.
pub(crate) fn resolve_decomposed_gate(
    name: &str,
    params: &[f64],
    qubits: &[usize],
    line_num: usize,
) -> Result<Option<Vec<Instruction>>> {
    match name {
        "mcx" => {
            if qubits.len() < 2 {
                return Err(PrismError::GateArity {
                    gate: name.to_string(),
                    expected: 2,
                    got: qubits.len(),
                });
            }
            let controls = qubits.len() - 1;
            if controls > u8::MAX as usize {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "`{name}` at line {line_num} supports at most {} controls, got {controls}",
                        u8::MAX
                    ),
                });
            }
            Ok(Some(vec![ig(
                Gate::mcu(Gate::X.matrix_2x2(), controls as u8),
                qubits,
            )]))
        }
        "rccx" => {
            check_arity(name, qubits, 3)?;
            let c0 = qubits[0];
            let c1 = qubits[1];
            let target = qubits[2];
            Ok(Some(vec![
                ig(Gate::H, &[target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c1, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::Cx, &[c0, target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c1, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::H, &[target]),
            ]))
        }
        "rc3x" | "rcccx" => {
            check_arity(name, qubits, 4)?;
            let c0 = qubits[0];
            let c1 = qubits[1];
            let c2 = qubits[2];
            let target = qubits[3];
            Ok(Some(vec![
                ig(Gate::H, &[target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c2, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::H, &[target]),
                ig(Gate::Cx, &[c0, target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c1, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::Cx, &[c0, target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c1, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::H, &[target]),
                ig(Gate::T, &[target]),
                ig(Gate::Cx, &[c2, target]),
                ig(Gate::Tdg, &[target]),
                ig(Gate::H, &[target]),
            ]))
        }
        "cswap" | "fredkin" => {
            check_arity(name, qubits, 3)?;
            Ok(Some(
                cswap_expansion(qubits[0], qubits[1], qubits[2]).into(),
            ))
        }
        "iswap" => {
            check_arity(name, qubits, 2)?;
            Ok(Some(iswap_expansion(qubits[0], qubits[1]).into()))
        }
        "dcx" => {
            check_arity(name, qubits, 2)?;
            let q0 = qubits[0];
            let q1 = qubits[1];
            Ok(Some(vec![ig(Gate::Cx, &[q0, q1]), ig(Gate::Cx, &[q1, q0])]))
        }
        "u1" => {
            expect_param_count(name, params, 1, line_num)?;
            check_arity(name, qubits, 1)?;
            Ok(Some(vec![ig(Gate::P(params[0]), &[qubits[0]])]))
        }
        "u2" => {
            expect_param_count(name, params, 2, line_num)?;
            check_arity(name, qubits, 1)?;
            let phi = params[0];
            let lam = params[1];
            let isqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let one = Complex64::new(isqrt2, 0.0);
            let mat = [
                [one, -Complex64::from_polar(isqrt2, lam)],
                [
                    Complex64::from_polar(isqrt2, phi),
                    Complex64::from_polar(isqrt2, phi + lam),
                ],
            ];
            Ok(Some(vec![ig(Gate::Fused(Box::new(mat)), &[qubits[0]])]))
        }
        "u3" | "u" | "U" => {
            expect_param_count(name, params, 3, line_num)?;
            check_arity(name, qubits, 1)?;
            let theta = params[0];
            let phi = params[1];
            let lam = params[2];
            let mat = u_matrix(theta, phi, lam);
            Ok(Some(vec![ig(Gate::Fused(Box::new(mat)), &[qubits[0]])]))
        }
        _ => Ok(None),
    }
}

/// Controlled swap as a Toffoli between two CNOTs.
pub(crate) fn cswap_expansion(control: usize, t1: usize, t2: usize) -> [Instruction; 3] {
    [
        ig(Gate::Cx, &[t2, t1]),
        ig(Gate::mcu(Gate::X.matrix_2x2(), 2), &[control, t1, t2]),
        ig(Gate::Cx, &[t2, t1]),
    ]
}

/// iSWAP as two phase gates, two Hadamards and two CNOTs.
pub(crate) fn iswap_expansion(q0: usize, q1: usize) -> [Instruction; 6] {
    [
        ig(Gate::S, &[q0]),
        ig(Gate::S, &[q1]),
        ig(Gate::H, &[q0]),
        ig(Gate::Cx, &[q0, q1]),
        ig(Gate::Cx, &[q1, q0]),
        ig(Gate::H, &[q1]),
    ]
}

fn check_arity(name: &str, qubits: &[usize], expected: usize) -> Result<()> {
    if qubits.len() != expected {
        return Err(PrismError::GateArity {
            gate: name.to_string(),
            expected,
            got: qubits.len(),
        });
    }
    Ok(())
}

pub(crate) fn u_matrix(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex64::new(c, 0.0), -Complex64::from_polar(s, lam)],
        [
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lam),
        ],
    ]
}

fn r_matrix(theta: f64, phi: f64) -> [[Complex64; 2]; 2] {
    let zero_phase = Complex64::new((theta / 2.0).cos(), 0.0);
    let off = Complex64::new(0.0, -1.0) * (theta / 2.0).sin();
    [
        [zero_phase, off * Complex64::from_polar(1.0, -phi)],
        [off * Complex64::from_polar(1.0, phi), zero_phase],
    ]
}

pub(crate) fn cu_target_matrix(theta: f64, phi: f64, lam: f64, gamma: f64) -> [[Complex64; 2]; 2] {
    let phase = Complex64::from_polar(1.0, gamma);
    let u = u_matrix(theta, phi, lam);
    [
        [phase * u[0][0], phase * u[0][1]],
        [phase * u[1][0], phase * u[1][1]],
    ]
}

pub(crate) fn xx_plus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, -(theta / 2.0).sin());
    [
        [one, zero, zero, zero],
        [zero, c, s * Complex64::from_polar(1.0, -beta), zero],
        [zero, s * Complex64::from_polar(1.0, beta), c, zero],
        [zero, zero, zero, one],
    ]
}

pub(crate) fn xx_minus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, -(theta / 2.0).sin());
    [
        [c, zero, zero, s * Complex64::from_polar(1.0, -beta)],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [s * Complex64::from_polar(1.0, beta), zero, zero, c],
    ]
}

fn gpi_matrix(phi: f64) -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    [
        [
            zero,
            Complex64::from_polar(1.0, -std::f64::consts::TAU * phi),
        ],
        [
            Complex64::from_polar(1.0, std::f64::consts::TAU * phi),
            zero,
        ],
    ]
}

fn gpi2_matrix(phi: f64) -> [[Complex64; 2]; 2] {
    let one = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let off = Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
    [
        [
            one,
            off * Complex64::from_polar(1.0, -std::f64::consts::TAU * phi),
        ],
        [
            off * Complex64::from_polar(1.0, std::f64::consts::TAU * phi),
            one,
        ],
    ]
}

pub(crate) fn ms_matrix(phi0: f64, phi1: f64, theta: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let c = Complex64::new((std::f64::consts::PI * theta).cos(), 0.0);
    let s = Complex64::new(0.0, -(std::f64::consts::PI * theta).sin());
    let sum = phi0 + phi1;
    let diff = phi0 - phi1;
    [
        [
            c,
            zero,
            zero,
            s * Complex64::from_polar(1.0, -std::f64::consts::TAU * sum),
        ],
        [
            zero,
            c,
            s * Complex64::from_polar(1.0, -std::f64::consts::TAU * diff),
            zero,
        ],
        [
            zero,
            s * Complex64::from_polar(1.0, std::f64::consts::TAU * diff),
            c,
            zero,
        ],
        [
            s * Complex64::from_polar(1.0, std::f64::consts::TAU * sum),
            zero,
            zero,
            c,
        ],
    ]
}

/// Echoed cross-resonance, `(XI - YX) / sqrt(2)` with `targets[0]` the
/// leading factor.
pub(crate) fn ecr_matrix() -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let r = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let i = Complex64::new(0.0, std::f64::consts::FRAC_1_SQRT_2);
    [
        [zero, zero, r, i],
        [zero, zero, i, r],
        [r, -i, zero, zero],
        [-i, r, zero, zero],
    ]
}

/// XY interaction, a half-angle rotation inside the single-excitation
/// subspace.
pub(crate) fn xy_matrix(theta: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, (theta / 2.0).sin());
    [
        [one, zero, zero, zero],
        [zero, c, s, zero],
        [zero, s, c, zero],
        [zero, zero, zero, one],
    ]
}

/// Phased SWAP: a swap whose exchanged amplitudes pick up `e^{i theta}`.
/// The angle is not halved.
pub(crate) fn pswap_matrix(theta: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let phase = Complex64::from_polar(1.0, theta);
    [
        [one, zero, zero, zero],
        [zero, zero, phase, zero],
        [zero, phase, zero, zero],
        [zero, zero, zero, one],
    ]
}

/// Diagonal two-qubit phase on the single basis state `index`, where bit 1
/// of `index` is `targets[0]`. `index` 3 is the ordinary controlled phase.
pub(crate) fn cphaseshift_matrix(index: usize, theta: f64) -> [[Complex64; 4]; 4] {
    let mut mat = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (i, row) in mat.iter_mut().enumerate() {
        row[i] = if i == index {
            Complex64::from_polar(1.0, theta)
        } else {
            Complex64::new(1.0, 0.0)
        };
    }
    mat
}

pub(crate) fn syc_matrix() -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    [
        [one, zero, zero, zero],
        [zero, zero, neg_i, zero],
        [zero, neg_i, zero, zero],
        [
            zero,
            zero,
            zero,
            Complex64::from_polar(1.0, -std::f64::consts::PI / 6.0),
        ],
    ]
}

pub(crate) fn sqrt_iswap_matrix(sign: f64) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let half = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let off = Complex64::new(0.0, sign * std::f64::consts::FRAC_1_SQRT_2);
    [
        [one, zero, zero, zero],
        [zero, half, off, zero],
        [zero, off, half, zero],
        [zero, zero, zero, one],
    ]
}

pub(crate) fn expect_param_count(
    gate: &str,
    params: &[f64],
    expected: usize,
    line_num: usize,
) -> Result<()> {
    if params.len() != expected {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "`{gate}` at line {line_num} requires {expected} parameter(s), got {}",
                params.len()
            ),
        });
    }
    Ok(())
}
