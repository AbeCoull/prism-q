//! OpenQASM 3.0 export from a [`Circuit`].
//!
//! Inverts [`openqasm::parse`](super::openqasm::parse) over the constructs that
//! parser produces: re-parsing an export yields the same instruction stream, with
//! gate matrices agreeing to floating-point round-off rather than bit for bit.
//! Angles carried inline (`rx`, `rz`, `rzz`, `p`) survive exactly.
//!
//! [`Gate::QftBlock`] expands to its textbook sequence before emission.
//! [`Gate::PauliRot`] keeps its native form, spelled `r` followed by the Pauli
//! letters (`rxyz(0.7) q[0], q[1], q[2];`), which generalizes the `rzz` the
//! parser already takes and is what makes the round trip preserve the gate
//! rather than a lowering of it. That spelling is a PRISM-Q extension: for
//! output another toolchain reads, run
//! [`expand_pauli_rotations`](super::expand_pauli_rotations) first and export
//! the CNOT ladder instead. Fusion payloads have no
//! OpenQASM spelling and are rejected: `MultiFused`, `Multi2q`,
//! `BatchPhase`, `BatchRzz`, `DiagonalBatch`, a `Fused2q` outside the two-qubit
//! families the parser builds, and a single-qubit matrix carrying a global phase
//! no named rotation absorbs.

use num_complex::Complex64;
use std::collections::BTreeSet;
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt::Write;

use super::openqasm::Parser;
use super::{Circuit, ClassicalCondition, Instruction, expand_qft_blocks};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Agreement bound for recognizing a matrix as a named gate, matching the
/// parser's own `resolve_controlled` threshold.
const EPS: f64 = 1e-12;

/// Render `circuit` as an OpenQASM 3.0 program.
///
/// Qubits become one `qubit[n] q` register. Classical bits become one `bit[m] c`
/// register, cut at the boundaries of every register-valued condition so each
/// condition still names a whole register; the pieces are then `c0`, `c1`, and so
/// on, in bit order.
///
/// # Errors
/// Returns [`PrismError::ExportUnsupported`], naming the instruction's index in
/// the stream, for a gate with no OpenQASM 3.0 spelling.
///
/// # Examples
/// ```
/// use prism_q::circuit::{openqasm, qasm_export};
/// use prism_q::{Circuit, Gate};
///
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
///
/// let qasm = qasm_export::to_qasm3(&circuit).expect("export");
/// assert_eq!(openqasm::parse(&qasm).expect("reparse").gate_count(), 2);
/// ```
pub fn to_qasm3(circuit: &Circuit) -> Result<String> {
    let expanded = expand_qft_blocks(circuit);
    let circuit = expanded.as_ref();
    let cregs = CregLayout::of(circuit)?;

    let mut out = String::with_capacity(96 + circuit.instructions.len() * 24);
    out.push_str("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    if circuit.num_qubits > 0 {
        let _ = writeln!(out, "qubit[{}] q;", circuit.num_qubits);
    }
    for (&(_, size), name) in cregs.segments.iter().zip(&cregs.names) {
        let _ = writeln!(out, "bit[{size}] {name};");
    }

    for (index, inst) in circuit.instructions.iter().enumerate() {
        emit(&mut out, inst, index, &cregs, 0)?;
    }
    Ok(out)
}

/// Write one instruction, recursing into guarded regions as a braced `if`.
///
/// `index` names the top-level instruction in any error, so an unexportable
/// gate nested in a region is still located by the position a caller can see.
fn emit(
    out: &mut String,
    inst: &Instruction,
    index: usize,
    cregs: &CregLayout,
    depth: usize,
) -> Result<()> {
    let pad = "  ".repeat(depth);
    match inst {
        Instruction::Gate { gate, targets } => {
            let _ = writeln!(
                out,
                "{pad}{} {};",
                require_head(gate, index)?,
                args(targets)
            );
        }
        Instruction::Measure {
            qubit,
            classical_bit,
        } => {
            let _ = writeln!(
                out,
                "{pad}{} = measure q[{qubit}];",
                cregs.bit(*classical_bit)
            );
        }
        Instruction::Reset { qubit } => {
            let _ = writeln!(out, "{pad}reset q[{qubit}];");
        }
        Instruction::Barrier { qubits } => {
            if qubits.is_empty() {
                return Err(PrismError::ExportUnsupported {
                    index,
                    reason: "barrier over no qubits".to_string(),
                });
            }
            let _ = writeln!(out, "{pad}barrier {};", args(qubits));
        }
        Instruction::Conditional {
            condition,
            gate,
            targets,
        } => {
            let _ = writeln!(
                out,
                "{pad}if ({}) {} {};",
                cregs.condition(condition, index)?,
                require_head(gate, index)?,
                args(targets)
            );
        }
        Instruction::Region(region) => {
            let _ = writeln!(
                out,
                "{pad}if ({}) {{",
                cregs.condition(region.condition(), index)?
            );
            for inner in region.body() {
                emit(out, inner, index, cregs, depth + 1)?;
            }
            let _ = writeln!(out, "{pad}}}");
        }
    }
    Ok(())
}

fn args(targets: &[usize]) -> String {
    targets
        .iter()
        .map(|q| format!("q[{q}]"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn require_head(gate: &Gate, index: usize) -> Result<String> {
    gate_head(gate).ok_or_else(|| PrismError::ExportUnsupported {
        index,
        reason: format!("gate `{}` has no OpenQASM 3.0 spelling", gate.name()),
    })
}

/// The gate name, parameters, and any `ctrl @` prefix, without the qubit
/// arguments.
fn gate_head(gate: &Gate) -> Option<String> {
    Some(match gate {
        Gate::Id => "id".to_string(),
        Gate::X => "x".to_string(),
        Gate::Y => "y".to_string(),
        Gate::Z => "z".to_string(),
        Gate::H => "h".to_string(),
        Gate::S => "s".to_string(),
        Gate::Sdg => "sdg".to_string(),
        Gate::T => "t".to_string(),
        Gate::Tdg => "tdg".to_string(),
        Gate::SX => "sx".to_string(),
        Gate::SXdg => "sxdg".to_string(),
        Gate::Rx(theta) => format!("rx({theta})"),
        Gate::Ry(theta) => format!("ry({theta})"),
        Gate::Rz(theta) => format!("rz({theta})"),
        Gate::P(theta) => format!("p({theta})"),
        Gate::Rzz(theta) => format!("rzz({theta})"),
        Gate::PauliRot(data) => {
            let letters: String = data
                .axes()
                .iter()
                .map(|axis| axis.letter().to_ascii_lowercase())
                .collect();
            format!("r{letters}({})", data.theta())
        }
        Gate::Cx => "cx".to_string(),
        Gate::Cz => "cz".to_string(),
        Gate::Swap => "swap".to_string(),
        Gate::Fused(mat) => spell_1q(mat)?,
        Gate::Fused2q(mat) => spell_2q(mat)?,
        Gate::Cu(mat) => match gate.controlled_phase() {
            Some(phase) => format!("cp({})", phase.arg()),
            None => spell_cu(mat)?,
        },
        Gate::Mcu(data) if data.mat == Gate::X.matrix_2x2() => {
            if data.num_controls == 2 {
                "ccx".to_string()
            } else {
                "mcx".to_string()
            }
        }
        Gate::Mcu(data) if data.num_controls >= 2 => format!(
            "{}{}",
            "ctrl @ ".repeat(data.num_controls as usize - 1),
            spell_cu(&data.mat)?
        ),
        _ => return None,
    })
}

/// A single-qubit matrix as `p`, `rz`, or `u`, or `None` when it carries a
/// global phase none of those three absorb.
fn spell_1q(mat: &[[Complex64; 2]; 2]) -> Option<String> {
    let phase = mat[1][1].arg();
    if close_2x2(&Gate::P(phase).matrix_2x2(), mat) {
        return Some(format!("p({phase})"));
    }
    let theta = 2.0 * phase;
    if close_2x2(&Gate::Rz(theta).matrix_2x2(), mat) {
        return Some(format!("rz({theta})"));
    }
    let (theta, phi, lam, gamma) = zyz(mat);
    if gamma.abs() < EPS && close_2x2(&Parser::u_matrix(theta, phi, lam), mat) {
        return Some(format!("u({theta}, {phi}, {lam})"));
    }
    None
}

/// The controlled form, whose fourth parameter carries the global phase `u`
/// cannot express.
fn spell_cu(mat: &[[Complex64; 2]; 2]) -> Option<String> {
    let (theta, phi, lam, gamma) = zyz(mat);
    close_2x2(&Parser::cu_target_matrix(theta, phi, lam, gamma), mat)
        .then(|| format!("cu({theta}, {phi}, {lam}, {gamma})"))
}

/// A two-qubit matrix as one of the named families the parser builds.
fn spell_2q(mat: &[[Complex64; 4]; 4]) -> Option<String> {
    if close_4x4(&Parser::syc_matrix(), mat) {
        return Some("syc".to_string());
    }
    if close_4x4(&Parser::sqrt_iswap_matrix(1.0), mat) {
        return Some("sqrt_iswap".to_string());
    }
    if close_4x4(&Parser::sqrt_iswap_matrix(-1.0), mat) {
        return Some("sqrt_iswap_inv".to_string());
    }
    let (theta, beta) = xy_params(mat[1][1], mat[1][2]);
    if close_4x4(&Parser::xx_plus_yy_matrix(theta, beta), mat) {
        return Some(format!("xx_plus_yy({theta}, {beta})"));
    }
    let (theta, beta) = xy_params(mat[0][0], mat[0][3]);
    if close_4x4(&Parser::xx_minus_yy_matrix(theta, beta), mat) {
        return Some(format!("xx_minus_yy({theta}, {beta})"));
    }
    let (theta, phi0, phi1) = ms_params(mat);
    if close_4x4(&Parser::ms_matrix(phi0, phi1, theta), mat) {
        return Some(format!("ms({phi0}, {phi1}, {theta})"));
    }
    None
}

/// Euler angles and global phase of a single-qubit unitary, with the
/// convention `mat = e^{i gamma} u(theta, phi, lambda)`.
///
/// `theta` lands in `[0, pi]` so `cos(theta / 2)` is the non-negative modulus of
/// `mat[0][0]` and the phase of that entry is the whole of `gamma`. The two
/// degenerate shapes take their own branch: an anti-diagonal matrix leaves
/// `gamma` free and fixes it to zero, a diagonal one leaves `phi` free.
fn zyz(mat: &[[Complex64; 2]; 2]) -> (f64, f64, f64, f64) {
    let cos_half = mat[0][0].norm();
    let sin_half = mat[1][0].norm();
    let theta = 2.0 * sin_half.atan2(cos_half);
    if cos_half < EPS {
        (theta, mat[1][0].arg(), (-mat[0][1]).arg(), 0.0)
    } else if sin_half < EPS {
        let gamma = mat[0][0].arg();
        (theta, 0.0, mat[1][1].arg() - gamma, gamma)
    } else {
        let gamma = mat[0][0].arg();
        (
            theta,
            mat[1][0].arg() - gamma,
            (-mat[0][1]).arg() - gamma,
            gamma,
        )
    }
}

/// `(theta, beta)` of an XX+YY or XX-YY block from its `cos(theta / 2)` entry
/// and the off-diagonal `-i sin(theta / 2) e^{-i beta}`.
fn xy_params(cos_entry: Complex64, off: Complex64) -> (f64, f64) {
    let theta = 2.0 * off.norm().atan2(cos_entry.re);
    let beta = if off.norm() < EPS {
        0.0
    } else {
        -(off.arg() + FRAC_PI_2)
    };
    (theta, beta)
}

/// `(theta, phi0, phi1)` of a Mølmer-Sørensen block, whose two off-diagonal
/// entries carry the phase sum and the phase difference in turns.
fn ms_params(mat: &[[Complex64; 4]; 4]) -> (f64, f64, f64) {
    let theta = mat[0][3].norm().atan2(mat[0][0].re) / PI;
    let turns = |entry: Complex64| {
        if entry.norm() < EPS {
            0.0
        } else {
            -(entry.arg() + FRAC_PI_2) / TAU
        }
    };
    let sum = turns(mat[0][3]);
    let diff = turns(mat[1][2]);
    (theta, (sum + diff) / 2.0, (sum - diff) / 2.0)
}

fn close_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> bool {
    a.iter()
        .zip(b)
        .all(|(ra, rb)| ra.iter().zip(rb).all(|(x, y)| (x - y).norm() < EPS))
}

fn close_4x4(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4]) -> bool {
    a.iter()
        .zip(b)
        .all(|(ra, rb)| ra.iter().zip(rb).all(|(x, y)| (x - y).norm() < EPS))
}

/// Classical bits split into the registers the emitted conditions name.
struct CregLayout {
    segments: Vec<(usize, usize)>,
    names: Vec<String>,
}

/// Record the register boundaries every condition in `inst` needs, descending
/// into region bodies so a nested condition still names a whole register.
fn cut_register_conditions(
    inst: &Instruction,
    index: usize,
    total: usize,
    cuts: &mut BTreeSet<usize>,
) -> Result<()> {
    let condition = match inst {
        Instruction::Conditional { condition, .. } => condition,
        Instruction::Region(region) => {
            for inner in region.body() {
                cut_register_conditions(inner, index, total, cuts)?;
            }
            region.condition()
        }
        _ => return Ok(()),
    };
    let (offset, size) = match condition {
        ClassicalCondition::RegisterEquals { offset, size, .. }
        | ClassicalCondition::RegisterNotEquals { offset, size, .. } => (offset, size),
        // A parity names bits rather than a range, so it cuts nothing. Its bounds
        // still need checking here: `CregLayout::bit` panics on a bit past the
        // declared registers, and export of a hand-built circuit is not API
        // misuse.
        ClassicalCondition::Parity { bits, .. } => {
            if let Some(&bit) = bits.iter().find(|&&bit| bit >= total) {
                return Err(PrismError::ExportUnsupported {
                    index,
                    reason: format!("condition reads bit {bit} but the circuit has {total}"),
                });
            }
            return Ok(());
        }
        _ => return Ok(()),
    };
    if offset + size > total {
        return Err(PrismError::ExportUnsupported {
            index,
            reason: format!(
                "condition covers bits {offset}..{} but the circuit has {total}",
                offset + size
            ),
        });
    }
    cuts.insert(*offset);
    cuts.insert(offset + size);
    Ok(())
}

impl CregLayout {
    fn of(circuit: &Circuit) -> Result<Self> {
        let total = circuit.num_classical_bits;
        let mut cuts = BTreeSet::from([0, total]);
        for (index, inst) in circuit.instructions.iter().enumerate() {
            cut_register_conditions(inst, index, total, &mut cuts)?;
        }

        let bounds: Vec<usize> = cuts.into_iter().collect();
        let segments: Vec<(usize, usize)> = bounds
            .windows(2)
            .map(|pair| (pair[0], pair[1] - pair[0]))
            .collect();
        let names = if segments.len() == 1 {
            vec!["c".to_string()]
        } else {
            (0..segments.len()).map(|i| format!("c{i}")).collect()
        };
        Ok(Self { segments, names })
    }

    /// # Panics
    /// Panics if `bit` is past the circuit's classical register, which
    /// [`Circuit::add_measure`] already rejects.
    fn bit(&self, bit: usize) -> String {
        let (position, &(offset, _)) = self
            .segments
            .iter()
            .enumerate()
            .find(|&(_, &(offset, size))| bit >= offset && bit < offset + size)
            .expect("classical bit within the declared register");
        format!("{}[{}]", self.names[position], bit - offset)
    }

    fn condition(&self, condition: &ClassicalCondition, index: usize) -> Result<String> {
        Ok(match condition {
            ClassicalCondition::BitIsOne(bit) => self.bit(*bit),
            ClassicalCondition::BitIsZero(bit) => format!("!{}", self.bit(*bit)),
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            } => format!("{} == {value}", self.register(*offset, *size, index)?),
            ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => format!("{} != {value}", self.register(*offset, *size, index)?),
            // A parity over one bit is a bit test, and only the bit spelling
            // reparses: the parser routes an expression to the parity form on
            // the `^` that a single term does not have.
            ClassicalCondition::Parity { bits, expected } => match bits.as_ref() {
                [] => {
                    return Err(PrismError::ExportUnsupported {
                        index,
                        reason: "parity condition over no bits has no OpenQASM form".to_string(),
                    });
                }
                [bit] if *expected => self.bit(*bit),
                [bit] => format!("!{}", self.bit(*bit)),
                _ => {
                    let terms = bits
                        .iter()
                        .map(|&bit| self.bit(bit))
                        .collect::<Vec<_>>()
                        .join(" ^ ");
                    if *expected {
                        terms
                    } else {
                        format!("({terms}) == 0")
                    }
                }
            },
        })
    }

    fn register(&self, offset: usize, size: usize, index: usize) -> Result<&str> {
        self.segments
            .iter()
            .position(|&segment| segment == (offset, size))
            .map(|position| self.names[position].as_str())
            .ok_or_else(|| PrismError::ExportUnsupported {
                index,
                reason: format!(
                    "condition covers bits {offset}..{}, which overlaps another condition's register",
                    offset + size
                ),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::openqasm;
    use crate::circuit::{SmallVec, smallvec};

    fn reparse(circuit: &Circuit) -> Circuit {
        openqasm::parse(&to_qasm3(circuit).expect("export")).expect("reparse")
    }

    fn one_gate(gate: Gate, targets: &[usize]) -> Circuit {
        let mut c = Circuit::new(targets.iter().max().unwrap() + 1, 0);
        c.add_gate(gate, targets);
        c
    }

    #[test]
    fn named_gates_keep_their_name() {
        for gate in [Gate::H, Gate::T, Gate::Sdg, Gate::SX, Gate::Id] {
            assert_eq!(gate_head(&gate).unwrap(), gate.name());
        }
    }

    #[test]
    fn angles_survive_exactly() {
        let round = reparse(&one_gate(Gate::Rx(0.735_193_628_1), &[0]));
        let Instruction::Gate { gate, .. } = &round.instructions[0] else {
            panic!("expected a gate");
        };
        assert_eq!(*gate, Gate::Rx(0.735_193_628_1));
    }

    #[test]
    fn u3_matrix_round_trips() {
        let mat = Parser::u_matrix(0.7, -1.3, 2.1);
        let round = reparse(&one_gate(Gate::Fused(Box::new(mat)), &[0]));
        let Instruction::Gate { gate, .. } = &round.instructions[0] else {
            panic!("expected a gate");
        };
        assert!(close_2x2(&gate.matrix_2x2(), &mat));
    }

    #[test]
    fn global_phase_on_a_fused_matrix_is_rejected() {
        let phase = Complex64::from_polar(1.0, PI / 3.0);
        let h = Gate::H.matrix_2x2();
        let mat = [
            [phase * h[0][0], phase * h[0][1]],
            [phase * h[1][0], phase * h[1][1]],
        ];
        let circuit = one_gate(Gate::Fused(Box::new(mat)), &[0]);
        assert!(matches!(
            to_qasm3(&circuit),
            Err(PrismError::ExportUnsupported { index: 0, .. })
        ));
    }

    #[test]
    fn rz_carries_the_phase_u3_cannot() {
        assert_eq!(
            spell_1q(&Gate::Rz(0.9).matrix_2x2()).unwrap(),
            format!("rz({})", 0.9_f64)
        );
    }

    #[test]
    fn controlled_unitary_keeps_its_global_phase() {
        let mat = Gate::Rz(1.1).matrix_2x2();
        let round = reparse(&one_gate(Gate::cu(mat), &[0, 1]));
        let Instruction::Gate { gate, .. } = &round.instructions[0] else {
            panic!("expected a gate");
        };
        assert!(close_4x4(&gate.matrix_4x4(), &Gate::cu(mat).matrix_4x4()));
    }

    #[test]
    fn multi_controlled_x_uses_the_named_forms() {
        assert_eq!(
            gate_head(&Gate::mcu(Gate::X.matrix_2x2(), 2)).unwrap(),
            "ccx"
        );
        assert_eq!(
            gate_head(&Gate::mcu(Gate::X.matrix_2x2(), 4)).unwrap(),
            "mcx"
        );
    }

    #[test]
    fn multi_controlled_unitary_chains_ctrl() {
        let head = gate_head(&Gate::mcu(Gate::H.matrix_2x2(), 3)).unwrap();
        assert!(head.starts_with("ctrl @ ctrl @ cu("), "{head}");
    }

    #[test]
    fn fusion_payloads_are_rejected() {
        let mut circuit = Circuit::new(2, 0);
        circuit.instructions.push(Instruction::Gate {
            gate: Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())),
            targets: smallvec![0, 1],
        });
        assert!(matches!(
            to_qasm3(&circuit),
            Err(PrismError::ExportUnsupported { index: 0, .. })
        ));
    }

    #[test]
    fn conditions_split_the_classical_register() {
        let mut circuit = Circuit::new(1, 4);
        circuit.add_measure(0, 0);
        circuit.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::RegisterEquals {
                offset: 2,
                size: 2,
                value: 3,
            },
            gate: Gate::X,
            targets: SmallVec::from_slice(&[0]),
        });
        let qasm = to_qasm3(&circuit).expect("export");
        assert!(qasm.contains("bit[2] c0;\nbit[2] c1;"), "{qasm}");
        assert!(qasm.contains("if (c1 == 3) x q[0];"), "{qasm}");
        assert_eq!(
            openqasm::parse(&qasm).expect("reparse").num_classical_bits,
            4
        );
    }

    #[test]
    fn empty_barrier_has_no_spelling() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_barrier(&[]);
        assert!(to_qasm3(&circuit).is_err());
    }
}
