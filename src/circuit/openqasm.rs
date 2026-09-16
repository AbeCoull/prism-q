//! OpenQASM 3.0 parser, v0 subset.
//!
//! The front end reads a token stream: the `qasm` module turns source text into
//! tokens and a syntax tree, and this module walks that tree into a [`Circuit`].
//! A statement ends at its `;` and a block at its `}` wherever the newlines
//! fall.
//!
//! # Supported constructs
//!
//! | Construct | Example | Notes |
//! |-----------|---------|-------|
//! | Header | `OPENQASM 3.0;` | 2.0 also accepted for compat; a later major version is rejected rather than read as 3 |
//! | Include | `include "stdgates.inc";` | Accepted, ignored (gates built-in) |
//! | Qubit declaration | `qubit[4] q;` | OQ3 syntax (primary) |
//! | Bit declaration | `bit[4] c;` | OQ3 syntax (primary) |
//! | Legacy qreg/creg | `qreg q[4]; creg c[4];` | OQ2 compat |
//! | Input parameter | `input float[64] theta;` | One named slot; [`parse_parametric`] returns them |
//! | Output declaration | `output bit[4] c;` | Declares the register; every bit is reported anyway |
//! | 1-qubit gates | `h q[0]; x q[1];` | id/i, x, y, z, h, s, sdg/si, t, tdg/ti, sx/v, sxdg/vi, p/phase/phaseshift, r/prx, gpi, gpi2, u/U forms |
//! | Parametric gates | `rx(pi/4) q[0];` | rx, ry, rz, cu, ms, arithmetic expressions with `pi`, math functions |
//! | 2-qubit gates | `cx q[0], q[1];` | cx/cnot, cy, cz, ch, cs, csdg, cp/cphase/cphaseshift, cphaseshift00/01/10, crx, cry, crz, csx/cv, swap, pswap, xy, xx_plus_yy, xx_minus_yy, ecr, iswap, dcx, syc, sqrt_iswap |
//! | Pauli rotation | `rzz(t) q[0], q[1];` `rxyz(t) q[0], q[1], q[2];` | `r` plus the Pauli letters, one per qubit argument; `rxx`, `ryy`, `rzz` are the two-letter cases, also spelled `xx`, `yy`, `zz` |
//! | Multi-qubit gates | `ccx q[0], q[1], q[2];` | ccx/toffoli/ccnot, ccz, cswap/fredkin, c3x, c4x, mcx, rccx, rc3x/rcccx |
//! | Gate modifiers | `inv @ h q[0];` | `inv @`, `ctrl @` and `negctrl @` (chainable), `pow(k) @` for any real k. A control applies to whatever the gate expanded to, so it reaches a user `gate` and a lowered gate as well as a direct one; a fractional `pow` is the principal power of what the call expanded to |
//! | Global phase | `gphase(pi/2);` `ctrl @ gphase(pi/2) q[0];` | Carried rather than dropped: it is observable through a `state_vector` result and under a control |
//! | Classical declaration | `int n = 3;` `const float t = pi/4;` | `int`, `uint`, `bool`, `float`, `angle`, with an optional width. Folded at parse time, so the value reads as an index, a loop bound, a gate angle or a condition operand |
//! | Classical assignment | `n = n + 1;` `n += 1;` | Plain and compound forms on a declared, non-`const` name |
//! | Register slice | `h q[0:2];` `h q[0:2:6];` `h q[{0, 3}];` | Inclusive range with an optional step in the middle, or an explicit index set. Broadcasts like a whole register |
//! | Register alias | `let a = q[0:1];` `let a = q[2] ++ q[0];` | Names qubits or bits in the order written; an alias is itself sliceable |
//! | Physical qubits | `h $0;` `cx $0, $1;` | Absolute indices with no declaration; the register is as wide as the highest one named, and a declared register alongside is rejected |
//! | Block comments | `/* ... */` | Skipped by the lexer; they do not nest |
//! | Measurement (OQ3) | `c[0] = measure q[0];` | Assignment syntax (primary) |
//! | Measurement (OQ2) | `measure q[0] -> c[0];` | Arrow syntax (compat) |
//! | Register broadcast | `h q;` / `cx q, r;` | Applies gate to all qubits in register |
//! | Conditional (OQ2) | `if(c==1) x q[0];` | Classical register equality |
//! | Conditional (OQ3) | `if (c[0]) x q[0];` | Single classical bit test |
//! | Conditional inequality | `if (c != 0) x q[0];` | Register or bit `!=` |
//! | Conditional bit literal | `if (c[0] == 1) x q[0];` | Bit equality vs `0` / `1` |
//! | Conditional negation | `if (!c[0]) x q[0];` | Negated bit truthy test |
//! | Guarded region | `if (c[0]) { x q[0]; measure q[1] -> c[1]; }` | Braced body, any statement, nestable |
//! | Conditional parity | `if (c[0] ^ c[2]) x q[0];` | Parity over bits, optionally `(...) == 0` |
//! | Else arm | `if (c[0]) { ... } else { ... }` | Lowers to a second guard on the negated condition |
//! | Else-if chain | `if (c[0]) { ... } else if (c[1]) { ... }` | Nests under the negated arm |
//! | Switch | `switch (c) { case 0 { ... } default { ... } }` | Lowers to one guard per case label |
//! | Hex / binary literals | `if (c == 0xff) ...` | `0x`, `0b`, `0o` integer prefixes with optional `_` separators |
//! | Boolean literals | `rx(true * pi) ...` | `true` / `false` evaluate to `1.0` / `0.0` |
//! | Gate definition | `gate rxx(t) a,b { ... }` | User-defined gates |
//! | Subroutine definition | `def myg(qubit a, float t) { ... }` | Unitary `def` bodies, inlined at the call site |
//! | Static for loop | `for int i in [0:n] { ... }` | Inclusive ranges, optional step, set form `{a,b,c}` |
//! | Barrier | `barrier q[0], q[1];` | |
//! | Line comments | `// comment` | |
//! | Expression operators | `rx(2 ** -1 % 3)` | `+ - * / %` and `**`, which is right associative and binds tighter than unary minus |
//! | Expression builtins | `rx(mod(7, 3))` | `arcsin`, `arccos`, `arctan`, `ceiling`, `cos`, `exp`, `floor`, `log`, `mod`, `popcount`, `pow`, `sin`, `sqrt`, `tan`, plus the shorter C spellings |
//! | Angle dialect | [`parse_with`] | `gpi`, `gpi2`, `ms` take turns under [`Dialect::Native`] and radians under [`Dialect::Braket`] |
//! | Result pragma | `#pragma braket result expectation z(q[0])` | [`Dialect::Braket`] only; reaches the caller through [`parse_braket`] |
//! | Noise pragma | `#pragma braket noise bit_flip(0.1) q[0]` | Builds a [`NoiseModel`] event after the preceding instruction |
//! | Inline unitary | `#pragma braket unitary([[0, 1], [1, 0]]) q[0]` | One or two targets; wider has no matrix gate variant |
//! | Verbatim box | `#pragma braket verbatim` then `box { ... }` | The body runs as written; a `box` without the pragma is rejected |
//!
//! # Unsupported constructs (return `PrismError::UnsupportedConstruct`)
//!
//! - `defcal`, `extern`, `opaque`, `while`, `return`, `break`, and a `box`
//!   that no `#pragma braket verbatim` precedes
//! - `def` bodies that contain `measure`, `reset`, `bit`, `creg`, `return`,
//!   or the `=measure` assignment shape (V1 supports unitary subroutines only)
//! - `def` declarations with a return type
//! - `ctrl @` on a `def` call: a subroutine is not a gate and the language
//!   gives it no controlled form
//! - `ctrl @` and a fractional `pow(k) @` on a call spanning more than four
//!   qubits: both reduce the call to a dense matrix first, and that matrix is
//!   where the width bound sits
//! - a `qubit` or `qreg` declaration in a program that also names physical
//!   qubits (`$0`)
//! - Bit literal comparisons against integers other than `0` / `1`
//! - Negative integer literals in `if` register comparisons
//! - `else` whose `if` body measures into a bit the condition reads, and
//!   `switch` whose arm measures into the switched register: both lowerings
//!   re-read the bits after an earlier body ran
//! - `switch` with a `default` and more case labels than the region depth bound
//! - `duration`, `stretch` outside `def` parameter lists
//! - `input` of any type but `float` and `angle`, and `output` of any type but
//!   `bit`
//! - an `input` anywhere but as the whole angle argument of a top-level
//!   parametric gate: an expression over one, two on one gate, a modified gate,
//!   a gate carrying no rotation angle, and a use inside a `gate`, `def`,
//!   `for`, or guarded-region body all reject
//!
//! # Error behaviour
//!
//! All parse failures return `PrismError::Parse` or `PrismError::UnsupportedConstruct`
//! with the source line number. The parser never panics on user input.
//!
//! The reverse direction is [`qasm_export`](super::qasm_export).

use num_complex::Complex64;

use super::braket::{self, NoiseSpec, ResultSpec};
use crate::circuit::qasm::ast::{self, DefParam};
use crate::circuit::synthesis;
use crate::circuit::{
    Circuit, ClassicalCondition, Instruction, MAX_REGION_DEPTH, ParamLink, Parameters, SmallVec,
    guarded, pauli_rotation_gate, smallvec,
};
use crate::error::{PrismError, Result};
use crate::gates::{Gate, spectral};
use crate::sim::noise::{NoiseEvent, NoiseModel};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};
use std::collections::HashMap;

fn parse_error(line: usize, message: impl Into<String>) -> PrismError {
    PrismError::Parse {
        line,
        message: message.into(),
    }
}

/// Angle convention a program is read under.
///
/// Vendors disagree on the units of the hardware-native family (`gpi`, `gpi2`,
/// `ms`): IonQ's transpiler emits turns, Amazon Braket emits radians, and the
/// gate names are the same either way. Every other gate takes radians under
/// both.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Dialect {
    /// IonQ's reading: `gpi`, `gpi2`, and `ms` take turns.
    #[default]
    Native,
    /// Amazon Braket's reading: `gpi`, `gpi2`, and `ms` take radians.
    Braket,
}

impl Dialect {
    /// Convert a source angle for the hardware-native family into turns, the
    /// unit [`Parser::ms_matrix`] and its peers take.
    fn native_turns(self, angle: f64) -> f64 {
        match self {
            Dialect::Native => angle,
            Dialect::Braket => angle / std::f64::consts::TAU,
        }
    }
}

/// Parse an OpenQASM 3.0 string into a PRISM-Q [`Circuit`].
///
/// This is the primary input entrypoint. The entire parse happens in-memory
/// from the provided `&str`, no file I/O. Reads the source under
/// [`Dialect::Native`]; use [`parse_with`] to select another.
///
/// # Errors
///
/// Returns structured [`PrismError`] for any parse failure or unsupported
/// construct, and [`PrismError::InvalidParameter`] when the program declares an
/// `input`, whose value this entry point has nowhere to take. Use
/// [`parse_parametric`] for those.
pub fn parse(input: &str) -> Result<Circuit> {
    parse_with(input, Dialect::Native)
}

/// Parse an OpenQASM 3.0 string under an explicit [`Dialect`].
///
/// # Errors
///
/// Same conditions as [`parse`].
pub fn parse_with(input: &str, dialect: Dialect) -> Result<Circuit> {
    let (circuit, params) = parse_parametric_with(input, dialect)?;
    if params.num_slots() > 0 {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "program declares {} `input` parameter(s); parse it with `parse_parametric` and bind them",
                params.num_slots()
            ),
        });
    }
    Ok(circuit)
}

/// Parse an OpenQASM 3.0 string into a [`Circuit`] plus the [`Parameters`] its
/// `input` declarations name.
///
/// Slots are ordered by declaration and carry the declared names, so
/// [`Parameters::slot_of`] resolves an angle by the name the source used. The
/// returned circuit is a template holding zero for every input; bind it before
/// running, since dispatch reads the angles it is given.
///
/// An `input` may only be the whole angle argument of a directly named
/// parametric gate at the top level. An expression over one, a use inside a
/// gate, `def`, `for`, or guarded-region body, and a use on a gate carrying no
/// rotation angle all return [`PrismError::UnsupportedConstruct`].
///
/// # Errors
///
/// Same conditions as [`parse`], less the `input` rejection.
pub fn parse_parametric(input: &str) -> Result<(Circuit, Parameters)> {
    parse_parametric_with(input, Dialect::Native)
}

/// Parse an OpenQASM 3.0 string plus its `input` [`Parameters`] under an
/// explicit [`Dialect`].
///
/// # Errors
///
/// Same conditions as [`parse_parametric`].
pub fn parse_parametric_with(input: &str, dialect: Dialect) -> Result<(Circuit, Parameters)> {
    Parser::new_with(input, dialect).parse()
}

/// A Braket program: the circuit, its free parameters, and everything the
/// `#pragma braket` lines declared beside it.
#[derive(Debug, Clone)]
pub struct BraketProgram {
    pub circuit: Circuit,
    pub parameters: Parameters,
    /// `#pragma braket result` requests in declaration order, empty when the
    /// program measures instead.
    pub results: Vec<ResultSpec>,
    /// The model `#pragma braket noise` lines built, `None` when there were
    /// none.
    pub noise: Option<NoiseModel>,
}

/// Parse an OpenQASM 3.0 string under [`Dialect::Braket`], keeping what the
/// pragmas declare.
///
/// This is the entry point for Braket programs. [`parse`] and
/// [`parse_parametric`] read [`Dialect::Native`], where a pragma is an
/// unsupported construct, and [`parse_with`] under [`Dialect::Braket`] reads
/// the pragmas but has nowhere to return what they declared, so it drops them.
///
/// # Errors
///
/// Same conditions as [`parse_parametric`], plus a malformed pragma and a
/// noise pragma with no preceding instruction to follow.
pub fn parse_braket(input: &str) -> Result<BraketProgram> {
    let mut parser = Parser::new_with(input, Dialect::Braket);
    let (circuit, parameters, results, noise) = parser.parse_program()?;
    Ok(BraketProgram {
        circuit,
        parameters,
        results,
        noise,
    })
}

#[derive(Clone, Copy)]
enum Modifier {
    Inv,
    /// `pow(k)`. A real exponent, so a fractional power of a single-qubit gate
    /// reaches [`Gate::matrix_power_real`].
    Pow(f64),
    /// `ctrl` and `negctrl`, which differ only in the control polarity.
    Ctrl {
        negated: bool,
    },
}

impl Modifier {
    /// Control polarity when this is a control modifier, `None` otherwise.
    fn control(&self) -> Option<bool> {
        match self {
            Modifier::Ctrl { negated } => Some(*negated),
            _ => None,
        }
    }
}

struct Register {
    offset: usize,
    size: usize,
}

struct GateDefinition<'a> {
    params: Vec<&'a str>,
    qubits: Vec<&'a str>,
    body: ast::Block<'a>,
}

struct DefDefinition<'a> {
    args: Vec<DefParam<'a>>,
    body: ast::Block<'a>,
}

/// A `let` alias: the qubits or bits it names, in the order it named them.
struct Alias {
    kind: ast::RegisterKind,
    indices: Vec<usize>,
}

/// Type of a classical variable, which decides how its value is read back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ClassicalType {
    Int,
    Bool,
    Float,
}

struct ClassicalDecl {
    ty: ClassicalType,
    constant: bool,
}

/// The error an out-of-range subscript raises, on the side of the register
/// wall the reference sits on.
fn invalid_index(kind: ast::RegisterKind, index: usize, register_size: usize) -> PrismError {
    match kind {
        ast::RegisterKind::Qubit => PrismError::InvalidQubit {
            index,
            register_size,
        },
        ast::RegisterKind::Classical => PrismError::InvalidClassicalBit {
            index,
            register_size,
        },
    }
}

pub(crate) struct Parser<'a> {
    input: &'a str,
    qregs: HashMap<&'a str, Register>,
    cregs: HashMap<&'a str, Register>,
    gate_defs: HashMap<&'a str, GateDefinition<'a>>,
    def_defs: HashMap<&'a str, DefDefinition<'a>>,
    total_qubits: usize,
    total_cbits: usize,
    gate_expansion_depth: usize,
    region_depth: usize,
    param_vars: Option<HashMap<&'a str, f64>>,
    /// `input` slot per declared name, and the names in slot order.
    inputs: HashMap<&'a str, usize>,
    input_names: Vec<String>,
    links: Vec<ParamLink>,
    /// Slot the statement just run reads, handed up to the block walk, which
    /// is the only place that knows the instruction index the link needs.
    pending_input_slot: Option<usize>,
    /// True while parsing a block body, whose instruction indices are local to
    /// that body and so cannot carry a top-level parameter link.
    nested: bool,
    dialect: Dialect,
    /// `#pragma braket result` requests, in declaration order.
    results: Vec<ResultSpec>,
    /// `#pragma braket noise` events, each paired with the top-level
    /// instruction index it follows.
    noise_specs: Vec<(usize, NoiseSpec)>,
    /// Noise the statement just run declared, handed up to the block walk,
    /// which is the only place that knows the instruction index it follows.
    pending_noise: Option<NoiseSpec>,
    /// True once a verbatim pragma has been seen and before its `box` opens.
    verbatim_pending: bool,
    /// The program names physical qubits (`$0`) instead of declaring registers.
    physical: bool,
    /// `let` aliases by name; the values live beside the registers they index.
    aliases: HashMap<&'a str, Alias>,
    /// Declared classical variables. The values sit in `param_vars`, which is
    /// where every expression already reads them.
    classical: HashMap<&'a str, ClassicalDecl>,
}

const MAX_GATE_EXPANSION_DEPTH: usize = 32;
const MAX_FOR_ITERATIONS: i64 = 1_000_000;

/// Largest whole-number `pow(k)` exponent, which bounds both the repetition of
/// an expansion and the matrix product a single-qubit gate takes. Without it a
/// literal exponent aborts the process rather than returning an error.
const MAX_POW_REPEATS: i64 = 1_000_000;

/// Pauli letters of an `r<letters>` rotation name, or `None` when the name is
/// not one.
///
/// Covers the `rxx`, `ryy`, and `rzz` OpenQASM already spells as well as the
/// wider strings it does not, so one rule serves the whole family. `rzz` still
/// resolves to `Gate::Rzz` and a weight-1 name to `Rx`/`Ry`/`Rz`, because
/// `pauli_rotation_gate` lowers those; only the residual strings build the
/// native multi-qubit gate. Names like `rccx` do not match, `c` being no Pauli
/// letter.
fn pauli_rotation_axes(name: &str) -> Option<Vec<PauliAxis>> {
    let letters = name.strip_prefix('r')?;
    if letters.len() < 2 {
        return None;
    }
    letters.chars().map(PauliAxis::from_letter).collect()
}

/// The `r`-prefixed spelling of a two-qubit Pauli rotation Braket writes bare.
///
/// Exact names only: `xy` is Braket's XY interaction, not `rxy`.
fn bare_pauli_rotation(name: &str) -> Option<&'static str> {
    match name {
        "xx" => Some("rxx"),
        "yy" => Some("ryy"),
        "zz" => Some("rzz"),
        _ => None,
    }
}

/// Wrap `body` in one negated-equality region per case label, which is how a
/// conjunction is spelled in a condition language that has no `and`.
fn nest_default_arm(
    offset: usize,
    size: usize,
    labels: &[u64],
    body: Vec<Instruction>,
    region_depth: usize,
    line_num: usize,
) -> Result<Vec<Instruction>> {
    // The nesting the default costs is on top of wherever the `switch` sits, so
    // the bound is the sum rather than the label count alone.
    if region_depth + labels.len() > MAX_REGION_DEPTH {
        return Err(parse_error(
            line_num,
            format!(
                "`switch` with a `default` nests one region per case label and would \
                 pass the depth bound of {MAX_REGION_DEPTH} at {} labels",
                labels.len()
            ),
        ));
    }
    let mut current = body;
    for &value in labels.iter().rev() {
        let condition = ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        };
        current = guarded(condition, current).into_iter().collect();
    }
    Ok(current)
}

impl<'a> Parser<'a> {
    fn new_with(input: &'a str, dialect: Dialect) -> Self {
        Self {
            dialect,
            input,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            def_defs: HashMap::new(),
            total_qubits: 0,
            total_cbits: 0,
            gate_expansion_depth: 0,
            region_depth: 0,
            param_vars: None,
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: false,
            results: Vec::new(),
            noise_specs: Vec::new(),
            pending_noise: None,
            verbatim_pending: false,
            physical: false,
            aliases: HashMap::new(),
            classical: HashMap::new(),
        }
    }

    fn parse(mut self) -> Result<(Circuit, Parameters)> {
        let (circuit, params, _, _) = self.parse_program()?;
        Ok((circuit, params))
    }

    fn build_noise_model(&mut self, circuit: &Circuit) -> Result<Option<NoiseModel>> {
        if self.noise_specs.is_empty() {
            return Ok(None);
        }
        let mut model = NoiseModel {
            after_gate: vec![Vec::new(); circuit.instructions.len()],
            readout: vec![None; circuit.num_classical_bits],
        };
        for (index, spec) in std::mem::take(&mut self.noise_specs) {
            model.after_gate[index].push(NoiseEvent {
                channel: spec.channel,
                qubits: spec.qubits.into_iter().collect(),
            });
        }
        model.validate_for(circuit)?;
        Ok(Some(model))
    }

    fn bind_classical(&mut self, name: &'a str, ty: ClassicalType, value: f64, constant: bool) {
        self.classical.insert(name, ClassicalDecl { ty, constant });
        self.param_vars
            .get_or_insert_with(HashMap::new)
            .insert(name, value);
    }

    fn reject_redeclaration(&self, name: &str, line_num: usize) -> Result<()> {
        let clash = if self.classical.contains_key(name) {
            "a classical variable"
        } else if self.aliases.contains_key(name) {
            "an alias"
        } else if self.qregs.contains_key(name) || self.cregs.contains_key(name) {
            "a register"
        } else if self.inputs.contains_key(name) {
            "an input"
        } else {
            return Ok(());
        };
        Err(parse_error(
            line_num,
            format!("`{name}` is already {clash}"),
        ))
    }

    /// A physical qubit is an absolute index and a declared one is an offset
    /// into a register, so a program using both has two meanings for `0`.
    fn reject_mixed_addressing(&self, line_num: usize) -> Result<()> {
        if self.physical {
            return Err(PrismError::UnsupportedConstruct {
                construct:
                    "a qubit register beside physical qubits (`$0`), which address the same hardware two ways"
                        .to_string(),
                line: line_num,
            });
        }
        Ok(())
    }

    fn build_measurements(
        qubits: SmallVec<[usize; 4]>,
        cbits: SmallVec<[usize; 4]>,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if qubits.len() != cbits.len() {
            return Err(parse_error(
                line_num,
                format!(
                    "register size mismatch in measure: {} qubits vs {} classical bits",
                    qubits.len(),
                    cbits.len()
                ),
            ));
        }
        Ok(qubits
            .into_iter()
            .zip(cbits)
            .map(|(qubit, classical_bit)| Instruction::Measure {
                qubit,
                classical_bit,
            })
            .collect())
    }

    /// `gphase(theta)` multiplies the state by `e^(i theta)`.
    ///
    /// A global phase is observable through a `state_vector` result and under a
    /// control, where `ctrl(e^(i theta) I)` is a phase gate on the control, so
    /// it is carried rather than dropped. With controls it becomes exactly that
    /// phase; without, an identity scaled by the phase on the first qubit,
    /// which is the same operator wherever it is placed.
    fn resolve_global_phase(
        &self,
        params: &[f64],
        modifiers: &[Modifier],
        qubits: &[usize],
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        Self::expect_param_count("gphase", params, 1, line_num)?;
        let controls: Vec<bool> = modifiers.iter().filter_map(Modifier::control).collect();
        if qubits.len() != controls.len() {
            return Err(PrismError::GateArity {
                gate: "gphase".to_string(),
                expected: controls.len(),
                got: qubits.len(),
            });
        }
        let mut angle = params[0];
        for modifier in modifiers {
            match modifier {
                Modifier::Inv => angle = -angle,
                Modifier::Pow(k) => angle *= k,
                Modifier::Ctrl { .. } => {}
            }
        }

        if controls.is_empty() {
            if self.total_qubits == 0 {
                return Err(parse_error(
                    line_num,
                    "`gphase` needs a qubit to carry the phase, and none is declared",
                ));
            }
            let scale = Complex64::from_polar(1.0, angle);
            let zero = Complex64::new(0.0, 0.0);
            return Ok(vec![Self::ig(
                Gate::Fused(Box::new([[scale, zero], [zero, scale]])),
                &[0],
            )]);
        }

        let negated: Vec<usize> = qubits
            .iter()
            .zip(&controls)
            .filter(|&(_, &negated)| negated)
            .map(|(&qubit, _)| qubit)
            .collect();
        let mut out: Vec<Instruction> = negated.iter().map(|&q| Self::ig(Gate::X, &[q])).collect();
        let phase = Gate::P(angle);
        out.push(match qubits.split_last() {
            Some((&last, [])) => Self::ig(phase, &[last]),
            Some((_, rest)) => Self::ig(Gate::mcu(phase.matrix_2x2(), rest.len() as u8), qubits),
            None => unreachable!("the control-free case returned above"),
        });
        out.extend(negated.iter().map(|&q| Self::ig(Gate::X, &[q])));
        Ok(out)
    }

    /// Build the instruction an `r<letters>` application names, checking the
    /// arity and distinctness `pauli_rotation_gate` would otherwise panic on.
    fn resolve_pauli_rotation(
        gate_name: &str,
        axes: &[PauliAxis],
        params: &[f64],
        qubits: &SmallVec<[usize; 4]>,
        line_num: usize,
    ) -> Result<Instruction> {
        Self::expect_param_count(gate_name, params, 1, line_num)?;
        if qubits.len() != axes.len() {
            return Err(PrismError::GateArity {
                gate: gate_name.to_string(),
                expected: axes.len(),
                got: qubits.len(),
            });
        }
        let mut seen: SmallVec<[usize; 4]> = qubits.clone();
        seen.sort_unstable();
        if let Some(pair) = seen.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(parse_error(
                line_num,
                format!("Pauli rotation `{gate_name}` names qubit {} twice", pair[0]),
            ));
        }
        let factors: Vec<PauliTerm> = qubits
            .iter()
            .zip(axes)
            .map(|(&qubit, &axis)| PauliTerm::new(qubit, axis))
            .collect();
        let (gate, targets) = pauli_rotation_gate(params[0], &factors);
        Ok(Instruction::Gate { gate, targets })
    }

    /// A slot writes one angle onto each instruction it links, so every
    /// instruction a gate reading an `input` produced has to carry one.
    fn check_every_instruction_is_bindable(
        instrs: &[Instruction],
        gate_name: &str,
        line_num: usize,
    ) -> Result<()> {
        let bindable = instrs.iter().all(|instr| {
            matches!(instr, Instruction::Gate { gate, .. } if gate.pauli_generator().is_some())
        });
        if !bindable {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`input` on `{gate_name}`, which carries no rotation angle to bind"
                ),
                line: line_num,
            });
        }
        Ok(())
    }

    /// Build what one gate call names, splitting off the control qubits a
    /// `ctrl` or `negctrl` chain consumes.
    ///
    /// The arity a controlled call spells includes those qubits, so the body is
    /// resolved on what is left and the controls are added to whatever it
    /// expanded to. `inv` and `pow` commute with a control, so they apply to
    /// the body first and the result is the same either way.
    fn resolve_gate_application_once(
        &self,
        gate_name: &str,
        params: &[f64],
        modifiers: &[Modifier],
        qubits: &SmallVec<[usize; 4]>,
        has_input: bool,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        let polarity: Vec<bool> = modifiers.iter().filter_map(Modifier::control).collect();
        if polarity.is_empty() {
            return self
                .resolve_gate_body(gate_name, params, modifiers, qubits, has_input, line_num);
        }
        if qubits.len() <= polarity.len() {
            return Err(PrismError::GateArity {
                gate: gate_name.to_string(),
                expected: polarity.len() + 1,
                got: qubits.len(),
            });
        }
        let (controls, rest) = qubits.split_at(polarity.len());
        let unitary: Vec<Modifier> = modifiers
            .iter()
            .filter(|m| m.control().is_none())
            .copied()
            .collect();
        let body = self.resolve_gate_body(
            gate_name,
            params,
            &unitary,
            &rest.iter().copied().collect(),
            has_input,
            line_num,
        )?;
        let controlled = Self::control_expansion(body, controls, line_num)?;
        Ok(Self::conjugate_negations(controlled, controls, &polarity))
    }

    fn resolve_gate_body(
        &self,
        gate_name: &str,
        params: &[f64],
        modifiers: &[Modifier],
        qubits: &SmallVec<[usize; 4]>,
        has_input: bool,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if let Some(instrs) = Self::resolve_decomposed_gate(gate_name, params, qubits, line_num)? {
            // A lowering folds the angle into its own arithmetic, so binding a
            // slot afterwards would write the raw value over a derived one.
            if has_input {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`input` on `{gate_name}`, which lowers to a gate sequence"),
                    line: line_num,
                });
            }
            return Self::modify_expansion(instrs, modifiers, gate_name, line_num);
        }

        if let Some(instrs) = self.expand_user_gate(gate_name, params, qubits, line_num)? {
            // Expansion substitutes the numeric value into the body, so a body
            // writing `rx(2 * a)` would take a bound angle whole.
            if has_input {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`input` on user-defined gate `{gate_name}`"),
                    line: line_num,
                });
            }
            return Self::modify_expansion(instrs, modifiers, gate_name, line_num);
        }

        if let Some(axes) = pauli_rotation_axes(bare_pauli_rotation(gate_name).unwrap_or(gate_name))
        {
            let instr = Self::resolve_pauli_rotation(gate_name, &axes, params, qubits, line_num)?;
            return Self::modify_expansion(vec![instr], modifiers, gate_name, line_num);
        }

        let mut gate = Self::resolve_gate(gate_name, params, self.dialect, line_num)?;
        let expected = gate.num_qubits();
        if qubits.len() != expected {
            return Err(PrismError::GateArity {
                gate: gate_name.to_string(),
                expected,
                got: qubits.len(),
            });
        }
        // A matrix power needs a matrix, which only the single-qubit gates
        // carry. Wider gates take a whole-number power as repetition.
        let foldable = expected == 1 || modifiers.iter().all(|m| !matches!(m, Modifier::Pow(_)));
        if !foldable {
            let single = vec![Instruction::Gate {
                gate,
                targets: qubits.clone(),
            }];
            return Self::modify_expansion(single, modifiers, gate_name, line_num);
        }
        for modifier in modifiers.iter().rev() {
            gate = Self::apply_modifier(gate, modifier);
        }
        Ok(vec![Instruction::Gate {
            gate,
            targets: qubits.clone(),
        }])
    }

    /// Determine the broadcast length from resolved qubit arguments.
    /// All multi-element args must have the same length. Single-element args broadcast.
    fn broadcast_length(
        &self,
        resolved: &[SmallVec<[usize; 4]>],
        gate_name: &str,
        line_num: usize,
    ) -> Result<usize> {
        let mut broadcast_len = 1usize;
        for arg in resolved {
            if arg.len() > 1 {
                if broadcast_len == 1 {
                    broadcast_len = arg.len();
                } else if arg.len() != broadcast_len {
                    return Err(parse_error(
                        line_num,
                        format!(
                            "register size mismatch in `{gate_name}`: \
                             expected {broadcast_len} qubits but got {}",
                            arg.len()
                        ),
                    ));
                }
            }
        }
        Ok(broadcast_len)
    }

    fn classical_copy(&self) -> HashMap<&'a str, ClassicalDecl> {
        self.classical
            .iter()
            .map(|(name, decl)| {
                (
                    *name,
                    ClassicalDecl {
                        ty: decl.ty,
                        constant: decl.constant,
                    },
                )
            })
            .collect()
    }

    /// Accept the OpenQASM versions this parser implements, and name the one it
    /// was handed otherwise.
    ///
    /// A 2.0 program is read as the compatible subset. Anything past 3 is
    /// rejected rather than read as 3 and silently misparsed where the
    /// languages differ.
    fn check_version_number(version: &str, line_num: usize) -> Result<()> {
        let major = version
            .split_once('.')
            .map_or(version, |(major, _)| major)
            .trim();
        match major.parse::<u32>() {
            Ok(2) | Ok(3) => Ok(()),
            Ok(_) => Err(PrismError::UnsupportedConstruct {
                construct: format!("OPENQASM {version}, where this parser reads 2 and 3"),
                line: line_num,
            }),
            Err(_) => Err(parse_error(
                line_num,
                format!("`{version}` is not an OpenQASM version"),
            )),
        }
    }

    /// A square matrix read the other way round, which swaps row-major and
    /// column-major without changing what it means.
    fn transposed(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
        for row in 0..dim {
            for column in 0..dim {
                out[column * dim + row] = matrix[row * dim + column];
            }
        }
        out
    }

    fn ctrl_on_expansion_error(name: &str, line_num: usize) -> PrismError {
        PrismError::UnsupportedConstruct {
            construct: format!(
                "ctrl @ `{name}`, a subroutine rather than a gate, so it has no controlled form"
            ),
            line: line_num,
        }
    }

    /// Repeat an expansion `k` times, inverting it first for a negative `k`.
    ///
    /// A whole-number exponent is repetition whatever the gate's width, which
    /// is why a multi-qubit `pow` needs no matrix power.
    fn repeat_expansion(
        instrs: Vec<Instruction>,
        k: f64,
        name: &str,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if k.fract() != 0.0 {
            // A fraction is a matrix power rather than a repetition, so the
            // whole expansion is composed into one matrix first, whether it
            // is a single gate or a lowering of several.
            let Some((span, matrix)) = synthesis::expansion_matrix(&instrs) else {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "pow({k}) @ `{name}`, whose lowering spans more than {} qubits or \
                         carries an instruction with no matrix",
                        synthesis::MAX_COMPOSED_QUBITS
                    ),
                    line: line_num,
                });
            };
            let dim = 1usize << span.len();
            let powered = spectral::unitary_power(&Self::transposed(&matrix, dim), dim, k);
            return Ok(synthesis::dense_unitary(
                &Self::transposed(&powered, dim),
                &span,
            ));
        }
        let base = if k < 0.0 {
            Self::invert_expansion(instrs, name, line_num)?
        } else {
            instrs
        };
        let repeats = k.abs() as usize;
        let mut powered = Vec::with_capacity(base.len() * repeats);
        for _ in 0..repeats {
            powered.extend(base.iter().cloned());
        }
        Ok(powered)
    }

    /// Add `controls` to every instruction of an expansion.
    ///
    /// `ctrl(U1 U2 ... Un) = ctrl(U1) ctrl(U2) ... ctrl(Un)`, so a gate with no
    /// controlled form of its own still has one wherever its lowering does.
    fn control_expansion(
        instrs: Vec<Instruction>,
        controls: &[usize],
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        let mut out = Vec::with_capacity(instrs.len());
        for instr in instrs {
            if let Instruction::Gate { targets, .. } = &instr
                && let Some(&shared) = targets.iter().find(|target| controls.contains(target))
            {
                return Err(parse_error(
                    line_num,
                    format!("qubit {shared} is both a control and a target"),
                ));
            }
            out.extend(Self::control_instruction(instr, controls, line_num)?);
        }
        Ok(out)
    }

    /// One instruction under `controls`, as one or more instructions.
    fn control_instruction(
        instr: Instruction,
        controls: &[usize],
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        let Instruction::Gate { gate, targets } = instr else {
            return Err(PrismError::UnsupportedConstruct {
                construct: "ctrl @ a body that measures, resets, or branches".to_string(),
                line: line_num,
            });
        };
        let extra = controls.len();
        let prefixed = |rest: &[usize]| -> SmallVec<[usize; 4]> {
            controls
                .iter()
                .copied()
                .chain(rest.iter().copied())
                .collect()
        };
        let widen = |num_controls: usize| -> Result<u8> {
            num_controls
                .checked_add(extra)
                .and_then(|total| u8::try_from(total).ok())
                .ok_or_else(|| PrismError::UnsupportedConstruct {
                    construct: format!("ctrl @ chain past {} controls", u8::MAX),
                    line: line_num,
                })
        };
        let controlled = match &gate {
            g if g.num_qubits() == 1 => {
                let mat = gate.matrix_2x2();
                vec![Self::ig(
                    Self::controlled_unitary(mat, widen(0)?),
                    &prefixed(&targets),
                )]
            }
            Gate::Cx => vec![Self::ig(
                Self::controlled_unitary(Gate::X.matrix_2x2(), widen(1)?),
                &prefixed(&targets),
            )],
            Gate::Cz => vec![Self::ig(
                Self::controlled_unitary(Gate::Z.matrix_2x2(), widen(1)?),
                &prefixed(&targets),
            )],
            Gate::Cu(mat) => vec![Self::ig(
                Self::controlled_unitary(**mat, widen(1)?),
                &prefixed(&targets),
            )],
            Gate::Mcu(data) => vec![Self::ig(
                Self::controlled_unitary(data.mat, widen(data.num_controls as usize)?),
                &prefixed(&targets),
            )],
            // Fredkin: conjugating by a CNOT turns the swap into one extra
            // control on a Toffoli rather than three controlled CNOTs.
            Gate::Swap => {
                let (a, b) = (targets[0], targets[1]);
                vec![
                    Self::ig(Gate::Cx, &[b, a]),
                    Self::ig(
                        Self::controlled_unitary(Gate::X.matrix_2x2(), widen(1)?),
                        &prefixed(&[a, b]),
                    ),
                    Self::ig(Gate::Cx, &[b, a]),
                ]
            }
            // `Rzz` is `Rz` conjugated by a CNOT pair, and the conjugation
            // needs no control of its own: with the control low the pair
            // cancels.
            Gate::Rzz(theta) => {
                let (a, b) = (targets[0], targets[1]);
                vec![
                    Self::ig(Gate::Cx, &[a, b]),
                    Self::ig(
                        Self::controlled_unitary(Gate::Rz(*theta).matrix_2x2(), widen(0)?),
                        &prefixed(&[b]),
                    ),
                    Self::ig(Gate::Cx, &[a, b]),
                ]
            }
            // Everything else two-qubit is carried as a matrix, and a
            // controlled one has no gate variant wide enough to hold it.
            // `ctrl(V D V*) = V ctrl(D) V*` reaches it through the variants
            // that do exist, at a cost independent of the control count.
            other if other.num_qubits() == 2 => {
                let mat = other.matrix_4x4();
                let flat: Vec<Complex64> = mat.iter().flat_map(|row| row.iter().copied()).collect();
                synthesis::controlled_dense(&flat, controls, &targets)
            }
            // A wider Pauli rotation has no matrix accessor, but its CNOT
            // ladder is an expansion like any other, so the same rule applies
            // one level down.
            Gate::PauliRot(data) => {
                let mut ladder = Vec::new();
                crate::circuit::pauli_rotation_lowering(
                    data.theta(),
                    &targets,
                    data.axes(),
                    |gate, wires| ladder.push(Self::ig(gate, wires)),
                );
                Self::control_expansion(ladder, controls, line_num)?
            }
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "ctrl @ {}, which has no controlled form and no lowering to control",
                        other.name()
                    ),
                    line: line_num,
                });
            }
        };
        Ok(controlled)
    }

    /// A unitary under `num_controls` controls, taking the named two-qubit
    /// variant where one exists.
    fn controlled_unitary(mat: [[num_complex::Complex64; 2]; 2], num_controls: u8) -> Gate {
        if num_controls == 1 {
            Self::resolve_controlled(mat)
        } else {
            Gate::mcu(mat, num_controls)
        }
    }

    /// Wrap `instrs` in `X` on each negated control, which is what turns a
    /// `ctrl` into a `negctrl`: the gate fires on `|0>` instead of `|1>`.
    fn conjugate_negations(
        instrs: Vec<Instruction>,
        controls: &[usize],
        polarity: &[bool],
    ) -> Vec<Instruction> {
        let negated: Vec<usize> = polarity
            .iter()
            .zip(controls)
            .filter(|(negated, _)| **negated)
            .map(|(_, &qubit)| qubit)
            .collect();
        if negated.is_empty() {
            return instrs;
        }
        let flips = || negated.iter().map(|&qubit| Self::ig(Gate::X, &[qubit]));
        flips().chain(instrs).chain(flips()).collect()
    }

    /// Apply modifiers to the instruction sequence a `gate` or `def` body
    /// expanded to, innermost modifier first.
    fn modify_expansion(
        instrs: Vec<Instruction>,
        modifiers: &[Modifier],
        name: &str,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        let mut instrs = instrs;
        for modifier in modifiers.iter().rev() {
            instrs = match modifier {
                Modifier::Inv => Self::invert_expansion(instrs, name, line_num)?,
                Modifier::Pow(k) => Self::repeat_expansion(instrs, *k, name, line_num)?,
                Modifier::Ctrl { .. } => return Err(Self::ctrl_on_expansion_error(name, line_num)),
            };
        }
        Ok(instrs)
    }

    fn invert_expansion(
        instrs: Vec<Instruction>,
        name: &str,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        instrs
            .into_iter()
            .rev()
            .map(|instr| match instr {
                Instruction::Gate { gate, targets } => Ok(Instruction::Gate {
                    gate: gate.inverse(),
                    targets,
                }),
                _ => Err(PrismError::UnsupportedConstruct {
                    construct: format!("inv @ `{name}`, whose body is not a gate sequence"),
                    line: line_num,
                }),
            })
            .collect()
    }

    fn apply_modifier(gate: Gate, modifier: &Modifier) -> Gate {
        match modifier {
            Modifier::Inv => gate.inverse(),
            Modifier::Pow(k) => gate.matrix_power_real(*k),
            Modifier::Ctrl { .. } => unreachable!("controls are applied to the expansion"),
        }
    }

    fn resolve_controlled(mat: [[num_complex::Complex64; 2]; 2]) -> Gate {
        use num_complex::Complex64;
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let eps = 1e-12;

        // CX: mat = X = [[0,1],[1,0]]
        if (mat[0][0] - zero).norm() < eps
            && (mat[0][1] - one).norm() < eps
            && (mat[1][0] - one).norm() < eps
            && (mat[1][1] - zero).norm() < eps
        {
            return Gate::Cx;
        }
        // CZ: mat = Z = [[1,0],[0,-1]]
        if (mat[0][0] - one).norm() < eps
            && (mat[0][1] - zero).norm() < eps
            && (mat[1][0] - zero).norm() < eps
            && (mat[1][1] + one).norm() < eps
        {
            return Gate::Cz;
        }
        Gate::cu(mat)
    }

    fn resolve_gate(name: &str, params: &[f64], dialect: Dialect, line_num: usize) -> Result<Gate> {
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
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Rx(params[0]))
            }
            "ry" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Ry(params[0]))
            }
            "rz" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Rz(params[0]))
            }
            "p" | "phase" | "phaseshift" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::P(params[0]))
            }
            "r" | "prx" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused(Box::new(Self::r_matrix(params[0], params[1]))))
            }
            "sx" | "v" => Ok(Gate::SX),
            "sxdg" | "vi" => Ok(Gate::SXdg),
            "cp" | "cphase" | "cphaseshift" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cphase(params[0]))
            }
            "cx" | "CX" | "cnot" => Ok(Gate::Cx),
            "cy" => Ok(Gate::cu(Gate::Y.matrix_2x2())),
            "cs" => Ok(Gate::cu(Gate::S.matrix_2x2())),
            "csdg" => Ok(Gate::cu(Gate::Sdg.matrix_2x2())),
            "ch" => Ok(Gate::cu(Gate::H.matrix_2x2())),
            "cu" => {
                Self::expect_param_count(name, params, 4, line_num)?;
                Ok(Gate::cu(Self::cu_target_matrix(
                    params[0], params[1], params[2], params[3],
                )))
            }
            "crx" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Rx(params[0]).matrix_2x2()))
            }
            "cry" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Ry(params[0]).matrix_2x2()))
            }
            "crz" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Rz(params[0]).matrix_2x2()))
            }
            "csx" | "cv" => Ok(Gate::cu(Gate::SX.matrix_2x2())),
            "cz" => Ok(Gate::Cz),
            "swap" => Ok(Gate::Swap),
            "ccx" | "toffoli" | "ccnot" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 2)),
            "ccz" => Ok(Gate::mcu(Gate::Z.matrix_2x2(), 2)),
            "c3x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 3)),
            "c4x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 4)),
            "xx_plus_yy" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::xx_plus_yy_matrix(
                    params[0], params[1],
                ))))
            }
            "xx_minus_yy" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::xx_minus_yy_matrix(
                    params[0], params[1],
                ))))
            }
            "gpi" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                let phi = dialect.native_turns(params[0]);
                Ok(Gate::Fused(Box::new(Self::gpi_matrix(phi))))
            }
            "gpi2" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                let phi = dialect.native_turns(params[0]);
                Ok(Gate::Fused(Box::new(Self::gpi2_matrix(phi))))
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
                Ok(Gate::Fused2q(Box::new(Self::ms_matrix(
                    dialect.native_turns(params[0]),
                    dialect.native_turns(params[1]),
                    theta,
                ))))
            }
            "ecr" => Ok(Gate::Fused2q(Box::new(Self::ecr_matrix()))),
            "xy" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::xy_matrix(params[0]))))
            }
            "pswap" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::pswap_matrix(params[0]))))
            }
            "cphaseshift00" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::cphaseshift_matrix(
                    0, params[0],
                ))))
            }
            "cphaseshift01" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::cphaseshift_matrix(
                    1, params[0],
                ))))
            }
            "cphaseshift10" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::cphaseshift_matrix(
                    2, params[0],
                ))))
            }
            "syc" => Ok(Gate::Fused2q(Box::new(Self::syc_matrix()))),
            "sqrt_iswap" => Ok(Gate::Fused2q(Box::new(Self::sqrt_iswap_matrix(1.0)))),
            "sqrt_iswap_inv" => Ok(Gate::Fused2q(Box::new(Self::sqrt_iswap_matrix(-1.0)))),
            _ => Err(PrismError::UnsupportedConstruct {
                construct: name.to_string(),
                line: line_num,
            }),
        }
    }

    /// Decomposition-body shorthand for a gate instruction.
    fn ig(gate: Gate, targets: &[usize]) -> Instruction {
        Instruction::Gate {
            gate,
            targets: SmallVec::from_slice(targets),
        }
    }

    /// Handle gates that decompose into multiple instructions at parse time.
    ///
    /// Returns `Ok(None)` if the gate name is not a decomposed gate (caller
    /// should fall through to `resolve_gate`).
    fn resolve_decomposed_gate(
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
                Ok(Some(vec![Self::ig(
                    Gate::mcu(Gate::X.matrix_2x2(), controls as u8),
                    qubits,
                )]))
            }
            "rccx" => {
                Self::check_arity(name, qubits, 3)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let target = qubits[2];
                Ok(Some(vec![
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                ]))
            }
            "rc3x" | "rcccx" => {
                Self::check_arity(name, qubits, 4)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let c2 = qubits[2];
                let target = qubits[3];
                Ok(Some(vec![
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c2, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c2, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                ]))
            }
            "cswap" | "fredkin" => {
                Self::check_arity(name, qubits, 3)?;
                let ctrl = qubits[0];
                let t1 = qubits[1];
                let t2 = qubits[2];
                Ok(Some(vec![
                    Self::ig(Gate::Cx, &[t2, t1]),
                    Self::ig(Gate::mcu(Gate::X.matrix_2x2(), 2), &[ctrl, t1, t2]),
                    Self::ig(Gate::Cx, &[t2, t1]),
                ]))
            }
            "iswap" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::S, &[q0]),
                    Self::ig(Gate::S, &[q1]),
                    Self::ig(Gate::H, &[q0]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Cx, &[q1, q0]),
                    Self::ig(Gate::H, &[q1]),
                ]))
            }
            "dcx" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Cx, &[q1, q0]),
                ]))
            }
            "u1" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                Ok(Some(vec![Self::ig(Gate::P(params[0]), &[qubits[0]])]))
            }
            "u2" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Self::check_arity(name, qubits, 1)?;
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
                Ok(Some(vec![Self::ig(
                    Gate::Fused(Box::new(mat)),
                    &[qubits[0]],
                )]))
            }
            "u3" | "u" | "U" => {
                Self::expect_param_count(name, params, 3, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                let theta = params[0];
                let phi = params[1];
                let lam = params[2];
                let mat = Self::u_matrix(theta, phi, lam);
                Ok(Some(vec![Self::ig(
                    Gate::Fused(Box::new(mat)),
                    &[qubits[0]],
                )]))
            }
            _ => Ok(None),
        }
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

    pub(crate) fn cu_target_matrix(
        theta: f64,
        phi: f64,
        lam: f64,
        gamma: f64,
    ) -> [[Complex64; 2]; 2] {
        let phase = Complex64::from_polar(1.0, gamma);
        let u = Self::u_matrix(theta, phi, lam);
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

    fn expect_param_count(
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
}

#[path = "openqasm_eval.rs"]
mod eval;

#[cfg(test)]
#[path = "openqasm_tests.rs"]
mod tests;
