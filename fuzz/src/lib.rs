//! Input decoders and checks shared by the fuzz targets and the replay test in
//! `tests/integration/fuzz_replay.rs`, which includes this file by path.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, SmallVec, expand_qft_blocks, openqasm, qasm_export};
use prism_q::gates::Gate;
use prism_q::{
    BackendKind, ClassicalCondition, Instruction, Limits, PauliAxis, PauliTerm, Query, SaveSpec,
    run_on, run_on_state, simulate,
};

/// Fused and unfused amplitudes must agree to this, per amplitude.
pub const FUSION_EPS: f64 = 1e-10;

/// A `simulate` run and its reference must agree to this, per probability. Looser than
/// [`FUSION_EPS`] because a run under a statevector cap lands on the sparse or MPS routes.
pub const SIMULATE_EPS: f64 = 1e-8;

/// A measurement or reset counts as certain when the qubit sits within this of a basis
/// state.
const CERTAIN_EPS: f64 = 1e-9;

const MIN_SIMULATE_QUBITS: usize = 4;
const MAX_SIMULATE_QUBITS: usize = 7;
const MAX_IDLE_QUBITS: usize = 4;
const MAX_SIMULATE_OPS: usize = 40;

const MIN_QUBITS: usize = 10;
const MAX_QUBITS: usize = 18;
const MAX_OPS: usize = 160;

/// Operation budget for a circuit on `n` qubits: [`MAX_OPS`] through 16 qubits, halved
/// per qubit above that, so an input costs about the same at every width.
fn max_ops(n: usize) -> usize {
    MAX_OPS >> n.saturating_sub(16)
}

/// Parse `text` through every public QASM entry point. Errors are expected; only a
/// panic is a failure.
pub fn check_parse(text: &str) {
    if let Ok((circuit, _)) = openqasm::parse_parametric(text) {
        let _ = qasm_export::to_qasm3(&circuit);
    }
    let _ = openqasm::parse_with(text, openqasm::Dialect::Braket);
    let _ = openqasm::parse_braket(text);
}

/// Parse raw fuzzer bytes, invalid UTF-8 replaced.
pub fn check_parse_bytes(data: &[u8]) {
    check_parse(&String::from_utf8_lossy(data));
}

/// Parse a program assembled from [`TOKENS`], one token per input byte.
pub fn check_parse_tokens(data: &[u8]) {
    check_parse(&qasm_from_tokens(data));
}

/// QASM fragments the token target draws from. Whole statements appear beside bare
/// tokens so a short input can still reach the evaluator.
pub const TOKENS: &[&str] = &[
    "OPENQASM 3.0;",
    "OPENQASM 2.0;",
    "include \"stdgates.inc\";",
    "qubit[4] q;",
    "qubit[2] r;",
    "bit[4] c;",
    "qreg q[4];",
    "creg c[4];",
    "qubit",
    "bit",
    "qreg",
    "creg",
    "q",
    "r",
    "c",
    "a",
    "b",
    "q[0]",
    "q[1]",
    "q[3]",
    "c[0]",
    "c[1]",
    "$0",
    "$1",
    "[",
    "]",
    "(",
    ")",
    "{",
    "}",
    ";",
    ",",
    ":",
    "@",
    "=",
    "==",
    "!=",
    "!",
    "^",
    "+",
    "-",
    "*",
    "/",
    "%",
    "**",
    "++",
    "+=",
    "->",
    "<",
    ">",
    "&&",
    "||",
    "0",
    "1",
    "2",
    "3",
    "7",
    "-1",
    "0.5",
    "1e308",
    "1e-308",
    "0x1f",
    "0b101",
    "0o7",
    "1_000",
    "pi",
    "tau",
    "euler",
    "π",
    "true",
    "false",
    "h",
    "x",
    "t",
    "s",
    "cx",
    "ccx",
    "mcx",
    "swap",
    "cswap",
    "rx",
    "ry",
    "rz",
    "rzz",
    "rxyz",
    "cp",
    "u",
    "U",
    "gpi",
    "ms",
    "gphase",
    "inv",
    "ctrl",
    "negctrl",
    "pow",
    "gate",
    "def",
    "for",
    "int",
    "uint",
    "float",
    "float[64]",
    "angle",
    "bool",
    "const",
    "input",
    "output",
    "in",
    "if",
    "else",
    "switch",
    "case",
    "default",
    "while",
    "return",
    "let",
    "measure",
    "reset",
    "barrier",
    "box",
    "delay",
    "duration",
    "sin",
    "arccos",
    "sqrt",
    "mod",
    "popcount",
    "#pragma braket result expectation z(q[0])",
    "#pragma braket noise bit_flip(0.1) q[0]",
    "#pragma braket unitary([[0, 1], [1, 0]]) q[0]",
    "#pragma braket verbatim",
    "#pragma braket",
    "h q[0];",
    "cx q[0], q[1];",
    "c[0] = measure q[0];",
    "measure q -> c;",
    "if (c[0]) x q[1];",
    "for int i in [0:3] { h q[i]; }",
    "gate g(t) a, b { rx(t) a; cx a, b; }",
    "def f(qubit a, float t) { rz(t) a; }",
    "\"",
    "//",
    "/*",
    "*/",
    "\n",
];

/// Join one token per input byte into a program.
pub fn qasm_from_tokens(data: &[u8]) -> String {
    let mut text = String::with_capacity(data.len() * 4);
    for &byte in data {
        text.push_str(TOKENS[byte as usize % TOKENS.len()]);
        text.push(' ');
    }
    text
}

/// Byte cursor that reads zeros once the input runs out, so every input decodes.
pub struct Bytes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bytes<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    fn below(&mut self, n: usize) -> usize {
        self.byte() as usize % n
    }

    /// An angle that lands on a value fusion treats specially one time in four.
    fn angle(&mut self) -> f64 {
        const SPECIAL: [f64; 12] = [
            0.0,
            FRAC_PI_4,
            FRAC_PI_2,
            PI,
            1.5 * PI,
            2.0 * PI,
            -FRAC_PI_2,
            -PI,
            4.0 * PI,
            1e-13,
            -1e-13,
            FRAC_PI_2 + 1e-13,
        ];
        let tag = self.byte();
        if tag.is_multiple_of(4) {
            return SPECIAL[self.below(SPECIAL.len())];
        }
        let raw = u16::from_le_bytes([self.byte(), self.byte()]);
        (f64::from(raw) / 65536.0) * 4.0 * PI - 2.0 * PI
    }

    fn distinct(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut picked = Vec::with_capacity(k);
        let mut q = self.below(n);
        while picked.len() < k {
            while picked.contains(&q) {
                q = (q + 1) % n;
            }
            picked.push(q);
            q = (q + 1 + self.below(n)) % n;
        }
        picked
    }
}

fn u3(theta: f64, phi: f64, lambda: f64) -> [[Complex64; 2]; 2] {
    let (s, c) = (theta / 2.0).sin_cos();
    [
        [Complex64::new(c, 0.0), -Complex64::from_polar(s, lambda)],
        [
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lambda),
        ],
    ]
}

/// Dense `2^k` unitary `H^{⊗k} · diag(phases)`, row-major.
fn hadamard_diag(phases: &[f64]) -> Vec<Complex64> {
    let dim = phases.len();
    let scale = (dim as f64).sqrt().recip();
    let mut mat = Vec::with_capacity(dim * dim);
    for row in 0..dim {
        for (col, &phase) in phases.iter().enumerate() {
            let sign = if (row & col).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            mat.push(Complex64::from_polar(sign * scale, phase));
        }
    }
    mat
}

fn kron_cx(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> Vec<Complex64> {
    let mut ab = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in ab.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = a[r >> 1][c >> 1] * b[r & 1][c & 1];
        }
    }
    // Right-multiply by CX with the first target (the high bit) as control.
    let perm = [0, 1, 3, 2];
    let mut out = Vec::with_capacity(16);
    for row in &ab {
        out.extend(perm.iter().map(|&c| row[c]));
    }
    out
}

/// Decode a unitary circuit on 10 to 18 qubits from a fixed gate menu. The top width
/// reaches the post-phase 1q batching pass, which starts at 18 qubits.
pub fn circuit_from_bytes(bytes: &mut Bytes<'_>) -> Circuit {
    let n = MIN_QUBITS + bytes.below(MAX_QUBITS - MIN_QUBITS + 1);
    let budget = max_ops(n);
    let mut circuit = Circuit::new(n, 0);
    while !bytes.exhausted() && circuit.instructions.len() < budget {
        push_op(&mut circuit, bytes);
    }
    circuit
}

fn push_op(c: &mut Circuit, b: &mut Bytes<'_>) {
    let n = c.num_qubits;
    let op = b.below(34);
    match op {
        0..=10 => {
            let gate = [
                Gate::Id,
                Gate::X,
                Gate::Y,
                Gate::Z,
                Gate::H,
                Gate::S,
                Gate::Sdg,
                Gate::T,
                Gate::Tdg,
                Gate::SX,
                Gate::SXdg,
            ][op]
                .clone();
            let q = b.below(n);
            c.add_gate(gate, &[q]);
        }
        11..=14 => {
            let theta = b.angle();
            let gate = match op {
                11 => Gate::Rx(theta),
                12 => Gate::Ry(theta),
                13 => Gate::Rz(theta),
                _ => Gate::P(theta),
            };
            let q = b.below(n);
            c.add_gate(gate, &[q]);
        }
        15 => {
            let m = u3(b.angle(), b.angle(), b.angle());
            let q = b.below(n);
            c.add_gate(Gate::Fused(Box::new(m)), &[q]);
        }
        16..=20 => {
            let gate = match op {
                16 | 17 => Gate::Cx,
                18 => Gate::Cz,
                19 => Gate::Swap,
                _ => Gate::Rzz(b.angle()),
            };
            let t = b.distinct(n, 2);
            c.add_gate(gate, &t);
        }
        21 => {
            let gate = Gate::cphase(b.angle());
            let t = b.distinct(n, 2);
            c.add_gate(gate, &t);
        }
        22 => {
            let gate = Gate::cu(u3(b.angle(), b.angle(), b.angle()));
            let t = b.distinct(n, 2);
            c.add_gate(gate, &t);
        }
        23 => {
            let controls = 2 + b.below(2);
            let mat = if b.byte().is_multiple_of(2) {
                Gate::X.matrix_2x2()
            } else {
                u3(b.angle(), b.angle(), b.angle())
            };
            let t = b.distinct(n, controls + 1);
            c.add_gate(Gate::mcu(mat, controls as u8), &t);
        }
        24 => {
            let mat = kron_cx(
                u3(b.angle(), b.angle(), b.angle()),
                u3(b.angle(), b.angle(), b.angle()),
            );
            let t = b.distinct(n, 2);
            c.add_unitary(mat, &t).expect("menu matrix is unitary");
        }
        25 => {
            let k = 3 + b.below(2);
            let dim = 1usize << k;
            let phases: Vec<f64> = (0..dim).map(|_| b.angle()).collect();
            let mat = if b.byte().is_multiple_of(2) {
                let mut diag = vec![Complex64::new(0.0, 0.0); dim * dim];
                for (i, &p) in phases.iter().enumerate() {
                    diag[i * (dim + 1)] = Complex64::from_polar(1.0, p);
                }
                diag
            } else {
                hadamard_diag(&phases)
            };
            let t = b.distinct(n, k);
            c.add_unitary(mat, &t).expect("menu matrix is unitary");
        }
        26 | 27 => {
            let weight = 2 + b.below(3);
            let theta = b.angle();
            let factors: Vec<PauliTerm> = b
                .distinct(n, weight)
                .into_iter()
                .map(|q| {
                    let axis = [PauliAxis::X, PauliAxis::Y, PauliAxis::Z][b.below(3)];
                    PauliTerm::new(q, axis)
                })
                .collect();
            c.add_pauli_rotation(theta, &factors);
        }
        28 => {
            let num = 2 + b.below(n - 1);
            let start = b.below(n - num + 1);
            let targets: Vec<usize> = (start..start + num).collect();
            c.add_gate(
                Gate::QftBlock {
                    start: start as u8,
                    num: num as u8,
                    inverse: false,
                    swaps: true,
                    big_endian: false,
                },
                &targets,
            );
        }
        29 => {
            let k = 1 + b.below(n);
            let qubits = b.distinct(n, k);
            c.add_barrier(&qubits);
        }
        _ => {
            // A run of one gate family across many qubits, the shape the batching
            // and tiled passes look for.
            let family = b.below(4);
            let theta = b.angle();
            let stride = 1 + b.below(3);
            let offset = b.below(n);
            for i in (0..n).step_by(stride) {
                let q = (i + offset) % n;
                let r = (q + 1) % n;
                match family {
                    0 => c.add_gate(Gate::Ry(theta), &[q]),
                    1 => c.add_gate(Gate::Rzz(theta), &[q, r]),
                    2 => c.add_gate(Gate::cphase(theta), &[q, r]),
                    _ => c.add_gate(Gate::Cx, &[q, r]),
                }
            }
        }
    }
}

/// Normalized pseudo-random amplitudes seeded from the input, qubit 0 the LSB.
fn random_state(num_qubits: usize, seed: u64) -> Vec<Complex64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut amps: Vec<Complex64> = (0..1usize << num_qubits)
        .map(|_| Complex64::new(next(), next()))
        .collect();
    let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut amps {
        *a /= norm;
    }
    amps
}

/// Run a decoded circuit through the fusing `run_on` route and through a plain
/// `Backend::apply` loop, and panic if any amplitude differs by more than
/// [`FUSION_EPS`]. Half the inputs start from a random state instead of `|0...0>`.
pub fn check_fusion(data: &[u8]) {
    let mut bytes = Bytes::new(data);
    let start = bytes.byte();
    let seed = u64::from(bytes.byte()) | (u64::from(bytes.byte()) << 8);
    let circuit = circuit_from_bytes(&mut bytes);
    let n = circuit.num_qubits;
    let initial = (start % 2 == 1).then(|| random_state(n, seed));

    let mut fused = StatevectorBackend::new(42);
    match &initial {
        Some(state) => run_on_state(&mut fused, &circuit, state).expect("fused run"),
        None => run_on(&mut fused, &circuit).expect("fused run"),
    };
    let fused = fused.export_statevector().expect("fused export");

    let mut unfused = StatevectorBackend::new(42);
    match &initial {
        Some(state) => unfused.init_from_amplitudes(state.clone(), 0),
        None => unfused.init(n, 0),
    }
    .expect("unfused init");
    for instr in &expand_qft_blocks(&circuit).instructions {
        unfused.apply(instr).expect("unfused apply");
    }
    let unfused = unfused.export_statevector().expect("unfused export");

    assert_eq!(fused.len(), unfused.len());
    let (worst, diff) = fused
        .iter()
        .zip(&unfused)
        .map(|(f, u)| (*f - *u).norm())
        .enumerate()
        .fold((0, 0.0), |acc, (i, d)| if d > acc.1 { (i, d) } else { acc });
    assert!(
        diff <= FUSION_EPS,
        "fused and unfused runs differ at amplitude {worst}: fused {} unfused {} (|diff| {diff:e}) \
         on {n} qubits, start {}\n{circuit:#?}",
        fused[worst],
        unfused[worst],
        if initial.is_some() { "random" } else { "zero" },
    );
}

/// One decoded input for [`check_simulate_case`]: a circuit with measurements, resets,
/// conditionals and save points, and the request it runs under.
#[derive(Debug)]
pub struct SimulateCase {
    /// The circuit on 4 to 7 qubits, under the width where a run splits into blocks.
    pub base: Circuit,
    /// `base` with up to four idle qubits above it, which is what carries it across that
    /// width. Equal to `base` when the input asks for none.
    pub padded: Circuit,
    pub kind: BackendKind,
    /// Statevector cap below the circuit width, which sends `Auto` to a route that holds
    /// no dense state.
    pub statevector_cap: Option<usize>,
}

/// Decode a [`SimulateCase`]. Half the inputs keep to gates that map basis states to
/// basis states, so every outcome is certain and a bit-vector model is the reference.
pub fn simulate_case_from_bytes(data: &[u8]) -> SimulateCase {
    let mut bytes = Bytes::new(data);
    let flags = bytes.byte();
    let basis_only = flags & 1 == 1;
    let kind = if flags & 2 == 0 {
        BackendKind::Auto
    } else {
        BackendKind::Statevector
    };
    let capped = flags & 0b1110 == 0b1100;
    let width = MIN_SIMULATE_QUBITS + bytes.below(MAX_SIMULATE_QUBITS - MIN_SIMULATE_QUBITS + 1);
    let idle = bytes.below(MAX_IDLE_QUBITS + 1);
    let classical_bits = 1 + bytes.below(4);
    let statevector_cap = capped.then(|| 2 + bytes.below(2));

    let mut base = Circuit::new(width, classical_bits);
    while !bytes.exhausted() && base.instructions.len() < MAX_SIMULATE_OPS {
        push_dynamic_op(&mut base, &mut bytes, basis_only);
    }
    let mut padded = Circuit::new(width + idle, classical_bits);
    for instruction in &base.instructions {
        match instruction {
            // A save lists the whole register, so it is rebuilt at the padded width.
            Instruction::Save { spec, label, .. } => padded.add_save(*spec, label.clone()),
            other => padded.instructions.push(other.clone()),
        }
    }
    SimulateCase {
        base,
        padded,
        kind,
        statevector_cap,
    }
}

fn push_dynamic_op(c: &mut Circuit, b: &mut Bytes<'_>, basis_only: bool) {
    let n = c.num_qubits;
    let bits = c.num_classical_bits;
    match b.below(10) {
        0..=4 if basis_only => push_basis_op(c, b),
        0..=4 => push_op(c, b),
        5 | 6 => {
            let qubit = b.below(n);
            c.add_measure(qubit, b.below(bits));
        }
        7 => c.add_reset(b.below(n)),
        8 => {
            let condition = condition_from_bytes(b, bits);
            let gate = if basis_only {
                [Gate::X, Gate::Z][b.below(2)].clone()
            } else {
                [Gate::X, Gate::Z, Gate::H, Gate::S, Gate::Ry(FRAC_PI_4)][b.below(5)].clone()
            };
            c.instructions.push(Instruction::Conditional {
                condition,
                gate,
                targets: SmallVec::from_slice(&[b.below(n)]),
            });
        }
        _ => {
            if b.below(4) == 0 {
                let label = format!("s{}", c.instructions.len());
                c.add_save(SaveSpec::Probabilities, label);
            } else {
                let qubit = b.below(n);
                c.add_measure(qubit, b.below(bits));
            }
        }
    }
}

/// A gate that maps each basis state to one basis state, up to a phase.
fn push_basis_op(c: &mut Circuit, b: &mut Bytes<'_>) {
    let n = c.num_qubits;
    match b.below(8) {
        0 | 1 => c.add_gate(Gate::X, &[b.below(n)]),
        2 | 3 => c.add_gate(Gate::Cx, &b.distinct(n, 2)),
        4 => c.add_gate(Gate::Swap, &b.distinct(n, 2)),
        5 => c.add_gate(
            [Gate::Z, Gate::S, Gate::T][b.below(3)].clone(),
            &[b.below(n)],
        ),
        6 => c.add_gate(Gate::Cz, &b.distinct(n, 2)),
        _ => {
            let theta = b.angle();
            c.add_gate(Gate::Rz(theta), &[b.below(n)]);
        }
    }
}

fn condition_from_bytes(b: &mut Bytes<'_>, bits: usize) -> ClassicalCondition {
    match b.below(5) {
        0 => ClassicalCondition::BitIsOne(b.below(bits)),
        1 => ClassicalCondition::BitIsZero(b.below(bits)),
        shape @ (2 | 3) => {
            let offset = b.below(bits);
            let size = 1 + b.below(bits - offset);
            let value = b.below(1 << size) as u64;
            if shape == 2 {
                ClassicalCondition::RegisterEquals {
                    offset,
                    size,
                    value,
                }
            } else {
                ClassicalCondition::RegisterNotEquals {
                    offset,
                    size,
                    value,
                }
            }
        }
        _ => {
            let count = 1 + b.below(bits);
            ClassicalCondition::Parity {
                bits: b.distinct(bits, count).into_boxed_slice(),
                expected: b.byte() % 2 == 1,
            }
        }
    }
}

/// What a run must report when every outcome in the circuit is certain.
struct Reference {
    classical_bits: Vec<bool>,
    probabilities: Vec<f64>,
}

/// Written apart from the crate's own condition evaluation so the reference shares no
/// code with the routes under test.
fn condition_holds(condition: &ClassicalCondition, bits: &[bool]) -> bool {
    let register = |offset: usize, size: usize| {
        (0..size).fold(0u64, |acc, i| acc | (u64::from(bits[offset + i]) << i))
    };
    match condition {
        ClassicalCondition::BitIsOne(bit) => bits[*bit],
        ClassicalCondition::BitIsZero(bit) => !bits[*bit],
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => register(*offset, *size) == *value,
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => register(*offset, *size) != *value,
        ClassicalCondition::Parity {
            bits: listed,
            expected,
        } => listed.iter().fold(false, |acc, &bit| acc ^ bits[bit]) == *expected,
        other => unreachable!("condition outside the menu: {other:?}"),
    }
}

/// Bit-vector model of a circuit built from [`push_basis_op`], measurements, resets and
/// conditionals. Shares nothing with the simulator.
fn basis_reference(circuit: &Circuit) -> Reference {
    fn permute(gate: &Gate, targets: &[usize], state: &mut [bool]) {
        if matches!(gate, Gate::X) {
            state[targets[0]] ^= true;
        } else if matches!(gate, Gate::Cx) {
            state[targets[1]] ^= state[targets[0]];
        } else if matches!(gate, Gate::Swap) {
            state.swap(targets[0], targets[1]);
        }
    }

    let mut state = vec![false; circuit.num_qubits];
    let mut classical_bits = vec![false; circuit.num_classical_bits];
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, targets } => permute(gate, targets, &mut state),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => classical_bits[*classical_bit] = state[*qubit],
            Instruction::Reset { qubit } => state[*qubit] = false,
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition_holds(condition, &classical_bits) {
                    permute(gate, targets, &mut state);
                }
            }
            Instruction::Barrier { .. } | Instruction::Save { .. } => {}
            Instruction::Region(_) => unreachable!("the decoder emits no regions"),
        }
    }
    let index = state
        .iter()
        .enumerate()
        .fold(0usize, |acc, (q, &one)| acc | (usize::from(one) << q));
    let mut probabilities = vec![0.0; 1 << circuit.num_qubits];
    probabilities[index] = 1.0;
    Reference {
        classical_bits,
        probabilities,
    }
}

/// A plain `Backend::apply` loop, or `None` once a measurement or reset meets a qubit
/// that is not in a basis state: its outcome is then a draw, and two routes need not
/// draw alike.
fn apply_reference(circuit: &Circuit) -> Option<Reference> {
    let mut backend = StatevectorBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("reference init");
    for instruction in &expand_qft_blocks(circuit).instructions {
        // The simulation layer records saves between instructions; a backend declines one.
        if matches!(instruction, Instruction::Save { .. }) {
            continue;
        }
        if let Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } = instruction {
            let one = backend.qubit_probability(*qubit).expect("reference probe");
            if one.min(1.0 - one) > CERTAIN_EPS {
                return None;
            }
        }
        backend.apply(instruction).expect("reference apply");
    }
    Some(Reference {
        classical_bits: backend.classical_results().to_vec(),
        probabilities: backend.probabilities().expect("reference probabilities"),
    })
}

fn is_basis_circuit(circuit: &Circuit) -> bool {
    circuit
        .instructions
        .iter()
        .all(|instruction| match instruction {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => matches!(
                gate,
                Gate::X
                    | Gate::Cx
                    | Gate::Swap
                    | Gate::Z
                    | Gate::S
                    | Gate::T
                    | Gate::Cz
                    | Gate::Rz(_)
            ),
            _ => true,
        })
}

/// Run both widths of a case through `simulate` and panic on a mismatch.
///
/// Three checks: no entry panics; `plan` predicts a failure exactly when `run` returns
/// one; and, when every outcome is certain, the classical bits and probabilities equal
/// the reference at both widths, the idle qubits staying in `|0>`.
pub fn check_simulate_case(case: &SimulateCase) {
    let reference = if is_basis_circuit(&case.base) {
        Some(basis_reference(&case.base))
    } else {
        apply_reference(&case.base)
    };
    check_simulate_run(case, "base", &case.base, reference.as_ref());
    if case.padded.num_qubits > case.base.num_qubits {
        check_simulate_run(case, "padded", &case.padded, reference.as_ref());
    }
}

fn check_simulate_run(
    case: &SimulateCase,
    label: &str,
    circuit: &Circuit,
    reference: Option<&Reference>,
) {
    let limits = match case.statevector_cap {
        Some(cap) => Limits::new().max_statevector_qubits(cap),
        None => Limits::new(),
    };
    let request = || simulate(circuit).backend(case.kind.clone()).limits(limits);
    let plan = request().plan(Query::Run);
    let run = request().seed(42).run();

    let planned_failure = plan.as_ref().map_or(true, |plan| plan.would_fail.is_some());
    assert_eq!(
        planned_failure,
        run.is_err(),
        "{label}: plan and run disagree on failure\nplan: {:?}\nrun: {:?}\n{case:#?}",
        plan.as_ref().map(|plan| &plan.would_fail),
        run.as_ref().map(|_| ()),
    );
    let (Ok(outcome), Some(reference)) = (run, reference) else {
        return;
    };
    assert_eq!(
        outcome.classical_bits, reference.classical_bits,
        "{label}: classical bits differ from the reference\n{case:#?}"
    );
    let Some(probabilities) = outcome.probabilities else {
        return;
    };
    let probabilities = probabilities.to_vec();
    assert_eq!(probabilities.len(), 1 << circuit.num_qubits);
    for (index, &got) in probabilities.iter().enumerate() {
        let want = reference.probabilities.get(index).copied().unwrap_or(0.0);
        assert!(
            (got - want).abs() <= SIMULATE_EPS,
            "{label}: probability {index} is {got}, reference {want}\n{case:#?}"
        );
    }
}

/// Decode and check one fuzzer input.
pub fn check_simulate(data: &[u8]) {
    check_simulate_case(&simulate_case_from_bytes(data));
}
