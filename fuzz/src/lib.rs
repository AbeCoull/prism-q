//! Input decoders and checks shared by the fuzz targets and the replay test in
//! `tests/fuzz_replay.rs`, which includes this file by path.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, expand_qft_blocks, openqasm, qasm_export};
use prism_q::gates::Gate;
use prism_q::{PauliAxis, PauliTerm, run_on, run_on_state};

/// Fused and unfused amplitudes must agree to this, per amplitude.
pub const FUSION_EPS: f64 = 1e-10;

const MIN_QUBITS: usize = 10;
const MAX_QUBITS: usize = 16;
const MAX_OPS: usize = 160;

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

/// Decode a unitary circuit on 10 to 16 qubits from a fixed gate menu.
pub fn circuit_from_bytes(bytes: &mut Bytes<'_>) -> Circuit {
    let n = MIN_QUBITS + bytes.below(MAX_QUBITS - MIN_QUBITS + 1);
    let mut circuit = Circuit::new(n, 0);
    while !bytes.exhausted() && circuit.instructions.len() < MAX_OPS {
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
