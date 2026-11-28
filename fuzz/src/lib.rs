//! Input decoders and checks shared by the fuzz targets and the replay test in
//! `tests/integration/fuzz_replay.rs`, which includes this file by path.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use num_complex::Complex64;
#[cfg(feature = "distributed")]
use prism_q::DistributedContext;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{
    Circuit, SmallVec, expand_pauli_rotations, expand_qft_blocks, openqasm, qasm_export,
};
use prism_q::gates::Gate;
use prism_q::sim::run_batch;
use prism_q::{
    BackendKind, ClassicalCondition, DensityMatrixBackend, Exactness, GateFilter, Instruction,
    LimitKnob, Limits, NoiseBuilder, NoiseChannel, NoiseModel, PauliAxis, PauliTerm, Placement,
    Plan, PlannedExactness, PrismError, Probabilities, Query, ResolvedBackend, RouteRule,
    RunMetadata, RunOutcome, SaveSpec, SavedValue, SpdTruncation, run_on, run_on_state, simulate,
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
        if self.byte().is_multiple_of(4) {
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
    let ((s, c), p) = ((theta / 2.0).sin_cos(), Complex64::from_polar);
    [
        [Complex64::new(c, 0.0), -p(s, lambda)],
        [p(s, phi), p(c, phi + lambda)],
    ]
}

/// Dense `2^k` unitary `H^{⊗k} · diag(phases)`, row-major.
fn hadamard_diag(phases: &[f64]) -> Vec<Complex64> {
    let (dim, scale) = (phases.len(), (phases.len() as f64).sqrt().recip());
    (0..dim)
        .flat_map(|row| {
            phases.iter().enumerate().map(move |(col, &phase)| {
                let sign = if (row & col).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                Complex64::from_polar(sign * scale, phase)
            })
        })
        .collect()
}

fn kron_cx(a: [[Complex64; 2]; 2], b: [[Complex64; 2]; 2]) -> Vec<Complex64> {
    let mut ab = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in ab.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = a[r >> 1][c >> 1] * b[r & 1][c & 1];
        }
    }
    ab.iter()
        .flat_map(|row| [0, 1, 3, 2].map(|c| row[c]))
        .collect()
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

fn one_qubit_named(op: usize) -> Gate {
    match op {
        0 => Gate::Id,
        1 => Gate::X,
        2 => Gate::Y,
        3 => Gate::Z,
        4 => Gate::H,
        5 => Gate::S,
        6 => Gate::Sdg,
        7 => Gate::T,
        8 => Gate::Tdg,
        9 => Gate::SX,
        _ => Gate::SXdg,
    }
}

fn pauli_terms_from_bytes(
    b: &mut Bytes<'_>,
    n: usize,
    weight: usize,
    sorted: bool,
) -> Vec<PauliTerm> {
    let mut qs = b.distinct(n, weight);
    if sorted {
        qs.sort_unstable();
    }
    qs.into_iter()
        .map(|q| PauliTerm::new(q, [PauliAxis::X, PauliAxis::Y, PauliAxis::Z][b.below(3)]))
        .collect()
}

fn add_1q_rot(c: &mut Circuit, b: &mut Bytes<'_>, kind: usize) {
    let theta = b.angle();
    let gate = match kind {
        0 => Gate::Rx(theta),
        1 => Gate::Ry(theta),
        2 => Gate::Rz(theta),
        _ => Gate::P(theta),
    };
    let q = b.below(c.num_qubits);
    c.add_gate(gate, &[q]);
}

fn push_unitary_op(c: &mut Circuit, b: &mut Bytes<'_>, mode: u8) {
    let n = c.num_qubits;
    let op = b.below(match mode {
        0 => 12,
        1 => 15,
        _ => 18,
    });
    match (mode, op) {
        (0 | 1, 0..=8) => {
            let q = b.below(n);
            c.add_gate(one_qubit_named(if op >= 7 { op + 2 } else { op }), &[q]);
        }
        (1, 9..=11) => {
            let q = b.below(n);
            c.add_gate(if op == 9 { Gate::Tdg } else { Gate::T }, &[q]);
        }
        (2, 0..=10) => {
            let q = b.below(n);
            c.add_gate(one_qubit_named(op), &[q]);
        }
        (2, 11) => {
            let k = b.below(4);
            add_1q_rot(c, b, k);
        }
        (0, 9..=11) | (1 | 2, 12..=14) => {
            let gate = [Gate::Cx, Gate::Cz, Gate::Swap][b.below(3)].clone();
            let t = b.distinct(n, 2);
            c.add_gate(gate, &t);
        }
        (2, 15) => {
            let t = b.distinct(n, 2);
            c.add_gate(Gate::Rzz(b.angle()), &t);
        }
        (2, 16) => {
            let t = b.distinct(n, 2);
            c.add_gate(Gate::cphase(b.angle()), &t);
        }
        _ => {
            let weight = (2 + b.below(2)).min(n);
            let theta = b.angle();
            let factors = pauli_terms_from_bytes(b, n, weight, false);
            c.add_pauli_rotation(theta, &factors);
        }
    }
}

fn push_op(c: &mut Circuit, b: &mut Bytes<'_>) {
    let n = c.num_qubits;
    let op = b.below(34);
    match op {
        0..=10 => {
            let q = b.below(n);
            c.add_gate(one_qubit_named(op), &[q]);
        }
        11..=14 => add_1q_rot(c, b, op - 11),
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
            let t = b.distinct(n, 2);
            c.add_gate(Gate::cphase(b.angle()), &t);
        }
        22 => {
            let t = b.distinct(n, 2);
            c.add_gate(Gate::cu(u3(b.angle(), b.angle(), b.angle())), &t);
        }
        23 => {
            let controls = (2 + b.below(2)).min(n - 1);
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
            let k = (3 + b.below(2)).min(n);
            let dim = 1usize << k;
            let phases: Vec<f64> = (0..dim).map(|_| b.angle()).collect();
            let mat = if b.byte().is_multiple_of(2) {
                (0..dim * dim)
                    .map(|idx| {
                        if idx % (dim + 1) == 0 {
                            Complex64::from_polar(1.0, phases[idx / (dim + 1)])
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect()
            } else {
                hadamard_diag(&phases)
            };
            let t = b.distinct(n, k);
            c.add_unitary(mat, &t).expect("menu matrix is unitary");
        }
        26 | 27 => {
            let weight = (2 + b.below(3)).min(n);
            let theta = b.angle();
            let factors = pauli_terms_from_bytes(b, n, weight, false);
            c.add_pauli_rotation(theta, &factors);
        }
        28 => {
            let num = 2 + b.below(n - 1);
            let start = b.below(n - num + 1);
            let gate = Gate::QftBlock {
                start: start as u8,
                num: num as u8,
                inverse: false,
                swaps: true,
                big_endian: false,
            };
            c.add_gate(gate, &(start..start + num).collect::<Vec<_>>());
        }
        29 => {
            let k = 1 + b.below(n);
            c.add_barrier(&b.distinct(n, k));
        }
        _ => {
            let (family, theta, stride, offset) =
                (b.below(4), b.angle(), 1 + b.below(3), b.below(n));
            for i in (0..n).step_by(stride) {
                let (q, r) = ((i + offset) % n, (i + offset + 1) % n);
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

    let (mut fused, mut unfused) = (StatevectorBackend::new(42), StatevectorBackend::new(42));
    match &initial {
        Some(state) => {
            run_on_state(&mut fused, &circuit, state).expect("fused run");
            unfused.init_from_amplitudes(state.clone(), 0)
        }
        None => {
            run_on(&mut fused, &circuit).expect("fused run");
            unfused.init(n, 0)
        }
    }
    .expect("unfused init");
    for instr in &expand_qft_blocks(&circuit).instructions {
        unfused.apply(instr).expect("unfused apply");
    }
    let (fused, unfused) = (
        fused.export_statevector().expect("fused export"),
        unfused.export_statevector().expect("unfused export"),
    );

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

/// Terminal query exercised on a decoded [`SimulateCase`].
#[derive(Debug, Clone)]
pub enum FuzzQuery {
    Run,
    Shots(usize),
    Counts(usize),
    Marginals,
    Expectation(Vec<Vec<PauliTerm>>),
}

/// Noise recipe decoded from the Stage 2 byte tape and compiled per circuit width.
#[derive(Debug, Clone, Copy)]
pub struct FuzzNoiseSpec {
    pub kind: u8,
    pub with_readout: bool,
}

impl FuzzNoiseSpec {
    pub fn build_for(&self, circuit: &Circuit) -> NoiseModel {
        let build = |b: NoiseBuilder| b.build(circuit).expect("noise model");
        let ag = |ch| build(NoiseBuilder::new().after_gates(GateFilter::all(), ch));
        let ag2 = |ch| build(NoiseBuilder::new().after_gates_joint(GateFilter::all().arity(2), ch));
        let qs = |axis, weights| ag(NoiseChannel::QuasiStatic { axis, weights });
        let c = |re: f64| Complex64::new(re, 0.0);
        let mut model = match self.kind % 16 {
            0 => NoiseModel::uniform_depolarizing(circuit, 0.02),
            1 => ag(NoiseChannel::Pauli {
                px: 0.01,
                py: 0.01,
                pz: 0.02,
            }),
            2 => NoiseModel::with_amplitude_damping(circuit, 0.05),
            3 => ag(NoiseChannel::PhaseDamping { gamma: 0.05 }),
            4 => ag(NoiseChannel::ThermalRelaxation {
                t1: 100e-6,
                t2: 80e-6,
                gate_time: 1e-6,
                excited_population: 0.1,
            }),
            5 => ag2(NoiseChannel::TwoQubitDepolarizing { p: 0.03 }),
            6 => {
                let (k0, k1) = (0.95_f64.sqrt(), 0.05_f64.sqrt());
                ag(NoiseChannel::Custom {
                    kraus: vec![
                        [[c(1.0), c(0.0)], [c(0.0), c(k0)]],
                        [[c(0.0), c(k1)], [c(0.0), c(0.0)]],
                    ],
                })
            }
            7 => {
                let diag = |scale: f64, signs: [f64; 4]| {
                    let mut m = [[c(0.0); 4]; 4];
                    for (t, &s) in signs.iter().enumerate() {
                        m[t][t] = c(scale * s);
                    }
                    m
                };
                ag2(NoiseChannel::Kraus2q {
                    kraus: vec![
                        diag(0.97_f64.sqrt(), [1.0, 1.0, 1.0, 1.0]),
                        diag(0.03_f64.sqrt(), [1.0, -1.0, -1.0, 1.0]),
                    ],
                })
            }
            8 => ag(NoiseChannel::Leakage { p: 0.05 }),
            9 => build(
                NoiseBuilder::new()
                    .after_gates(GateFilter::all(), NoiseChannel::Leakage { p: 0.05 })
                    .after_resets(NoiseChannel::Seepage { p: 0.25 }),
            ),
            10 => build(
                NoiseBuilder::new()
                    .after_gates(GateFilter::all(), NoiseChannel::Leakage { p: 0.04 })
                    .after_gates_joint(
                        GateFilter::all().arity(2),
                        NoiseChannel::LeakageTransport { p: 0.1 },
                    ),
            ),
            11 => qs(PauliAxis::Z, Vec::new()),
            12 => qs(PauliAxis::X, vec![(0, 0.05)]),
            13 => qs(PauliAxis::Z, vec![(1 << 30, 0.03)]),
            _ => {
                let mut b = if self.kind % 16 == 14 {
                    NoiseBuilder::new()
                        .after_resets(NoiseChannel::Depolarizing { p: 0.01 })
                        .on_idle_qubits(NoiseChannel::PhaseDamping { gamma: 0.01 })
                } else {
                    NoiseBuilder::new()
                        .after_gates(GateFilter::all(), NoiseChannel::Depolarizing { p: 0.01 })
                        .after_resets(NoiseChannel::AmplitudeDamping { gamma: 0.02 })
                        .on_idle_qubits(NoiseChannel::Depolarizing { p: 0.005 })
                };
                let b_no_meas = b.clone();
                b = b.before_measurements(NoiseChannel::Depolarizing { p: 0.015 });
                b.build(circuit).unwrap_or_else(|_| build(b_no_meas))
            }
        };
        if self.with_readout && circuit.num_classical_bits > 0 {
            model.with_readout_error(0.05, 0.05);
        }
        model
    }
}

#[derive(Debug)]
pub struct SimulateCase {
    pub base: Circuit,
    pub padded: Circuit,
    pub kind: BackendKind,
    #[allow(dead_code)]
    pub statevector_cap: Option<usize>,
    pub query: FuzzQuery,
    pub limits: Limits,
    pub has_request_limits: bool,
    pub base_noise: Option<NoiseModel>,
    pub padded_noise: Option<NoiseModel>,
    pub require_exact: bool,
    pub check_batch: bool,
}

fn decode_backend_kind(kind_idx: usize, param: u8) -> BackendKind {
    let p = param as usize;
    let thresh = |epsilon, max_terms| SpdTruncation::Threshold { epsilon, max_terms };
    let (epsilon, max_terms) = if param.is_multiple_of(2) {
        (0.0, 0)
    } else {
        (1e-3, 16)
    };
    match kind_idx % 16 {
        0 => BackendKind::Auto,
        1 => BackendKind::Statevector,
        2 => BackendKind::Stabilizer,
        3 => BackendKind::Sparse,
        4 => BackendKind::Mps {
            max_bond_dim: [1, 2, 4, 16][p % 4],
        },
        5 => BackendKind::ProductState,
        6 => BackendKind::TensorNetwork,
        7 => BackendKind::TensorNetworkBounded {
            tolerance: [1e-3, 1e-1][p % 2],
        },
        8 => BackendKind::Factored,
        9 => BackendKind::StabilizerRank,
        10 => BackendKind::FactoredStabilizer,
        11 => BackendKind::DensityMatrix,
        12 => BackendKind::StochasticPauli {
            num_samples: [32, 64][p % 2],
        },
        13 => BackendKind::DeterministicPauli {
            truncation: match param % 3 {
                0 => thresh(0.0, 0),
                1 => thresh(1e-3, 64),
                _ => SpdTruncation::Budget { max_terms: 16 },
            },
        },
        14 => BackendKind::PauliPath { epsilon, max_terms },
        _ => {
            #[cfg(feature = "distributed")]
            return BackendKind::StatevectorDistributed {
                context: DistributedContext::serial(),
            };
            #[cfg(not(feature = "distributed"))]
            BackendKind::Auto
        }
    }
}

fn decode_limits(tag: usize, val: u8) -> (Limits, bool) {
    let (v, l) = (val as usize, Limits::new());
    match tag % 16 {
        0..=4 => (l, false),
        5 => (l.max_statevector_qubits(2 + (v % 3)), true),
        6 => (
            l.auto_mps_bond_dim([1, 2, 4, 8][v % 4])
                .max_mps_workspace_qubits(2 + (v % 3)),
            true,
        ),
        7 => (
            l.max_tn_peak_qubits(1 + (v % 4))
                .max_tn_slices([1, 2, 4][v % 3]),
            true,
        ),
        8 => (l.memory_budget_bytes([256, 512, 1024, 4096][v % 4]), true),
        9 => (l.max_density_matrix_qubits(1 + (v % 3)), true),
        10 => (l.max_probability_qubits(1 + (v % 4)), true),
        11 => (l.max_export_qubits(1 + (v % 4)), true),
        12 => (l.max_sparse_qubits(1 + (v % 3)), true),
        13 => (l.max_factored_merge_qubits(2 + (v % 2)), true),
        14 => (l.max_stabilizer_cluster_qubits(2 + (v % 2)), true),
        _ => (
            l.max_tn_peak_qubits(4)
                .max_tn_slices(4)
                .max_probability_qubits(3),
            true,
        ),
    }
}

/// Decode a [`SimulateCase`]. Stage 1 inputs (`(flags & 0xf0) == 0`) preserve exact
/// Stage 1 decoding; Stage 2 inputs (`(flags & 0xf0) != 0`) expand across all
/// backends, queries, noise models, limits, conditional wide gates and batch checks.
pub fn simulate_case_from_bytes(data: &[u8]) -> SimulateCase {
    let mut bytes = Bytes::new(data);
    let flags = bytes.byte();
    let basis_only = flags & 1 == 1;
    let stage2 = (flags & 0xf0) != 0;

    if !stage2 {
        let kind = if flags & 2 == 0 {
            BackendKind::Auto
        } else {
            BackendKind::Statevector
        };
        let capped = flags & 0b1110 == 0b1100;
        let width =
            MIN_SIMULATE_QUBITS + bytes.below(MAX_SIMULATE_QUBITS - MIN_SIMULATE_QUBITS + 1);
        let idle = bytes.below(MAX_IDLE_QUBITS + 1);
        let classical_bits = 1 + bytes.below(4);
        let statevector_cap = capped.then(|| 2 + bytes.below(2));
        let limits = statevector_cap
            .map_or_else(Limits::new, |cap| Limits::new().max_statevector_qubits(cap));
        let mut base = Circuit::new(width, classical_bits);
        while !bytes.exhausted() && base.instructions.len() < MAX_SIMULATE_OPS {
            push_dynamic_op(&mut base, &mut bytes, basis_only, false);
        }
        let padded = pad_circuit(&base, idle);
        return SimulateCase {
            base,
            padded,
            kind,
            statevector_cap,
            query: FuzzQuery::Run,
            limits,
            has_request_limits: statevector_cap.is_some(),
            base_noise: None,
            padded_noise: None,
            require_exact: false,
            check_batch: false,
        };
    }

    let (check_batch, require_exact) = ((flags & 0x20) != 0, (flags & 0x80) != 0);
    let mut has_noise = (flags & 0x40) != 0;
    let profile_tag = (flags >> 1) & 0x07;
    let (kind_idx, kind_param) = (bytes.below(16), bytes.byte());
    let kind = decode_backend_kind(kind_idx, kind_param);
    let raw_term = bytes.byte();
    let term_idx = if profile_tag == 0 {
        raw_term as usize % 5
    } else {
        match (&kind, raw_term % 3) {
            (BackendKind::PauliPath { .. }, 0 | 1) => 4,
            (
                BackendKind::StochasticPauli { .. } | BackendKind::DeterministicPauli { .. },
                0 | 1,
            ) => 3 + (raw_term as usize & 1),
            _ => raw_term as usize % 5,
        }
    };
    let (shots_byte, limit_tag, limit_val) = (bytes.byte(), bytes.below(16), bytes.byte());
    let (limits, has_request_limits) = decode_limits(limit_tag, limit_val);
    let noise_spec = if has_noise {
        let (nk, nf) = (bytes.byte(), bytes.byte());
        let drop_noise = profile_tag >= 4
            && matches!(
                kind,
                BackendKind::StabilizerRank
                    | BackendKind::StochasticPauli { .. }
                    | BackendKind::DeterministicPauli { .. }
            );
        if drop_noise {
            has_noise = false;
            None
        } else {
            let readout_off = profile_tag != 0 && term_idx >= 3 && (nf & 3 != 0);
            let kind_adj = if profile_tag == 0 || nf & 2 != 0 {
                nk
            } else if matches!(kind, BackendKind::PauliPath { .. }) && term_idx == 4 {
                nk % 6
            } else if matches!(kind, BackendKind::DensityMatrix) && !matches!(term_idx, 1 | 2) {
                nk % 8
            } else {
                nk
            };
            Some(FuzzNoiseSpec {
                kind: kind_adj,
                with_readout: !readout_off && (nf & 1 == 1),
            })
        }
    } else {
        None
    };

    let width = if matches!(kind, BackendKind::DensityMatrix) {
        2 + bytes.below(4)
    } else {
        2 + bytes.below(5)
    };
    let raw_idle = bytes.below(4);
    let idle = match &kind {
        BackendKind::DensityMatrix => raw_idle.min(6usize.saturating_sub(width)),
        BackendKind::TensorNetwork | BackendKind::TensorNetworkBounded { .. } => {
            raw_idle.min(7usize.saturating_sub(width))
        }
        _ => raw_idle,
    };
    let raw_cbits = 1 + bytes.below(4);
    let max_ops = match &kind {
        BackendKind::TensorNetwork
        | BackendKind::TensorNetworkBounded { .. }
        | BackendKind::DensityMatrix => 14,
        BackendKind::StabilizerRank
        | BackendKind::PauliPath { .. }
        | BackendKind::DeterministicPauli { .. }
        | BackendKind::StochasticPauli { .. } => 10,
        _ => 18,
    };
    let mode = if profile_tag == 0 || (basis_only && profile_tag <= 3) {
        3
    } else {
        match &kind {
            BackendKind::StabilizerRank if profile_tag <= 5 => 1,
            BackendKind::DeterministicPauli { .. }
            | BackendKind::StochasticPauli { .. }
            | BackendKind::PauliPath { .. }
                if profile_tag <= 6 =>
            {
                2
            }
            BackendKind::Stabilizer | BackendKind::FactoredStabilizer if profile_tag <= 4 => 0,
            BackendKind::DensityMatrix
                if has_noise && !matches!(term_idx, 1 | 2) && profile_tag <= 5 =>
            {
                2
            }
            _ => [3, 3, 2, 2, 1, 1, 0][(profile_tag - 1) as usize],
        }
    };
    let classical_bits = if mode == 1
        && matches!(kind, BackendKind::StabilizerRank)
        && term_idx == 0
        && profile_tag.is_multiple_of(2)
    {
        0
    } else {
        raw_cbits
    };

    let mut base = Circuit::new(width, classical_bits);
    while !bytes.exhausted() && base.instructions.len() < max_ops {
        if mode < 3 {
            push_unitary_op(&mut base, &mut bytes, mode);
        } else {
            push_dynamic_op(&mut base, &mut bytes, basis_only, true);
        }
    }
    if mode == 1 && matches!(kind, BackendKind::StabilizerRank) && !base.has_t_gates() {
        base.add_gate(Gate::T, &[0]);
    }
    if ((mode == 1 && matches!(kind, BackendKind::StabilizerRank))
        || (mode == 0
            && matches!(
                kind,
                BackendKind::Stabilizer | BackendKind::FactoredStabilizer
            )))
        && matches!(term_idx, 1 | 2)
        && bytes.byte().is_multiple_of(2)
    {
        for q in 0..width.min(classical_bits) {
            base.add_measure(q, q);
        }
    }
    let padded = pad_circuit(&base, idle);

    let query = match term_idx {
        0 => FuzzQuery::Run,
        1 => FuzzQuery::Shots(4 + (shots_byte as usize % 5)),
        2 => FuzzQuery::Counts(4 + (shots_byte as usize % 5)),
        3 => FuzzQuery::Marginals,
        _ => {
            let count = 1 + bytes.below(2);
            let obs = (0..count)
                .map(|_| {
                    let w = 1 + bytes.below(width.min(3));
                    pauli_terms_from_bytes(&mut bytes, width, w, true)
                })
                .collect();
            FuzzQuery::Expectation(obs)
        }
    };

    let base_noise = noise_spec.map(|spec| spec.build_for(&base));
    let padded_noise = noise_spec.map(|spec| {
        if padded.num_qubits == base.num_qubits {
            base_noise.clone().expect("base_noise built")
        } else {
            spec.build_for(&padded)
        }
    });

    SimulateCase {
        base,
        padded,
        kind,
        statevector_cap: limits.max_statevector_qubits,
        query,
        limits,
        has_request_limits,
        base_noise,
        padded_noise,
        require_exact,
        check_batch,
    }
}

fn pad_circuit(base: &Circuit, idle: usize) -> Circuit {
    let mut padded = Circuit::new(base.num_qubits + idle, base.num_classical_bits);
    for instruction in &base.instructions {
        match instruction {
            // A save lists the whole register, so it is rebuilt at the padded width.
            Instruction::Save { spec, label, .. } => padded.add_save(*spec, label.clone()),
            other => padded.instructions.push(other.clone()),
        }
    }
    padded
}

fn push_dynamic_op(c: &mut Circuit, b: &mut Bytes<'_>, basis_only: bool, stage2: bool) {
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
            let small = |b: &mut Bytes<'_>| {
                [Gate::X, Gate::Z, Gate::H, Gate::S, Gate::Ry(FRAC_PI_4)][b.below(5)].clone()
            };
            let (gate, targets) = if stage2 && !basis_only {
                match b.below(7) {
                    0..=4 => {
                        let g = small(b);
                        (g, SmallVec::from_slice(&[b.below(n)]))
                    }
                    5 => {
                        let num = 2 + b.below((n - 1).min(2));
                        let start = b.below(n - num + 1);
                        let extra = b.byte();
                        let g = Gate::QftBlock {
                            start: start as u8,
                            num: num as u8,
                            inverse: extra & 1 == 1,
                            swaps: extra & 2 == 0,
                            big_endian: extra & 4 == 4,
                        };
                        (g, (start..start + num).collect())
                    }
                    _ => {
                        let weight = 2 + b.below((n - 1).min(2));
                        let theta = b.angle();
                        let mut factors = pauli_terms_from_bytes(b, n, weight, false);
                        if weight == 2 && factors.iter().all(|f| f.axis == PauliAxis::Z) {
                            factors[0].axis = PauliAxis::X;
                        }
                        let mut scratch = Circuit::new(n, 0);
                        scratch.add_pauli_rotation(theta, &factors);
                        let Some(Instruction::Gate { gate, targets }) = scratch.instructions.pop()
                        else {
                            return;
                        };
                        (gate, targets)
                    }
                }
            } else {
                let g = if basis_only {
                    [Gate::X, Gate::Z][b.below(2)].clone()
                } else {
                    small(b)
                };
                (g, SmallVec::from_slice(&[b.below(n)]))
            };
            c.instructions.push(Instruction::Conditional {
                condition,
                gate,
                targets,
            });
        }
        _ => {
            if b.below(4) == 0 {
                let label = format!("s{}", c.instructions.len());
                let spec = if stage2 {
                    [
                        SaveSpec::Probabilities,
                        SaveSpec::StateVector,
                        SaveSpec::DensityMatrix,
                    ][b.below(3)]
                } else {
                    SaveSpec::Probabilities
                };
                c.add_save(spec, label);
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
    marginals: Vec<(f64, f64)>,
}

fn marginals_from_probs(num_qubits: usize, probs: &[f64]) -> Vec<(f64, f64)> {
    (0..num_qubits)
        .map(|q| {
            let p1 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| i & (1 << q) != 0)
                .map(|(_, &p)| p)
                .sum::<f64>()
                .clamp(0.0, 1.0);
            (1.0 - p1, p1)
        })
        .collect()
}

/// Written apart from the crate's own condition evaluation so the reference shares no
/// code with the routes under test.
fn condition_holds(condition: &ClassicalCondition, bits: &[bool]) -> bool {
    let reg = |offset: usize, size: usize| {
        (0..size).fold(0u64, |acc, i| acc | (u64::from(bits[offset + i]) << i))
    };
    match condition {
        ClassicalCondition::BitIsOne(bit) => bits[*bit],
        ClassicalCondition::BitIsZero(bit) => !bits[*bit],
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => reg(*offset, *size) == *value,
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => reg(*offset, *size) != *value,
        ClassicalCondition::Parity {
            bits: listed,
            expected,
        } => listed.iter().fold(false, |acc, &b| acc ^ bits[b]) == *expected,
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
    let (mut state, mut classical_bits) = (
        vec![false; circuit.num_qubits],
        vec![false; circuit.num_classical_bits],
    );
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
    let marginals = marginals_from_probs(circuit.num_qubits, &probabilities);
    Reference {
        classical_bits,
        probabilities,
        marginals,
    }
}

/// A plain `Backend::apply` loop, or `None` once a measurement or reset meets a qubit
/// that is not in a basis state: its outcome is then a draw, and two routes need not
/// draw alike.
fn apply_reference(
    circuit: &Circuit,
    observables: Option<&[Vec<PauliTerm>]>,
) -> Option<(Reference, Option<Vec<f64>>)> {
    let mut backend = StatevectorBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("reference init");
    let qft_expanded = expand_qft_blocks(circuit);
    for instruction in &expand_pauli_rotations(&qft_expanded).instructions {
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
    let probabilities = backend.probabilities().expect("reference probabilities");
    let marginals = marginals_from_probs(circuit.num_qubits, &probabilities);
    let exp_values = observables.map(|obs| {
        backend
            .pauli_expectations(obs)
            .expect("reference pauli_expectations")
    });
    let r = Reference {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
        marginals,
    };
    Some((r, exp_values))
}

fn ref_kraus_1q(channel: &NoiseChannel) -> Vec<[[Complex64; 2]; 2]> {
    let c = Complex64::new;
    let r = |a, b, d, e| [[c(a, 0.0), c(b, 0.0)], [c(d, 0.0), c(e, 0.0)]];
    let id = r(1.0, 0.0, 0.0, 1.0);
    let scale = |m: [[Complex64; 2]; 2], s: f64| {
        let s = c(s, 0.0);
        [[m[0][0] * s, m[0][1] * s], [m[1][0] * s, m[1][1] * s]]
    };
    let pauli = |px: f64, py: f64, pz: f64| {
        let y = [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]];
        let pi = (1.0 - px - py - pz).max(0.0);
        vec![
            scale(id, pi.sqrt()),
            scale(r(0.0, 1.0, 1.0, 0.0), px.sqrt()),
            scale(y, py.sqrt()),
            scale(r(1.0, 0.0, 0.0, -1.0), pz.sqrt()),
        ]
    };
    let ad = |gamma: f64, excited: f64| {
        let (s, g) = ((1.0 - gamma).max(0.0).sqrt(), gamma.max(0.0).sqrt());
        let (cold, hot) = ((1.0 - excited).max(0.0).sqrt(), excited.max(0.0).sqrt());
        vec![
            r(cold, 0.0, 0.0, cold * s),
            r(0.0, cold * g, 0.0, 0.0),
            r(hot * s, 0.0, 0.0, hot),
            r(0.0, 0.0, hot * g, 0.0),
        ]
    };
    let pd = |gamma: f64| {
        let (s, g) = ((1.0 - gamma).max(0.0).sqrt(), gamma.max(0.0).sqrt());
        vec![r(1.0, 0.0, 0.0, s), r(0.0, 0.0, 0.0, g)]
    };
    match channel {
        NoiseChannel::Pauli { px, py, pz } => pauli(*px, *py, *pz),
        NoiseChannel::Depolarizing { p } => pauli(p / 3.0, p / 3.0, p / 3.0),
        NoiseChannel::AmplitudeDamping { gamma } => ad(*gamma, 0.0),
        NoiseChannel::PhaseDamping { gamma } => pd(*gamma),
        NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        } => {
            let (gad, gpd) = if *gate_time > 0.0 && *t1 > 0.0 && *t2 > 0.0 {
                let gad = 1.0 - (-*gate_time / *t1).exp();
                let gpd = if 0.5 * *t2 >= *t1 {
                    0.0
                } else {
                    (1.0 - (*gate_time / *t1 - 2.0 * (*gate_time / *t2)).exp()).clamp(0.0, 1.0)
                };
                (gad, gpd)
            } else {
                (0.0, 0.0)
            };
            let p_ops = pd(gpd);
            ad(gad, *excited_population)
                .iter()
                .flat_map(|a| {
                    p_ops.iter().map(move |p| {
                        [
                            [
                                a[0][0] * p[0][0] + a[0][1] * p[1][0],
                                a[0][0] * p[0][1] + a[0][1] * p[1][1],
                            ],
                            [
                                a[1][0] * p[0][0] + a[1][1] * p[1][0],
                                a[1][0] * p[0][1] + a[1][1] * p[1][1],
                            ],
                        ]
                    })
                })
                .collect()
        }
        NoiseChannel::Custom { kraus } => kraus.clone(),
        NoiseChannel::QuasiStatic { weights, .. } if weights.is_empty() => vec![id],
        _ => unreachable!("non-mixture channel in density-matrix reference"),
    }
}

/// Independent plain `DensityMatrixBackend` apply loop for noisy exact mixture checks.
fn dm_apply_reference(
    circuit: &Circuit,
    noise: &NoiseModel,
    observables: Option<&[Vec<PauliTerm>]>,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut last = vec![usize::MAX; circuit.num_qubits];
    for (idx, inst) in circuit.instructions.iter().enumerate() {
        if let Instruction::Measure { qubit, .. } = *inst {
            last[qubit] = idx;
        }
    }
    let mut dm = DensityMatrixBackend::new(42);
    dm.init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("dm reference init");
    for (idx, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Measure { .. }
            | Instruction::Barrier { .. }
            | Instruction::Save { .. } => {}
            Instruction::Gate {
                gate: Gate::QftBlock { .. } | Gate::PauliRot(_),
                ..
            } => {
                let mut single = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
                single.instructions.push(inst.clone());
                let qft_exp = expand_qft_blocks(&single);
                for sub in &expand_pauli_rotations(&qft_exp).instructions {
                    dm.apply(sub).expect("dm reference expanded apply");
                }
            }
            _ => dm.apply(inst).expect("dm reference apply"),
        }
        for event in noise.after_gate.get(idx).into_iter().flatten() {
            if event
                .qubits
                .iter()
                .all(|&q| last.get(q).is_some_and(|&m| m <= idx))
            {
                continue;
            }
            match &event.channel {
                NoiseChannel::TwoQubitDepolarizing { p } => {
                    dm.apply_2q_depolarizing(event.qubits[0], event.qubits[1], *p);
                }
                NoiseChannel::Kraus2q { kraus } => {
                    dm.apply_2q_kraus(event.qubits[0], event.qubits[1], kraus);
                }
                other => dm.apply_1q_kraus(event.qubits[0], &ref_kraus_1q(other)),
            }
        }
    }
    let probs = dm.probabilities().expect("dm reference probabilities");
    let exps = observables.map(|obs| {
        dm.pauli_expectations(obs)
            .expect("dm reference pauli_expectations")
    });
    (probs, exps)
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

fn has_only_terminal_measures_no_resets(circuit: &Circuit) -> bool {
    let mut seen_measure = false;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Measure { .. } => seen_measure = true,
            Instruction::Reset { .. }
            | Instruction::Conditional { .. }
            | Instruction::Region(_) => return false,
            Instruction::Gate { .. } if seen_measure => return false,
            _ => {}
        }
    }
    true
}

fn is_memoryless_no_readout(circuit: &Circuit, noise: &NoiseModel) -> bool {
    let memoryless = !noise.after_gate.iter().flatten().any(|e| {
        matches!(
            e.channel,
            NoiseChannel::Leakage { .. }
                | NoiseChannel::Seepage { .. }
                | NoiseChannel::LeakageTransport { .. }
        ) || matches!(&e.channel, NoiseChannel::QuasiStatic { weights, .. } if !weights.is_empty())
    });
    let inert = noise
        .readout
        .iter()
        .take(circuit.num_classical_bits)
        .all(|r| r.as_ref().is_none_or(|ro| ro.p01 == 0.0 && ro.p10 == 0.0));
    memoryless && inert
}

struct RunCtx<'a> {
    case: &'a SimulateCase,
    label: &'a str,
    circuit: &'a Circuit,
}

impl RunCtx<'_> {
    fn check_meta(&self, plan: &Plan, meta: &RunMetadata) {
        let (label, case) = (self.label, self.case);
        let ok = (meta.backend == plan.backend
            && meta.placement == plan.placement
            && meta.reason == plan.rule)
            || plan.fallbacks.iter().any(|fb| {
                fb.would_fail.is_none()
                    && meta.backend == fb.backend
                    && meta.placement == fb.placement
                    && meta.reason == fb.rule
            })
            || (meta.reason == RouteRule::DeviceInitializationFallback
                && meta.placement == Placement::Host
                && meta.backend == plan.backend);
        assert!(
            ok,
            "{label}: plan/metadata mismatch\nplan: {plan:#?}\nmeta: {meta:#?}\n{case:#?}"
        );
        let exact_ok = match plan.exactness {
            PlannedExactness::Exact => meta.exactness == Exactness::Exact,
            PlannedExactness::Approximate { .. } => {
                matches!(
                    meta.exactness,
                    Exactness::Exact | Exactness::Approximate { .. }
                )
            }
            PlannedExactness::Estimated { samples } => {
                assert_eq!(meta.shots, Some(samples), "{label}: shot count mismatch");
                meta.exactness
                    == Exactness::Approximate {
                        fidelity_lower_bound: None,
                    }
            }
            _ => true,
        };
        assert!(
            exact_ok,
            "{label}: planned {:?}, got {:?}\n{case:#?}",
            plan.exactness, meta.exactness
        );
    }

    fn check_saves(&self, outcome: &RunOutcome) {
        let expected: Vec<(SaveSpec, &str)> = self
            .circuit
            .instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::Save { spec, label, .. } => Some((*spec, label.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            outcome.saves.len(),
            expected.len(),
            "{}: save count\n{:#?}",
            self.label,
            self.case
        );
        let n = self.circuit.num_qubits;
        for (rec, (spec, want)) in outcome.saves.iter().zip(expected) {
            assert_eq!(rec.label, want, "{}: save label mismatch", self.label);
            let len = match (spec, &rec.value) {
                (SaveSpec::Probabilities, SavedValue::Probabilities(v)) => (v.len(), 1 << n),
                (SaveSpec::StateVector, SavedValue::StateVector(v)) => (v.len(), 1 << n),
                (SaveSpec::DensityMatrix, SavedValue::DensityMatrix(v)) => (v.len(), 1 << (2 * n)),
                other => panic!("{}: saved value variant mismatch: {other:?}", self.label),
            };
            assert_eq!(len.0, len.1, "{}: saved buffer len", self.label);
        }
    }

    fn vec_close(&self, what: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= SIMULATE_EPS,
                "{}: {what}[{i}] {g} vs {w}\n{:#?}",
                self.label,
                self.case
            );
        }
    }

    fn probs_close(&self, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), 1 << self.circuit.num_qubits);
        for (i, &g) in got.iter().enumerate() {
            let w = want.get(i).copied().unwrap_or(0.0);
            assert!(
                (g - w).abs() <= SIMULATE_EPS,
                "{}: prob[{i}] {g} vs {w}\n{:#?}",
                self.label,
                self.case
            );
        }
    }

    fn fidelity(&self, bound: f64, got: Option<&Probabilities>, want: &[f64]) {
        assert!(
            bound.is_finite() && (0.0..=1.0).contains(&bound),
            "{}: bad bound {bound}\n{:#?}",
            self.label,
            self.case
        );
        let Some(got) = got else { return };
        let bhat_sqrt: f64 = got
            .to_vec()
            .iter()
            .enumerate()
            .map(|(i, &p)| (p.max(0.0) * want.get(i).copied().unwrap_or(0.0).max(0.0)).sqrt())
            .sum();
        let bhat = bhat_sqrt * bhat_sqrt;
        assert!(
            bhat + 1e-6 >= bound,
            "{}: {bhat} < {bound}\n{:#?}",
            self.label,
            self.case
        );
    }

    fn plan_parity<T>(&self, name: &str, plan: &prism_q::Result<Plan>, res: &prism_q::Result<T>) {
        let planned_fail = plan.as_ref().map_or(true, |p| p.would_fail.is_some());
        let runtime_limit = matches!(
            res,
            Err(PrismError::ResourceLimit(rl)) if matches!(
                rl.knob,
                Some(LimitKnob::TnPeakQubits | LimitKnob::SparseQubits | LimitKnob::FactoredMergeQubits
                    | LimitKnob::StabilizerClusterQubits | LimitKnob::MpsWorkspaceQubits)
            )
        );
        if !planned_fail
            && runtime_limit
            && !(matches!(self.case.kind, BackendKind::TensorNetwork)
                && matches!(
                    self.case.query,
                    FuzzQuery::Expectation(_) | FuzzQuery::Marginals
                ))
        {
            return;
        }
        assert_eq!(
            planned_fail,
            res.is_err(),
            "{}: plan and {name} disagree\nplan: {:?}\nres: {:?}\n{:#?}",
            self.label,
            plan.as_ref().map(|p| &p.would_fail),
            res.as_ref().map(|_| ()),
            self.case
        );
    }

    fn finish<T>(
        &self,
        name: &str,
        plan: &prism_q::Result<Plan>,
        res: prism_q::Result<T>,
        meta: impl FnOnce(&T) -> &RunMetadata,
    ) -> Option<T> {
        self.plan_parity(name, plan, &res);
        if self.circuit.save_count() > 0 {
            assert!(res.is_err(), "{}: {name} succeeded with saves", self.label);
        }
        let val = res.ok()?;
        if let Ok(plan) = plan {
            self.check_meta(plan, meta(&val));
        }
        Some(val)
    }
}

/// Run both widths of a case through `simulate` and panic on a mismatch.
pub fn check_simulate_case(case: &SimulateCase) {
    let observables = match &case.query {
        FuzzQuery::Expectation(obs) => Some(obs.as_slice()),
        _ => None,
    };
    let ref_bundle = if is_basis_circuit(&case.base) && observables.is_none() {
        Some((basis_reference(&case.base), None))
    } else {
        apply_reference(&case.base, observables)
    };
    let (reference, ref_exp) = match &ref_bundle {
        Some((r, exps)) => (Some(r), exps.as_deref()),
        None => (None, None),
    };
    let run_w = |lbl, c, n| check_simulate_run(case, lbl, c, n, reference, ref_exp);
    let base_run = run_w("base", &case.base, case.base_noise.as_ref());
    let padded_run = (case.padded.num_qubits > case.base.num_qubits)
        .then(|| run_w("padded", &case.padded, case.padded_noise.as_ref()))
        .flatten();
    if case.check_batch
        && !case.has_request_limits
        && !case.require_exact
        && case.base_noise.is_none()
        && matches!(case.query, FuzzQuery::Run)
    {
        check_batch_oracle(case, base_run, padded_run);
    }
}

fn check_simulate_run(
    case: &SimulateCase,
    label: &str,
    circuit: &Circuit,
    noise: Option<&NoiseModel>,
    reference: Option<&Reference>,
    ref_expectations: Option<&[f64]>,
) -> Option<Result<RunOutcome, PrismError>> {
    let ctx = RunCtx {
        case,
        label,
        circuit,
    };
    let mut req = simulate(circuit)
        .backend(case.kind.clone())
        .limits(case.limits);
    if let Some(model) = noise {
        req = req.noise(model);
    }
    if case.require_exact {
        req = req.require_exact();
    }
    let query = match &case.query {
        FuzzQuery::Run => Query::Run,
        FuzzQuery::Shots(s) => Query::Shots(*s),
        FuzzQuery::Counts(s) => Query::Counts(*s),
        FuzzQuery::Marginals => Query::Marginals,
        FuzzQuery::Expectation(obs) => Query::Expectation(obs),
    };
    let plan = req.plan(query);
    let req = req.seed(42);
    match &case.query {
        FuzzQuery::Run => {
            let run = req.run();
            ctx.plan_parity("run", &plan, &run);
            if let Err(PrismError::ResourceLimit(rl)) = &run {
                assert!(
                    rl.required > rl.limit,
                    "{label}: invalid ResourceLimit {rl:?}"
                );
            }
            let Ok(outcome) = &run else { return Some(run) };
            if let Ok(plan) = &plan {
                ctx.check_meta(plan, &outcome.metadata);
            }
            if case
                .limits
                .max_probability_qubits
                .is_some_and(|cap| circuit.num_qubits > cap)
            {
                assert!(
                    !matches!(outcome.probabilities, Some(Probabilities::Dense(_))),
                    "{label}: dense probs past cap\n{case:#?}"
                );
            }
            if noise.is_some() && circuit.save_count() > 0 {
                panic!("{label}: noisy run succeeded despite save points\n{case:#?}");
            }
            ctx.check_saves(outcome);
            if let Some(model) = noise {
                if matches!(case.kind, BackendKind::DensityMatrix)
                    && circuit.num_qubits <= 5
                    && is_memoryless_no_readout(circuit, model)
                    && let Some(probs) = &outcome.probabilities
                {
                    let (dm_probs, _) = dm_apply_reference(circuit, model, None);
                    ctx.vec_close("dm probability", &probs.to_vec(), &dm_probs);
                }
                return Some(run);
            }
            if let Some(reference) = reference {
                match outcome.metadata.exactness {
                    Exactness::Exact => {
                        assert_eq!(
                            outcome.classical_bits, reference.classical_bits,
                            "{label}: bits\n{case:#?}"
                        );
                        if let Some(p) = &outcome.probabilities {
                            ctx.probs_close(&p.to_vec(), &reference.probabilities);
                        }
                    }
                    Exactness::Approximate {
                        fidelity_lower_bound: Some(bound),
                    } => {
                        ctx.fidelity(
                            bound,
                            outcome.probabilities.as_ref(),
                            &reference.probabilities,
                        );
                    }
                    _ => {}
                }
            }
            Some(run)
        }
        FuzzQuery::Shots(n_shots) => {
            let shots = ctx.finish("shots", &plan, req.shots(*n_shots), |s| &s.metadata)?;
            assert_eq!(shots.shots.len(), *n_shots, "{label}: shot row count");
            for row in &shots.shots {
                assert_eq!(row.len(), circuit.num_classical_bits);
            }
            if noise.is_none()
                && shots.metadata.exactness == Exactness::Exact
                && has_only_terminal_measures_no_resets(circuit)
                && let Some(reference) = reference
            {
                for row in &shots.shots {
                    assert_eq!(
                        row, &reference.classical_bits,
                        "{label}: deterministic shot\n{case:#?}"
                    );
                }
            }
            None
        }
        FuzzQuery::Counts(n_shots) => {
            let counts = ctx.finish("sample_counts", &plan, req.sample_counts(*n_shots), |c| {
                &c.metadata
            })?;
            assert_eq!(counts.num_classical_bits, circuit.num_classical_bits);
            assert_eq!(
                counts.counts.values().sum::<u64>(),
                *n_shots as u64,
                "{label}: count sum"
            );
            None
        }
        FuzzQuery::Marginals => {
            let res = ctx.finish("marginals", &plan, req.marginals(), |m| &m.metadata)?;
            assert_eq!(res.marginals.len(), circuit.num_qubits);
            for &(p0, p1) in &res.marginals {
                assert!(p0.is_finite() && p1.is_finite() && (p0 + p1 - 1.0).abs() <= 1e-6);
            }
            let check_m = |got: &[(f64, f64)], want: &[(f64, f64)], tag: &str| {
                for (q, (&(g0, g1), &(w0, w1))) in got.iter().zip(want).enumerate() {
                    assert!(
                        (g0 - w0).abs() <= SIMULATE_EPS && (g1 - w1).abs() <= SIMULATE_EPS,
                        "{label}: {tag} q{q} ({g0}, {g1}) vs ({w0}, {w1})\n{case:#?}"
                    );
                }
            };
            if let Some(model) = noise {
                if matches!(case.kind, BackendKind::DensityMatrix)
                    && circuit.num_qubits <= 5
                    && is_memoryless_no_readout(circuit, model)
                {
                    let (dm_probs, _) = dm_apply_reference(circuit, model, None);
                    check_m(
                        &res.marginals,
                        &marginals_from_probs(circuit.num_qubits, &dm_probs),
                        "noisy marginal",
                    );
                }
                return None;
            }
            if res.metadata.exactness == Exactness::Exact
                && let Some(reference) = reference
            {
                check_m(
                    &res.marginals[..case.base.num_qubits],
                    &reference.marginals,
                    "marginal",
                );
                for (q, &(g0, g1)) in res.marginals.iter().enumerate().skip(case.base.num_qubits) {
                    assert!(
                        (g0 - 1.0).abs() <= SIMULATE_EPS && g1.abs() <= SIMULATE_EPS,
                        "{label}: idle q{q} ({g0}, {g1})\n{case:#?}"
                    );
                }
            }
            None
        }
        FuzzQuery::Expectation(obs) => {
            let res = ctx.finish(
                "expectation_values",
                &plan,
                req.expectation_values_reported(obs),
                |e| &e.metadata,
            )?;
            assert_eq!(res.values.len(), obs.len());
            for &v in &res.values {
                assert!(v.is_finite(), "{label}: non-finite expectation {v}");
            }
            if let Some(model) = noise {
                if circuit.num_qubits <= 5
                    && is_memoryless_no_readout(circuit, model)
                    && res.metadata.exactness == Exactness::Exact
                    && (matches!(case.kind, BackendKind::DensityMatrix)
                        || matches!(case.kind, BackendKind::PauliPath { epsilon, max_terms: 0 } if epsilon == 0.0))
                {
                    let (_, dm_exps) = dm_apply_reference(circuit, model, Some(obs));
                    ctx.vec_close("noisy expectation", &res.values, &dm_exps.expect("dm_exps"));
                }
                return None;
            }
            if let Some(want) = ref_expectations {
                if res.metadata.exactness == Exactness::Exact {
                    assert!(
                        res.std_errors.is_none(),
                        "{label}: exact expectation carried std_errors"
                    );
                    ctx.vec_close("expectation", &res.values, want);
                } else if matches!(res.metadata.backend, ResolvedBackend::StochasticPauli) {
                    let se = res
                        .std_errors
                        .as_ref()
                        .expect("StochasticPauli reports std_errors");
                    for (i, (&g, &w)) in res.values.iter().zip(want).enumerate() {
                        assert!(
                            (g - w).abs() <= 6.0 * se[i] + 0.05,
                            "{label}: stochastic[{i}] {g} vs {w} (se {})\n{case:#?}",
                            se[i]
                        );
                    }
                }
            }
            None
        }
    }
}

fn check_batch_oracle(
    case: &SimulateCase,
    base_solo: Option<Result<RunOutcome, PrismError>>,
    padded_solo: Option<Result<RunOutcome, PrismError>>,
) {
    let base_solo = base_solo.expect("base_solo computed for Run");
    let circuits = if case.padded.num_qubits > case.base.num_qubits {
        vec![case.base.clone(), case.padded.clone()]
    } else {
        vec![case.base.clone(); 2]
    };
    let batch = run_batch(&circuits, case.kind.clone(), 42);
    if let Some(want_err) = base_solo
        .as_ref()
        .err()
        .or_else(|| padded_solo.as_ref().and_then(|r| r.as_ref().err()))
    {
        assert_eq!(
            &batch.expect_err("batch must fail"),
            want_err,
            "batch error differs\n{case:#?}"
        );
        return;
    }
    let batch = batch.expect("batch must succeed when solo succeeds");
    assert_eq!(batch.len(), circuits.len());
    let base_ok = base_solo.expect("checked Ok above");
    let second_ok = padded_solo
        .as_ref()
        .map(|r| r.as_ref().expect("checked Ok"))
        .unwrap_or(&base_ok);
    let ctx = RunCtx {
        case,
        label: "batch",
        circuit: &case.base,
    };
    for (idx, (got, want)) in batch.iter().zip([&base_ok, second_ok]).enumerate() {
        assert_eq!(
            got.classical_bits, want.classical_bits,
            "batch[{idx}] bits\n{case:#?}"
        );
        assert_eq!(got.saves, want.saves, "batch[{idx}] saves\n{case:#?}");
        match (&got.probabilities, &want.probabilities) {
            (None, None) => {}
            (Some(gp), Some(wp)) => ctx.vec_close("prob", &gp.to_vec(), &wp.to_vec()),
            _ => panic!("batch[{idx}] probabilities presence mismatch\n{case:#?}"),
        }
        let (g, w) = (&got.metadata, &want.metadata);
        let same_meta = g.backend == w.backend
            && g.reason == w.reason
            && g.exactness == w.exactness
            && g.placement == w.placement
            && g.engine == w.engine
            && g.shots == w.shots
            && g.bond == w.bond
            && g.blocks == w.blocks;
        assert!(
            same_meta,
            "batch[{idx}] metadata: got={g:?} want={w:?}\n{case:#?}"
        );
    }
}

/// Decode and check one fuzzer input.
pub fn check_simulate(data: &[u8]) {
    check_simulate_case(&simulate_case_from_bytes(data));
}
