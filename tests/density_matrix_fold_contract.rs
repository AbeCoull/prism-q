//! Differential check of the exact noisy density-matrix walk against a dense
//! reference that applies every gate and channel in program order.

use num_complex::Complex64;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{BackendKind, NoiseChannel, NoiseEvent, NoiseModel, PauliTerm, ResolvedBackend, sim};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

type C = Complex64;

const TOL: f64 = 1e-10;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// Buffer offsets of the `2^k` local values over bit positions `ps`, `ps[0]`
/// the most significant bit of the local index.
fn offsets(ps: &[usize]) -> Vec<usize> {
    let k = ps.len();
    (0..1usize << k)
        .map(|t| {
            ps.iter()
                .enumerate()
                .filter(|&(i, _)| (t >> (k - 1 - i)) & 1 == 1)
                .map(|(_, &p)| 1usize << p)
                .sum()
        })
        .collect()
}

/// Apply the row-major square matrix `m` to the local values at `offs` of
/// every group whose `mask` bits are clear.
fn apply_on_offsets(buf: &mut [C], m: &[C], offs: &[usize], mask: usize) {
    let dim = offs.len();
    let mut tmp = vec![C::new(0.0, 0.0); dim];
    for base in 0..buf.len() {
        if base & mask != 0 {
            continue;
        }
        for (slot, &off) in tmp.iter_mut().zip(offs) {
            *slot = buf[base | off];
        }
        for (r, &off) in offs.iter().enumerate() {
            buf[base | off] = (0..dim).map(|col| m[r * dim + col] * tmp[col]).sum();
        }
    }
}

/// `rho` is the `4^n` buffer with index `(r << n) | c` holding `<r|rho|c>`.
struct Reference {
    n: usize,
    rho: Vec<C>,
}

impl Reference {
    fn new(n: usize) -> Self {
        let mut rho = vec![C::new(0.0, 0.0); 1 << (2 * n)];
        rho[0] = C::new(1.0, 0.0);
        Self { n, rho }
    }

    fn row_bits(&self, qs: &[usize]) -> Vec<usize> {
        qs.iter().map(|q| q + self.n).collect()
    }

    /// `rho -> U rho U^dagger`: `U` on the row bits, `conj(U)` on the column bits.
    fn unitary(&mut self, op: &[C], qs: &[usize]) {
        let rows = self.row_bits(qs);
        let mask = |ps: &[usize]| ps.iter().map(|&p| 1usize << p).sum::<usize>();
        apply_on_offsets(&mut self.rho, op, &offsets(&rows), mask(&rows));
        let conj: Vec<C> = op.iter().map(|z| z.conj()).collect();
        apply_on_offsets(&mut self.rho, &conj, &offsets(qs), mask(qs));
    }

    /// `rho -> sum_k K rho K^dagger`, as the matrix acting on each `(row, col)`
    /// block of `qs`: `L[(r, c)][(r', c')] = sum_k K[r][r'] conj(K[c][c'])`.
    fn kraus(&mut self, ops: &[Vec<C>], qs: &[usize]) {
        let d = 1usize << qs.len();
        let mut liouville = vec![C::new(0.0, 0.0); d * d * d * d];
        for k in ops {
            for (r, rp, col, cp) in
                (0..d * d * d * d).map(|i| (i / (d * d * d), i / (d * d) % d, i / d % d, i % d))
            {
                liouville[(r * d + col) * d * d + rp * d + cp] +=
                    k[r * d + rp] * k[col * d + cp].conj();
            }
        }
        let rows = offsets(&self.row_bits(qs));
        let cols = offsets(qs);
        let block: Vec<usize> = rows
            .iter()
            .flat_map(|&r| cols.iter().map(move |&col| r | col))
            .collect();
        let mask = rows[d - 1] | cols[d - 1];
        apply_on_offsets(&mut self.rho, &liouville, &block, mask);
    }

    /// Populations relax as `e1` toward `excited`, coherences scale by `e2`.
    fn relax(&mut self, q: usize, e1: f64, e2: f64, excited: f64) {
        let (col, row) = (1usize << q, 1usize << (q + self.n));
        for base in 0..self.rho.len() {
            if base & (col | row) != 0 {
                continue;
            }
            let trace = self.rho[base] + self.rho[base | row | col];
            let p1 = trace * (excited * (1.0 - e1)) + self.rho[base | row | col] * e1;
            self.rho[base | row | col] = p1;
            self.rho[base] = trace - p1;
            self.rho[base | col] *= e2;
            self.rho[base | row] *= e2;
        }
    }

    fn channel(&mut self, channel: &RefChannel, qs: &[usize]) {
        match channel {
            RefChannel::Kraus(ops) => self.kraus(ops, qs),
            RefChannel::Relax { e1, e2, excited } => self.relax(qs[0], *e1, *e2, *excited),
        }
    }

    fn probabilities(&self) -> Vec<f64> {
        let d = 1usize << self.n;
        (0..d).map(|i| self.rho[(i << self.n) | i].re).collect()
    }
}

enum RefChannel {
    Kraus(Vec<Vec<C>>),
    Relax { e1: f64, e2: f64, excited: f64 },
}

fn identity(dim: usize) -> Vec<C> {
    let mut m = vec![C::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        m[i * dim + i] = C::new(1.0, 0.0);
    }
    m
}

fn kron(a: &[C], b: &[C]) -> Vec<C> {
    let da = (a.len() as f64).sqrt() as usize;
    let db = (b.len() as f64).sqrt() as usize;
    let d = da * db;
    let mut out = vec![C::new(0.0, 0.0); d * d];
    for r in 0..d {
        for col in 0..d {
            out[r * d + col] = a[(r / db) * da + col / db] * b[(r % db) * db + col % db];
        }
    }
    out
}

fn scale(m: &[C], s: f64) -> Vec<C> {
    m.iter().map(|z| z * s).collect()
}

fn pauli(axis: usize) -> Vec<C> {
    let (o, z, i) = (c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0));
    match axis {
        0 => vec![o, z, z, o],
        1 => vec![z, o, o, z],
        2 => vec![z, -i, i, z],
        _ => vec![o, z, z, -o],
    }
}

/// Columns of a random `rows x cols` complex matrix, orthonormalized: an
/// isometry whose `rows / cols` square blocks form a trace-preserving Kraus set.
fn random_isometry(rng: &mut ChaCha8Rng, rows: usize, cols: usize) -> Vec<Vec<C>> {
    let mut columns: Vec<Vec<C>> = Vec::with_capacity(cols);
    while columns.len() < cols {
        let mut v: Vec<C> = (0..rows)
            .map(|_| c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        for u in &columns {
            let dot: C = u.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
            for (x, y) in v.iter_mut().zip(u) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-6 {
            columns.push(v.iter().map(|z| z / norm).collect());
        }
    }
    columns
}

/// Square blocks of a random isometry, each row-major `dim x dim`.
fn random_kraus(rng: &mut ChaCha8Rng, dim: usize, count: usize) -> Vec<Vec<C>> {
    let columns = random_isometry(rng, dim * count, dim);
    (0..count)
        .map(|k| {
            let mut m = vec![C::new(0.0, 0.0); dim * dim];
            for r in 0..dim {
                for (col, column) in columns.iter().enumerate() {
                    m[r * dim + col] = column[k * dim + r];
                }
            }
            m
        })
        .collect()
}

fn random_unitary(rng: &mut ChaCha8Rng, dim: usize) -> Vec<C> {
    random_kraus(rng, dim, 1).swap_remove(0)
}

fn to_2x2(m: &[C]) -> [[C; 2]; 2] {
    [[m[0], m[1]], [m[2], m[3]]]
}

fn to_4x4(m: &[C]) -> [[C; 4]; 4] {
    std::array::from_fn(|r| std::array::from_fn(|col| m[r * 4 + col]))
}

fn random_1q_gate(rng: &mut ChaCha8Rng, diagonal_only: bool) -> (Gate, Vec<C>) {
    let (o, z, i) = (c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0));
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let theta: f64 = rng.random_range(-3.0..3.0);
    let (ch, sh) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let phase = C::from_polar(1.0, theta);
    let t = C::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let pick = if diagonal_only {
        [3, 5, 6, 7, 8, 13, 14][rng.random_range(0..7)]
    } else {
        rng.random_range(0..16)
    };
    match pick {
        0 => (Gate::H, vec![c(h, 0.0), c(h, 0.0), c(h, 0.0), c(-h, 0.0)]),
        1 => (Gate::X, pauli(1)),
        2 => (Gate::Y, pauli(2)),
        3 => (Gate::Z, pauli(3)),
        4 => (
            Gate::SX,
            vec![c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)],
        ),
        5 => (Gate::S, vec![o, z, z, i]),
        6 => (Gate::Sdg, vec![o, z, z, -i]),
        7 => (Gate::T, vec![o, z, z, t]),
        8 => (Gate::Tdg, vec![o, z, z, t.conj()]),
        9 => (
            Gate::SXdg,
            vec![c(0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, -0.5)],
        ),
        10 => (
            Gate::Rx(theta),
            vec![c(ch, 0.0), c(0.0, -sh), c(0.0, -sh), c(ch, 0.0)],
        ),
        11 => (
            Gate::Ry(theta),
            vec![c(ch, 0.0), c(-sh, 0.0), c(sh, 0.0), c(ch, 0.0)],
        ),
        12 => {
            let u = random_unitary(rng, 2);
            (Gate::Fused(Box::new(to_2x2(&u))), u)
        }
        13 => (
            Gate::Rz(theta),
            vec![
                C::from_polar(1.0, -theta / 2.0),
                z,
                z,
                C::from_polar(1.0, theta / 2.0),
            ],
        ),
        14 => (Gate::P(theta), vec![o, z, z, phase]),
        _ => (Gate::Id, identity(2)),
    }
}

/// One-qubit channel with its reference. `diagonal_only` keeps to channels
/// whose superoperator is diagonal.
fn random_1q_channel(rng: &mut ChaCha8Rng, diagonal_only: bool) -> (NoiseChannel, RefChannel) {
    let pick = if diagonal_only {
        [1, 3][rng.random_range(0..2)]
    } else {
        rng.random_range(0..6)
    };
    let p: f64 = rng.random_range(0.01..0.2);
    match pick {
        0 => {
            let kraus = vec![
                scale(&identity(2), (1.0 - p).sqrt()),
                scale(&pauli(1), (p / 3.0).sqrt()),
                scale(&pauli(2), (p / 3.0).sqrt()),
                scale(&pauli(3), (p / 3.0).sqrt()),
            ];
            (NoiseChannel::Depolarizing { p }, RefChannel::Kraus(kraus))
        }
        1 => {
            let (px, py, pz) = if diagonal_only {
                (0.0, 0.0, p)
            } else {
                (p, p / 2.0, p / 3.0)
            };
            let kraus = vec![
                scale(&identity(2), (1.0 - px - py - pz).sqrt()),
                scale(&pauli(1), px.sqrt()),
                scale(&pauli(2), py.sqrt()),
                scale(&pauli(3), pz.sqrt()),
            ];
            (NoiseChannel::Pauli { px, py, pz }, RefChannel::Kraus(kraus))
        }
        2 => {
            let (o, z) = (c(1.0, 0.0), c(0.0, 0.0));
            let kraus = vec![
                vec![o, z, z, c((1.0 - p).sqrt(), 0.0)],
                vec![z, c(p.sqrt(), 0.0), z, z],
            ];
            (
                NoiseChannel::AmplitudeDamping { gamma: p },
                RefChannel::Kraus(kraus),
            )
        }
        3 => (
            NoiseChannel::PhaseDamping { gamma: p },
            RefChannel::Relax {
                e1: 1.0,
                e2: (1.0 - p).sqrt(),
                excited: 0.0,
            },
        ),
        4 => {
            let (t1, gate_time) = (50.0, rng.random_range(0.5..5.0));
            let t2 = rng.random_range(20.0..100.0);
            let excited = [0.0, 0.1, 0.5][rng.random_range(0..3)];
            (
                NoiseChannel::ThermalRelaxation {
                    t1,
                    t2,
                    gate_time,
                    excited_population: excited,
                },
                RefChannel::Relax {
                    e1: (-gate_time / t1).exp(),
                    e2: (-gate_time / t2).exp(),
                    excited,
                },
            )
        }
        _ => {
            let count = rng.random_range(2..4);
            let kraus = random_kraus(rng, 2, count);
            (
                NoiseChannel::Custom {
                    kraus: kraus.iter().map(|k| to_2x2(k)).collect(),
                },
                RefChannel::Kraus(kraus),
            )
        }
    }
}

fn random_2q_channel(rng: &mut ChaCha8Rng) -> (NoiseChannel, RefChannel) {
    if rng.random_bool(0.5) {
        let p: f64 = rng.random_range(0.01..0.2);
        let mut kraus = vec![scale(&identity(4), (1.0 - p).sqrt())];
        for a in 0..4 {
            for b in 0..4 {
                if a + b > 0 {
                    kraus.push(scale(&kron(&pauli(a), &pauli(b)), (p / 15.0).sqrt()));
                }
            }
        }
        (
            NoiseChannel::TwoQubitDepolarizing { p },
            RefChannel::Kraus(kraus),
        )
    } else {
        let kraus = random_kraus(rng, 4, 2);
        (
            NoiseChannel::Kraus2q {
                kraus: kraus.iter().map(|k| to_4x4(k)).collect(),
            },
            RefChannel::Kraus(kraus),
        )
    }
}

#[derive(Clone, Copy, Debug)]
enum GateMix {
    OneQubit,
    CxChain,
    Diagonal,
    Swap,
    Dense,
    Flush,
    Everything,
}

#[derive(Clone, Copy, Debug)]
enum NoiseMix {
    Mixed,
    OneQubitGatesOnly,
    QubitZeroOnly,
    EveryTarget,
    OffTarget,
    PairAndIdle,
    Silent,
}

enum RefOp {
    Unitary(Vec<C>, Vec<usize>),
    Reset(usize),
    Barrier,
}

type PlacedChannels = Vec<(RefChannel, Vec<usize>)>;

struct Case {
    circuit: Circuit,
    noise: NoiseModel,
    ops: Vec<(RefOp, PlacedChannels)>,
    has_reset: bool,
}

fn distinct_pair(rng: &mut ChaCha8Rng, n: usize) -> (usize, usize) {
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n - 1);
    if b >= a {
        b += 1;
    }
    (a, b)
}

fn random_2q_gate(
    rng: &mut ChaCha8Rng,
    circuit: &mut Circuit,
    mix: GateMix,
    n: usize,
    step: usize,
) -> (Vec<C>, Vec<usize>) {
    let (o, z) = (c(1.0, 0.0), c(0.0, 0.0));
    let (a, b) = match mix {
        GateMix::CxChain => {
            let q = step % (n - 1);
            if rng.random_bool(0.5) {
                (q, q + 1)
            } else {
                (q + 1, q)
            }
        }
        _ => distinct_pair(rng, n),
    };
    let theta: f64 = rng.random_range(-3.0..3.0);
    let pick = match mix {
        GateMix::CxChain => 0,
        GateMix::Diagonal => [1, 3, 4][rng.random_range(0..3)],
        GateMix::Swap => 2,
        GateMix::Dense => rng.random_range(4..8),
        _ => rng.random_range(0..8),
    };
    let mat = match pick {
        0 => {
            circuit.add_gate(Gate::Cx, &[a, b]);
            vec![o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z]
        }
        1 => {
            circuit.add_gate(Gate::Cz, &[a, b]);
            vec![o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, -o]
        }
        2 => {
            circuit.add_gate(Gate::Swap, &[a, b]);
            vec![o, z, z, z, z, z, o, z, z, o, z, z, z, z, z, o]
        }
        3 => {
            circuit.add_gate(Gate::Rzz(theta), &[a, b]);
            let (s, d) = (
                C::from_polar(1.0, -theta / 2.0),
                C::from_polar(1.0, theta / 2.0),
            );
            vec![s, z, z, z, z, d, z, z, z, z, d, z, z, z, z, s]
        }
        4 => {
            let u = if matches!(mix, GateMix::Diagonal) {
                vec![o, z, z, C::from_polar(1.0, theta)]
            } else {
                random_unitary(rng, 2)
            };
            circuit.add_gate(Gate::cu(to_2x2(&u)), &[a, b]);
            vec![o, z, z, z, z, o, z, z, z, z, u[0], u[1], z, z, u[2], u[3]]
        }
        5 => {
            let u = random_unitary(rng, 4);
            circuit.add_gate(Gate::Fused2q(Box::new(to_4x4(&u))), &[a, b]);
            u
        }
        6 => {
            let u = random_unitary(rng, 4);
            circuit.add_gate(Gate::unitary(u.clone(), 2).unwrap(), &[a, b]);
            u
        }
        _ => {
            let (pa, pb) = (rng.random_range(1..4), rng.random_range(1..3));
            let term = |axis: usize, q: usize| match axis {
                1 => PauliTerm::x(q),
                2 => PauliTerm::y(q),
                _ => PauliTerm::z(q),
            };
            circuit.add_pauli_rotation(theta, &[term(pa, a), term(pb, b)]);
            let p = kron(&pauli(pa), &pauli(pb));
            let (ch, sh) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            let mut m: Vec<C> = p.iter().map(|e| e * c(0.0, -sh)).collect();
            for d in 0..4 {
                m[d * 4 + d] += ch;
            }
            m
        }
    };
    (mat, vec![a, b])
}

fn three_distinct(rng: &mut ChaCha8Rng, n: usize) -> Vec<usize> {
    let mut qs: Vec<usize> = Vec::with_capacity(3);
    while qs.len() < 3 {
        let q = rng.random_range(0..n);
        if !qs.contains(&q) {
            qs.push(q);
        }
    }
    qs
}

type Events = Vec<(NoiseChannel, RefChannel, Vec<usize>)>;

fn events_for(
    rng: &mut ChaCha8Rng,
    mix: NoiseMix,
    n: usize,
    targets: &[usize],
    one_qubit_gate: bool,
    diagonal: bool,
) -> Events {
    let mut out: Events = Vec::new();
    let one = |rng: &mut ChaCha8Rng, out: &mut Events, q: usize| {
        let (channel, reference) = random_1q_channel(rng, diagonal);
        out.push((channel, reference, vec![q]));
    };
    let off_target = |rng: &mut ChaCha8Rng| {
        let free: Vec<usize> = (0..n).filter(|q| !targets.contains(q)).collect();
        (!free.is_empty()).then(|| free[rng.random_range(0..free.len())])
    };
    match mix {
        NoiseMix::Silent => {}
        NoiseMix::OneQubitGatesOnly => {
            if one_qubit_gate {
                one(rng, &mut out, targets[0]);
            }
        }
        NoiseMix::QubitZeroOnly => {
            if rng.random_bool(0.7) {
                one(rng, &mut out, 0);
            }
        }
        NoiseMix::EveryTarget => {
            for &q in targets {
                one(rng, &mut out, q);
                if rng.random_bool(0.3) {
                    one(rng, &mut out, q);
                }
            }
        }
        NoiseMix::OffTarget => {
            if let Some(q) = off_target(rng) {
                one(rng, &mut out, q);
            }
        }
        NoiseMix::PairAndIdle => {
            for &q in targets {
                one(rng, &mut out, q);
            }
            if targets.len() == 2 && !diagonal {
                let qs = if rng.random_bool(0.5) {
                    targets.to_vec()
                } else {
                    vec![targets[1], targets[0]]
                };
                let (channel, reference) = random_2q_channel(rng);
                out.push((channel, reference, qs));
                if rng.random_bool(0.3) {
                    let q = targets[rng.random_range(0..2)];
                    one(rng, &mut out, q);
                }
            }
            if !diagonal && rng.random_bool(0.2) {
                if let Some(q) = off_target(rng) {
                    let (channel, reference) = random_2q_channel(rng);
                    out.push((channel, reference, vec![targets[0], q]));
                    one(rng, &mut out, targets[0]);
                }
            }
            for _ in 0..rng.random_range(0..3) {
                if let Some(q) = off_target(rng) {
                    one(rng, &mut out, q);
                }
            }
        }
        NoiseMix::Mixed => match rng.random_range(0..10) {
            0..=2 => {}
            3..=6 => {
                for &q in targets {
                    one(rng, &mut out, q);
                }
                if rng.random_bool(0.3) {
                    one(rng, &mut out, targets[0]);
                }
            }
            7 => {
                let q = targets[rng.random_range(0..targets.len())];
                one(rng, &mut out, q);
            }
            8 => {
                if let Some(q) = off_target(rng) {
                    one(rng, &mut out, q);
                }
            }
            _ => {
                if diagonal || n < 2 {
                    one(rng, &mut out, targets[0]);
                } else {
                    let qs = if targets.len() == 2 {
                        targets.to_vec()
                    } else {
                        let (a, b) = distinct_pair(rng, n);
                        vec![a, b]
                    };
                    let (channel, reference) = random_2q_channel(rng);
                    out.push((channel, reference, qs));
                }
            }
        },
    }
    out
}

fn build_case(n: usize, seed: u64, gates: GateMix, noise: NoiseMix) -> Case {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n, 0);
    let mut ops = Vec::new();
    let mut after_gate = Vec::new();
    let mut has_reset = false;
    let diagonal = matches!(gates, GateMix::Diagonal);
    for step in 0..3 * n + 8 {
        let before = circuit.instructions.len();
        let roll = rng.random_range(0..100);
        let (op, targets, one_qubit_gate) = if diagonal && step < n {
            let theta = 0.3 + 0.2 * step as f64;
            circuit.add_gate(Gate::Ry(theta), &[step]);
            let (ch, sh) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            let m = vec![c(ch, 0.0), c(-sh, 0.0), c(sh, 0.0), c(ch, 0.0)];
            (RefOp::Unitary(m, vec![step]), vec![step], true)
        } else {
            let two_qubit = n >= 2 && !matches!(gates, GateMix::OneQubit) && roll >= 50;
            let wide = n >= 3 && matches!(gates, GateMix::Flush | GateMix::Everything);
            if wide && roll < 8 {
                let qs = three_distinct(&mut rng, n);
                let u = random_unitary(&mut rng, 2);
                circuit.add_gate(Gate::mcu(to_2x2(&u), 2), &qs);
                let mut m = identity(8);
                for r in 0..2 {
                    for col in 0..2 {
                        m[(6 + r) * 8 + 6 + col] = u[r * 2 + col];
                    }
                }
                (RefOp::Unitary(m, qs.clone()), qs, false)
            } else if wide && roll < 12 {
                let qs = three_distinct(&mut rng, n);
                let u = random_unitary(&mut rng, 8);
                circuit.add_gate(Gate::unitary(u.clone(), 3).unwrap(), &qs);
                (RefOp::Unitary(u, qs.clone()), qs, false)
            } else if matches!(gates, GateMix::Flush) && roll < 20 {
                let q = rng.random_range(0..n);
                circuit.add_reset(q);
                has_reset = true;
                (RefOp::Reset(q), vec![q], false)
            } else if wide && roll < 25 {
                let qs: Vec<usize> = (0..n).collect();
                circuit.add_barrier(&qs);
                (RefOp::Barrier, qs, false)
            } else if two_qubit {
                let (m, qs) = random_2q_gate(&mut rng, &mut circuit, gates, n, step);
                (RefOp::Unitary(m, qs.clone()), qs, false)
            } else {
                let q = rng.random_range(0..n);
                let (gate, m) = random_1q_gate(&mut rng, diagonal);
                circuit.add_gate(gate, &[q]);
                (RefOp::Unitary(m, vec![q]), vec![q], true)
            }
        };
        assert_eq!(circuit.instructions.len(), before + 1);
        let events = events_for(&mut rng, noise, n, &targets, one_qubit_gate, diagonal);
        after_gate.push(
            events
                .iter()
                .map(|(channel, _, qs)| NoiseEvent {
                    channel: channel.clone(),
                    qubits: qs.iter().copied().collect(),
                })
                .collect::<Vec<_>>(),
        );
        ops.push((
            op,
            events
                .into_iter()
                .map(|(_, reference, qs)| (reference, qs))
                .collect(),
        ));
    }
    Case {
        circuit,
        noise: NoiseModel {
            after_gate,
            readout: Vec::new(),
        },
        ops,
        has_reset,
    }
}

fn reference_state(case: &Case) -> Reference {
    let mut reference = Reference::new(case.circuit.num_qubits);
    let (o, z) = (c(1.0, 0.0), c(0.0, 0.0));
    for (op, channels) in &case.ops {
        match op {
            RefOp::Unitary(m, qs) => reference.unitary(m, qs),
            RefOp::Reset(q) => reference.kraus(&[vec![o, z, z, z], vec![z, o, z, z]], &[*q]),
            RefOp::Barrier => {}
        }
        for (channel, qs) in channels {
            reference.channel(channel, qs);
        }
    }
    reference
}

fn check_case(n: usize, seed: u64, gates: GateMix, noise: NoiseMix) {
    let case = build_case(n, seed, gates, noise);
    let reference = reference_state(&case);
    let label = format!("{n} qubits, seed {seed}, {gates:?} gates, {noise:?} noise");
    let simulate = || {
        sim::simulate(&case.circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&case.noise)
            .seed(42)
    };
    if case.has_reset {
        let outcome = simulate().run().unwrap();
        assert_eq!(
            outcome.metadata.backend,
            ResolvedBackend::DensityMatrix,
            "{label}"
        );
        let got = outcome.probabilities.unwrap().to_vec();
        for (i, (g, w)) in got.iter().zip(reference.probabilities()).enumerate() {
            assert!(
                (g - w).abs() < TOL,
                "{label}: probability {i} walk {g} vs reference {w}"
            );
        }
    } else {
        let all: Vec<usize> = (0..n).collect();
        let rdm = simulate().reduced_density_matrix(&all).unwrap();
        assert_eq!(
            rdm.metadata.backend,
            ResolvedBackend::DensityMatrix,
            "{label}"
        );
        for (i, (g, w)) in rdm.data.iter().zip(&reference.rho).enumerate() {
            assert!(
                (g - w).norm() < TOL,
                "{label}: entry {i} walk {g} vs reference {w}"
            );
        }
    }
}

const GATE_MIXES: [GateMix; 7] = [
    GateMix::OneQubit,
    GateMix::CxChain,
    GateMix::Diagonal,
    GateMix::Swap,
    GateMix::Dense,
    GateMix::Flush,
    GateMix::Everything,
];

const NOISE_MIXES: [NoiseMix; 7] = [
    NoiseMix::Mixed,
    NoiseMix::OneQubitGatesOnly,
    NoiseMix::QubitZeroOnly,
    NoiseMix::EveryTarget,
    NoiseMix::OffTarget,
    NoiseMix::PairAndIdle,
    NoiseMix::Silent,
];

fn sweep(n: usize, seeds: std::ops::Range<u64>, noise_mixes: &[NoiseMix]) {
    for seed in seeds {
        for gates in GATE_MIXES {
            for &noise in noise_mixes {
                check_case(n, seed, gates, noise);
            }
        }
    }
}

#[test]
fn native_walk_below_the_deferral_floor() {
    for n in 2..=6 {
        sweep(n, 42..44, &NOISE_MIXES);
    }
}

#[test]
fn selective_fold_at_seven_qubits() {
    sweep(7, 42..44, &NOISE_MIXES);
}

#[test]
fn selective_fold_at_eight_qubits() {
    sweep(8, 42..43, &NOISE_MIXES);
}

#[test]
fn fold_any_pending_map_at_nine_qubits() {
    sweep(9, 42..43, &[NoiseMix::Mixed, NoiseMix::EveryTarget]);
}

#[test]
fn fold_pair_and_idle_channels_at_nine_qubits() {
    sweep(9, 42..43, &[NoiseMix::PairAndIdle]);
}
