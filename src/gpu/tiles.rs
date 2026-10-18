//! Device lowering for [`FusionProfile::Device`](crate::circuit::fusion::FusionProfile):
//! a window of gates becomes [`TileOp`]s, and [`plan`] schedules them into shared-memory
//! tiles, each applying many ops in one pass over the state.

use std::ops::Range;

use num_complex::Complex64;

use crate::error::Result;
use crate::gates::{Gate, is_antidiagonal_2x2, is_diagonal_2x2};
use crate::gpu::GpuState;
use crate::gpu::kernels::dense::{self as k, Route2q, route_2q};
use crate::gpu::kernels::{check_pair, check_qubit};

/// Narrowest state the device lowering runs on. On a GTX 1080 Ti a three-layer QAOA ran
/// 12% slower tiled than batched at 17 qubits and 7% faster at 18, where the ansatz and
/// random circuits already ran 2x faster.
pub(crate) const MIN_QUBITS: usize = 18;

/// Whether an `n`-qubit device state takes the tiled lowering.
pub(crate) fn enabled(n: usize) -> bool {
    n >= MIN_QUBITS
}

/// One gate of a lowered window, on global qubits.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TileOp<'a> {
    /// A 2x2 matrix that is neither diagonal nor antidiagonal.
    Mat1 {
        q: usize,
        m: [[Complex64; 2]; 2],
    },
    /// `diag(d[0], d[1])`.
    Diag1 {
        q: usize,
        d: [Complex64; 2],
    },
    /// `[[0, d[0]], [d[1], 0]]`.
    Anti1 {
        q: usize,
        d: [Complex64; 2],
    },
    Cx {
        control: usize,
        target: usize,
    },
    Swap {
        a: usize,
        b: usize,
    },
    /// Diagonal on two qubits, `d[2 * bit(a) + bit(b)]`.
    Diag2 {
        a: usize,
        b: usize,
        d: [Complex64; 4],
    },
    /// A 4x4 in the `apply_fused_2q` basis (`hi` the high bit), on its own kernel.
    Dense2q {
        hi: usize,
        lo: usize,
        m: &'a [[Complex64; 4]; 4],
    },
    /// A gate no tile carries, dispatched as the per-gate path would.
    Gate {
        gate: &'a Gate,
        targets: &'a [usize],
    },
}

const ONE: Complex64 = Complex64::new(1.0, 0.0);

impl TileOp<'_> {
    /// Qubits the op reads or writes, and the subset it moves between basis states.
    fn masks(&self) -> (u64, u64) {
        let bit = |q: usize| 1u64 << q;
        match *self {
            TileOp::Mat1 { q, .. } | TileOp::Anti1 { q, .. } => (bit(q), bit(q)),
            TileOp::Diag1 { q, .. } => (bit(q), 0),
            TileOp::Cx { control, target } => (bit(control) | bit(target), bit(target)),
            TileOp::Swap { a, b } => (bit(a) | bit(b), bit(a) | bit(b)),
            TileOp::Diag2 { a, b, .. } => (bit(a) | bit(b), 0),
            TileOp::Dense2q { hi, lo, .. } => (bit(hi) | bit(lo), bit(hi) | bit(lo)),
            TileOp::Gate { targets, .. } => {
                let all = targets.iter().fold(0, |m, &q| m | bit(q));
                (all, all)
            }
        }
    }

    /// The qubit checks the per-gate launchers make, so a tile never indexes past the
    /// state; [`TileOp::Gate`] is left to its own launcher.
    fn check(&self, n: usize) -> Result<()> {
        match *self {
            TileOp::Mat1 { q, .. } | TileOp::Diag1 { q, .. } | TileOp::Anti1 { q, .. } => {
                check_qubit(n, q)
            }
            TileOp::Cx {
                control: a,
                target: b,
            }
            | TileOp::Swap { a, b }
            | TileOp::Diag2 { a, b, .. }
            | TileOp::Dense2q { hi: a, lo: b, .. } => check_pair(n, a, b),
            TileOp::Gate { .. } => Ok(()),
        }
    }

    /// Kernel parameter doubles the op takes in a tile, `None` when no tile carries it.
    fn coefs(&self) -> Option<usize> {
        match self {
            TileOp::Mat1 { .. } | TileOp::Diag2 { .. } => Some(8),
            TileOp::Diag1 { .. } | TileOp::Anti1 { .. } => Some(4),
            TileOp::Cx { .. } | TileOp::Swap { .. } => Some(0),
            TileOp::Dense2q { .. } | TileOp::Gate { .. } => None,
        }
    }
}

fn push_1q<'a>(ops: &mut Vec<TileOp<'a>>, q: usize, m: [[Complex64; 2]; 2]) {
    ops.push(if is_diagonal_2x2(&m) {
        TileOp::Diag1 {
            q,
            d: [m[0][0], m[1][1]],
        }
    } else if is_antidiagonal_2x2(&m) {
        TileOp::Anti1 {
            q,
            d: [m[0][1], m[1][0]],
        }
    } else {
        TileOp::Mat1 { q, m }
    });
}

fn push_4x4<'a>(ops: &mut Vec<TileOp<'a>>, hi: usize, lo: usize, m: &'a [[Complex64; 4]; 4]) {
    ops.push(match route_2q(m) {
        Route2q::Cx { control_high: true } => TileOp::Cx {
            control: hi,
            target: lo,
        },
        Route2q::Cx {
            control_high: false,
        } => TileOp::Cx {
            control: lo,
            target: hi,
        },
        Route2q::Swap => TileOp::Swap { a: hi, b: lo },
        Route2q::Cz => controlled_phase(hi, lo, -ONE),
        Route2q::ControlledPhase(p) => controlled_phase(hi, lo, p),
        Route2q::Monomial {
            src: [0, 1, 2, 3],
            scale,
        } => TileOp::Diag2 {
            a: hi,
            b: lo,
            d: scale,
        },
        Route2q::Monomial { .. } | Route2q::Dense => TileOp::Dense2q { hi, lo, m },
    });
}

fn controlled_phase<'a>(a: usize, b: usize, phase: Complex64) -> TileOp<'a> {
    TileOp::Diag2 {
        a,
        b,
        d: [ONE, ONE, ONE, phase],
    }
}

/// Append the ops `gate` on `targets` lowers to.
pub(crate) fn lower<'a>(ops: &mut Vec<TileOp<'a>>, gate: &'a Gate, targets: &'a [usize]) {
    match gate {
        Gate::Id => {}
        Gate::X
        | Gate::Y
        | Gate::Z
        | Gate::H
        | Gate::S
        | Gate::Sdg
        | Gate::T
        | Gate::Tdg
        | Gate::SX
        | Gate::SXdg
        | Gate::Rx(_)
        | Gate::Ry(_)
        | Gate::Rz(_)
        | Gate::P(_)
        | Gate::Fused(_) => push_1q(ops, targets[0], gate.matrix_2x2()),
        Gate::Cx => ops.push(TileOp::Cx {
            control: targets[0],
            target: targets[1],
        }),
        Gate::Cz => ops.push(controlled_phase(targets[0], targets[1], -ONE)),
        Gate::Swap => ops.push(TileOp::Swap {
            a: targets[0],
            b: targets[1],
        }),
        Gate::Rzz(_) => {
            let m = gate.matrix_4x4();
            ops.push(TileOp::Diag2 {
                a: targets[0],
                b: targets[1],
                d: [m[0][0], m[1][1], m[2][2], m[3][3]],
            });
        }
        Gate::Cu(mat) => match gate.controlled_phase() {
            Some(p) => ops.push(controlled_phase(targets[0], targets[1], p)),
            None if is_diagonal_2x2(mat) => ops.push(TileOp::Diag2 {
                a: targets[0],
                b: targets[1],
                d: [ONE, ONE, mat[0][0], mat[1][1]],
            }),
            None => ops.push(TileOp::Gate { gate, targets }),
        },
        Gate::Fused2q(m) => push_4x4(ops, targets[0], targets[1], m),
        Gate::Multi2q(data) => {
            for (q0, q1, m) in data.gates() {
                push_4x4(ops, *q0, *q1, m);
            }
        }
        Gate::MultiFused(data) => {
            for &(q, m) in &data.gates {
                push_1q(ops, q, m);
            }
        }
        Gate::BatchRzz(data) => {
            for &(a, b, theta) in &data.edges {
                let m = Gate::Rzz(theta).matrix_4x4();
                ops.push(TileOp::Diag2 {
                    a,
                    b,
                    d: [m[0][0], m[1][1], m[2][2], m[3][3]],
                });
            }
        }
        Gate::BatchPhase(data) => {
            for &(target, p) in &data.phases {
                ops.push(controlled_phase(targets[0], target, p));
            }
        }
        Gate::Mcu(_)
        | Gate::DiagonalBatch(_)
        | Gate::QftBlock { .. }
        | Gate::PauliRot(_)
        | Gate::MultiPauliRot(_)
        | Gate::Unitary(_) => ops.push(TileOp::Gate { gate, targets }),
    }
}

/// The tile shape the planner fills: the qubits every tile holds, how many others it
/// may add, and the parameter budget of one launch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub(crate) anchor: u64,
    pub(crate) budget: u32,
    pub(crate) max_ops: usize,
    pub(crate) max_coefs: usize,
}

/// One launch of a planned window. `ops` indexes [`Plan::order`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// One tile pass over `qubits` (anchor included, filler not) applying the ops in order.
    Tile { qubits: u64, ops: Range<usize> },
    /// One op on its own kernel.
    Single(u32),
}

/// A window's schedule: every op index exactly once, in the order the steps apply them.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub(crate) order: Vec<u32>,
    pub(crate) steps: Vec<Step>,
}

/// An op as the planner sees it.
#[derive(Clone, Copy, Debug)]
struct Shape {
    touched: u64,
    moved: u64,
    /// Parameter doubles in a tile, `u32::MAX` for an op no tile carries.
    coefs: u32,
}

/// Fewest ops worth a tile pass; one op runs faster on its own kernel.
const MIN_TILE_OPS: usize = 2;
/// Candidate qubit sets one growth step scores.
const MAX_CANDIDATES: usize = 16;

/// Schedule `ops` on an `n`-qubit state into tiles of `geometry` and single launches.
///
/// Each step grows a tile from the anchor over the ops whose dependencies are met: it
/// repeatedly adds the missing qubits of a frontier op that let the tile take the most
/// ops per qubit added, until no addition helps or the budget is spent. Two ops commute
/// when every qubit they share is moved by neither, so an op may run ahead of an earlier
/// one it commutes with. A tile that takes fewer than [`MIN_TILE_OPS`] ops yields the
/// first remaining op as a single launch instead.
pub(crate) fn plan(ops: &[TileOp<'_>], n: usize, geometry: &Geometry, out: &mut Plan) {
    let shapes: Vec<Shape> = ops
        .iter()
        .map(|op| {
            let (touched, moved) = op.masks();
            Shape {
                touched,
                moved,
                coefs: op.coefs().map_or(u32::MAX, |c| c as u32),
            }
        })
        .collect();
    let mut planner = Planner {
        shapes: &shapes,
        geometry,
        all: if n >= 64 { u64::MAX } else { (1 << n) - 1 },
        skip_limit: 4 * n,
        rem: (0..ops.len() as u32).collect(),
        taken: Vec::new(),
        front: Vec::new(),
    };
    out.order.clear();
    out.steps.clear();
    while !planner.rem.is_empty() {
        let high = planner.grow();
        let mut taken = std::mem::take(&mut planner.taken);
        planner.scan(geometry.anchor | high, &mut taken, false);
        if taken.len() >= MIN_TILE_OPS {
            let start = out.order.len();
            out.order.extend_from_slice(&taken);
            out.steps.push(Step::Tile {
                qubits: geometry.anchor | high,
                ops: start..out.order.len(),
            });
        } else {
            taken.clear();
            taken.push(planner.rem[0]);
            out.order.push(planner.rem[0]);
            out.steps.push(Step::Single(planner.rem[0]));
        }
        let mut next = taken.iter().peekable();
        planner.rem.retain(|k| {
            if next.peek() == Some(&k) {
                next.next();
                false
            } else {
                true
            }
        });
        planner.taken = taken;
    }
}

struct Planner<'s> {
    shapes: &'s [Shape],
    geometry: &'s Geometry,
    all: u64,
    skip_limit: usize,
    rem: Vec<u32>,
    taken: Vec<u32>,
    front: Vec<u64>,
}

impl Planner<'_> {
    /// Fill `taken` with the ops a tile over `allowed` takes from `rem`, in order: each
    /// one that fits, stays within the launch budget, and conflicts with no op skipped
    /// before it. With `front`, also collect the missing qubits of each skipped op
    /// nothing blocked that the budget could still admit.
    fn scan(&mut self, allowed: u64, taken: &mut Vec<u32>, front: bool) {
        taken.clear();
        if front {
            self.front.clear();
        }
        let room = self.geometry.budget - (allowed & !self.geometry.anchor).count_ones();
        let (mut blocked_touched, mut blocked_moved) = (0u64, 0u64);
        let (mut skipped, mut coefs) = (0usize, 0usize);
        for &k in &self.rem {
            let op = self.shapes[k as usize];
            let blocked = (op.touched & blocked_moved) | (op.moved & blocked_touched) != 0;
            if !blocked && op.coefs != u32::MAX {
                let missing = op.touched & !allowed;
                let fits_launch = taken.len() < self.geometry.max_ops
                    && coefs + op.coefs as usize <= self.geometry.max_coefs;
                if missing == 0 && fits_launch {
                    taken.push(k);
                    coefs += op.coefs as usize;
                    continue;
                }
                if front && missing != 0 && missing.count_ones() <= room {
                    self.front.push(missing);
                }
            }
            blocked_touched |= op.touched;
            blocked_moved |= op.moved;
            skipped += 1;
            if skipped > self.skip_limit || blocked_moved == self.all {
                break;
            }
        }
    }

    /// The high qubits of the next tile, grown greedily from none.
    fn grow(&mut self) -> u64 {
        let anchor = self.geometry.anchor;
        let mut high = 0u64;
        let mut taken = std::mem::take(&mut self.taken);
        self.scan(anchor, &mut taken, true);
        let mut base = taken.len();
        loop {
            let mut seen: smallvec::SmallVec<[u64; MAX_CANDIDATES]> = smallvec::SmallVec::new();
            let mut best: Option<(u64, usize)> = None;
            let front = std::mem::take(&mut self.front);
            for &missing in &front {
                if seen.len() == MAX_CANDIDATES {
                    break;
                }
                if seen.contains(&missing) {
                    continue;
                }
                seen.push(missing);
                self.scan(anchor | high | missing, &mut taken, false);
                let count = taken.len();
                if count <= base {
                    continue;
                }
                let better = best.is_none_or(|(b_missing, b_count)| {
                    (count - base) * b_missing.count_ones() as usize
                        > (b_count - base) * missing.count_ones() as usize
                });
                if better {
                    best = Some((missing, count));
                }
            }
            self.front = front;
            let Some((missing, count)) = best else {
                break;
            };
            high |= missing;
            base = count;
            self.scan(anchor | high, &mut taken, true);
        }
        self.taken = taken;
        high
    }
}

/// Lowered gates of one window, planned and launched on [`Window::flush`].
#[derive(Default)]
pub(crate) struct Window<'a> {
    ops: Vec<TileOp<'a>>,
    plan: Plan,
}

impl<'a> Window<'a> {
    /// Lower `gate` for an `n`-qubit state, rejecting a qubit its launcher would reject.
    pub(crate) fn push(&mut self, gate: &'a Gate, targets: &'a [usize], n: usize) -> Result<()> {
        let start = self.ops.len();
        lower(&mut self.ops, gate, targets);
        self.ops[start..].iter().try_for_each(|op| op.check(n))
    }

    /// Plan and launch every pushed op, sending each [`TileOp::Gate`] to `dispatch`.
    pub(crate) fn flush(
        &mut self,
        state: &mut GpuState,
        mut dispatch: impl FnMut(&mut GpuState, &Gate, &[usize]) -> Result<()>,
    ) -> Result<()> {
        if self.ops.is_empty() {
            return Ok(());
        }
        let ctx = state.context().clone();
        plan(
            &self.ops,
            state.num_qubits(),
            &k::MIXED_TILE_GEOMETRY,
            &mut self.plan,
        );
        for step in &self.plan.steps {
            match step {
                Step::Tile { qubits, ops } => {
                    let tile_ops = self.plan.order[ops.clone()]
                        .iter()
                        .map(|&i| &self.ops[i as usize]);
                    k::launch_apply_mixed_tile(&ctx, state, *qubits, tile_ops)?;
                }
                Step::Single(i) => match self.ops[*i as usize] {
                    TileOp::Gate { gate, targets } => dispatch(state, gate, targets)?,
                    op => launch_single(&ctx, state, &op)?,
                },
            }
        }
        self.ops.clear();
        Ok(())
    }
}

fn launch_single(
    ctx: &crate::gpu::GpuContext,
    state: &mut GpuState,
    op: &TileOp<'_>,
) -> Result<()> {
    let zero = Complex64::new(0.0, 0.0);
    match *op {
        TileOp::Mat1 { q, m } => k::launch_apply_gate_1q(ctx, state, q, m),
        TileOp::Diag1 { q, d } => k::launch_apply_diagonal_1q(ctx, state, q, d[0], d[1]),
        TileOp::Anti1 { q, d } => {
            k::launch_apply_gate_1q(ctx, state, q, [[zero, d[0]], [d[1], zero]])
        }
        TileOp::Cx { control, target } => k::launch_apply_cx(ctx, state, control, target),
        TileOp::Swap { a, b } => k::launch_apply_swap(ctx, state, a, b),
        TileOp::Diag2 { a, b, d } => {
            let mut m = [[zero; 4]; 4];
            for (r, &v) in d.iter().enumerate() {
                m[r][r] = v;
            }
            k::launch_apply_matrix_2q(ctx, state, a, b, &m)
        }
        TileOp::Dense2q { hi, lo, m } => k::launch_apply_matrix_2q(ctx, state, hi, lo, m),
        TileOp::Gate { .. } => unreachable!("gates dispatch through the backend"),
    }
}

#[cfg(test)]
#[path = "tiles_tests.rs"]
mod tests;
