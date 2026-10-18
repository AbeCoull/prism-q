use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Instruction;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn apply_op(state: &mut [Complex64], op: &TileOp<'_>) {
    let zero = Complex64::new(0.0, 0.0);
    let bit = |i: usize, q: usize| (i >> q) & 1;
    match *op {
        TileOp::Mat1 { q, m } => apply_1q(state, q, m),
        TileOp::Diag1 { q, d } => apply_1q(state, q, [[d[0], zero], [zero, d[1]]]),
        TileOp::Anti1 { q, d } => apply_1q(state, q, [[zero, d[0]], [d[1], zero]]),
        TileOp::Cx { control, target } => {
            for i in 0..state.len() {
                if bit(i, control) == 1 && bit(i, target) == 0 {
                    state.swap(i, i | 1 << target);
                }
            }
        }
        TileOp::Swap { a, b } => {
            for i in 0..state.len() {
                if bit(i, a) == 1 && bit(i, b) == 0 {
                    state.swap(i, i ^ (1 << a) ^ (1 << b));
                }
            }
        }
        TileOp::Diag2 { a, b, d } => {
            for (i, amp) in state.iter_mut().enumerate() {
                *amp *= d[2 * bit(i, a) + bit(i, b)];
            }
        }
        TileOp::Dense2q { hi, lo, m } => {
            for i in 0..state.len() {
                if bit(i, hi) == 0 && bit(i, lo) == 0 {
                    let idx = [i, i | 1 << lo, i | 1 << hi, i | 1 << hi | 1 << lo];
                    let v = idx.map(|j| state[j]);
                    for (r, &j) in idx.iter().enumerate() {
                        state[j] = (0..4).map(|c| m[r][c] * v[c]).sum();
                    }
                }
            }
        }
        TileOp::Gate { gate, targets } => {
            let mut host = StatevectorBackend::new(42);
            host.init_from_amplitudes(state.to_vec(), 0).unwrap();
            host.apply(&Instruction::Gate {
                gate: gate.clone(),
                targets: targets.iter().copied().collect(),
            })
            .unwrap();
            state.copy_from_slice(&host.export_statevector().unwrap());
        }
    }
}

fn apply_1q(state: &mut [Complex64], q: usize, m: [[Complex64; 2]; 2]) {
    for i in 0..state.len() {
        if (i >> q) & 1 == 0 {
            let (a, b) = (state[i], state[i | 1 << q]);
            state[i] = m[0][0] * a + m[0][1] * b;
            state[i | 1 << q] = m[1][0] * a + m[1][1] * b;
        }
    }
}

fn random_state(n: usize, rng: &mut ChaCha8Rng) -> Vec<Complex64> {
    (0..1 << n)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect()
}

fn random_1q(rng: &mut ChaCha8Rng) -> [[Complex64; 2]; 2] {
    let (a, b, c) = (
        rng.random::<f64>(),
        rng.random::<f64>(),
        rng.random::<f64>(),
    );
    crate::gates::mat_mul_2x2(&Gate::Ry(b).matrix_2x2(), &Gate::Rz(a + c).matrix_2x2())
}

fn random_ops<'a>(
    n: usize,
    count: usize,
    dense: &'a [[Complex64; 4]; 4],
    rng: &mut ChaCha8Rng,
) -> Vec<TileOp<'a>> {
    let phase = |rng: &mut ChaCha8Rng| Complex64::from_polar(1.0, rng.random::<f64>() * 6.0);
    (0..count)
        .map(|_| {
            let a = rng.random_range(0..n);
            let b = (a + rng.random_range(1..n)) % n;
            match rng.random_range(0..7) {
                0 => TileOp::Mat1 {
                    q: a,
                    m: random_1q(rng),
                },
                1 => TileOp::Diag1 {
                    q: a,
                    d: [phase(rng), phase(rng)],
                },
                2 => TileOp::Anti1 {
                    q: a,
                    d: [phase(rng), phase(rng)],
                },
                3 => TileOp::Cx {
                    control: a,
                    target: b,
                },
                4 => TileOp::Swap { a, b },
                5 => TileOp::Diag2 {
                    a,
                    b,
                    d: [phase(rng), phase(rng), phase(rng), phase(rng)],
                },
                _ => TileOp::Dense2q {
                    hi: a,
                    lo: b,
                    m: dense,
                },
            }
        })
        .collect()
}

fn assert_close(got: &[Complex64], want: &[Complex64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).norm() < 1e-12, "amplitude {i}: {g} vs {w}");
    }
}

fn run_plan(state: &mut [Complex64], ops: &[TileOp<'_>], plan: &Plan) {
    for step in &plan.steps {
        match step {
            Step::Tile { ops: range, .. } => {
                for &i in &plan.order[range.clone()] {
                    apply_op(state, &ops[i as usize]);
                }
            }
            Step::Single(i) => apply_op(state, &ops[*i as usize]),
        }
    }
}

fn check_plan(ops: &[TileOp<'_>], n: usize, geometry: &Geometry, plan: &Plan) {
    let mut seen = vec![false; ops.len()];
    for &i in &plan.order {
        assert!(!seen[i as usize], "op {i} scheduled twice");
        seen[i as usize] = true;
    }
    assert!(seen.iter().all(|&s| s), "an op was dropped");
    let all = (1u64 << n) - 1;
    for step in &plan.steps {
        if let Step::Tile { qubits, ops: range } = step {
            assert_eq!(qubits & geometry.anchor, geometry.anchor);
            assert!((qubits & !geometry.anchor).count_ones() <= geometry.budget);
            assert_eq!(qubits & !all, 0);
            assert!(range.len() >= MIN_TILE_OPS && range.len() <= geometry.max_ops);
            let mut coefs = 0;
            for &i in &plan.order[range.clone()] {
                let op = &ops[i as usize];
                assert_eq!(op.masks().0 & !qubits, 0, "{op:?} outside tile {qubits:b}");
                coefs += op.coefs().expect("only tileable ops ride in a tile");
            }
            assert!(coefs <= geometry.max_coefs);
        }
    }
}

const SMALL: Geometry = Geometry {
    anchor: 0b11,
    budget: 3,
    max_ops: 12,
    max_coefs: 48,
};

#[test]
#[ignore = "deterministic schedule census"]
fn diagonal_run_census() {
    use crate::circuit::fusion::{FusionProfile, fuse_circuit_for_profile};
    let seed = 0xDEAD_BEEF;
    for n in [18, 20, 24] {
        for (name, circuit) in [
            (
                "hea_l5",
                crate::circuits::hardware_efficient_ansatz(n, 5, seed),
            ),
            ("qaoa_l3", crate::circuits::qaoa_circuit(n, 3, seed)),
            (
                "diag_mixed_l6",
                crate::circuits::diagonal_mixed_circuit(n, 6, seed),
            ),
            ("random_d10", crate::circuits::random_circuit(n, 10, seed)),
            ("qv", crate::circuits::quantum_volume_circuit(n, n, seed)),
        ] {
            let fused = fuse_circuit_for_profile(&circuit, true, n, FusionProfile::Device);
            let mut ops = Vec::new();
            for inst in &fused.instructions {
                if let Instruction::Gate { gate, targets } = inst {
                    lower(&mut ops, gate, targets);
                }
            }
            let mut schedule = Plan::default();
            plan(&ops, n, &k::MIXED_TILE_GEOMETRY, &mut schedule);
            let (mut diagonal_ops, mut runs, mut combined, mut longest) = (0, 0, 0, 0);
            for step in &schedule.steps {
                if let Step::Tile { ops: range, .. } = step {
                    let mut len = 0;
                    for i in schedule.order[range.clone()].iter().map(Some).chain([None]) {
                        if i.is_some_and(|&i| {
                            matches!(ops[i as usize], TileOp::Diag1 { .. } | TileOp::Diag2 { .. })
                        }) {
                            len += 1;
                            diagonal_ops += 1;
                        } else if len > 0 {
                            runs += 1;
                            combined += usize::from(len > 1);
                            longest = longest.max(len);
                            len = 0;
                        }
                    }
                }
            }
            println!(
                "{name}/{n}: launches={} diagonal_ops={diagonal_ops} runs={runs} combined_runs={combined} longest={longest} shared_access_upper_bound={}->{}",
                schedule.steps.len(),
                2 * diagonal_ops,
                2 * runs,
            );
        }
    }
}

#[test]
fn separated_diagonal_runs_survive_planning() {
    for n in [18, 21] {
        let mut ops = Vec::new();
        for q in 0..n {
            push_1q(&mut ops, q, Gate::H.matrix_2x2());
        }
        for _ in 0..4 {
            for k in 0..6 {
                let (a, b) = if k % 2 == 0 { (0, n - 1) } else { (n - 1, 0) };
                ops.extend([
                    TileOp::Diag1 { q: a, d: [ONE; 2] },
                    TileOp::Diag1 { q: b, d: [ONE; 2] },
                    TileOp::Diag2 { a, b, d: [ONE; 4] },
                    TileOp::Diag2 {
                        a: b,
                        b: a,
                        d: [ONE; 4],
                    },
                    TileOp::Diag2 { a, b, d: [ONE; 4] },
                ]);
            }
            push_1q(&mut ops, n - 1, Gate::H.matrix_2x2());
            push_1q(&mut ops, 0, Gate::Y.matrix_2x2());
            ops.push(TileOp::Cx {
                control: n - 1,
                target: 0,
            });
            ops.push(TileOp::Swap { a: 0, b: n - 2 });
        }
        let mut schedule = Plan::default();
        plan(&ops, n, &k::MIXED_TILE_GEOMETRY, &mut schedule);
        let (mut combined, mut longest) = (0, 0);
        for step in &schedule.steps {
            if let Step::Tile { ops: range, .. } = step {
                let mut len = 0;
                for i in schedule.order[range.clone()].iter().map(Some).chain([None]) {
                    if i.is_some_and(|&i| {
                        matches!(ops[i as usize], TileOp::Diag1 { .. } | TileOp::Diag2 { .. })
                    }) {
                        len += 1;
                    } else {
                        combined += usize::from(len > 1);
                        longest = longest.max(len);
                        len = 0;
                    }
                }
            }
        }
        assert!(combined >= 4 && longest >= 20);
        println!(
            "separated/{n}: launches={} combined_runs={combined} longest={longest}",
            schedule.steps.len()
        );
    }
}

#[test]
fn plan_matches_in_order_application_on_random_windows() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let dense = {
        let mut m = Gate::Cx.matrix_4x4();
        m = crate::gates::mat_mul_4x4(
            &m,
            &crate::gates::kron_2x2(&random_1q(&mut rng), &random_1q(&mut rng)),
        );
        m
    };
    let mut plan_out = Plan::default();
    for (n, count) in [(4, 30), (7, 60), (9, 120), (9, 400)] {
        let ops = random_ops(n, count, &dense, &mut rng);
        plan(&ops, n, &SMALL, &mut plan_out);
        check_plan(&ops, n, &SMALL, &plan_out);
        let start = random_state(n, &mut rng);
        let mut want = start.clone();
        for op in &ops {
            apply_op(&mut want, op);
        }
        let mut got = start;
        run_plan(&mut got, &ops, &plan_out);
        assert_close(&got, &want);
    }
}

#[test]
fn plan_tiles_a_ladder_in_a_few_passes() {
    let n = 12;
    let circuit = crate::circuits::hardware_efficient_ansatz(n, 3, 42);
    let fused = crate::circuit::fusion::fuse_circuit_for_profile(
        &circuit,
        true,
        n,
        crate::circuit::fusion::FusionProfile::Device,
    );
    let mut ops = Vec::new();
    for inst in &fused.instructions {
        if let Instruction::Gate { gate, targets } = inst {
            lower(&mut ops, gate, targets);
        }
    }
    assert!(ops.iter().all(|op| op.coefs().is_some()));
    let geometry = Geometry {
        anchor: 0b11,
        budget: 5,
        max_ops: 128,
        max_coefs: 440,
    };
    let mut out = Plan::default();
    plan(&ops, n, &geometry, &mut out);
    check_plan(&ops, n, &geometry, &out);
    assert!(
        out.steps.len() * 8 <= ops.len(),
        "{} steps for {} ops",
        out.steps.len(),
        ops.len()
    );
}

#[test]
fn lowering_matches_the_host_statevector() {
    let n = 6;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut insts = Vec::new();
    let g = |gate: Gate, targets: &[usize]| Instruction::Gate {
        gate,
        targets: targets.iter().copied().collect(),
    };
    for q in 0..n {
        insts.push(g(Gate::Ry(0.3 + 0.2 * q as f64), &[q]));
        insts.push(g(Gate::Rz(0.1 * q as f64), &[q]));
    }
    let m1 = random_1q(&mut rng);
    let cu = crate::gates::mat_mul_2x2(&m1, &Gate::X.matrix_2x2());
    let dense = crate::gates::mat_mul_4x4(
        &Gate::Cx.matrix_4x4(),
        &crate::gates::kron_2x2(&m1, &Gate::H.matrix_2x2()),
    );
    let gates = [
        (Gate::X, vec![0]),
        (Gate::Y, vec![5]),
        (Gate::H, vec![2]),
        (Gate::T, vec![3]),
        (Gate::Fused(Box::new(m1)), vec![4]),
        (Gate::Cx, vec![4, 1]),
        (Gate::Cz, vec![0, 5]),
        (Gate::Swap, vec![2, 4]),
        (Gate::Rzz(0.7), vec![3, 1]),
        (Gate::cphase(0.9), vec![5, 2]),
        (Gate::Cu(Box::new(cu)), vec![1, 3]),
        (Gate::Cu(Box::new(Gate::S.matrix_2x2())), vec![0, 2]),
        (Gate::Fused2q(Box::new(dense)), vec![5, 0]),
        (Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())), vec![1, 4]),
        (Gate::Fused2q(Box::new(Gate::Swap.matrix_4x4())), vec![3, 0]),
        (
            Gate::Multi2q(Box::new(crate::gates::Multi2qData::new(vec![
                (2, 3, Gate::Cz.matrix_4x4()),
                (0, 4, dense),
            ]))),
            vec![0, 2, 3, 4],
        ),
        (
            Gate::MultiFused(Box::new(crate::gates::MultiFusedData::new(vec![
                (1, m1),
                (2, Gate::S.matrix_2x2()),
            ]))),
            vec![1, 2],
        ),
        (
            Gate::BatchRzz(Box::new(crate::gates::BatchRzzData::new(vec![
                (0, 1, 0.4),
                (2, 5, -1.1),
            ]))),
            vec![0, 1, 2, 5],
        ),
        (
            Gate::BatchPhase(Box::new(crate::gates::BatchPhaseData::new([
                (1, Complex64::from_polar(1.0, 0.3)),
                (4, Complex64::from_polar(1.0, -0.8)),
            ]))),
            vec![3],
        ),
    ];
    for (gate, targets) in gates {
        insts.push(g(gate, &targets));
    }
    let mut ops = Vec::new();
    for inst in &insts {
        if let Instruction::Gate { gate, targets } = inst {
            lower(&mut ops, gate, targets);
        }
    }
    let mut host = StatevectorBackend::new(42);
    host.init(n, 0).unwrap();
    for inst in &insts {
        host.apply(inst).unwrap();
    }
    let mut got = vec![Complex64::new(0.0, 0.0); 1 << n];
    got[0] = Complex64::new(1.0, 0.0);
    for op in &ops {
        apply_op(&mut got, op);
    }
    assert_close(&got, &host.export_statevector().unwrap());
}
