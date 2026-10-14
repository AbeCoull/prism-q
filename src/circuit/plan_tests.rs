use super::*;
use crate::backend::{Backend, statevector::StatevectorBackend};
use crate::circuit::fusion::{FusionProfile, fuse_diagonal_batch};
use crate::circuit::fusion_phase::fuse_controlled_phases;
use crate::circuit::parameter::{Parameters, angle_mut};
use std::borrow::Cow;

fn finish(fused: &Circuit, tracer: Tracer) -> FusionPlan {
    assert!(!tracer.bailed);
    assert_eq!(fused.instructions.len(), tracer.input.len());
    let mut steps = Vec::new();
    let mut sites = Vec::new();
    for (i, (inst, prov)) in fused.instructions.iter().zip(&tracer.input).enumerate() {
        assert!(record_sites(i as u32, inst, prov, &mut steps, &mut sites));
    }
    FusionPlan {
        steps,
        sites,
        guard_steps: tracer.guard_steps,
        guards: tracer.guards,
    }
}

fn assert_states(a: &Circuit, b: &Circuit) {
    let run = |c: &Circuit| {
        let mut backend = StatevectorBackend::new(42);
        backend.init(c.num_qubits, 0).unwrap();
        for q in 0..c.num_qubits {
            backend
                .apply(&Instruction::Gate {
                    gate: Gate::H,
                    targets: crate::circuit::smallvec![q],
                })
                .unwrap();
        }
        backend.apply_instructions(&c.instructions).unwrap();
        backend.state_vector().to_vec()
    };
    for (x, y) in run(a).iter().zip(run(b)) {
        assert!((*x - y).norm() < 1e-12, "{x} != {y}");
    }
}

#[test]
fn phase_replay_tracks_products_rerooting_and_lone_outputs() {
    let mut template = Circuit::new(5, 0);
    for (a, b, theta) in [(0, 4, 0.2), (0, 4, 0.3), (1, 4, 0.4), (2, 4, 0.5)] {
        template.add_gate(Gate::cphase(theta), &[a, b]);
    }
    template.add_gate(Gate::H, &[4]);
    template.add_gate(Gate::cphase(0.6), &[3, 1]);
    let mut tracer = Tracer::tracking(&template);
    let mut fused = fuse_controlled_phases(Cow::Borrowed(&template), &mut tracer).into_owned();
    assert!(fused.instructions.iter().any(|inst| matches!(
        inst,
        Instruction::Gate {
            gate: Gate::BatchPhase(_),
            ..
        }
    )));
    let plan = finish(&fused, tracer);
    for shift in [0.1, 0.7, -0.9] {
        let mut bound = template.clone();
        for (i, inst) in bound.instructions.iter_mut().enumerate() {
            if let Instruction::Gate {
                gate: gate @ Gate::Cu(_),
                ..
            } = inst
            {
                *gate = Gate::cphase(shift + i as f64 * 0.13);
            }
        }
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
    let mut nonphase = template.clone();
    let Instruction::Gate { gate, .. } = &mut nonphase.instructions[0] else {
        unreachable!()
    };
    *gate = Gate::cu(Gate::Rz(0.4).matrix_2x2());
    assert!(!plan.replay(&nonphase, &mut fused));
}

#[test]
fn diagonal_replay_rebinds_phases_and_parity() {
    let mut template = Circuit::new(4, 0);
    template.add_gate(Gate::Rz(0.2), &[0]);
    template.add_gate(Gate::P(0.3), &[1]);
    template.add_gate(Gate::Rzz(0.4), &[2, 3]);
    template.add_gate(Gate::cphase(0.5), &[0, 2]);
    let mut tracer = Tracer::tracking(&template);
    let mut fused = fuse_diagonal_batch(Cow::Borrowed(&template), &mut tracer).into_owned();
    assert!(matches!(
        fused.instructions[0],
        Instruction::Gate {
            gate: Gate::DiagonalBatch(_),
            ..
        }
    ));
    let plan = finish(&fused, tracer);
    let params = Parameters::all_rotations(&template);
    for values in [[0.0, 0.0, 0.0], [0.9, -0.7, 0.5], [-1.3, 0.8, 2.1]] {
        let bound = params.bind(&template, &values).unwrap();
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}

#[test]
fn phase_replay_preserves_chunk_boundaries() {
    let count = crate::gates::BatchPhaseData::MAX_PHASES + 1;
    let mut template = Circuit::new(count + 1, 0);
    for q in 1..=count {
        template.add_gate(Gate::cphase(0.1 + q as f64 * 0.01), &[0, q]);
    }
    let mut tracer = Tracer::tracking(&template);
    let mut fused = fuse_controlled_phases(Cow::Borrowed(&template), &mut tracer).into_owned();
    assert_eq!(fused.instructions.len(), 2);
    let plan = finish(&fused, tracer);
    for inst in &mut template.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            unreachable!()
        };
        *gate = Gate::cphase(-0.2 * targets[1] as f64);
    }
    assert!(plan.replay(&template, &mut fused));
    for inst in &fused.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            unreachable!()
        };
        match gate {
            Gate::BatchPhase(data) => {
                for &(q, phase) in &data.phases {
                    assert!((phase - Complex64::from_polar(1.0, -0.2 * q as f64)).norm() < 1e-12);
                }
            }
            Gate::Cu(_) => {
                assert!(
                    (gate.controlled_phase().unwrap()
                        - Complex64::from_polar(1.0, -0.2 * targets[1] as f64))
                    .norm()
                        < 1e-12
                );
            }
            _ => panic!("phase chunk changed kind"),
        }
    }
}

#[test]
fn diagonal_projection_rebuilds_the_complete_pair_product() {
    let mut template = Circuit::new(3, 0);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Rz(0.31), &[1]);
    template.add_gate(Gate::Rx(0.0), &[1]);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Rz(0.27), &[2]);
    let mut tracer = Tracer::tracking(&template);
    tracer.begin();
    tracer.merge(&[
        (0, Place::Plain),
        (1, Place::High),
        (2, Place::High),
        (3, Place::Plain),
    ]);
    tracer.keep(4);
    tracer.commit();
    let mut intermediate = Circuit::new(3, 0);
    intermediate.add_gate(
        Gate::Fused2q(Box::new(Gate::Rzz(0.31).matrix_4x4())),
        &[0, 1],
    );
    intermediate.add_gate(Gate::Rz(0.27), &[2]);
    let mut fused = fuse_diagonal_batch(Cow::Borrowed(&intermediate), &mut tracer).into_owned();
    let plan = finish(&fused, tracer);
    for theta in [0.0, 0.8, -1.4] {
        let mut bound = template.clone();
        *angle_mut(&mut bound.instructions[1]) = theta;
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
    let mut nondiagonal = template.clone();
    *angle_mut(&mut nondiagonal.instructions[2]) = 0.4;
    assert!(!plan.replay(&nondiagonal, &mut fused));
}

#[test]
fn staged_diagonal_subpasses_preserve_replay_sources() {
    use crate::circuit::fusion_stage::{TileGeometry, plan_with};
    let mut template = Circuit::new(4, 0);
    template.add_gate(Gate::Rz(0.2), &[0]);
    template.add_gate(Gate::P(0.3), &[1]);
    template.add_gate(Gate::Rzz(0.4), &[2, 3]);
    template.add_gate(Gate::cphase(0.5), &[0, 2]);
    let mut tracer = Tracer::tracking(&template);
    let mut fused = plan_with(
        Cow::Borrowed(&template),
        TileGeometry::new(4, 1, 2),
        &mut tracer,
    )
    .into_owned();
    assert!(fused.instructions.iter().any(|inst| matches!(
        inst,
        Instruction::Gate {
            gate: Gate::DiagonalBatch(_),
            ..
        }
    )));
    let plan = finish(&fused, tracer);
    let params = Parameters::all_rotations(&template);
    for values in [[0.0, 0.0, 0.0], [0.9, -0.7, 0.5], [-1.3, 0.8, 2.1]] {
        let bound = params.bind(&template, &values).unwrap();
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}

#[test]
fn rewritten_rzz_replays_as_a_pair_matrix() {
    let mut template = Circuit::new(4, 0);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Rz(0.31), &[1]);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Rz(0.27), &[2]);
    template.add_gate(Gate::P(0.21), &[3]);
    let (mut fused, plan) = FusionPlan::capture(&template, 24, FusionProfile::Host);
    let plan = plan.expect("rewritten Rzz captures a plan");
    for theta in [0.0, 0.8, -1.4] {
        let mut bound = template.clone();
        *angle_mut(&mut bound.instructions[1]) = theta;
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}
