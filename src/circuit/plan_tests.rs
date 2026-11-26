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
        assert!(record_sites(
            i as u32,
            inst,
            prov,
            &tracer.constant_sources,
            &mut steps,
            &mut sites
        ));
    }
    build_plan(fused, steps, sites, tracer.guard_steps, tracer.guards)
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
    let mut tracer = Tracer::tracking_all(&template);
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
    let params = Parameters::all_rotations(&template);
    let mut tracer = Tracer::tracking(&template, &params);
    let mut fused = fuse_diagonal_batch(Cow::Borrowed(&template), &mut tracer).into_owned();
    assert!(matches!(
        fused.instructions[0],
        Instruction::Gate {
            gate: Gate::DiagonalBatch(_),
            ..
        }
    ));
    let plan = finish(&fused, tracer);
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
    let mut tracer = Tracer::tracking_all(&template);
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
    let params = Parameters::all_rotations(&template);
    let mut tracer = Tracer::tracking(&template, &params);
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
    let params = Parameters::all_rotations(&template);
    let mut tracer = Tracer::tracking(&template, &params);
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

#[test]
fn constant_named_and_identity_runs_preserve_replay() {
    let mut template = Circuit::new(4, 0);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::T, &[1]);
    template.add_gate(Gate::T, &[1]);
    template.add_gate(Gate::Rz(0.23), &[2]);
    template.add_gate(Gate::Rz(0.41), &[2]);
    let (mut fused, plan) = FusionPlan::capture(&template, 16, FusionProfile::Host);
    let plan = plan.expect("constant collapses retain a plan");
    let parameters = Parameters::all_rotations(&template);
    for values in [[0.31, 0.47], [-0.27, 0.61]] {
        let bound = parameters.bind(&template, &values).unwrap();
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}

#[test]
fn parameter_dependent_named_and_identity_runs_decline_capture() {
    for angle in [0.0, std::f64::consts::FRAC_PI_4] {
        let mut template = Circuit::new(4, 0);
        template.add_gate(Gate::P(angle), &[0]);
        template.add_gate(Gate::P(angle), &[0]);
        let (_, plan) = FusionPlan::capture(&template, 16, FusionProfile::Host);
        assert!(plan.is_none());
    }
}

#[test]
fn diagonal_mixed_benchmark_bindings_replay_without_fusion() {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let seed = 0xDEAD_BEEFu64;
    let template = crate::circuits::diagonal_mixed_circuit(16, 6, seed);
    let parameters = Parameters::all_rotations(&template);
    let (mut fused, plan) = FusionPlan::capture(&template, 16, FusionProfile::Host);
    let plan = plan.expect("mixed diagonal fixture captures a plan");
    for k in 0..100 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(k));
        let values: Vec<_> = (0..parameters.num_slots())
            .map(|_| rng.random::<f64>() * std::f64::consts::TAU)
            .collect();
        let bound = parameters.bind(&template, &values).unwrap();
        assert!(plan.replay(&bound, &mut fused), "binding {k} fell back");
    }
}

#[test]
fn link_aware_constant_sources_capture_partially_bound_templates() {
    let mut template = Circuit::new(4, 0);
    template.add_gate(Gate::P(0.0), &[0]);
    template.add_gate(Gate::P(0.0), &[0]);
    template.add_gate(Gate::P(std::f64::consts::FRAC_PI_4), &[1]);
    template.add_gate(Gate::P(std::f64::consts::FRAC_PI_4), &[1]);
    template.add_gate(Gate::Ry(0.35), &[2]);
    template.add_gate(Gate::Rz(0.25), &[2]);

    // Linking every rotation declines capture because the unlinked P pairs collapse.
    let all = Parameters::all_rotations(&template);
    assert!(
        FusionPlan::capture_with_params(&template, &all, 16, FusionProfile::Host)
            .1
            .is_none()
    );

    // Linking only the active Ry/Rz pair marks the P pairs constant and captures a plan.
    let partial = Parameters::from_links(
        vec![
            crate::circuit::ParamLink::new(4, 0),
            crate::circuit::ParamLink::new(5, 1),
        ],
        2,
    );
    let tracer = Tracer::tracking(&template, &partial);
    assert_eq!(
        tracer.constant_sources,
        vec![true, true, true, true, false, false]
    );

    let (mut fused, plan) =
        FusionPlan::capture_with_params(&template, &partial, 16, FusionProfile::Host);
    let plan = plan.expect("partially bound template captures a plan");
    assert!(plan.guards.is_empty());

    for values in [[0.31, -0.47], [-0.92, 1.15]] {
        let bound = partial.bind(&template, &values).unwrap();
        assert!(plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}

#[test]
fn constant_sites_and_guards_are_skipped() {
    use crate::sim::unified_pauli::PauliTerm;

    // Empty Parameters skips every site and guard across single- and multi-qubit gates.
    let mut template = Circuit::new(14, 0);
    template.add_gate(Gate::Rz(0.2), &[0]);
    template.add_gate(Gate::P(0.3), &[1]);
    template.add_gate(Gate::Rzz(0.4), &[2, 3]);
    template.add_gate(Gate::cphase(0.5), &[0, 4]);
    template.add_gate(Gate::cphase(0.6), &[0, 5]);
    template.add_gate(Gate::H, &[6]);
    template.add_gate(Gate::Rx(0.7), &[6]);
    template.add_gate(Gate::Cx, &[7, 8]);
    template.add_gate(Gate::Ry(0.8), &[7]);
    template.add_pauli_rotation(0.9, &[PauliTerm::x(9), PauliTerm::x(10), PauliTerm::x(11)]);
    template.add_pauli_rotation(1.1, &[PauliTerm::z(9), PauliTerm::z(10), PauliTerm::z(11)]);

    let empty = Parameters::new(0);
    let (_, plan) = FusionPlan::capture_with_params(&template, &empty, 14, FusionProfile::Host);
    let plan = plan.expect("all-constant circuit captures an empty plan");
    assert!(plan.sites.is_empty());
    assert!(plan.guards.is_empty());
    assert!(plan.steps.is_empty());
    assert!(plan.guard_steps.is_empty());
    assert_eq!(plan.shared_2x2_count, 0);
    assert_eq!(plan.shared_4x4_count, 0);

    // Mixed constant/linked circuit records sites only for linked sources.
    let partial = Parameters::from_links(vec![crate::circuit::ParamLink::new(6, 0)], 1);
    let (mut fused, mixed_plan) =
        FusionPlan::capture_with_params(&template, &partial, 14, FusionProfile::Host);
    let mixed_plan = mixed_plan.expect("mixed circuit captures a plan");
    assert_eq!(mixed_plan.sites.len(), 1);
    assert!(mixed_plan.guards.is_empty());

    for theta in [0.15, -1.23] {
        let bound = partial.bind(&template, &[theta]).unwrap();
        assert!(mixed_plan.replay(&bound, &mut fused));
        assert_states(&fused, &bound);
    }
}

#[test]
fn diagonal_projection_and_absorbed_1q_share_slots_and_deduplicate_guards() {
    // Projected Fused2q inside DiagonalBatch retains one Guard::Diag4q and shares one 4x4 slot
    // across the guard and all three Step::Project entries.
    let mut proj_template = Circuit::new(3, 0);
    proj_template.add_gate(Gate::Cx, &[0, 1]);
    proj_template.add_gate(Gate::Rz(0.31), &[1]);
    proj_template.add_gate(Gate::Rx(0.0), &[1]);
    proj_template.add_gate(Gate::Cx, &[0, 1]);
    proj_template.add_gate(Gate::Rz(0.27), &[2]);
    let proj_params = Parameters::all_rotations(&proj_template);
    let mut tracer = Tracer::tracking(&proj_template, &proj_params);
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
    let mut proj_fused =
        fuse_diagonal_batch(Cow::Borrowed(&intermediate), &mut tracer).into_owned();
    let proj_plan = finish(&proj_fused, tracer);
    assert_eq!(proj_plan.guards.len(), 1);
    assert!(matches!(proj_plan.guards[0], Guard::Diag4q(ref sr, true) if sr.slot == Some(0)));
    assert_eq!(proj_plan.shared_4x4_count, 1);
    assert_eq!(proj_plan.shared_2x2_count, 0);
    for site in &proj_plan.sites {
        if let Recipe::Mat2(sr, _) | Recipe::Phase(sr) = &site.recipe {
            let slice = &proj_plan.steps[sr.range.start as usize..sr.range.end as usize];
            if let [Step::Project { slot, .. }] = slice {
                assert_eq!(*slot, Some(0));
            }
        }
    }
    let bound = proj_params
        .bind(&proj_template, &[0.65, 0.0, -0.4])
        .unwrap();
    assert!(proj_plan.replay(&bound, &mut proj_fused));
    assert_states(&proj_fused, &bound);

    // Absorbed Gate::Fused inside DiagonalBatch keeps Guard::Fuses1q and shares one 2x2 slot
    // between the guard and DiagonalBatch's Recipe::Mat2.
    let mut diag_1q = Circuit::new(14, 0);
    diag_1q.add_gate(Gate::cphase(0.25), &[0, 2]);
    diag_1q.add_gate(Gate::Rz(0.21), &[0]);
    diag_1q.add_gate(Gate::P(0.37), &[0]);
    let diag_params = Parameters::all_rotations(&diag_1q);
    let (mut diag_fused, diag_plan) =
        FusionPlan::capture_with_params(&diag_1q, &diag_params, 14, FusionProfile::Host);
    let diag_plan = diag_plan.expect("absorbed diagonal 1q captures a plan");
    assert_eq!(diag_plan.guards.len(), 1);
    assert!(matches!(diag_plan.guards[0], Guard::Fuses1q(ref sr, true) if sr.slot == Some(0)));
    assert_eq!(diag_plan.shared_2x2_count, 1);

    // Collapsing Rz(0.0) + P(0.0) to Id trips the retained Guard::Fuses1q.
    let collapsed = diag_params.bind(&diag_1q, &[0.0, 0.0]).unwrap();
    assert!(!diag_plan.replay(&collapsed, &mut diag_fused));
    // Collapsing Rz(0.0) + P(PI/2) to S trips the retained Guard::Fuses1q.
    let named = diag_params
        .bind(&diag_1q, &[0.0, std::f64::consts::FRAC_PI_2])
        .unwrap();
    assert!(!diag_plan.replay(&named, &mut diag_fused));

    let generic = diag_params.bind(&diag_1q, &[0.55, -0.25]).unwrap();
    assert!(diag_plan.replay(&generic, &mut diag_fused));
    assert_states(&diag_fused, &generic);
}

#[test]
fn subsumed_guards_are_dropped_while_site_bailouts_hold() {
    // Pass 1+2 Fuses1q guards on Fused/MultiFused and Phase2q/Diag4q guards on Fused2q/BatchPhase
    // are all subsumed by their sites, leaving zero guards and zero shared slots.
    let mut template = Circuit::new(10, 0);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::Rz(0.3), &[0]);
    template.add_gate(Gate::Cx, &[1, 2]);
    template.add_gate(Gate::Rx(0.4), &[1]);
    template.add_gate(Gate::cphase(0.25), &[3, 4]);
    template.add_gate(Gate::cphase(0.35), &[3, 5]);
    template.add_gate(Gate::Rx(0.2), &[6]);
    template.add_gate(Gate::P(0.5), &[6]);
    let params = Parameters::all_rotations(&template);
    let (mut fused, plan) =
        FusionPlan::capture_with_params(&template, &params, 10, FusionProfile::Host);
    let plan = plan.expect("subsumed circuit captures a plan");
    assert!(plan.guards.is_empty());
    assert_eq!(plan.shared_2x2_count, 0);
    assert_eq!(plan.shared_4x4_count, 0);

    // Even without Guard::Fuses1q, write_2x2's matrix_2x2_still_fuses trips when Rx(0)+P(0) -> Id
    // or P(PI)*Rx(0) -> Z.
    let id_bound = params.bind(&template, &[0.3, 0.4, 0.0, 0.0]).unwrap();
    assert!(!plan.replay(&id_bound, &mut fused));

    let mut fresh = fused.clone();
    let named_bound = params
        .bind(&template, &[0.3, 0.4, 0.0, std::f64::consts::PI])
        .unwrap();
    assert!(!plan.replay(&named_bound, &mut fresh));

    let mut clean = fused.clone();
    let generic = params.bind(&template, &[-0.6, 0.7, 0.35, -0.45]).unwrap();
    assert!(plan.replay(&generic, &mut clean));
    assert_states(&clean, &generic);
}

#[test]
fn post_phase_1q_folding_handles_single_and_multi_qubit_runs() {
    use crate::circuit::fusion_phase::batch_post_phase_1q;

    // >= 2 distinct qubits with repeated qubit 0 and constant T on qubit 2.
    let mut multi = Circuit::new(4, 0);
    multi.add_gate(Gate::H, &[0]);
    multi.add_gate(Gate::Rz(0.3), &[1]);
    multi.add_gate(Gate::Rx(0.4), &[0]);
    multi.add_gate(Gate::T, &[2]);
    let multi_params = Parameters::all_rotations(&multi);
    let mut tracer = Tracer::tracking(&multi, &multi_params);
    let mut multi_fused = batch_post_phase_1q(Cow::Borrowed(&multi), &mut tracer).into_owned();
    assert!(matches!(
        multi_fused.instructions.as_slice(),
        [Instruction::Gate {
            gate: Gate::MultiFused(d),
            ..
        }] if d.gates.len() == 3
    ));
    let multi_plan = finish(&multi_fused, tracer);
    assert_eq!(multi_plan.sites.len(), 2);
    for values in [[0.7, -0.5], [-1.1, 0.85]] {
        let bound = multi_params.bind(&multi, &values).unwrap();
        assert!(multi_plan.replay(&bound, &mut multi_fused));
        assert_states(&multi_fused, &bound);
    }

    // Single distinct qubit with >= 2 gates folds into Gate::Fused instead of 1-entry MultiFused.
    let mut single = Circuit::new(2, 0);
    single.add_gate(Gate::H, &[0]);
    single.add_gate(Gate::Rz(0.4), &[0]);
    let single_params = Parameters::all_rotations(&single);
    let mut tracer = Tracer::tracking(&single, &single_params);
    let mut single_fused = batch_post_phase_1q(Cow::Borrowed(&single), &mut tracer).into_owned();
    assert!(matches!(
        single_fused.instructions.as_slice(),
        [Instruction::Gate {
            gate: Gate::Fused(_),
            ..
        }]
    ));
    let single_plan = finish(&single_fused, tracer);
    assert_eq!(single_plan.sites.len(), 1);
    for theta in [0.9, -1.25] {
        let bound = single_params.bind(&single, &[theta]).unwrap();
        assert!(single_plan.replay(&bound, &mut single_fused));
        assert_states(&single_fused, &bound);
    }
    // Collapsing H * Rz(0.0) to named gate H trips write_2x2.
    let collapsed = single_params.bind(&single, &[0.0]).unwrap();
    assert!(!single_plan.replay(&collapsed, &mut single_fused));
}
