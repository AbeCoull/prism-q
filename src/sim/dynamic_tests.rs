use super::*;
use crate::circuit::dynamic::{BinaryOp, ClassicalExpr, ClassicalType, DynamicProgramBuilder};
use crate::gates::Gate;
use crate::sim::ResolvedBackend;

fn rus() -> DynamicProgram {
    let mut b = DynamicProgramBuilder::new(2, 1);
    b.add_gate(Gate::H, &[0]).add_gate(Gate::Cx, &[0, 1]);
    b.add_measure(0, 0);
    b.begin_while("retry", ClassicalExpr::Bit(0));
    b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.end().unwrap();
    b.build().unwrap()
}

#[test]
fn repeat_until_success_always_ends_on_zero() {
    let shots = simulate_program(&rus()).seed(7).shots(200).unwrap();
    assert!(shots.shots.iter().all(|shot| !shot[0]));
    assert_eq!(shots.metadata.backend, ResolvedBackend::Stabilizer);
}

#[test]
fn a_loop_past_the_step_bound_names_itself() {
    let mut b = DynamicProgramBuilder::new(1, 0);
    b.begin_while("forever", ClassicalExpr::from(true));
    b.add_gate(Gate::X, &[0]);
    b.end().unwrap();
    let program = b.build().unwrap();
    let err = simulate_program(&program)
        .max_steps(50)
        .seed(1)
        .shots(2)
        .unwrap_err();
    match err {
        PrismError::StepLimit { region, max_steps } => {
            assert_eq!(region, "loop `forever`");
            assert_eq!(max_steps, 50);
        }
        other => panic!("expected a step limit, got {other:?}"),
    }
}

#[test]
fn a_reused_statevector_matches_a_fresh_one_per_shot() {
    let mut b = DynamicProgramBuilder::new(2, 2);
    let n = b.declare("n", ClassicalType::Uint { width: 4 }, 0i64.into());
    b.add_gate(Gate::Ry(0.7), &[0]).add_gate(Gate::Cx, &[0, 1]);
    b.add_measure(0, 0);
    b.begin_while(
        "count",
        ClassicalExpr::binary(BinaryOp::Lt, n.into(), 3i64.into()),
    );
    b.add_gate(Gate::Ry(0.4), &[1]).add_measure(1, 1);
    b.assign(
        n,
        ClassicalExpr::binary(BinaryOp::Add, n.into(), 1i64.into()),
    );
    b.end().unwrap();
    let program = b.build().unwrap();
    let prepared = Prepared::new(
        &program,
        &BackendKind::Statevector,
        3,
        100,
        &ResolvedLimits::process_default(),
        None,
    )
    .unwrap();
    let reused = prepared
        .run_per_shot(&BackendKind::Statevector, 64, 3)
        .unwrap();
    let fresh: Vec<Vec<bool>> = (0..64)
        .map(|i| {
            let shot_seed = mix_seed(3, i);
            let mut backend = StatevectorBackend::new(shot_seed);
            prepared
                .run_shot(&mut backend, &mut Vec::new(), shot_seed)
                .unwrap()
                .0
        })
        .collect();
    assert_eq!(reused.shots, fresh);
}
