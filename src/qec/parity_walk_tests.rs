use super::*;
use crate::qec::QecProgram;
use crate::qec::noise::lower_qec_program_to_deferred_circuit;

fn lower(text: &str) -> (QecDeferredProgram, QecParityProjection) {
    let program = QecProgram::from_text(text).unwrap();
    let projection = QecParityProjection::new(
        program.num_measurements(),
        &program.detector_rows().unwrap(),
        &program.observable_rows().unwrap(),
    );
    (
        lower_qec_program_to_deferred_circuit(&program).unwrap(),
        projection,
    )
}

fn sites(text: &str) -> Vec<Vec<(f64, Vec<u32>)>> {
    let (deferred, projection) = lower(text);
    let sites = compile_fault_sites(&deferred, &projection).unwrap();
    sites
        .sites()
        .map(|site| site.map(|(p, outputs)| (p, outputs.to_vec())).collect())
        .collect()
}

/// Total rate of each nonempty output set a site flips. Branches of one event are
/// exclusive, so rates of branches with the same flips add.
fn symptoms(site: &[(f64, Vec<u32>)]) -> Vec<(Vec<u32>, u64)> {
    let mut totals = std::collections::BTreeMap::<Vec<u32>, f64>::new();
    for (p, outputs) in site {
        if !outputs.is_empty() {
            *totals.entry(outputs.clone()).or_default() += p;
        }
    }
    totals
        .into_iter()
        .map(|(outputs, p)| (outputs, (p * 1e9).round() as u64))
        .collect()
}

#[test]
fn x_error_before_a_z_measurement_flips_its_detector() {
    let sites = sites("X_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]");
    assert_eq!(sites, vec![vec![(0.125, vec![0])]]);
}

#[test]
fn depolarize1_flips_a_z_measurement_at_two_thirds_of_its_rate() {
    // X and Y anticommute with the Z measurement; Z commutes and flips nothing.
    let sites = sites("DEPOLARIZE1(0.3) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]");
    assert_eq!(sites.len(), 1);
    assert_eq!(symptoms(&sites[0]), vec![(vec![0, 1], 200_000_000)]);
}

#[test]
fn depolarize2_branches_flip_the_measurements_their_x_part_reaches() {
    // Of the 15 non-identity Paulis on two qubits at 0.01 each, four have an X or Y on
    // qubit 0 only, four on qubit 1 only, four on both, and three flip neither record.
    let sites = sites("DEPOLARIZE2(0.15) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
    assert_eq!(sites.len(), 1);
    assert_eq!(
        symptoms(&sites[0]),
        vec![
            (vec![0], 40_000_000),
            (vec![0, 1], 40_000_000),
            (vec![1], 40_000_000),
        ]
    );
}

#[test]
fn fault_sites_come_in_program_order() {
    let sites = sites(
        "X_ERROR(0.1) 0\nX_ERROR(0.2) 1\nH 2\nX_ERROR(0.3) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    );
    let order: Vec<(f64, Vec<u32>)> = sites.iter().map(|site| site[0].clone()).collect();
    assert_eq!(
        order,
        vec![
            (0.1, vec![0]),
            (0.2, vec![1]),
            (0.3, vec![0]),
            (0.3, vec![1]),
        ]
    );
}

#[test]
fn a_cnot_carries_an_x_fault_onto_its_target() {
    let sites = sites("X_ERROR(0.1) 0\nCX 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
    assert_eq!(sites, vec![vec![(0.1, vec![0, 1])]]);
}

#[test]
fn fault_sites_name_the_first_random_output() {
    for (text, expected) in [
        ("H 0\nM 0\nDETECTOR rec[-1]", "detector 0 "),
        (
            "H 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
            "detector 0 ",
        ),
        (
            "H 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            "observable 0 ",
        ),
    ] {
        let (deferred, projection) = lower(text);
        let Err(err) = compile_fault_sites(&deferred, &projection) else {
            panic!("{text}: a random output must be refused");
        };
        assert!(err.to_string().contains(expected), "{text}: {err}");
    }
}

#[test]
fn parity_compile_returns_the_noiseless_pattern_or_none() {
    let (deferred, projection) =
        lower("X 0\nX_ERROR(0.1) 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
    let (pattern, noise) = compile_parity_noise(&deferred, &projection, 42, None)
        .unwrap()
        .unwrap();
    assert_eq!(pattern, vec![0b01]);
    assert_eq!(noise.event_count(), 1);

    let (deferred, projection) = lower("H 0\nM 0\nDETECTOR rec[-1]");
    assert!(
        compile_parity_noise(&deferred, &projection, 42, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn record_events_come_in_draw_order_with_their_targets() {
    let (deferred, _) = lower("X_ERROR(0.1) 1\nCX 0 1\nX_ERROR(0.2) 0\nM 0 1");
    let (events, _) = compile_record_events(&deferred).unwrap();
    let sites: Vec<(u32, [u32; 2])> = events.iter().map(|e| (e.position, e.targets)).collect();
    assert_eq!(sites, vec![(1, [0, u32::MAX]), (0, [1, u32::MAX])]);
}

#[test]
fn record_events_drop_faults_that_flip_no_record() {
    let (deferred, _) = lower("Z_ERROR(0.1) 0\nM 0");
    let (events, _) = compile_record_events(&deferred).unwrap();
    assert!(events.is_empty());
}

#[test]
fn flipping_births_are_the_aliases_whose_z_reaches_a_record() {
    // A Z at the birth of qubit 0 becomes an X through H and flips its record; qubit 1
    // is measured in Z directly, so its birth Z does nothing.
    let (deferred, _) = lower("H 0\nM 0 1");
    let (_, births) = compile_record_events(&deferred).unwrap();
    assert_eq!(births, vec![0]);
}

#[test]
fn walk_plan_windows_tile_every_position_once() {
    let (deferred, projection) = lower("H 0\nS 0\nH 0\nS 0\nH 0\nM 0\nDETECTOR rec[-1]");
    let gates = deferred.gates.len();
    assert!(gates >= 5);
    for (requested, expected) in [
        (Some(0), 1),
        (Some(1), 1),
        (Some(3), 3),
        (Some(gates + 1), gates + 1),
        (Some(gates + 10), gates + 1),
    ] {
        let plan = WalkPlan::new(&deferred, &projection, requested).unwrap();
        assert_eq!(plan.windows.len(), expected, "{requested:?}");
        assert_eq!(plan.windows[0][0], 0);
        assert_eq!(plan.windows.last().unwrap()[1], gates);
        for pair in plan.windows.windows(2) {
            assert_eq!(
                pair[1][0],
                pair[0][1] + 1,
                "{requested:?}: {:?}",
                plan.windows
            );
        }
        assert!(plan.windows.iter().all(|&[lo, hi]| lo <= hi));
    }
}

#[test]
fn walk_plan_numbers_candidates_in_visit_order() {
    // Positions descending, program order within a position, one candidate per target.
    let (deferred, projection) =
        lower("X_ERROR(0.1) 0 1\nH 0\nX_ERROR(0.2) 0\nZ_ERROR(0.3) 1\nM 0 1");
    let plan = WalkPlan::new(&deferred, &projection, Some(1)).unwrap();
    assert_eq!(plan.candidates, 4);
    let bases: Vec<(usize, u32)> = plan
        .events_by_position
        .iter()
        .zip(&plan.candidate_base)
        .map(|(&event, &base)| (deferred.noise_events[event as usize].position, base))
        .collect();
    assert_eq!(bases, vec![(0, 2), (1, 0), (1, 1)]);
}

#[test]
fn walk_plan_rejects_an_event_past_the_last_gate() {
    let (mut deferred, projection) = lower("X_ERROR(0.1) 0\nM 0");
    deferred.noise_events[0].position = deferred.gates.len() + 1;
    let Err(err) = WalkPlan::new(&deferred, &projection, None) else {
        panic!("an event past the gate list must be refused");
    };
    assert!(
        err.to_string().contains("exceeds deferred gate count"),
        "{err}"
    );
}

#[test]
fn candidate_count_follows_the_channel_arity() {
    let event = |channel, targets: Vec<usize>| QecDeferredNoiseEvent {
        channel,
        targets,
        position: 0,
    };
    assert_eq!(
        candidate_count(&event(
            QecDeferredChannel::Single([0.1, 0.0, 0.0]),
            vec![0, 1, 2]
        )),
        3
    );
    assert_eq!(
        candidate_count(&event(
            QecDeferredChannel::Depolarize2(0.1),
            vec![0, 1, 2, 3]
        )),
        2
    );
    assert_eq!(
        candidate_count(&event(
            QecDeferredChannel::PairTable { p: 0.1, table: 0 },
            vec![0, 1]
        )),
        1
    );
    assert_eq!(
        candidate_count(&event(
            QecDeferredChannel::Products { p: 0.1, table: 0 },
            vec![0, 1, 2]
        )),
        1
    );
}
