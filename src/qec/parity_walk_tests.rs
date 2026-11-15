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
    let (events, _) = compile_record_events(&deferred, None).unwrap();
    let sites: Vec<(u32, [u32; 2])> = events.iter().map(|e| (e.position, e.targets)).collect();
    assert_eq!(sites, vec![(1, [0, u32::MAX]), (0, [1, u32::MAX])]);
}

#[test]
fn record_events_drop_faults_that_flip_no_record() {
    let (deferred, _) = lower("Z_ERROR(0.1) 0\nM 0");
    let (events, _) = compile_record_events(&deferred, None).unwrap();
    assert!(events.is_empty());
}

#[test]
fn flipping_births_are_the_aliases_whose_z_reaches_a_record() {
    // A Z at the birth of qubit 0 becomes an X through H and flips its record; qubit 1
    // is measured in Z directly, so its birth Z does nothing.
    let (deferred, _) = lower("H 0\nM 0 1");
    let (_, births) = compile_record_events(&deferred, None).unwrap();
    assert_eq!(births, vec![0]);
}

/// The record events as the output walk derives them over no outputs: its kept events,
/// each mapped back through its candidate to the deferred event and target it came from.
fn record_events_through_output_walk(
    deferred: &QecDeferredProgram,
    windows: Option<usize>,
) -> Vec<QecRecordEvent> {
    let projection = QecParityProjection::new(deferred.measurement_qubits.len(), &[], &[]);
    let plan = WalkPlan::new(deferred, &projection, windows).unwrap();
    let walk = walk_outputs(&plan).unwrap();
    let mut site_of_candidate = vec![(0u32, 0u32); plan.candidates as usize];
    for (slot, &index) in plan.events_by_position.iter().enumerate() {
        let event = &deferred.noise_events[index as usize];
        let base = plan.candidate_base[slot] as usize;
        for (offset, site) in site_of_candidate[base..base + candidate_count(event)]
            .iter_mut()
            .enumerate()
        {
            *site = (index, offset as u32);
        }
    }
    walk.events
        .iter()
        .map(|kept| {
            let (index, offset) = site_of_candidate[kept.kept.candidate as usize];
            let event = &deferred.noise_events[index as usize];
            let offset = offset as usize;
            let pair = matches!(
                event.channel,
                QecDeferredChannel::Depolarize2(_) | QecDeferredChannel::PairTable { .. }
            );
            let targets = if pair {
                [
                    event.targets[2 * offset] as u32,
                    event.targets[2 * offset + 1] as u32,
                ]
            } else if matches!(event.channel, QecDeferredChannel::Products { .. }) {
                [index, u32::MAX]
            } else {
                [event.targets[offset] as u32, u32::MAX]
            };
            QecRecordEvent {
                draw: kept.kept.draw,
                position: event.position as u32,
                targets,
            }
        })
        .collect()
}

#[test]
fn record_events_match_the_output_walk_for_any_window_count() {
    let noise = crate::qec::QecCircuitNoise::uniform(0.01);
    let mut programs = vec![
        QecProgram::surface_memory(3, 4, crate::qec::QecBasis::Z, &noise).unwrap(),
        QecProgram::color_memory(3, 3, crate::qec::QecBasis::X, &noise).unwrap(),
        QecProgram::repetition_memory(5, 3, &noise).unwrap(),
    ];
    programs.push(
        QecProgram::from_text(
            "RX 0
H 1
CX 0 1 1 2
PAULI_CHANNEL_2(0, 0.25, 0, 0.25, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.05) 0 1 1 2
             E(0.1) X0 Z2
ELSE_CORRELATED_ERROR(0.2) Y1
HERALDED_ERASE(0.1) 2
             PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0 1 2
MPP X0*Z1 Y2
Z_ERROR(0.2) 0
MX 0
             DEPOLARIZE2(0.1) 1 2
S 1
X_ERROR(0.1) 1 2
MR 1 2
H 1
Y_ERROR(0.1) 1
M 1",
        )
        .unwrap(),
    );
    for program in &programs {
        let deferred = lower_qec_program_to_deferred_circuit(program).unwrap();
        let gates = deferred.gates.len();
        let (_, mut one_window_births) = compile_record_events(&deferred, Some(1)).unwrap();
        one_window_births.sort_unstable();
        for windows in [
            None,
            Some(1),
            Some(2),
            Some(5),
            Some(gates / 3),
            Some(gates + 1),
        ] {
            let (events, mut births) = compile_record_events(&deferred, windows).unwrap();
            let expected = record_events_through_output_walk(&deferred, windows);
            let key = |e: &QecRecordEvent| (e.draw, e.position, e.targets);
            assert_eq!(
                events.iter().map(key).collect::<Vec<_>>(),
                expected.iter().map(key).collect::<Vec<_>>(),
                "{windows:?}"
            );
            assert!(!events.is_empty());
            births.sort_unstable();
            assert_eq!(births, one_window_births, "{windows:?}");
        }
    }
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
        targets: targets.into(),
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
