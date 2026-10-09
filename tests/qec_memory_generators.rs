//! Memory-experiment generators: noiseless determinism, graphlike decomposition and
//! graphlike distance, code distance of the color code, logical error rates that fall
//! with distance, and the text round trips.

use std::collections::VecDeque;

use prism_q::{
    DetectorErrorModel, QecBasis, QecCircuitNoise, QecOptions, QecProgram, UnionFindDecoder,
    parse_qec_program, run_qec_program, run_qec_program_reference,
};

mod qec_common;

fn noiseless() -> QecCircuitNoise {
    QecCircuitNoise::default()
}

fn with_shots(mut program: QecProgram, shots: usize) -> QecProgram {
    program.set_options(QecOptions {
        shots,
        seed: qec_common::SEED,
        chunk_size: None,
        keep_measurements: false,
    });
    program
}

fn memories(noise: &QecCircuitNoise) -> Vec<(String, QecProgram)> {
    let mut out = Vec::new();
    for distance in 2..=5 {
        for rounds in [1, 3] {
            out.push((
                format!("repetition d{distance} r{rounds}"),
                QecProgram::repetition_memory(distance, rounds, noise).unwrap(),
            ));
            for basis in [QecBasis::Z, QecBasis::X] {
                out.push((
                    format!("surface {basis:?} d{distance} r{rounds}"),
                    QecProgram::surface_memory(distance, rounds, basis, noise).unwrap(),
                ));
            }
        }
    }
    for distance in [3, 5, 7] {
        for rounds in [1, 3] {
            for basis in [QecBasis::Z, QecBasis::X] {
                out.push((
                    format!("color {basis:?} d{distance} r{rounds}"),
                    QecProgram::color_memory(distance, rounds, basis, noise).unwrap(),
                ));
            }
        }
    }
    out
}

#[test]
fn noiseless_detectors_and_observables_are_deterministic() {
    for (label, program) in memories(&noiseless()) {
        let result = run_qec_program(&with_shots(program, 512)).unwrap();
        for shot in 0..result.total_shots {
            for detector in 0..result.detectors.num_measurements() {
                assert!(
                    !result.detectors.get_bit(shot, detector),
                    "{label}: detector {detector} fired"
                );
            }
            assert!(!result.observables.get_bit(shot, 0), "{label}: observable");
        }
    }
}

#[test]
fn reference_oracle_agrees_on_noiseless_determinism() {
    for program in [
        QecProgram::repetition_memory(3, 2, &noiseless()).unwrap(),
        QecProgram::surface_memory(2, 2, QecBasis::X, &noiseless()).unwrap(),
        QecProgram::surface_memory(3, 1, QecBasis::Z, &noiseless()).unwrap(),
        QecProgram::color_memory(3, 2, QecBasis::Z, &noiseless()).unwrap(),
    ] {
        let result = run_qec_program_reference(&with_shots(program, 16)).unwrap();
        for shot in 0..result.total_shots {
            for detector in 0..result.detectors.num_measurements() {
                assert!(!result.detectors.get_bit(shot, detector));
            }
            assert!(!result.observables.get_bit(shot, 0));
        }
    }
}

#[test]
fn layouts_have_the_documented_shape() {
    let noise = noiseless();
    let rep = QecProgram::repetition_memory(5, 4, &noise).unwrap();
    assert_eq!(rep.num_qubits(), 9);
    assert_eq!(rep.num_detectors(), 4 * 4 + 4);
    assert_eq!(rep.num_observables(), 1);

    let surface = QecProgram::surface_memory(5, 4, QecBasis::Z, &noise).unwrap();
    assert_eq!(surface.num_qubits(), 2 * 25 - 1);
    assert_eq!(surface.num_detectors(), 12 + 3 * 24 + 12);

    for distance in [3, 5, 7, 9] {
        let data = (3 * distance * distance + 1) / 4;
        let faces = (data - 1) / 2;
        let color = QecProgram::color_memory(distance, 2, QecBasis::X, &noise).unwrap();
        assert_eq!(color.num_qubits(), data + faces);
        assert_eq!(color.num_detectors(), faces + 2 * faces + faces);
    }

    let model = surface.detector_error_model().unwrap();
    assert_eq!(model.detector_coords()[0].len(), 3);
    assert_eq!(model.detector_coords()[0][2], 0.0);
    assert_eq!(model.detector_coords().last().unwrap()[2], 4.0);
}

#[test]
fn invalid_parameters_are_rejected() {
    let noise = noiseless();
    assert!(QecProgram::repetition_memory(1, 3, &noise).is_err());
    assert!(QecProgram::repetition_memory(3, 0, &noise).is_err());
    assert!(QecProgram::surface_memory(3, 3, QecBasis::Y, &noise).is_err());
    assert!(QecProgram::color_memory(4, 3, QecBasis::Z, &noise).is_err());
    assert!(QecProgram::color_memory(1, 3, QecBasis::Z, &noise).is_err());
    let bad = QecCircuitNoise {
        after_reset_flip_probability: 1.5,
        ..noise
    };
    assert!(QecProgram::repetition_memory(3, 3, &bad).is_err());
}

/// Fewest graphlike mechanisms whose detectors cancel and whose observable flips: the
/// distance a matching decoder sees.
fn graphlike_distance(model: &DetectorErrorModel) -> usize {
    let boundary = model.num_detectors();
    let mut edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); boundary + 1];
    for mechanism in model.mechanisms() {
        let flips = mechanism.observables().contains(&0);
        match *mechanism.detectors() {
            [] if flips => return 1,
            [] => {}
            [a] => {
                edges[a].push((boundary, flips));
                edges[boundary].push((a, flips));
            }
            [a, b] => {
                edges[a].push((b, flips));
                edges[b].push((a, flips));
            }
            _ => panic!("model is not graphlike"),
        }
    }
    let mut distance = vec![[usize::MAX; 2]; boundary + 1];
    distance[boundary][0] = 0;
    let mut queue = VecDeque::from([(boundary, false)]);
    while let Some((node, parity)) = queue.pop_front() {
        let here = distance[node][usize::from(parity)];
        for &(next, flips) in &edges[node] {
            let next_parity = parity ^ flips;
            if distance[next][usize::from(next_parity)] == usize::MAX {
                distance[next][usize::from(next_parity)] = here + 1;
                queue.push_back((next, next_parity));
            }
        }
    }
    distance[boundary][1]
}

#[test]
fn repetition_and_surface_models_decompose_at_full_distance() {
    let noise = QecCircuitNoise::uniform(0.001);
    for distance in [3, 5] {
        let mut programs = vec![(
            "repetition",
            QecProgram::repetition_memory(distance, distance, &noise).unwrap(),
        )];
        for basis in [QecBasis::Z, QecBasis::X] {
            programs.push((
                "surface",
                QecProgram::surface_memory(distance, distance, basis, &noise).unwrap(),
            ));
        }
        for (label, program) in programs {
            let model = program.detector_error_model().unwrap();
            let graphlike = model.decompose_graphlike().unwrap();
            assert!(
                graphlike
                    .mechanisms()
                    .iter()
                    .all(|m| m.detectors().len() <= 2)
            );
            assert_eq!(
                graphlike_distance(&graphlike),
                distance,
                "{label} d{distance}"
            );
        }
    }
}

/// Fewest mechanisms whose detectors cancel and whose observable flips, by exhaustive
/// search up to `limit` mechanisms.
fn hypergraph_distance(model: &DetectorErrorModel, limit: usize) -> Option<usize> {
    let symptoms: Vec<(u128, bool)> = model
        .mechanisms()
        .iter()
        .map(|m| {
            let mask = m.detectors().iter().fold(0u128, |mask, &d| mask | 1 << d);
            (mask, m.observables().contains(&0))
        })
        .collect();
    fn search(
        symptoms: &[(u128, bool)],
        start: usize,
        left: usize,
        mask: u128,
        flips: bool,
    ) -> bool {
        if mask == 0 && flips {
            return true;
        }
        if left == 0 {
            return false;
        }
        (start..symptoms.len()).any(|at| {
            let (m, f) = symptoms[at];
            search(symptoms, at + 1, left - 1, mask ^ m, flips ^ f)
        })
    }
    (1..=limit).find(|&weight| search(&symptoms, 0, weight, 0, false))
}

#[test]
fn color_code_capacity_distance_is_the_code_distance() {
    let noise = QecCircuitNoise {
        before_round_data_depolarization: 0.01,
        ..noiseless()
    };
    for distance in [3, 5] {
        for basis in [QecBasis::Z, QecBasis::X] {
            let model = QecProgram::color_memory(distance, 1, basis, &noise)
                .unwrap()
                .detector_error_model()
                .unwrap();
            assert!(model.num_detectors() <= 128);
            assert_eq!(
                hypergraph_distance(&model, distance),
                Some(distance),
                "{basis:?} d{distance}"
            );
        }
    }
    let circuit_level =
        QecProgram::color_memory(5, 3, QecBasis::Z, &QecCircuitNoise::uniform(0.001))
            .unwrap()
            .detector_error_model()
            .unwrap();
    assert!(circuit_level.decompose_graphlike().is_err());
}

fn logical_error_rate(program: QecProgram, shots: usize) -> f64 {
    let model = program
        .detector_error_model()
        .unwrap()
        .decompose_graphlike()
        .unwrap();
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    let result = run_qec_program(&with_shots(program, shots)).unwrap();
    let predicted = decoder.decode_packed(&result.detectors).unwrap();
    let failures = (0..result.total_shots)
        .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
        .count();
    failures as f64 / shots as f64
}

#[test]
fn logical_error_rates_fall_with_distance_below_threshold() {
    let noise = QecCircuitNoise::uniform(0.01);
    let rates: Vec<f64> = [3, 5, 7]
        .iter()
        .map(|&d| logical_error_rate(QecProgram::repetition_memory(d, d, &noise).unwrap(), 50_000))
        .collect();
    assert!(
        rates[0] > rates[1] && rates[1] > rates[2],
        "repetition {rates:?}"
    );

    let noise = QecCircuitNoise::uniform(0.002);
    for basis in [QecBasis::Z, QecBasis::X] {
        let rates: Vec<f64> = [3, 5]
            .iter()
            .map(|&d| {
                logical_error_rate(
                    QecProgram::surface_memory(d, d, basis, &noise).unwrap(),
                    100_000,
                )
            })
            .collect();
        assert!(rates[0] > 2.0 * rates[1], "surface {basis:?} {rates:?}");
    }
}

#[test]
fn generated_programs_round_trip_through_text() {
    let noise = QecCircuitNoise::uniform(0.001);
    for (label, program) in memories(&noise) {
        let text = program.to_text().unwrap();
        let parsed = parse_qec_program(&text).unwrap();
        assert_eq!(parsed.num_qubits(), program.num_qubits(), "{label}");
        assert_eq!(parsed.ops(), program.ops(), "{label}");
    }
}

#[test]
fn generated_models_round_trip_through_dem_text() {
    let noise = QecCircuitNoise::uniform(0.001);
    for program in [
        QecProgram::repetition_memory(5, 5, &noise).unwrap(),
        QecProgram::surface_memory(5, 5, QecBasis::Z, &noise).unwrap(),
        QecProgram::color_memory(5, 3, QecBasis::X, &noise).unwrap(),
    ] {
        let model = program.detector_error_model().unwrap();
        let text = model.to_text();
        let parsed = DetectorErrorModel::from_text(&text).unwrap();
        assert_eq!(parsed.num_detectors(), model.num_detectors());
        assert_eq!(parsed.num_observables(), model.num_observables());
        assert_eq!(parsed.detector_coords(), model.detector_coords());
        assert_eq!(parsed.num_mechanisms(), model.num_mechanisms());
        for (a, b) in parsed.mechanisms().iter().zip(model.mechanisms()) {
            assert_eq!(a.probability(), b.probability());
            assert_eq!(a.detectors(), b.detectors());
            assert_eq!(a.observables(), b.observables());
        }
        assert_eq!(parsed.to_text(), text);
    }
}
