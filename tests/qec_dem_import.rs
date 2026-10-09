//! Detector error model text import: grammar coverage, `repeat` and shift expansion,
//! `^` decomposition suggestions, and the export, import, compare round trip.

use prism_q::{DetectorErrorModel, QecNoise, UnionFindDecoder, parse_qec_program};

mod qec_common;

#[test]
fn reads_the_grammar() {
    let model = DetectorErrorModel::from_text(
        "# leading comment
         error(0.125) D0 D1 ^ D2 L0   # trailing comment
         ERROR[tagged](0.25) D1 D1 D3
         detector(1, 2, 0) D0
         repeat 3 {
             error(0.01) D0 L1
             detector(0, 0, 1) D0
             shift_detectors(0, 0, 1) 1
         }
         detector D9
         logical_observable L4",
    )
    .unwrap();
    assert_eq!(model.num_detectors(), 13);
    assert_eq!(model.num_observables(), 5);
    assert_eq!(model.num_mechanisms(), 5);

    let first = &model.mechanisms()[0];
    assert_eq!(first.probability(), 0.125);
    assert_eq!(first.detectors(), [0, 1, 2]);
    assert_eq!(first.observables(), [0]);
    assert_eq!(
        first.suggested_decomposition(),
        [(vec![0, 1], vec![]), (vec![2], vec![0])]
    );

    let second = &model.mechanisms()[1];
    assert_eq!(second.detectors(), [3]);
    assert!(second.suggested_decomposition().is_empty());

    for (k, mechanism) in model.mechanisms()[2..].iter().enumerate() {
        assert_eq!(mechanism.detectors(), [k]);
        assert_eq!(mechanism.observables(), [1]);
    }
    let coords = model.detector_coords();
    assert_eq!(coords[0], [0.0, 0.0, 1.0]);
    assert_eq!(coords[1], [0.0, 0.0, 2.0]);
    assert_eq!(coords[2], [0.0, 0.0, 3.0]);
    assert!(coords[3].is_empty());
    assert!(coords[12].is_empty());
}

#[test]
fn nested_repeats_expand_in_order() {
    let model = DetectorErrorModel::from_text(
        "repeat 2 {
             repeat 3 {
                 error(0.1) D0
                 shift_detectors 1
             }
             shift_detectors 10
         }",
    )
    .unwrap();
    let detectors: Vec<usize> = model
        .mechanisms()
        .iter()
        .map(|m| m.detectors()[0])
        .collect();
    assert_eq!(detectors, [0, 1, 2, 13, 14, 15]);
    assert_eq!(model.num_detectors(), 16);
}

#[test]
fn rejects_malformed_text() {
    for text in [
        "error(1.5) D0",
        "error(0.1, 0.2) D0",
        "error D0",
        "error(0.1) X0",
        "error(0.1) D0 ^",
        "error(0.1) ^ D0",
        "detector(1, x) D0",
        "detector L0",
        "logical_observable D0",
        "detector_separator 1",
        "repeat 2 {\n error(0.1) D0",
        "}",
        "repeat {\n}",
        "error(0.1 D0",
        "repeat 4294967296 {\n repeat 4294967296 {\n error(0.1) D0\n }\n}",
    ] {
        assert!(
            DetectorErrorModel::from_text(text).is_err(),
            "accepted `{text}`"
        );
    }
}

fn derived_models() -> Vec<DetectorErrorModel> {
    let data: Vec<usize> = (0..9).collect();
    vec![
        qec_common::repetition_memory(4, 4, QecNoise::Depolarize2(0.02), 1)
            .detector_error_model()
            .unwrap(),
        qec_common::rotated_surface_memory(3, 3, QecNoise::Depolarize1(0.01), &data, 1)
            .detector_error_model()
            .unwrap(),
        qec_common::surface_memory_d3(2, QecNoise::Depolarize2(0.01), &data[..8], 1)
            .detector_error_model()
            .unwrap(),
    ]
}

#[test]
fn export_import_compare() {
    for model in derived_models() {
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
        assert_eq!(
            parsed.decompose_graphlike().unwrap().mechanisms().len(),
            model.decompose_graphlike().unwrap().mechanisms().len()
        );
    }
}

#[test]
fn hyperedges_export_their_decomposition() {
    let program =
        parse_qec_program("DEPOLARIZE2(0.1) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nX_ERROR(0.2) 2\nCX 2 3\nM 2 3\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-1] rec[-3]")
            .unwrap();
    let model = program.detector_error_model().unwrap();
    let text = model.to_text();
    let hyper: Vec<&str> = text.lines().filter(|line| line.contains('^')).collect();
    assert!(!hyper.is_empty(), "{text}");
    for line in &hyper {
        let components: Vec<&str> = line.split('^').collect();
        assert!(components.len() >= 2);
    }
    let parsed = DetectorErrorModel::from_text(&text).unwrap();
    assert_eq!(
        parsed.decompose_graphlike().unwrap(),
        DetectorErrorModel::from_text(&model.decompose_graphlike().unwrap().to_text()).unwrap()
    );
}

#[test]
fn suggestions_drive_decomposition() {
    let model = DetectorErrorModel::from_text(
        "error(0.1) D0 D1
         error(0.2) D0 D1 ^ D2 D3",
    )
    .unwrap();
    let graphlike = model.decompose_graphlike().unwrap();
    let mechanisms = graphlike.mechanisms();
    assert_eq!(mechanisms.len(), 2);
    assert_eq!(mechanisms[0].detectors(), [0, 1]);
    assert!((mechanisms[0].probability() - (0.1 * 0.8 + 0.2 * 0.9)).abs() < 1e-15);
    assert_eq!(mechanisms[1].detectors(), [2, 3]);
    assert_eq!(mechanisms[1].probability(), 0.2);
}

#[test]
fn imported_models_decode_like_the_originals() {
    let program = qec_common::repetition_memory(5, 5, QecNoise::Depolarize1(0.03), 4096);
    let original = program
        .detector_error_model()
        .unwrap()
        .decompose_graphlike()
        .unwrap();
    let imported = DetectorErrorModel::from_text(&original.to_text()).unwrap();
    let samples = prism_q::run_qec_program(&program).unwrap();
    let a = UnionFindDecoder::from_model(&original)
        .unwrap()
        .decode_packed(&samples.detectors)
        .unwrap();
    let b = UnionFindDecoder::from_model(&imported)
        .unwrap()
        .decode_packed(&samples.detectors)
        .unwrap();
    for shot in 0..samples.total_shots {
        assert_eq!(a.get_bit(shot, 0), b.get_bit(shot, 0));
    }
}
