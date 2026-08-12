//! Union-find decoder: construction errors, hand-walked corrections, exact
//! bands against the model's ML rate by full syndrome enumeration, and the
//! distance threshold on repetition memory at fixed seed.

mod qec_common;

use prism_q::{
    DetectorErrorModel, PackedShots, QecNoise, QecPauli, QecProgram, QecRecordRef, QecSampleResult,
    ShotLayout, UnionFindDecoder, run_qec_program,
};

const STAT_SHOTS: usize = 20_000;

// Fixed-seed golden values from the first passing run; a decoder change that
// shifts any of them is a loud regression signal.
const GOLDEN_REPETITION_D3_ANALYTIC_UF: f64 = 0.00971417964172128;
const GOLDEN_SURFACE_D3_ANALYTIC_UF: f64 = 0.03503902060385806;
const GOLDEN_REPETITION_D3_DECODE_FAILURES: usize = 23;
const GOLDEN_REPETITION_D5_DECODE_FAILURES: usize = 2;
const GOLDEN_SURFACE_D3_DECODE_FAILURES: usize = 639;

// Exact joint distribution over (syndrome, observable 0) under the model:
// XOR-convolve one two-point distribution per mechanism. Syndrome bits are
// LSB-first by detector index, the observable sits in bit `num_detectors`.
fn joint_distribution(model: &DetectorErrorModel) -> Vec<f64> {
    let num_detectors = model.num_detectors();
    let states = 1usize << (num_detectors + 1);
    let mut dist = vec![0.0f64; states];
    dist[0] = 1.0;
    for mechanism in model.mechanisms() {
        let mut flip = 0usize;
        for &detector in mechanism.detectors() {
            flip |= 1 << detector;
        }
        for &observable in mechanism.observables() {
            assert_eq!(observable, 0);
            flip |= 1 << num_detectors;
        }
        let q = mechanism.probability();
        let mut next = vec![0.0f64; states];
        for (state, &mass) in dist.iter().enumerate() {
            next[state] += mass * (1.0 - q);
            next[state ^ flip] += mass * q;
        }
        dist = next;
    }
    dist
}

// Bulk-decode bit-packed syndromes (at most 64 detectors) and return the
// predicted flip of observable 0 per syndrome.
fn decode_syndromes(decoder: &UnionFindDecoder, syndromes: &[usize]) -> Vec<bool> {
    let num_detectors = decoder.num_detectors();
    assert!(num_detectors <= 64);
    let data: Vec<u64> = syndromes.iter().map(|&s| s as u64).collect();
    let packed = PackedShots::from_shot_major(data, syndromes.len(), num_detectors);
    let decoded = decoder.decode_packed(&packed).unwrap();
    (0..syndromes.len())
        .map(|shot| decoded.get_bit(shot, 0))
        .collect()
}

struct ExactRates {
    ml: f64,
    uf: f64,
    at_least_two_faults: f64,
    single_fault_miss_mass: f64,
}

// Exact logical error rates under the model, by enumerating every reachable
// syndrome. `at_least_two_faults` plus `single_fault_miss_mass` bounds the
// union-find rate from above: zero faults give an empty syndrome (decoded to
// no flip), so a failure needs two simultaneous faults or a single-fault
// syndrome the decoder maps to the wrong class, and the latter mass is
// measured by decoding each mechanism's own syndrome.
fn exact_rates(model: &DetectorErrorModel, decoder: &UnionFindDecoder) -> ExactRates {
    let dist = joint_distribution(model);
    let observable_bit = 1usize << model.num_detectors();
    let feasible: Vec<usize> = (0..observable_bit)
        .filter(|&syndrome| dist[syndrome] > 0.0 || dist[syndrome | observable_bit] > 0.0)
        .collect();
    let predictions = decode_syndromes(decoder, &feasible);
    let mut ml = 0.0;
    let mut uf = 0.0;
    for (&syndrome, &flip) in feasible.iter().zip(&predictions) {
        let quiet = dist[syndrome];
        let flipped = dist[syndrome | observable_bit];
        ml += quiet.min(flipped);
        uf += if flip { quiet } else { flipped };
    }

    let none: f64 = model
        .mechanisms()
        .iter()
        .map(|m| 1.0 - m.probability())
        .product();
    let one: f64 = model
        .mechanisms()
        .iter()
        .map(|m| m.probability() / (1.0 - m.probability()))
        .sum::<f64>()
        * none;
    let at_least_two_faults = 1.0 - none - one;

    let singles: Vec<usize> = model
        .mechanisms()
        .iter()
        .map(|m| m.detectors().iter().fold(0usize, |acc, &d| acc | 1 << d))
        .collect();
    let single_predictions = decode_syndromes(decoder, &singles);
    let mut single_fault_miss_mass = 0.0;
    for (mechanism, &flip) in model.mechanisms().iter().zip(&single_predictions) {
        if flip != mechanism.observables().contains(&0) {
            single_fault_miss_mass += mechanism.probability();
        }
    }

    ExactRates {
        ml,
        uf,
        at_least_two_faults,
        single_fault_miss_mass,
    }
}

fn prediction_mismatches(predicted: &PackedShots, result: &QecSampleResult) -> usize {
    (0..result.total_shots)
        .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
        .count()
}

#[test]
fn decoder_rejects_hypergraph_models() {
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    for _ in 0..3 {
        let record = program.measure_pauli_product(&[QecPauli::z(0)]).unwrap();
        program.detector(&[QecRecordRef::absolute(record)]).unwrap();
    }
    let model = program.detector_error_model().unwrap();
    let err = UnionFindDecoder::from_model(&model)
        .unwrap_err()
        .to_string();
    assert!(err.contains("D0 D1 D2"), "names the symptom: {err}");
    assert!(
        err.contains("decompose_graphlike"),
        "points at the fix: {err}"
    );
}

#[test]
fn decoder_rejects_detector_count_mismatch() {
    let program = qec_common::repetition_memory(3, 1, QecNoise::Depolarize1(0.05), 16);
    let model = program.detector_error_model().unwrap();
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    let shots = PackedShots::from_shot_major(vec![0u64; 2], 2, 3);
    let err = decoder.decode_packed(&shots).unwrap_err().to_string();
    assert!(err.contains("4 detectors"), "{err}");
}

#[test]
fn decoder_rejects_impossible_syndromes() {
    // One mechanism flipping two detectors: a lone defect has no boundary
    // edge to absorb its odd parity.
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    for _ in 0..2 {
        let record = program.measure_pauli_product(&[QecPauli::z(0)]).unwrap();
        program.detector(&[QecRecordRef::absolute(record)]).unwrap();
    }
    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_mechanisms(), 1);
    let decoder = UnionFindDecoder::from_model(&model).unwrap();

    let possible = PackedShots::from_shot_major(vec![0b00, 0b11], 2, 2);
    let decoded = decoder.decode_packed(&possible).unwrap();
    assert_eq!(decoded.num_shots(), 2);
    assert_eq!(decoded.num_measurements(), 0);

    let impossible = PackedShots::from_shot_major(vec![0b01], 1, 2);
    let err = decoder.decode_packed(&impossible).unwrap_err().to_string();
    assert!(err.contains("impossible"), "{err}");
    assert!(err.contains("shot 0"), "{err}");
}

#[test]
fn decoder_hand_walked_corrections_on_repetition_d3_r1() {
    // Mechanisms pinned by `dem_depolarize1_merges_exclusive_branches`:
    // {D0 L0}, {D0 D1}, {D1}, each at 2p/3. Equal weights, so a lone D0
    // defect resolves to the boundary edge carrying L0, a lone D1 defect to
    // its own boundary edge, and the D0 D1 pair to the internal edge.
    let program = qec_common::repetition_memory(3, 1, QecNoise::Depolarize1(0.09), 16);
    let model = program.detector_error_model().unwrap();
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    let predictions = decode_syndromes(&decoder, &[0b00, 0b01, 0b10, 0b11]);
    assert_eq!(predictions, vec![false, true, false, false]);
}

#[test]
fn decoder_cannot_predict_detector_free_mechanisms() {
    // The X error on qubit 0 flips only the observable. It cannot enter the
    // decoding graph, and its mass is the exact floor for any decoder over
    // this model: ML itself predicts no flip on every syndrome.
    let program = QecProgram::from_text(
        "X_ERROR(0.1) 0
         X_ERROR(0.05) 1
         M 0
         M 1
         OBSERVABLE_INCLUDE(0) rec[-2]
         DETECTOR rec[-1]",
    )
    .unwrap();
    let model = program.detector_error_model().unwrap();
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    assert_eq!(decode_syndromes(&decoder, &[0b0, 0b1]), vec![false, false]);
    let rates = exact_rates(&model, &decoder);
    assert!((rates.ml - 0.1).abs() < 1e-12);
    assert!((rates.uf - 0.1).abs() < 1e-12);
}

#[test]
fn decoder_matches_exact_ml_on_repetition_d3() {
    let p = 0.05;
    let program = qec_common::repetition_memory(3, 3, QecNoise::Depolarize1(p), STAT_SHOTS);
    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_detectors(), 8);
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    let rates = exact_rates(&model, &decoder);

    assert!(
        rates.ml <= rates.uf + 1e-12,
        "ML is per-syndrome optimal: {} vs {}",
        rates.ml,
        rates.uf
    );
    assert!(
        rates.uf <= rates.at_least_two_faults + rates.single_fault_miss_mass + 1e-12,
        "union-find fails only on multi-fault shots and measured single-fault misses: \
         {} vs {} + {}",
        rates.uf,
        rates.at_least_two_faults,
        rates.single_fault_miss_mass
    );
    assert_eq!(
        rates.single_fault_miss_mass, 0.0,
        "every repetition single fault decodes to its own class"
    );
    assert!(rates.uf < p, "analytic decoded rate must beat physical p");
    assert!(
        (rates.uf - GOLDEN_REPETITION_D3_ANALYTIC_UF).abs() < 1e-12,
        "analytic union-find rate drifted: {:.17}",
        rates.uf
    );

    // Sampled agreement: 5 sigma of the analytic rate, plus slack for the
    // second-order gap between independent mechanisms and exclusive branches.
    let result = run_qec_program(&program).unwrap();
    let predicted = decoder.decode_packed(&result.detectors).unwrap();
    let empirical = prediction_mismatches(&predicted, &result) as f64 / result.total_shots as f64;
    let sigma = (rates.uf * (1.0 - rates.uf) / result.total_shots as f64).sqrt();
    assert!(
        (empirical - rates.uf).abs() < 5.0 * sigma + 0.005,
        "sampled union-find rate {empirical:.5} vs analytic {:.5}",
        rates.uf
    );
    assert!(empirical < p, "decode must beat the physical error rate");
}

#[test]
fn decoder_matches_exact_ml_on_surface_d3() {
    let p = 0.02;
    let program = qec_common::surface_memory_d3(
        2,
        QecNoise::Depolarize2(p),
        &[0, 1, 2, 3, 4, 5, 6, 7],
        STAT_SHOTS,
    );
    let model = program
        .detector_error_model()
        .unwrap()
        .decompose_graphlike()
        .unwrap();
    assert_eq!(model.num_detectors(), 16);
    let decoder = UnionFindDecoder::from_model(&model).unwrap();
    let rates = exact_rates(&model, &decoder);

    assert!(
        rates.ml <= rates.uf + 1e-12,
        "ML is per-syndrome optimal: {} vs {}",
        rates.ml,
        rates.uf
    );
    assert!(
        rates.uf <= rates.at_least_two_faults + rates.single_fault_miss_mass + 1e-12,
        "union-find fails only on multi-fault shots and measured single-fault misses: \
         {} vs {} + {}",
        rates.uf,
        rates.at_least_two_faults,
        rates.single_fault_miss_mass
    );
    assert!(
        (rates.uf - GOLDEN_SURFACE_D3_ANALYTIC_UF).abs() < 1e-12,
        "analytic union-find rate drifted: {:.17}",
        rates.uf
    );

    // Correlated two-qubit faults at distance 3 leave the decoded rate above
    // the per-pair noise rate; the exact relations above and the fixed-seed
    // golden carry the claim instead of a threshold statement.
    let result = run_qec_program(&program).unwrap();
    let predicted = decoder.decode_packed(&result.detectors).unwrap();
    let failures = prediction_mismatches(&predicted, &result);
    assert_eq!(failures, GOLDEN_SURFACE_D3_DECODE_FAILURES);
}

#[test]
fn decoder_logical_error_rate_falls_with_distance() {
    let p = 0.02;
    let mut rates = Vec::new();
    for (distance, golden) in [
        (3, GOLDEN_REPETITION_D3_DECODE_FAILURES),
        (5, GOLDEN_REPETITION_D5_DECODE_FAILURES),
    ] {
        let program =
            qec_common::repetition_memory(distance, 3, QecNoise::Depolarize1(p), STAT_SHOTS);
        let model = program.detector_error_model().unwrap();
        let decoder = UnionFindDecoder::from_model(&model).unwrap();
        let result = run_qec_program(&program).unwrap();
        let predicted = decoder.decode_packed(&result.detectors).unwrap();
        let failures = prediction_mismatches(&predicted, &result);
        assert_eq!(failures, golden, "d{distance} fixed-seed decode failures");
        let rate = failures as f64 / result.total_shots as f64;
        assert!(
            rate < p,
            "d{distance} decoded rate {rate:.5} must beat physical {p}"
        );
        rates.push(rate);
    }
    assert!(
        rates[1] < rates[0],
        "logical error rate must fall with distance: {rates:?}"
    );
}

#[test]
fn decoder_layouts_and_parallelism_agree() {
    let program = qec_common::repetition_memory(3, 3, QecNoise::Depolarize1(0.05), STAT_SHOTS);
    let model = program.detector_error_model().unwrap();
    let decoder = UnionFindDecoder::from_model(&model).unwrap();

    // Feasible syndromes touch bits 0..6 only; detectors 6 and 7 belong to
    // no mechanism in this fixture.
    let syndromes: Vec<usize> = (0..64).collect();
    let shot_major = PackedShots::from_shot_major(
        syndromes.iter().map(|&s| s as u64).collect(),
        syndromes.len(),
        8,
    );
    let mut columns = vec![0u64; 8];
    for (shot, &syndrome) in syndromes.iter().enumerate() {
        for (detector, column) in columns.iter_mut().enumerate() {
            if syndrome >> detector & 1 == 1 {
                *column |= 1u64 << shot;
            }
        }
    }
    let meas_major = PackedShots::from_meas_major(columns, syndromes.len(), 8);
    let from_shot_major = decoder.decode_packed(&shot_major).unwrap();
    let from_meas_major = decoder.decode_packed(&meas_major).unwrap();
    assert_eq!(from_shot_major.raw_data(), from_meas_major.raw_data());

    // The parallel bulk path and the serial path agree row for row.
    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.detectors.layout(), ShotLayout::ShotMajor);
    let full = decoder.decode_packed(&result.detectors).unwrap();
    let again = decoder.decode_packed(&result.detectors).unwrap();
    assert_eq!(full.raw_data(), again.raw_data());
    let head = PackedShots::from_shot_major(result.detectors.raw_data()[..512].to_vec(), 512, 8);
    let head_decoded = decoder.decode_packed(&head).unwrap();
    assert_eq!(head_decoded.raw_data(), &full.raw_data()[..512]);
}
