//! In-crate decoders (union-find, minimum-weight perfect matching, BP+OSD):
//! construction errors, hand-walked corrections, exact bands against the
//! model's ML rate by full syndrome enumeration, head-to-head failure counts on
//! shared samples, and distance thresholds at fixed seed.

mod qec_common;

use prism_q::{
    BpMethod, BpOsdDecoder, BpOsdOptions, DetectorErrorModel, MatchingDecoder, OsdMethod,
    PackedShots, QecNoise, QecPauli, QecProgram, QecRecordRef, QecSampleResult, ShotLayout,
    UnionFindDecoder, run_qec_program,
};

const STAT_SHOTS: usize = 20_000;

// Fixed-seed golden values from the first passing run; a decoder change that
// shifts any of them is a loud regression signal.
const GOLDEN_REPETITION_D3_ANALYTIC_UF: f64 = 0.00971417964172128;
const GOLDEN_SURFACE_D3_ANALYTIC_UF: f64 = 0.03503902060385806;
const GOLDEN_REPETITION_D3_DECODE_FAILURES: usize = 27;
const GOLDEN_REPETITION_D5_DECODE_FAILURES: usize = 1;
const GOLDEN_SURFACE_D3_DECODE_FAILURES: usize = 603;

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

// The batch surface shared by the three decoders, so the exact-rate machinery
// runs any of them.
trait Decode {
    fn detectors(&self) -> usize;
    fn decode(&self, shots: &PackedShots) -> prism_q::Result<PackedShots>;
}

impl Decode for UnionFindDecoder {
    fn detectors(&self) -> usize {
        self.num_detectors()
    }
    fn decode(&self, shots: &PackedShots) -> prism_q::Result<PackedShots> {
        self.decode_packed(shots)
    }
}

impl Decode for MatchingDecoder {
    fn detectors(&self) -> usize {
        self.num_detectors()
    }
    fn decode(&self, shots: &PackedShots) -> prism_q::Result<PackedShots> {
        self.decode_packed(shots)
    }
}

impl Decode for BpOsdDecoder {
    fn detectors(&self) -> usize {
        self.num_detectors()
    }
    fn decode(&self, shots: &PackedShots) -> prism_q::Result<PackedShots> {
        self.decode_packed(shots)
    }
}

// Bulk-decode bit-packed syndromes (at most 64 detectors) and return the
// predicted flip of observable 0 per syndrome.
fn decode_syndromes(decoder: &impl Decode, syndromes: &[usize]) -> Vec<bool> {
    let num_detectors = decoder.detectors();
    assert!(num_detectors <= 64);
    let data: Vec<u64> = syndromes.iter().map(|&s| s as u64).collect();
    let packed = PackedShots::from_shot_major(data, syndromes.len(), num_detectors);
    let decoded = decoder.decode(&packed).unwrap();
    (0..syndromes.len())
        .map(|shot| decoded.get_bit(shot, 0))
        .collect()
}

struct ExactRates {
    ml: f64,
    decoded: f64,
    at_least_two_faults: f64,
    single_fault_miss_mass: f64,
}

// Exact logical error rates under the model, by enumerating every reachable
// syndrome. `at_least_two_faults` plus `single_fault_miss_mass` bounds the
// union-find rate from above: zero faults give an empty syndrome (decoded to
// no flip), so a failure needs two simultaneous faults or a single-fault
// syndrome the decoder maps to the wrong class, and the latter mass is
// measured by decoding each mechanism's own syndrome.
fn exact_rates(model: &DetectorErrorModel, decoder: &impl Decode) -> ExactRates {
    let dist = joint_distribution(model);
    let observable_bit = 1usize << model.num_detectors();
    let feasible: Vec<usize> = (0..observable_bit)
        .filter(|&syndrome| dist[syndrome] > 0.0 || dist[syndrome | observable_bit] > 0.0)
        .collect();
    let predictions = decode_syndromes(decoder, &feasible);
    let mut ml = 0.0;
    let mut decoded = 0.0;
    for (&syndrome, &flip) in feasible.iter().zip(&predictions) {
        let quiet = dist[syndrome];
        let flipped = dist[syndrome | observable_bit];
        ml += quiet.min(flipped);
        decoded += if flip { quiet } else { flipped };
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
        decoded,
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
    assert!((rates.decoded - 0.1).abs() < 1e-12);
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
        rates.ml <= rates.decoded + 1e-12,
        "ML is per-syndrome optimal: {} vs {}",
        rates.ml,
        rates.decoded
    );
    assert!(
        rates.decoded <= rates.at_least_two_faults + rates.single_fault_miss_mass + 1e-12,
        "union-find fails only on multi-fault shots and measured single-fault misses: \
         {} vs {} + {}",
        rates.decoded,
        rates.at_least_two_faults,
        rates.single_fault_miss_mass
    );
    assert_eq!(
        rates.single_fault_miss_mass, 0.0,
        "every repetition single fault decodes to its own class"
    );
    assert!(
        rates.decoded < p,
        "analytic decoded rate must beat physical p"
    );
    assert!(
        (rates.decoded - GOLDEN_REPETITION_D3_ANALYTIC_UF).abs() < 1e-12,
        "analytic union-find rate drifted: {:.17}",
        rates.decoded
    );

    // Sampled agreement: 5 sigma of the analytic rate, plus slack for the
    // second-order gap between independent mechanisms and exclusive branches.
    let result = run_qec_program(&program).unwrap();
    let predicted = decoder.decode_packed(&result.detectors).unwrap();
    let empirical = prediction_mismatches(&predicted, &result) as f64 / result.total_shots as f64;
    let sigma = (rates.decoded * (1.0 - rates.decoded) / result.total_shots as f64).sqrt();
    assert!(
        (empirical - rates.decoded).abs() < 5.0 * sigma + 0.005,
        "sampled union-find rate {empirical:.5} vs analytic {:.5}",
        rates.decoded
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
        rates.ml <= rates.decoded + 1e-12,
        "ML is per-syndrome optimal: {} vs {}",
        rates.ml,
        rates.decoded
    );
    assert!(
        rates.decoded <= rates.at_least_two_faults + rates.single_fault_miss_mass + 1e-12,
        "union-find fails only on multi-fault shots and measured single-fault misses: \
         {} vs {} + {}",
        rates.decoded,
        rates.at_least_two_faults,
        rates.single_fault_miss_mass
    );
    assert!(
        (rates.decoded - GOLDEN_SURFACE_D3_ANALYTIC_UF).abs() < 1e-12,
        "analytic union-find rate drifted: {:.17}",
        rates.decoded
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
    let mut program = qec_common::repetition_memory(3, 3, QecNoise::Depolarize1(0.05), STAT_SHOTS);
    // Kept records route through the shot-major record path.
    program.set_options(qec_common::qec_options(STAT_SHOTS, 4096, true));
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

fn sampled_failures(decoder: &impl Decode, result: &QecSampleResult) -> usize {
    prediction_mismatches(&decoder.decode(&result.detectors).unwrap(), result)
}

fn repetition_model(distance: usize, rounds: usize, p: f64) -> (QecProgram, DetectorErrorModel) {
    let program =
        qec_common::repetition_memory(distance, rounds, QecNoise::Depolarize1(p), STAT_SHOTS);
    let model = program.detector_error_model().unwrap();
    (program, model)
}

fn surface_d3_model(p: f64) -> (QecProgram, DetectorErrorModel) {
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
    (program, model)
}

fn rotated_surface_model(distance: usize, p: f64) -> (QecProgram, DetectorErrorModel) {
    let data: Vec<usize> = (0..distance * distance).collect();
    let program = qec_common::rotated_surface_memory(
        distance,
        distance,
        QecNoise::Depolarize1(p),
        &data,
        STAT_SHOTS,
    );
    let model = program
        .detector_error_model()
        .unwrap()
        .decompose_graphlike()
        .unwrap();
    (program, model)
}

#[test]
fn matching_rejects_hypergraph_models_and_bad_inputs() {
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    for _ in 0..3 {
        let record = program.measure_pauli_product(&[QecPauli::z(0)]).unwrap();
        program.detector(&[QecRecordRef::absolute(record)]).unwrap();
    }
    let model = program.detector_error_model().unwrap();
    let err = MatchingDecoder::from_model(&model).unwrap_err().to_string();
    assert!(err.contains("D0 D1 D2"), "names the symptom: {err}");
    assert!(err.contains("matching decoding"), "{err}");
    assert!(err.contains("decompose_graphlike"), "{err}");

    let (_, model) = repetition_model(3, 1, 0.05);
    let decoder = MatchingDecoder::from_model(&model).unwrap();
    let shots = PackedShots::from_shot_major(vec![0u64; 2], 2, 3);
    let err = decoder.decode_packed(&shots).unwrap_err().to_string();
    assert!(err.contains("4 detectors"), "{err}");
}

#[test]
fn matching_rejects_impossible_syndromes() {
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    for _ in 0..2 {
        let record = program.measure_pauli_product(&[QecPauli::z(0)]).unwrap();
        program.detector(&[QecRecordRef::absolute(record)]).unwrap();
    }
    let model = program.detector_error_model().unwrap();
    let decoder = MatchingDecoder::from_model(&model).unwrap();
    let possible = PackedShots::from_shot_major(vec![0b00, 0b11], 2, 2);
    assert_eq!(decoder.decode_packed(&possible).unwrap().num_shots(), 2);
    let impossible = PackedShots::from_shot_major(vec![0b11, 0b01], 2, 2);
    let err = decoder.decode_packed(&impossible).unwrap_err().to_string();
    assert!(err.contains("impossible"), "{err}");
    assert!(err.contains("shot 1"), "{err}");
}

#[test]
fn matching_hand_walked_corrections_on_repetition_d3_r1() {
    let (_, model) = repetition_model(3, 1, 0.09);
    let decoder = MatchingDecoder::from_model(&model).unwrap();
    let predictions = decode_syndromes(&decoder, &[0b00, 0b01, 0b10, 0b11]);
    assert_eq!(predictions, vec![false, true, false, false]);
}

#[test]
fn matching_exact_rate_sits_between_ml_and_union_find() {
    for (label, (_, model)) in [
        ("repetition d3", repetition_model(3, 3, 0.05)),
        ("surface d3", surface_d3_model(0.02)),
    ] {
        let union_find = exact_rates(&model, &UnionFindDecoder::from_model(&model).unwrap());
        let matching = exact_rates(&model, &MatchingDecoder::from_model(&model).unwrap());
        assert!(
            matching.ml <= matching.decoded + 1e-12,
            "{label}: ML {} vs matching {}",
            matching.ml,
            matching.decoded
        );
        assert!(
            matching.decoded <= union_find.decoded + 1e-12,
            "{label}: matching {} vs union-find {}",
            matching.decoded,
            union_find.decoded
        );
        if label.starts_with("repetition") {
            assert_eq!(matching.single_fault_miss_mass, 0.0, "{label}");
        }
    }
}

#[test]
fn matching_never_decodes_worse_than_union_find_on_shared_samples() {
    let fixtures = [
        ("repetition d3", repetition_model(3, 3, 0.05)),
        ("repetition d5", repetition_model(5, 5, 0.05)),
        ("repetition d7", repetition_model(7, 7, 0.05)),
        ("surface d3", rotated_surface_model(3, 0.03)),
        ("surface d5", rotated_surface_model(5, 0.03)),
    ];
    for (label, (program, model)) in fixtures {
        let result = run_qec_program(&program).unwrap();
        let union_find = sampled_failures(&UnionFindDecoder::from_model(&model).unwrap(), &result);
        let matching = sampled_failures(&MatchingDecoder::from_model(&model).unwrap(), &result);
        assert!(
            matching <= union_find,
            "{label}: matching {matching} vs union-find {union_find} failures"
        );
    }
}

#[test]
fn matching_logical_error_rate_falls_with_distance() {
    let p = 0.05;
    let mut rates = Vec::new();
    for distance in [3, 5, 7] {
        let (program, model) = repetition_model(distance, distance, p);
        let decoder = MatchingDecoder::from_model(&model).unwrap();
        let result = run_qec_program(&program).unwrap();
        let rate = decoder
            .logical_error_rate(&result.detectors, &result.observables)
            .unwrap();
        assert_eq!(
            rate,
            sampled_failures(&decoder, &result) as f64 / result.total_shots as f64
        );
        assert!(rate < p, "d{distance} decoded rate {rate:.5} must beat {p}");
        rates.push(rate);
    }
    assert!(
        rates[0] > rates[1] && rates[1] > rates[2],
        "below threshold the rate falls with distance: {rates:?}"
    );

    let mut surface = Vec::new();
    for distance in [3, 5] {
        let (program, model) = rotated_surface_model(distance, 0.01);
        let decoder = MatchingDecoder::from_model(&model).unwrap();
        let result = run_qec_program(&program).unwrap();
        surface.push(
            decoder
                .logical_error_rate(&result.detectors, &result.observables)
                .unwrap(),
        );
    }
    assert!(surface[1] < surface[0], "surface rates {surface:?}");
}

#[test]
fn decoders_agree_across_layouts_and_batch_paths() {
    let (program, model) = rotated_surface_model(3, 0.03);
    let result = run_qec_program(&program).unwrap();
    let detectors = model.num_detectors();
    let matching = MatchingDecoder::from_model(&model).unwrap();
    let bposd = BpOsdDecoder::from_model(&model).unwrap();
    let words = detectors.div_ceil(64);
    for head_shots in [32, 512] {
        let mut rows = vec![0u64; head_shots * words];
        for shot in 0..head_shots {
            for d in 0..detectors {
                if result.detectors.get_bit(shot, d) {
                    rows[shot * words + d / 64] |= 1 << (d % 64);
                }
            }
        }
        let head = PackedShots::from_shot_major(rows, head_shots, detectors);
        let mut columns = vec![0u64; detectors * head_shots.div_ceil(64)];
        for shot in 0..head_shots {
            for d in 0..detectors {
                if head.get_bit(shot, d) {
                    columns[d * head_shots.div_ceil(64) + shot / 64] |= 1 << (shot % 64);
                }
            }
        }
        let meas_major = PackedShots::from_meas_major(columns, head_shots, detectors);
        let full_matching = matching.decode_packed(&result.detectors).unwrap();
        let full_bposd = bposd.decode_packed(&result.detectors).unwrap();
        assert_eq!(
            matching.decode_packed(&head).unwrap().raw_data(),
            &full_matching.raw_data()[..head_shots]
        );
        assert_eq!(
            matching.decode_packed(&meas_major).unwrap().raw_data(),
            &full_matching.raw_data()[..head_shots]
        );
        assert_eq!(
            bposd.decode_packed(&head).unwrap().raw_data(),
            &full_bposd.raw_data()[..head_shots]
        );
        assert_eq!(
            bposd.decode_packed(&meas_major).unwrap().raw_data(),
            &full_bposd.raw_data()[..head_shots]
        );
    }
}

fn bposd_options(osd_method: OsdMethod) -> BpOsdOptions {
    BpOsdOptions {
        osd_method,
        ..BpOsdOptions::default()
    }
}

#[test]
fn bposd_never_decodes_worse_than_union_find_on_graphlike_models() {
    for (label, (_, model)) in [
        ("repetition d3", repetition_model(3, 3, 0.05)),
        ("surface d3", surface_d3_model(0.02)),
    ] {
        let union_find = UnionFindDecoder::from_model(&model).unwrap();
        let bposd = BpOsdDecoder::from_model(&model).unwrap();
        let uf_rates = exact_rates(&model, &union_find);
        let bposd_rates = exact_rates(&model, &bposd);
        assert!(bposd_rates.ml <= bposd_rates.decoded + 1e-12, "{label}");
        assert!(
            bposd_rates.decoded <= uf_rates.decoded + 1e-12,
            "{label}: BP+OSD {} vs union-find {}",
            bposd_rates.decoded,
            uf_rates.decoded
        );
    }
    // Sampled comparisons use single-qubit noise, where the decomposed model is
    // the sampler's exact distribution; see the correlated case below.
    for (label, (program, model)) in [
        ("repetition d3", repetition_model(3, 3, 0.05)),
        ("surface d3", rotated_surface_model(3, 0.03)),
        ("surface d5", rotated_surface_model(5, 0.03)),
        ("repetition d7", repetition_model(7, 7, 0.05)),
    ] {
        let result = run_qec_program(&program).unwrap();
        let uf_failures = sampled_failures(&UnionFindDecoder::from_model(&model).unwrap(), &result);
        let bposd_failures = sampled_failures(&BpOsdDecoder::from_model(&model).unwrap(), &result);
        assert!(
            bposd_failures <= uf_failures,
            "{label}: BP+OSD {bposd_failures} vs union-find {uf_failures} failures"
        );
    }
}

#[test]
fn bposd_decodes_the_hypergraph_color_code_near_ml() {
    let p = 0.01;
    let program = qec_common::color_code_memory_d3(3, QecNoise::Depolarize1(p), STAT_SHOTS);
    let model = program.detector_error_model().unwrap();
    assert!(
        model.mechanisms().iter().any(|m| m.detectors().len() > 2),
        "a weight-3 Z plaquette syndrome makes the model a hypergraph"
    );
    assert!(UnionFindDecoder::from_model(&model).is_err());
    assert!(MatchingDecoder::from_model(&model).is_err());

    let result = run_qec_program(&program).unwrap();
    for options in [
        BpOsdOptions::default(),
        bposd_options(OsdMethod::Zero),
        BpOsdOptions {
            bp_method: BpMethod::ProductSum,
            ..BpOsdOptions::default()
        },
    ] {
        let decoder = BpOsdDecoder::with_options(&model, options).unwrap();
        let rates = exact_rates(&model, &decoder);
        assert!(rates.ml <= rates.decoded + 1e-12, "{options:?}");
        assert_eq!(
            rates.single_fault_miss_mass, 0.0,
            "{options:?}: every single fault decodes to its own class"
        );
        assert!(
            rates.decoded <= 1.5 * rates.ml,
            "{options:?}: BP+OSD {} vs ML {}",
            rates.decoded,
            rates.ml
        );
        let empirical = decoder
            .logical_error_rate(&result.detectors, &result.observables)
            .unwrap();
        let sigma = (rates.decoded * (1.0 - rates.decoded) / result.total_shots as f64).sqrt();
        assert!(
            (empirical - rates.decoded).abs() < 5.0 * sigma + 0.002,
            "{options:?}: sampled {empirical:.5} vs analytic {:.5}",
            rates.decoded
        );
        assert!(
            empirical < p,
            "{options:?}: decoded rate {empirical} must beat {p}"
        );
    }
}

#[test]
fn bposd_on_the_undecomposed_model_beats_graph_decoders_on_correlated_noise() {
    // Two-qubit depolarizing noise makes hyperedges that `decompose_graphlike`
    // splits, losing their correlations: on these samples exact matching over
    // the decomposed model fails more often than union-find, and BP+OSD over
    // the full model fails least.
    let (program, graphlike) = surface_d3_model(0.02);
    let full = program.detector_error_model().unwrap();
    let result = run_qec_program(&program).unwrap();
    let union_find = sampled_failures(&UnionFindDecoder::from_model(&graphlike).unwrap(), &result);
    let matching = sampled_failures(&MatchingDecoder::from_model(&graphlike).unwrap(), &result);
    let bposd = sampled_failures(&BpOsdDecoder::from_model(&full).unwrap(), &result);
    assert_eq!(union_find, GOLDEN_SURFACE_D3_DECODE_FAILURES);
    assert!(
        bposd < union_find && bposd < matching,
        "BP+OSD {bposd}, union-find {union_find}, matching {matching}"
    );
}
