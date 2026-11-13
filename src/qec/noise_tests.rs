use super::*;
use crate::qec::{QecBasis, QecCircuitNoise};

const SHOTS: usize = 20_000;

fn sample(sampler: &mut QecCompiledNoiseSampler, shots: usize) -> PackedShots {
    sampler.sample_measurements_packed(0, shots, shots).unwrap()
}

fn frame_and_compiled(program: &QecProgram) -> (PackedShots, PackedShots) {
    let mut frame = compile_qec_noisy_sampler(program).unwrap();
    let mut compiled = compile_qec_noisy_sampler_on_compiled_base(program).unwrap();
    compiled.reseed(0x5EED);
    (sample(&mut frame, SHOTS), sample(&mut compiled, SHOTS))
}

fn distribution_programs() -> Vec<(&'static str, QecProgram)> {
    let text = |text: &str| QecProgram::from_text(text).unwrap();
    let noiseless = QecCircuitNoise::uniform(0.0);
    let noisy = QecCircuitNoise::uniform(0.01);
    vec![
        ("h measure", text("H 0\nM 0")),
        ("bell", text("H 0\nCX 0 1\nM 0 1")),
        ("mpp", text("MPP X0*X1\nM 0")),
        ("x basis", text("RX 0\nMX 0")),
        ("entangled reset", text("H 0\nCX 0 1\nR 0\nM 1")),
        (
            "mpp then data",
            text("H 2\nMPP X0*X1 Z1*Z2\nCX 2 0\nM 0 1 2"),
        ),
        (
            "repetition d3 r3",
            QecProgram::repetition_memory(3, 3, &noiseless).unwrap(),
        ),
        (
            "repetition d3 r3 p0.01",
            QecProgram::repetition_memory(3, 3, &noisy).unwrap(),
        ),
        (
            "surface d3 r3 x",
            QecProgram::surface_memory(3, 3, QecBasis::X, &noiseless).unwrap(),
        ),
        (
            "surface d3 r2 z p0.01",
            QecProgram::surface_memory(3, 2, QecBasis::Z, &noisy).unwrap(),
        ),
    ]
}

/// Two-sample chi-square of full record strings, against a bound six standard
/// deviations above its degrees of freedom.
fn assert_same_strings(a: &PackedShots, b: &PackedShots, label: &str) {
    let key = |shots: &PackedShots, shot: usize| {
        (0..shots.num_measurements())
            .fold(0u64, |acc, m| acc << 1 | u64::from(shots.get_bit(shot, m)))
    };
    let mut counts: std::collections::HashMap<u64, [f64; 2]> = std::collections::HashMap::new();
    for shot in 0..a.num_shots() {
        counts.entry(key(a, shot)).or_default()[0] += 1.0;
        counts.entry(key(b, shot)).or_default()[1] += 1.0;
    }
    let chi2: f64 = counts
        .values()
        .map(|[x, y]| (x - y) * (x - y) / (x + y))
        .sum();
    let dof = counts.len().saturating_sub(1).max(1) as f64;
    assert!(
        chi2 < dof + 6.0 * (2.0 * dof).sqrt(),
        "{label}: chi2 {chi2:.1} over {dof} degrees of freedom"
    );
}

/// Every record's frequency of 1 and every pair's frequency of differing agree within
/// six standard deviations of the difference of two fair-coin estimates.
fn assert_same_pairs(a: &PackedShots, b: &PackedShots, label: &str) {
    let shots = a.num_shots();
    let m = a.num_measurements();
    let tolerance = 6.0 * (0.5 / shots as f64).sqrt();
    let frequency = |shots_of: &PackedShots, i: usize, j: Option<usize>| {
        (0..shots)
            .filter(|&shot| {
                shots_of.get_bit(shot, i) ^ j.is_some_and(|j| shots_of.get_bit(shot, j))
            })
            .count() as f64
            / shots as f64
    };
    for i in 0..m {
        for j in std::iter::once(None).chain((i + 1..m).map(Some)) {
            let (fa, fb) = (frequency(a, i, j), frequency(b, i, j));
            assert!(
                (fa - fb).abs() < tolerance,
                "{label}: records {i} {j:?} read {fa:.4} against {fb:.4}"
            );
        }
    }
}

#[test]
fn frame_records_match_the_compiled_base_in_distribution() {
    for (label, program) in distribution_programs() {
        let (frame, compiled) = frame_and_compiled(&program);
        assert_eq!(frame.num_measurements(), compiled.num_measurements());
        if program.num_measurements() <= 12 {
            assert_same_strings(&frame, &compiled, label);
        }
        assert_same_pairs(&frame, &compiled, label);
    }
}

#[test]
fn a_qubit_measured_without_a_reset_is_a_fair_coin() {
    let program = QecProgram::from_text("H 0\nM 0").unwrap();
    let mut sampler = compile_qec_noisy_sampler(&program).unwrap();
    let shots = sample(&mut sampler, SHOTS);
    let ones = (0..SHOTS).filter(|&shot| shots.get_bit(shot, 0)).count();
    assert!((ones as f64 / SHOTS as f64 - 0.5).abs() < 0.02, "{ones}");
}

#[test]
fn a_birth_the_fingerprint_drops_flips_no_record() {
    let noise = QecCircuitNoise::uniform(0.01);
    for (program, random_records) in [
        (QecProgram::repetition_memory(3, 3, &noise).unwrap(), false),
        (
            QecProgram::surface_memory(3, 3, QecBasis::Z, &noise).unwrap(),
            true,
        ),
        (
            QecProgram::from_text("H 0\nCX 0 1\nR 0\nM 1\nR 1\nMPP X0*X1\nM 0").unwrap(),
            true,
        ),
    ] {
        let deferred = lower_qec_program_to_deferred_circuit_inner(&program, false, true).unwrap();
        let mut noise = QecRecordNoise::compile(deferred, 7, true).unwrap().unwrap();
        let kept: std::collections::HashSet<u32> =
            noise.births.iter().map(|&(_, alias)| alias).collect();
        assert_eq!(!kept.is_empty(), random_records);
        let dropped: Vec<(u32, u32)> = (0..noise.deferred.num_aliases as u32)
            .filter(|alias| !kept.contains(alias))
            .map(|alias| (noise.deferred.alias_positions[alias as usize] as u32, alias))
            .collect();
        assert!(!dropped.is_empty());
        noise.births = dropped;
        let records = noise.deferred.measurement_qubits.len();
        let mut data = vec![0u64; records * 2];
        noise.apply(&mut data, 128, 0, 128, QecFrameParts::COINS);
        assert!(data.iter().all(|&word| word == 0));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn frame_records_do_not_depend_on_the_thread_count() {
    let run = |threads: usize, program: &QecProgram| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut sampler = compile_qec_noisy_sampler(program).unwrap();
            sample(&mut sampler, 100_000)
        })
    };
    for p in [0.0, 0.01] {
        let program =
            QecProgram::surface_memory(3, 3, QecBasis::Z, &QecCircuitNoise::uniform(p)).unwrap();
        let one = run(1, &program);
        let four = run(4, &program);
        assert_eq!(one.raw_data(), four.raw_data(), "p {p}");
    }
}
