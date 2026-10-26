//! Seeded record sampling is part of the output contract: the same program, seed and
//! shot count must produce the same records, detectors and observables from one
//! release to the next, whatever path the runner takes.

use prism_q::{PackedShots, QecNoise, QecOptions, QecProgram, parse_qec_program, run_qec_program};

use crate::qec_common;

const SEED: u64 = 42;

fn fnv1a(digest: &mut u64, byte: u8) {
    *digest ^= u64::from(byte);
    *digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
}

/// Layout-independent digest of the shots, shot-major bit order.
fn digest(shots: &PackedShots) -> u64 {
    let mut digest = 0xCBF2_9CE4_8422_2325u64;
    for shot in 0..shots.num_shots() {
        let mut byte = 0u8;
        for measurement in 0..shots.num_measurements() {
            byte = (byte << 1) | u8::from(shots.get_bit(shot, measurement));
            if measurement % 8 == 7 {
                fnv1a(&mut digest, byte);
                byte = 0;
            }
        }
        fnv1a(&mut digest, byte);
    }
    digest
}

struct Fixture {
    label: &'static str,
    program: QecProgram,
    shots: usize,
    chunk_size: Option<usize>,
    keep_measurements: bool,
    expected: [u64; 3],
}

fn fixtures() -> Vec<Fixture> {
    let data: Vec<usize> = (0..9).collect();
    let wide_data: Vec<usize> = (0..25).collect();
    let random_records = parse_qec_program(
        "H 0\nX_ERROR(0.1) 0 1\nCX 0 1\nDEPOLARIZE1(0.05) 1\nM 0 1\nDETECTOR rec[-1]\n\
         OBSERVABLE_INCLUDE(0) rec[-2]",
    )
    .unwrap();
    let postselected = parse_qec_program(
        "X_ERROR(0.2) 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n\
         POSTSELECT rec[-2]",
    )
    .unwrap();
    let random_records_chunked = random_records.clone();
    let postselected_chunked = postselected.clone();
    vec![
        Fixture {
            label: "repetition d5 r5 depolarize1, kept",
            program: qec_common::repetition_memory(5, 5, QecNoise::Depolarize1(0.01), 1),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[0],
        },
        Fixture {
            label: "repetition d4 r4 depolarize2, kept",
            program: qec_common::repetition_memory(4, 4, QecNoise::Depolarize2(0.02), 1),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[1],
        },
        Fixture {
            label: "surface d3 r3 depolarize2, kept",
            program: qec_common::rotated_surface_memory(
                3,
                3,
                QecNoise::Depolarize2(0.02),
                &data[..8],
                1,
            ),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[2],
        },
        Fixture {
            label: "surface d3 r12 depolarize1, kept",
            program: qec_common::rotated_surface_memory(
                3,
                12,
                QecNoise::Depolarize1(0.01),
                &data,
                1,
            ),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[3],
        },
        Fixture {
            label: "surface d3 r3 depolarize2, kept, chunk 10007 of 20000",
            program: qec_common::rotated_surface_memory(
                3,
                3,
                QecNoise::Depolarize2(0.02),
                &data[..8],
                1,
            ),
            shots: 20_000,
            chunk_size: Some(10_007),
            keep_measurements: true,
            expected: EXPECTED[4],
        },
        Fixture {
            label: "random records, dropped",
            program: random_records.clone(),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: false,
            expected: EXPECTED[5],
        },
        Fixture {
            label: "random records, kept",
            program: random_records,
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[6],
        },
        Fixture {
            label: "postselected, dropped",
            program: postselected,
            shots: 10_000,
            chunk_size: None,
            keep_measurements: false,
            expected: EXPECTED[7],
        },
        Fixture {
            label: "surface d5 r20 depolarize1, kept",
            program: qec_common::rotated_surface_memory(
                5,
                20,
                QecNoise::Depolarize1(0.005),
                &wide_data,
                1,
            ),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[8],
        },
        Fixture {
            label: "repetition d15 r40 depolarize1, kept",
            program: qec_common::repetition_memory(15, 40, QecNoise::Depolarize1(0.005), 1),
            shots: 10_000,
            chunk_size: None,
            keep_measurements: true,
            expected: EXPECTED[9],
        },
        Fixture {
            label: "random records, kept, chunk 64 of 20000",
            program: random_records_chunked,
            shots: 20_000,
            chunk_size: Some(64),
            keep_measurements: true,
            expected: EXPECTED[10],
        },
        Fixture {
            label: "postselected, dropped, chunk 100 of 20000",
            program: postselected_chunked,
            shots: 20_000,
            chunk_size: Some(100),
            keep_measurements: false,
            expected: EXPECTED[11],
        },
    ]
}

/// `(measurements, detectors, observables)` digests per fixture, in `fixtures` order.
const EXPECTED: [[u64; 3]; 12] = [
    [0x94ebadd33fd4ab7b, 0x92755cac06099a6a, 0x257edc7a6737bbe9],
    [0x81e47d05fbcfe3e9, 0x2593f7e5054387f4, 0xddddbb9400a2aa87],
    [0x769f17e43f390d6a, 0x6677e1ae56997da3, 0x093d469cd1c2419a],
    [0xbbbd1247c98b22de, 0x0a834e9d0bfa3af7, 0x4a5fb4cb662cbcc7],
    [0x9b6118ab4eb821ed, 0xe3ca17fa67259dd5, 0xf6d8497e5d8532cd],
    [0xcbf29ce484222325, 0x95e48ebc05109443, 0xd29150f644f7b417],
    [0xae18f78f34967bff, 0x95e48ebc05109443, 0xd29150f644f7b417],
    [0xcbf29ce484222325, 0x81ce69436c37a870, 0x81ce69436c37a870],
    [0xd6d0f74308528fe0, 0xea060843a5d0a46d, 0x548ac3f6a691e9d5],
    [0xfbc47415627c7118, 0xb06ce3ffe153d5e2, 0x14b766494ff5d670],
    [0x702c1e8ac04a5b79, 0xe006cae26b354ae9, 0xee45d0e3bd2a9d87],
    [0xcbf29ce484222325, 0x236fa6dc5f67c077, 0x236fa6dc5f67c077],
];

#[test]
fn qec_seeded_records_are_stable() {
    let mut mismatches = Vec::new();
    for fixture in fixtures() {
        let mut program = fixture.program;
        program.set_options(QecOptions {
            shots: fixture.shots,
            seed: SEED,
            chunk_size: fixture.chunk_size,
            keep_measurements: fixture.keep_measurements,
            ..QecOptions::default()
        });
        let result = run_qec_program(&program).unwrap();
        let actual = [
            digest(&result.measurements),
            digest(&result.detectors),
            digest(&result.observables),
        ];
        eprintln!(
            "{}: [{:#018x}, {:#018x}, {:#018x}]",
            fixture.label, actual[0], actual[1], actual[2]
        );
        if actual != fixture.expected {
            mismatches.push(fixture.label);
        }
    }
    assert!(mismatches.is_empty(), "digests changed for {mismatches:?}");
}

#[test]
fn qec_chunked_record_runs_match_the_unchunked_run() {
    // Detectors and observables here are deterministic in the noiseless circuit, so the
    // noise units alone decide them, and those draw the same bits whatever the chunking.
    // The repetition and postselected records are deterministic too; the surface
    // memory's first X round is not, and draws per chunk.
    let data: Vec<usize> = (0..9).collect();
    let programs = [
        (
            "repetition d5 r5 depolarize1, kept",
            qec_common::repetition_memory(5, 5, QecNoise::Depolarize1(0.01), 1),
            true,
        ),
        (
            "surface d3 r3 depolarize2, kept",
            qec_common::rotated_surface_memory(3, 3, QecNoise::Depolarize2(0.02), &data[..8], 1),
            false,
        ),
        (
            "postselected, kept",
            parse_qec_program(
                "X_ERROR(0.2) 0 1
DEPOLARIZE1(0.1) 2
CX 2 0
M 0 1 2
DETECTOR rec[-1]
                 OBSERVABLE_INCLUDE(0) rec[-3]
OBSERVABLE_INCLUDE(1) rec[-1]
                 POSTSELECT rec[-2]",
            )
            .unwrap(),
            true,
        ),
    ];
    let shots = 20_000;
    for (label, mut program, records_deterministic) in programs {
        let mut run = |chunk_size: Option<usize>| {
            program.set_options(QecOptions {
                shots,
                seed: SEED,
                chunk_size,
                keep_measurements: true,
                ..QecOptions::default()
            });
            run_qec_program(&program).unwrap()
        };
        let whole = run(None);
        if label.starts_with("postselected") {
            // POSTSELECT rec[-2] keeps the shots whose middle record reads 0.
            let records = &whole.measurements;
            let accepted: Vec<usize> = (0..shots).filter(|&s| !records.get_bit(s, 1)).collect();
            let errors = [0, 2].map(|record| {
                accepted
                    .iter()
                    .filter(|&&s| records.get_bit(s, record))
                    .count() as u64
            });
            assert_eq!(whole.accepted_shots, accepted.len());
            assert_eq!(whole.logical_errors, errors);
        }
        for chunk_size in [64, 100, 8192, 10_007] {
            let chunked = run(Some(chunk_size));
            let chunk = format!("{label}, chunk {chunk_size}");
            if records_deterministic {
                assert_eq!(
                    digest(&chunked.measurements),
                    digest(&whole.measurements),
                    "{chunk}"
                );
            }
            assert_eq!(
                digest(&chunked.detectors),
                digest(&whole.detectors),
                "{chunk}"
            );
            assert_eq!(
                digest(&chunked.observables),
                digest(&whole.observables),
                "{chunk}"
            );
            assert_eq!(chunked.accepted_shots, whole.accepted_shots, "{chunk}");
            assert_eq!(chunked.logical_errors, whole.logical_errors, "{chunk}");
        }
    }
}
