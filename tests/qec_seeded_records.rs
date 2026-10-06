//! Seeded record sampling is part of the output contract: the same program, seed and
//! shot count must produce the same records, detectors and observables from one
//! release to the next, whatever path the runner takes.

use prism_q::{PackedShots, QecNoise, QecOptions, QecProgram, parse_qec_program, run_qec_program};

mod qec_common;

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
    ]
}

/// `(measurements, detectors, observables)` digests per fixture, in `fixtures` order.
const EXPECTED: [[u64; 3]; 10] = [
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
