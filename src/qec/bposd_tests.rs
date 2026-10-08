use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;

const N: usize = 15;
const CHECKS: usize = 8;

/// Parity checks of the cyclic [15, 7, 5] BCH code, one bitmask over the 15
/// code bits per row: shifts of the reciprocal check polynomial
/// `h(x) = (x^15 + 1) / g(x)`, `g(x) = 1 + x^4 + x^6 + x^7 + x^8`.
fn bch_checks() -> Vec<u32> {
    let g = 0b1_1101_0001u32;
    let mut remainder = (1u32 << 15) | 1;
    let mut h = 0u32;
    for shift in (0..=7).rev() {
        if remainder >> (shift + 8) & 1 == 1 {
            remainder ^= g << shift;
            h |= 1 << shift;
        }
    }
    assert_eq!(remainder, 0, "g divides x^15 + 1");
    let h_reversed = (0..8).fold(0u32, |acc, bit| acc | (h >> bit & 1) << (7 - bit));
    let checks: Vec<u32> = (0..CHECKS).map(|row| h_reversed << row).collect();
    for row in 0..7 {
        let codeword = g << row;
        for &check in &checks {
            assert_eq!((codeword & check).count_ones() % 2, 0, "G H^T = 0");
        }
    }
    checks
}

fn syndrome_of(checks: &[u32], error: u32) -> u64 {
    checks.iter().enumerate().fold(0u64, |acc, (row, &check)| {
        acc | u64::from((check & error).count_ones() % 2) << row
    })
}

/// One column per code bit, each flipping observable `j` so the prediction is
/// the decoded error itself.
fn bch_decoder(priors: &[f64], options: BpOsdOptions) -> BpOsdDecoder {
    let checks = bch_checks();
    let detectors: Vec<Vec<usize>> = (0..N)
        .map(|bit| {
            (0..CHECKS)
                .filter(|&row| checks[row] >> bit & 1 == 1)
                .collect()
        })
        .collect();
    let observables: Vec<[usize; 1]> = (0..N).map(|bit| [bit]).collect();
    let columns: Vec<(&[usize], f64, &[usize])> = (0..N)
        .map(|bit| {
            (
                detectors[bit].as_slice(),
                priors[bit],
                observables[bit].as_slice(),
            )
        })
        .collect();
    BpOsdDecoder::from_columns(CHECKS, N, &columns, options)
}

fn all_options() -> Vec<BpOsdOptions> {
    let mut options = Vec::new();
    for bp_method in [BpMethod::ProductSum, BpMethod::MinSum { scaling: 0.625 }] {
        for osd_method in [
            OsdMethod::Zero,
            OsdMethod::CombinationSweep { order: 7 },
            OsdMethod::Exhaustive { order: 7 },
        ] {
            for max_iterations in [0, 30] {
                options.push(BpOsdOptions {
                    max_iterations,
                    bp_method,
                    osd_method,
                });
            }
        }
    }
    options
}

fn decode(decoder: &BpOsdDecoder, scratch: &mut BpOsdScratch, syndrome: u64) -> (u32, f64) {
    let mut out = [0u64; 1];
    let weight = decoder
        .solve_shot(&[syndrome], &mut out, scratch)
        .unwrap_or_else(|_| panic!("syndrome {syndrome:b} is in the column space"));
    (out[0] as u32, weight)
}

#[test]
fn bch_code_has_distance_five_and_full_rank() {
    let checks = bch_checks();
    let lightest = (1u32..1 << N)
        .filter(|&error| syndrome_of(&checks, error) == 0)
        .map(u32::count_ones)
        .min();
    assert_eq!(lightest, Some(5));
    let decoder = bch_decoder(&[0.05; N], BpOsdOptions::default());
    assert_eq!(decoder.rank, CHECKS);
}

#[test]
fn bposd_recovers_every_weight_one_and_two_error() {
    let checks = bch_checks();
    // OSD-0 on uniform priors has no reliability order to work from, so it
    // needs belief propagation; every sweep covers all weight-2 patterns.
    for options in all_options()
        .into_iter()
        .filter(|o| o.max_iterations > 0 || o.osd_method != OsdMethod::Zero)
    {
        let decoder = bch_decoder(&[0.05; N], options);
        let mut scratch = decoder.scratch();
        for a in 0..N {
            for b in a..N {
                let error = (1u32 << a) | (1u32 << b);
                let (decoded, _) = decode(&decoder, &mut scratch, syndrome_of(&checks, error));
                assert_eq!(decoded, error, "{options:?}: error {error:015b}");
            }
        }
    }
}

#[test]
fn exhaustive_osd_equals_brute_force_maximum_likelihood() {
    let checks = bch_checks();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..8 {
        let priors: Vec<f64> = (0..N).map(|_| rng.random_range(0.01..0.3)).collect();
        let weight_of = |error: u32| -> f64 {
            (0..N)
                .filter(|&bit| error >> bit & 1 == 1)
                .map(|bit| ((1.0 - priors[bit]) / priors[bit]).ln())
                .sum()
        };
        let mut best = vec![f64::INFINITY; 1 << CHECKS];
        for error in 0u32..1 << N {
            let syndrome = syndrome_of(&checks, error) as usize;
            best[syndrome] = best[syndrome].min(weight_of(error));
        }
        for bp_method in [BpMethod::ProductSum, BpMethod::MinSum { scaling: 0.625 }] {
            let exhaustive = bch_decoder(
                &priors,
                BpOsdOptions {
                    max_iterations: 30,
                    bp_method,
                    osd_method: OsdMethod::Exhaustive { order: N - CHECKS },
                },
            );
            let sweep = bch_decoder(
                &priors,
                BpOsdOptions {
                    max_iterations: 30,
                    bp_method,
                    osd_method: OsdMethod::CombinationSweep { order: 7 },
                },
            );
            let osd0 = bch_decoder(
                &priors,
                BpOsdOptions {
                    max_iterations: 30,
                    bp_method,
                    osd_method: OsdMethod::Zero,
                },
            );
            let mut scratch = exhaustive.scratch();
            for (syndrome, &optimum) in best.iter().enumerate() {
                let syndrome = syndrome as u64;
                let (error, weight) = decode(&exhaustive, &mut scratch, syndrome);
                assert_eq!(syndrome_of(&checks, error), syndrome);
                // A converged BP decision skips OSD and may be heavier than the optimum.
                if !converges(&exhaustive, &mut scratch, syndrome) {
                    assert!(
                        (weight - optimum).abs() < 1e-9,
                        "syndrome {syndrome:b}: {weight} vs ML {optimum}"
                    );
                }
                let (sweep_error, sweep_weight) = decode(&sweep, &mut scratch, syndrome);
                let (_, zero_weight) = decode(&osd0, &mut scratch, syndrome);
                assert_eq!(syndrome_of(&checks, sweep_error), syndrome);
                assert!(sweep_weight >= optimum - 1e-9);
                assert!(sweep_weight <= zero_weight + 1e-9);
            }
        }
    }
}

fn converges(decoder: &BpOsdDecoder, scratch: &mut BpOsdScratch, syndrome: u64) -> bool {
    for (d, bit) in scratch.syndrome.iter_mut().enumerate() {
        *bit = (syndrome >> d) as u8 & 1;
    }
    decoder.belief_propagation(scratch)
}

#[test]
fn bposd_rejects_syndromes_outside_the_column_space() {
    // Two checks over one column: only the all-zero and all-one syndromes exist.
    let detectors = [0usize, 1];
    let columns: Vec<(&[usize], f64, &[usize])> = vec![(&detectors, 0.1, &[])];
    let decoder = BpOsdDecoder::from_columns(2, 0, &columns, BpOsdOptions::default());
    let mut scratch = decoder.scratch();
    assert!(decoder.solve_shot(&[0b11], &mut [], &mut scratch).is_ok());
    assert!(decoder.solve_shot(&[0b01], &mut [], &mut scratch).is_err());
}
