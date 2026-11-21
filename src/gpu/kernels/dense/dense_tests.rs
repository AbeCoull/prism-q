use super::tiles::{QftPass, qft_pass_plan};
use super::{Route2q, route_2q};
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::gates::Gate;
use crate::gpu::GpuState;
use crate::sim;
use crate::test_support::gpu_context_or_skip;
use num_complex::Complex64;

#[test]
fn route_2q_reads_exact_structure_only() {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    let phase = Complex64::from_polar(1.0, 0.4);
    assert_eq!(
        route_2q(&Gate::Cx.matrix_4x4()),
        Route2q::Cx { control_high: true }
    );
    assert_eq!(
        route_2q(&[[o, z, z, z], [z, z, z, o], [z, z, o, z], [z, o, z, z]]),
        Route2q::Cx {
            control_high: false
        }
    );
    assert_eq!(route_2q(&Gate::Swap.matrix_4x4()), Route2q::Swap);
    assert_eq!(route_2q(&Gate::Cz.matrix_4x4()), Route2q::Cz);
    assert_eq!(
        route_2q(&[[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, phase]]),
        Route2q::ControlledPhase(phase)
    );
    assert_eq!(
        route_2q(&[[z, o, z, z], [phase, z, z, z], [z, z, o, z], [z, z, z, o]]),
        Route2q::Monomial {
            src: [1, 0, 2, 3],
            scale: [o, phase, o, o]
        }
    );
    let mut near = Gate::Cx.matrix_4x4();
    near[0][1] = Complex64::new(1e-300, 0.0);
    assert_eq!(route_2q(&near), Route2q::Dense);
    let mut singular = Gate::Swap.matrix_4x4();
    singular[3][3] = z;
    assert_eq!(route_2q(&singular), Route2q::Dense);
    let mut repeated = Gate::Swap.matrix_4x4();
    repeated[0] = [z, o, z, z];
    assert_eq!(route_2q(&repeated), Route2q::Dense);
}

#[test]
fn qft_pass_plan_splits_high_stages_evenly_above_the_tile() {
    let pass = |lo, bits| QftPass { lo, bits };
    assert_eq!(qft_pass_plan(0, 10), [pass(0, 10)]);
    assert_eq!(qft_pass_plan(0, 20), [pass(11, 9), pass(0, 11)]);
    assert_eq!(
        qft_pass_plan(0, 28),
        [pass(19, 9), pass(11, 8), pass(0, 11)]
    );
    assert_eq!(qft_pass_plan(1, 12), [pass(10, 2), pass(0, 10)]);
    assert_eq!(qft_pass_plan(5, 17), [pass(9, 8), pass(0, 9)]);
}

#[test]
fn plan_walk_assigns_thresholds_to_the_block_spanning_them() {
    use super::sample::{WalkPlan, plan_walk};
    let sums = [0.25, 0.0, 0.5, 0.25];
    let sorted = [0.0, 0.125, 0.25, 0.625, 0.75, 0.875];
    assert_eq!(
        plan_walk(&sums, &sorted),
        Some(WalkPlan {
            jobs: vec![0, 0, 2, 2, 2, 4, 3, 4, 6],
            offsets: vec![0.0, 0.125, 0.0, 0.375, 0.0, 0.125],
        })
    );
}

#[test]
fn plan_walk_sends_thresholds_at_the_total_to_the_last_weighted_block() {
    use super::sample::plan_walk;
    let tail = plan_walk(&[0.5, 0.5, 0.0], &[0.25, 1.0]).unwrap();
    assert_eq!(tail.jobs, vec![0, 0, 1, 1, 1, 2]);
    assert_eq!(tail.offsets, vec![0.25, f64::INFINITY]);

    let extended = plan_walk(&[0.5, 0.5, 0.0], &[0.75, 1.0]).unwrap();
    assert_eq!(extended.jobs, vec![1, 0, 2]);
    assert_eq!(extended.offsets, vec![0.25, f64::INFINITY]);
}

#[test]
fn plan_walk_declines_empty_or_broken_weights() {
    use super::sample::plan_walk;
    assert_eq!(plan_walk(&[0.0, 0.0], &[0.5]), None);
    assert_eq!(plan_walk(&[0.5, f64::NAN], &[0.5]), None);
    assert_eq!(plan_walk(&[0.5, f64::INFINITY], &[0.5]), None);
}

#[test]
fn plan_walk_splits_a_crowded_block_across_jobs() {
    use super::sample::plan_walk;
    let sorted: Vec<f64> = (0..9000).map(|i| i as f64 / 9000.0).collect();
    let plan = plan_walk(&[0.0, 1.0], &sorted).unwrap();
    assert_eq!(plan.jobs, vec![1, 0, 4096, 1, 4096, 8192, 1, 8192, 9000]);
}

// First index whose running weight exceeds `u * total`, else the last weighted one.
fn sequential_walk(weights: &[f64], sorted: &[f64]) -> Vec<u32> {
    let total: f64 = weights.iter().sum();
    let last = weights.iter().rposition(|&w| w > 0.0).unwrap();
    let mut out = Vec::with_capacity(sorted.len());
    let mut acc = 0.0;
    let mut i = 0;
    for &u in sorted {
        let v = u * total;
        while i < weights.len() && acc + weights[i] <= v {
            acc += weights[i];
            i += 1;
        }
        out.push(i.min(last) as u32);
    }
    out
}

fn weights_of(amps: &[Complex64]) -> Vec<f64> {
    amps.iter().map(|a| a.norm_sqr()).collect()
}

// Dyadic weights summing to one keep every running sum exact, so the blocked device
// scan and a sequential one agree on every threshold, boundaries included. The
// layouts cover leading and trailing zero runs, a whole zero block, one and two
// blocks, and a lone basis state.
#[test]
fn device_walk_matches_a_sequential_cdf_on_dyadic_states() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    for n in [12usize, 13, 14] {
        let dim = 1usize << n;
        let spread: Vec<usize> = (0..40).map(|k| 3 + k * (dim / 41)).collect();
        let mut layouts = vec![spread.clone(), vec![dim - 1]];
        if n == 14 {
            let shifted = spread
                .iter()
                .map(|&i| {
                    if (4096..8192).contains(&i) {
                        i + 4096
                    } else {
                        i
                    }
                })
                .collect();
            layouts.push(shifted);
        }
        for indices in layouts {
            let mut amps = vec![Complex64::new(0.0, 0.0); dim];
            if indices.len() == 1 {
                amps[indices[0]] = Complex64::new(1.0, 0.0);
            } else {
                for (k, &i) in indices.iter().enumerate() {
                    amps[i] = if k % 5 == 0 {
                        Complex64::new(0.0, 0.25)
                    } else {
                        Complex64::new(0.125, 0.0)
                    };
                }
            }
            let weights = weights_of(&amps);
            let mut sorted = vec![0.0, 1.0 - f64::EPSILON / 2.0, 0.3, 0.7];
            let mut acc = 0.0;
            for &w in &weights {
                if w > 0.0 {
                    sorted.extend([acc, acc + w / 2.0, acc + w - 2f64.powi(-40)]);
                    acc += w;
                }
            }
            sorted.retain(|&u| (0.0..1.0).contains(&u));
            sorted.sort_unstable_by(f64::total_cmp);

            let state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();
            let device = super::sample_sorted(&ctx, &state, &sorted)
                .unwrap()
                .unwrap();
            assert_eq!(
                device,
                sequential_walk(&weights, &sorted),
                "n={n}, {} weighted indices",
                indices.len()
            );
        }
    }
}

// One off dyadic weight at the start of scattered runs of 16 leaves zero weights
// right after each run and warp boundary, where the tree order of the scan rounds
// differently from the serial running sum. Thresholds a few ulps either side of every
// running sum must still land on a weighted index inside the register.
#[test]
fn device_walk_never_draws_a_zero_weight_at_scan_boundaries() {
    use rand::RngExt;
    use rand::SeedableRng;
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for n in [8usize, 10, 12, 14] {
        let dim = 1usize << n;
        for _ in 0..8 {
            let mut amps = vec![Complex64::new(0.0, 0.0); dim];
            for run in (0..dim).step_by(16) {
                if rng.random::<f64>() < 0.6 {
                    amps[run] = Complex64::new(rng.random(), rng.random());
                }
            }
            let weights = weights_of(&amps);
            let total: f64 = weights.iter().sum();
            let mut sorted = Vec::new();
            let mut acc = 0.0;
            for &w in &weights {
                if w > 0.0 {
                    acc += w;
                    let u = acc / total;
                    for step in -4i64..=4 {
                        let near = f64::from_bits((u.to_bits() as i64 + step) as u64);
                        if (0.0..1.0).contains(&near) {
                            sorted.push(near);
                        }
                    }
                }
            }
            sorted.sort_unstable_by(f64::total_cmp);
            let state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();
            let device = super::sample_sorted(&ctx, &state, &sorted)
                .unwrap()
                .unwrap();
            for (&i, &u) in device.iter().zip(&sorted) {
                let i = i as usize;
                assert!(i < dim && weights[i] > 0.0, "n={n} u={u} drew {i}");
            }
        }
    }
}

#[test]
fn device_walk_declines_a_nan_weight() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let mut amps = vec![Complex64::new(0.0, 0.0); 1 << 14];
    amps[0] = Complex64::new(1.0, 0.0);
    amps[9000] = Complex64::new(f64::NAN, 0.0);
    let state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();
    let walked = super::sample_sorted(&ctx, &state, &[0.5]).unwrap();
    assert_eq!(walked, None);
}

// Off dyadic weights round differently in the blocked scan, so a threshold within
// rounding of a boundary can land one weighted index over. A block offset or range
// error would move most thresholds instead.
#[test]
fn device_walk_tracks_a_sequential_cdf_off_dyadic() {
    use rand::RngExt;
    use rand::SeedableRng;
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let n = 16;
    let mut gpu = StatevectorBackend::new(42).with_gpu(ctx.clone());
    sim::run_on(&mut gpu, &crate::circuits::random_circuit(n, 6, 42)).unwrap();
    let weights = weights_of(&gpu.export_statevector().unwrap());
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut sorted: Vec<f64> = (0..200_000).map(|_| rng.random()).collect();
    sorted.sort_unstable_by(f64::total_cmp);

    let device = gpu.sample_sorted_on_device(&sorted).unwrap().unwrap();
    let host = sequential_walk(&weights, &sorted);
    let rank: Vec<usize> = weights
        .iter()
        .scan(0usize, |seen, &w| {
            *seen += usize::from(w > 0.0);
            Some(*seen)
        })
        .collect();
    let mut moved = 0;
    for (&d, &h) in device.iter().zip(&host) {
        if d != h {
            moved += 1;
            assert!(weights[d as usize] > 0.0, "drew zero weight index {d}");
            assert_eq!(
                rank[d as usize].abs_diff(rank[h as usize]),
                1,
                "device {d} host {h}"
            );
        }
    }
    assert!(
        moved * 1000 < sorted.len(),
        "{moved} of {} moved",
        sorted.len()
    );
}

#[cfg(feature = "bench-internal")]
#[test]
fn device_walk_reads_block_totals_and_indices_only() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let mut gpu = StatevectorBackend::new(42).with_gpu(ctx.clone());
    sim::run_on(&mut gpu, &crate::circuits::random_circuit(20, 2, 42)).unwrap();
    let sorted: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
    let before = ctx.pass_census();
    gpu.sample_sorted_on_device(&sorted).unwrap().unwrap();
    let census = ctx.pass_census() - before;
    assert_eq!((census.launches, census.drains), (2, 2));
}
