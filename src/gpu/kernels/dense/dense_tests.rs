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

// 2^14 amplitudes in four blocks read back 131,072 bytes; 10,919 shots and the four block
// totals move 131,060.
#[test]
fn device_walk_floor_counts_bytes_moved() {
    use super::walk_moves_fewer_bytes;
    assert!(walk_moves_fewer_bytes(14, 10_919));
    assert!(!walk_moves_fewer_bytes(14, 10_920));
    assert!(!walk_moves_fewer_bytes(14, 1_000_000));
    assert!(walk_moves_fewer_bytes(28, 1_000_000));
}

// One shot either side of the floor at 14 qubits: below it the route runs the two
// sampling kernels and reads back twice, above it one probability kernel and one readback.
#[cfg(feature = "bench-internal")]
#[test]
fn device_walk_yields_to_the_probability_readback_past_the_floor() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let n = 14;
    let mut circuit = crate::circuit::Circuit::new(n, n);
    circuit
        .instructions
        .extend(crate::circuits::hardware_efficient_ansatz(n, 1, 42).instructions);
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    let census = |shots: usize| {
        let before = ctx.pass_census();
        let counts = sim::simulate(&circuit)
            .gpu(ctx.clone())
            .seed(42)
            .sample_counts(shots)
            .unwrap()
            .counts;
        assert_eq!(counts.values().sum::<u64>(), shots as u64);
        ctx.pass_census() - before
    };
    let walked = census(10_919);
    let read_back = census(10_920);
    assert_eq!(
        (walked.launches, walked.drains),
        (read_back.launches + 1, read_back.drains + 1)
    );
}

fn asymmetric_complex_state(num_qubits: usize, seed: u64) -> Vec<Complex64> {
    use rand::RngExt;
    use rand::SeedableRng;
    let dim = 1usize << num_qubits;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut amps: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.37, rng.random::<f64>() - 0.61))
        .collect();
    let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut amps {
        *a /= norm;
    }
    amps
}

fn cpu_pauli_sums(amps: &[Complex64], pending_norm: f64, masks: &[(u64, u64)]) -> Vec<Complex64> {
    let scale = pending_norm * pending_norm;
    let triples: Vec<(usize, usize, u32)> = masks
        .iter()
        .map(|&(x, z)| (x as usize, z as usize, 0))
        .collect();
    crate::sim::pauli_sandwiches_from_masks(amps, amps, &triples)
        .into_iter()
        .map(|s| s * scale)
        .collect()
}

#[test]
fn pauli_blocks_per_mask_caps_blocks_and_partial_bytes() {
    use super::expect::{PAULI_MAX_BLOCKS, PAULI_MAX_PARTIAL_BYTES, pauli_blocks_per_mask};
    assert_eq!(pauli_blocks_per_mask(1, 1), 1);
    assert_eq!(pauli_blocks_per_mask(2, 1), 1);
    assert_eq!(pauli_blocks_per_mask(1 << 30, 1), PAULI_MAX_BLOCKS as u32);
    assert_eq!(pauli_blocks_per_mask(1 << 30, 0), PAULI_MAX_BLOCKS as u32);

    let many_masks = 2048;
    let capped = pauli_blocks_per_mask(1 << 30, many_masks);
    assert_eq!(capped, 512);
    assert!(2 * many_masks * capped as usize * size_of::<f64>() <= PAULI_MAX_PARTIAL_BYTES);

    assert_eq!(pauli_blocks_per_mask(1 << 30, 600_000), 1);
}

#[test]
fn pauli_sums_empty_and_zero_state() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let zero_amps = vec![Complex64::ZERO; 1 << 3];
    let state = GpuState::from_host_amplitudes(ctx.clone(), &zero_amps).unwrap();
    assert!(super::pauli_sums(&ctx, &state, &[]).unwrap().is_empty());

    let masks = [(0, 0), (0, 0b101), (0b011, 0b010), (0b111, 0b101)];
    let got = super::pauli_sums(&ctx, &state, &masks).unwrap();
    assert_eq!(got, vec![Complex64::ZERO; masks.len()]);
}

#[test]
fn pauli_sums_small_registers_0q_1q_2q_exhaustive() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };

    // 0-qubit register (dim = 1): x == 0 fast path with non-unit pending_norm.
    let mut state0 = GpuState::new(ctx.clone(), 0).unwrap();
    let amp0 = [Complex64::new(0.6, -0.8)];
    state0
        .buffer_mut()
        .copy_from_host(ctx.device(), &[amp0[0].re, amp0[0].im])
        .unwrap();
    state0.set_pending_norm(1.25);
    let got0 = super::pauli_sums(&ctx, &state0, &[(0, 0)]).unwrap();
    let want0 = cpu_pauli_sums(&amp0, 1.25, &[(0, 0)]);
    assert!(
        (got0[0] - want0[0]).norm() < 1e-12,
        "0q: {got0:?} vs {want0:?}"
    );
    assert_eq!(got0[0].im, 0.0);

    // 1-qubit register (dim = 2, dim / 2 = 1 pair): exhaustive (x, z).
    let amps1 = asymmetric_complex_state(1, 11);
    let mut state1 = GpuState::from_host_amplitudes(ctx.clone(), &amps1).unwrap();
    state1.set_pending_norm(1.75);
    let masks1 = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let got1 = super::pauli_sums(&ctx, &state1, &masks1).unwrap();
    let want1 = cpu_pauli_sums(&amps1, 1.75, &masks1);
    for (i, ((&(x, z), g), w)) in masks1.iter().zip(&got1).zip(&want1).enumerate() {
        assert!((g - w).norm() < 1e-12, "1q mask {i} ({x},{z}): {g} vs {w}");
        if (x & z).count_ones() & 1 == 0 {
            assert!(g.im.abs() < 1e-14, "1q even-Y mask ({x},{z}) im={}", g.im);
        } else {
            assert!(g.re.abs() < 1e-14, "1q odd-Y mask ({x},{z}) re={}", g.re);
        }
    }

    // 2-qubit register (dim = 4, dim / 2 = 2): exhaustive 16 masks.
    let amps2 = asymmetric_complex_state(2, 22);
    let mut state2 = GpuState::from_host_amplitudes(ctx.clone(), &amps2).unwrap();
    state2.set_pending_norm(0.6);
    let masks2: Vec<(u64, u64)> = (0..4).flat_map(|x| (0..4).map(move |z| (x, z))).collect();
    let got2 = super::pauli_sums(&ctx, &state2, &masks2).unwrap();
    let want2 = cpu_pauli_sums(&amps2, 0.6, &masks2);
    for ((&(x, z), g), w) in masks2.iter().zip(&got2).zip(&want2) {
        assert!((g - w).norm() < 1e-12, "2q mask ({x},{z}): {g} vs {w}");
    }
}

#[test]
fn pauli_sums_tz_edge_positions_even_and_odd_y() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let n = 10;
    let amps = asymmetric_complex_state(n, 33);
    let mut state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();
    state.set_pending_norm(1.3);

    let msb = 1u64 << (n - 1);
    let masks = [
        // x == 0 fast path
        (0, 0),
        (0, 1),
        (0, msb),
        (0, (1 << n) - 1),
        // tz(x) == 0 (LSB) with 0, 1, 2, 3 Ys
        (1, 0),
        (1, 1),
        (0b10_1011, 0b10_0001),
        (0b10_1011, 0b10_1001),
        // tz(x) middle with 0, 1, 2, 3, 4 Ys
        (1 << 5, 0),
        (1 << 5, 1 << 5),
        ((1 << 8) | (1 << 4), (1 << 8) | (1 << 4) | 1),
        (
            (1 << 8) | (1 << 4) | (1 << 2),
            (1 << 8) | (1 << 4) | (1 << 2),
        ),
        (0b11_1100_0000, 0b11_1100_0000),
        // tz(x) == n - 1 (MSB) with even and odd Y
        (msb, 0),
        (msb, msb),
        (msb, msb | 1),
        (msb, msb | (1 << 3) | (1 << 5)),
    ];
    let got = super::pauli_sums(&ctx, &state, &masks).unwrap();
    let want = cpu_pauli_sums(&amps, 1.3, &masks);
    for (i, ((&(x, z), g), w)) in masks.iter().zip(&got).zip(&want).enumerate() {
        assert!(
            (g - w).norm() < 1e-12,
            "mask {i} ({x:#x},{z:#x}): {g} vs {w}"
        );
        if (x & z).count_ones() & 1 == 0 {
            assert!(g.im.abs() < 1e-13, "even-Y mask {i} im={}", g.im);
        } else {
            assert!(g.re.abs() < 1e-13, "odd-Y mask {i} re={}", g.re);
        }
    }
}

#[test]
fn pauli_sums_multi_mask_groups_and_slot_ordering() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let n = 6;
    let amps = asymmetric_complex_state(n, 44);
    let state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();

    let mut ordered = Vec::new();
    for k in 0..17u64 {
        ordered.push((0, (k * 3 + 1) & 63));
    }
    for k in 0..17u64 {
        ordered.push((0b01_0110, (k * 5 + 2) & 63));
    }
    for k in 0..9u64 {
        ordered.push((1 << 5, (k * 7) & 63));
    }
    // Interleave deterministically so groups split across unsorted input slots.
    let masks: Vec<(u64, u64)> = (0..ordered.len())
        .map(|i| ordered[(i * 17 + 5) % ordered.len()])
        .collect();
    let got = super::pauli_sums(&ctx, &state, &masks).unwrap();
    let want = cpu_pauli_sums(&amps, 1.0, &masks);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).norm() < 1e-12, "slot {i}: {g} vs {w}");
    }
}

#[test]
fn pauli_sums_grid_stride_24q_and_bounded_vram() {
    let Some(ctx) = gpu_context_or_skip() else {
        return;
    };
    let n = 24;
    let amps = asymmetric_complex_state(n, 55);
    let mut state = GpuState::from_host_amplitudes(ctx.clone(), &amps).unwrap();
    state.set_pending_norm(1.15);

    let mut masks = vec![
        (0, 0),
        (0, 1 | (1 << 23)),
        (1, 0),
        (1, 1),
        ((1 << 12) | (1 << 15), 1 << 15),
        ((1 << 12) | (1 << 15), (1 << 12) | (1 << 15) | (1 << 20)),
        (1 << 23, 0),
        (1 << 23, 1 << 23),
    ];
    let shared_x = (1u64 << 3) | (1u64 << 19);
    for k in 0..9u64 {
        masks.push((shared_x, (k << 3) | (1 << 19)));
    }

    let got = super::pauli_sums(&ctx, &state, &masks).unwrap();
    let partials_len = ctx
        .launcher_scratch()
        .measure_partials
        .as_ref()
        .expect("measure_partials allocated")
        .len();
    let max_len = 2 * masks.len() * super::expect::PAULI_MAX_BLOCKS as usize;
    assert!(
        partials_len <= max_len,
        "partials_len {partials_len} exceeded bounded cap {max_len}"
    );

    let want = cpu_pauli_sums(&amps, 1.15, &masks);
    for (i, ((&(x, z), g), w)) in masks.iter().zip(&got).zip(&want).enumerate() {
        assert!(
            (g - w).norm() < 1e-9,
            "24q mask {i} ({x:#x},{z:#x}): {g} vs {w}"
        );
    }
}
