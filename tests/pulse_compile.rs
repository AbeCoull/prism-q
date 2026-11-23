//! Pulse compile-down: closed-form and independently integrated references for the
//! rotating-frame model, and the hand-off of its unitary to the statevector.

#![cfg(feature = "unstable-pulse")]

mod common;

use std::f64::consts::{PI, TAU};

use common::{assert_probs_close, run_and_probs};
use num_complex::Complex64;
use prism_q::gates::Gate;
use prism_q::pulse::{
    CompiledGate, Device, Envelope, PulseError, Schedule, Transmon, compile, evolve,
};
use prism_q::{Circuit, NoiseChannel};

const NS: f64 = 1e-9;
const ALPHA: f64 = -300e6;

fn transmon(frequency: f64, levels: usize) -> Transmon {
    Transmon {
        frequency,
        anharmonicity: ALPHA,
        levels,
    }
}

fn basis(dim: usize, index: usize) -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[index] = Complex64::new(1.0, 0.0);
    state
}

// Peak amplitude of a 20 ns, sigma 5 ns lifted Gaussian whose midpoint-sampled area at
// 16 segments per ns is a pi rotation: 2 pi * amp * sum(g) * h = pi.
fn pi_pulse_amp() -> f64 {
    let (duration, sigma, h) = (20.0 * NS, 5.0 * NS, NS / 16.0);
    let half = 0.5 * duration;
    let floor = (-0.5 * (half / sigma).powi(2)).exp();
    let area: f64 = (0..320)
        .map(|k| {
            let x = (k as f64 + 0.5) * h - half;
            ((-0.5 * (x / sigma).powi(2)).exp() - floor) / (1.0 - floor) * h
        })
        .sum();
    0.5 / area
}

fn pi_pulse(device: &Device, beta: f64) -> CompiledGate {
    let envelope = Envelope::Drag {
        amp: pi_pulse_amp(),
        sigma: 5.0 * NS,
        beta,
    };
    let mut schedule = Schedule::new(NS);
    schedule.play(0, 0.0, 20.0 * NS, envelope);
    compile(device, &schedule, 16).unwrap()
}

fn drag_beta() -> f64 {
    -1.0 / (TAU * ALPHA)
}

fn drag_pi() -> Envelope {
    Envelope::Drag {
        amp: pi_pulse_amp(),
        sigma: 5.0 * NS,
        beta: drag_beta(),
    }
}

fn from_pairs(entries: &[[f64; 2]]) -> Vec<Complex64> {
    entries
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect()
}

fn max_entry_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max)
}

#[test]
fn rabi_resonant_matches_sin_squared() {
    let device = Device::new([transmon(5e9, 2)]);
    let rabi = 10e6;
    let mut schedule = Schedule::new(NS);
    schedule.play(0, 0.0, 200.0 * NS, Envelope::Constant { amp: rabi });
    let rows = evolve(&device, &schedule, &basis(2, 0), 1).unwrap();
    assert_eq!(rows.len(), 200);
    let worst = rows
        .iter()
        .enumerate()
        .map(|(k, row)| {
            let t = (k + 1) as f64 * NS;
            (row[1] - (0.5 * TAU * rabi * t).sin().powi(2)).abs()
        })
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn drag_suppresses_leakage() {
    // Converged references from an independent numpy integrator of the same model:
    // classical RK4 in the drive frame at 1024 and 2048 steps per ns, Richardson
    // extrapolated. The bounds cover the error at 16 substeps, measured at 1.1e-7
    // relative on the plain pulse and 6.2e-5 on DRAG, whose leaked amplitude cancels to
    // first order and leaves the step error a larger share.
    const PLAIN: f64 = 1.345_473_394_671_226_8e-5;
    const DRAG: f64 = 8.324_755_572_059_022e-11;
    let device = Device::new([transmon(5e9, 3)]);
    let plain = pi_pulse(&device, 0.0).leakage()[0][1];
    let drag = pi_pulse(&device, drag_beta()).leakage()[0][1];
    assert!(
        (plain - PLAIN).abs() < 1e-6 * PLAIN,
        "plain leakage {plain:e}"
    );
    assert!((drag - DRAG).abs() < 2e-4 * DRAG, "drag leakage {drag:e}");
    assert!(plain / drag > 1e3);
}

#[test]
fn compiled_unitary_round_trips() {
    // DRAG keeps leakage near 1e-10: the polar factor matches the renormalized projection
    // only to the order of the leaked amplitudes, 1.5e-7 on the plain Gaussian.
    let device = Device::new([transmon(5e9, 3)]);
    let compiled = pi_pulse(&device, drag_beta());
    let mut schedule = Schedule::new(NS);
    schedule.play(
        0,
        0.0,
        20.0 * NS,
        Envelope::Drag {
            amp: pi_pulse_amp(),
            sigma: 5.0 * NS,
            beta: drag_beta(),
        },
    );

    let rows = evolve(&device, &schedule, &basis(3, 1), 16).unwrap();
    let last = rows.last().unwrap();
    let retained = 1.0 - compiled.leakage()[0][1];
    let projected = [last[0] / retained, last[1] / retained];
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_unitary(compiled.unitary(), &[0]).unwrap();
    let probs = run_and_probs(&circuit);
    let error = (0..2)
        .map(|i| (probs[i] - projected[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "round trip error {error:e}");

    let theta = 0.7f64;
    let initial = [
        Complex64::new((0.5 * theta).cos(), 0.0),
        Complex64::new((0.5 * theta).sin(), 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let last = evolve(&device, &schedule, &initial, 16)
        .unwrap()
        .pop()
        .unwrap();
    let kept = last[0] + last[1];
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Ry(theta), &[0]);
    circuit.add_unitary(compiled.unitary(), &[0]).unwrap();
    assert_probs_close(
        &run_and_probs(&circuit),
        &[last[0] / kept, last[1] / kept],
        1e-9,
        "round trip from a superposition",
    );
}

#[test]
fn coupled_pair_matches_closed_form_in_register_order() {
    // Levels 2, no drive: |00> and |11> are fixed in the frames, and the one-excitation
    // block is exp(-i (H - wbar) T) with H - wbar = [[D/2, G], [G, -D/2]] on (site 0
    // excited, site 1 excited), each row then rotated into its site's frame by
    // exp(+-i D T / 2). D = 2 pi (f0 - f1), G = 2 pi g.
    let (f0, f1, g, duration) = (5.0e9, 5.2e9, 10e6, 50.0 * NS);
    let device = Device::new([transmon(f0, 2), transmon(f1, 2)]).couple(0, 1, g);
    let mut schedule = Schedule::new(NS);
    schedule.play(0, 0.0, duration, Envelope::Constant { amp: 0.0 });
    let compiled = compile(&device, &schedule, 1).unwrap();

    let half_detuning = 0.5 * TAU * (f0 - f1);
    let coupling = TAU * g;
    let rate = (half_detuning.powi(2) + coupling.powi(2)).sqrt();
    let (cos, sin) = ((rate * duration).cos(), (rate * duration).sin() / rate);
    let i = Complex64::new(0.0, 1.0);
    let block = [
        [cos - i * sin * half_detuning, -i * sin * coupling],
        [-i * sin * coupling, cos + i * sin * half_detuning],
    ];
    let frame = [
        Complex64::from_polar(1.0, half_detuning * duration),
        Complex64::from_polar(1.0, -half_detuning * duration),
    ];
    // Gate::unitary packing: site 0 is the high bit, so site 0 excited is index 2.
    let packed = [2, 1];
    let mut expected = vec![Complex64::new(0.0, 0.0); 16];
    expected[0] = Complex64::new(1.0, 0.0);
    expected[15] = Complex64::new(1.0, 0.0);
    for (r, &row) in packed.iter().enumerate() {
        for (c, &column) in packed.iter().enumerate() {
            expected[row * 4 + column] = frame[r] * block[r][c];
        }
    }

    let unitary = compiled.unitary();
    let error = (0..16)
        .map(|k| (unitary[k] - expected[k]).norm())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-10, "max entry error {error:e}");
    assert!((expected[10] - expected[5]).norm() > 0.1);
    let ideal = Gate::unitary(expected, 2).unwrap();
    assert!((compiled.average_fidelity(&ideal) - 1.0).abs() < 1e-10);
    assert_eq!(compiled.leakage(), &[[0.0, 0.0], [0.0, 0.0]]);

    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_unitary(compiled.unitary(), &[0, 1]).unwrap();
    let probs = run_and_probs(&circuit);
    let stays = block[0][0].norm_sqr();
    assert_probs_close(&probs, &[0.0, stays, 1.0 - stays, 0.0], 1e-10, "swap order");
}

#[test]
fn leakage_from_counts_leakage_through_a_neighbour() {
    // Site 1 at f0 - alpha puts |11> and |02> on resonance. Site 0 keeps two levels, so
    // the exchange coupling, matrix element sqrt 2 g between them, is an exact two-level
    // problem: the |11> population leaks as sin^2(2 pi sqrt 2 g t). The one-excitation
    // manifold has no leaked state, so leakage() sees nothing.
    let (f1, g) = (4.9e9, 3e6);
    let device = Device::new([transmon(f1 + ALPHA, 2), transmon(f1, 3)]).couple(0, 1, g);
    for duration in [30.0 * NS, 100.0 * NS] {
        let mut schedule = Schedule::new(NS);
        schedule.play(0, 0.0, duration, Envelope::Constant { amp: 0.0 });
        let compiled = compile(&device, &schedule, 1).unwrap();
        let expected = (TAU * 2f64.sqrt() * g * duration).sin().powi(2);
        let leaked = compiled.leakage_from(&[1, 1]);
        assert!(
            (leaked - expected).abs() < 1e-12,
            "leaked {leaked}, expected {expected}"
        );
        let stays = compiled.leakage_from(&[0, 2]);
        assert!((stays - (1.0 - expected)).abs() < 1e-12, "stays {stays}");
        assert!(compiled.leakage_from(&[0, 1]) < 1e-12);
        assert!(compiled.leakage()[1][1] < 1e-6);
    }
}

#[test]
fn mixed_levels_factor_into_single_sites() {
    let single = pi_pulse(&Device::new([transmon(5e9, 3)]), drag_beta());
    let pair = pi_pulse(
        &Device::new([transmon(5e9, 3), transmon(5.3e9, 2)]),
        drag_beta(),
    );
    assert_eq!(pair.propagator().len(), 36);
    let one = single.unitary();
    let two = pair.unitary();
    for row in 0..4 {
        for column in 0..4 {
            let expected = if row % 2 == column % 2 {
                one[(row / 2) * 2 + column / 2]
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!((two[row * 4 + column] - expected).norm() < 1e-12);
        }
    }
    assert!((pair.leakage()[0][1] - single.leakage()[0][1]).abs() < 1e-15);
    assert_eq!(pair.leakage()[1], [0.0, 0.0]);
    assert_eq!(single.leakage_from(&[1]), single.leakage()[0][1]);
    assert!((pair.leakage_from(&[1, 1]) - single.leakage()[0][1]).abs() < 1e-15);
}

#[test]
fn carrier_off_the_integration_frame_converges() {
    // A zero-amplitude play on site 1 comes first, so the integration frame sits on its
    // carrier and site 0's drive rotates at 300 MHz in it. The reference is the pulse
    // alone, converged in site 0's own frame by an independent numpy integrator (RK4 at
    // 1024 and 2048 steps per ns, Richardson extrapolated). The step is fourth order, so
    // halving the segment should cut the error 16 times; measured 1.8e-6 at 8 substeps
    // and 1.1e-7 at 16.
    const ALONE: [[f64; 2]; 4] = [
        [-0.0028462976260905594, -0.06762587824090933],
        [-0.09016760004383467, -0.9936238941794349],
        [-0.09016760004383499, -0.9936238941794348],
        [0.009373535346375268, 0.06703355617256807],
    ];
    let one = from_pairs(&ALONE);
    let mut expected = vec![Complex64::new(0.0, 0.0); 16];
    for row in 0..4 {
        for column in (row % 2..4).step_by(2) {
            expected[row * 4 + column] = one[(row / 2) * 2 + column / 2];
        }
    }
    let error = |substeps: usize| {
        let mut both = Schedule::new(NS);
        both.play(1, 0.0, NS, Envelope::Constant { amp: 0.0 })
            .play(0, NS, 20.0 * NS, drag_pi());
        let pair = Device::new([transmon(5e9, 3), transmon(5.3e9, 2)]);
        max_entry_error(
            &compile(&pair, &both, substeps).unwrap().unitary(),
            &expected,
        )
    };
    let (coarse, fine) = (error(8), error(16));
    assert!(coarse / fine > 12.0, "coarse {coarse:e}, fine {fine:e}");
    assert!(fine < 2e-7, "fine {fine:e}");
}

#[test]
fn coupled_pair_on_two_carriers_converges() {
    // DRAG pi pulses on both sites of the cross resonance pair, each on its own carrier.
    // Site 1 plays first, so site 0's drive rotates at 100 MHz in the integration frame.
    // The reference integrates each line in its own frame, where the coupling rotates
    // instead (numpy RK4 at 1024 and 2048 steps per ns, Richardson extrapolated).
    // Measured 3.0e-8 at 8 substeps and 1.9e-9 at 16.
    const REFERENCE: [[f64; 2]; 16] = [
        [-0.0025525005963642636, 0.015714560274481666],
        [-0.06845914541323316, 0.006271981363857183],
        [-0.07378442948473556, 0.013405637933585277],
        [-0.9734498858972074, 0.20443094550254715],
        [-0.055686701084259084, 0.005164863691115011],
        [-0.03613312779051947, 0.007347812171527822],
        [-0.9808261403296492, 0.16566809668637578],
        [0.07639925376822221, -0.014492772260249],
        [-0.06412648338837455, 0.0026712809904773345],
        [-0.9813766387556742, 0.16253517494043024],
        [0.045101842080209176, -0.006495356262647314],
        [0.06310044979727092, -0.017459542410535833],
        [-0.9754916705713034, 0.20227583116528688],
        [0.06314481258496099, -0.02023256896039015],
        [0.05149834866906299, -0.01348574417363034],
        [0.0006108568182671796, 0.016417862579018963],
    ];
    let reference = from_pairs(&REFERENCE);
    let device = Device::new([transmon(CR_CONTROL, 3), transmon(CR_TARGET, 3)]).couple(0, 1, CR_G);
    let error = |substeps: usize| {
        let mut schedule = Schedule::new(NS);
        schedule
            .play(1, 0.0, 20.0 * NS, drag_pi())
            .play(0, NS, 20.0 * NS, drag_pi());
        max_entry_error(
            &compile(&device, &schedule, substeps).unwrap().unitary(),
            &reference,
        )
    };
    let (coarse, fine) = (error(8), error(16));
    assert!(coarse / fine > 12.0, "coarse {coarse:e}, fine {fine:e}");
    assert!(fine < 4e-9, "fine {fine:e}");
}

const CR_CONTROL: f64 = 5.0e9;
const CR_TARGET: f64 = 4.9e9;
const CR_G: f64 = 3e6;
const CR_AMP: f64 = 20e6;

fn cross_resonance() -> CompiledGate {
    let device = Device::new([transmon(CR_CONTROL, 3), transmon(CR_TARGET, 3)]).couple(0, 1, CR_G);
    let mut schedule = Schedule::new(NS);
    // A real drive leaves the unitary symmetric; the trailing virtual Z on the control
    // breaks that, so a transposed packing cannot match the reference.
    schedule
        .set_frequency(0, 0.0, CR_TARGET)
        .play(
            0,
            0.0,
            300.0 * NS,
            Envelope::GaussianSquare {
                amp: CR_AMP,
                sigma: 10.0 * NS,
                width: 260.0 * NS,
            },
        )
        .shift_phase(0, 300.0 * NS, 0.5);
    compile(&device, &schedule, 4).unwrap()
}

#[test]
fn cross_resonance_matches_reference() {
    // Converged nearest unitary from an independent numpy integrator of the same model:
    // classical RK4 in the drive frame at 256 and 512 steps per ns, Richardson
    // extrapolated, which agrees with 512 and 1024 to 2e-14. The bounds cover the error
    // at 4 substeps, measured at 3.6e-9 on the unitary, 1.5e-6 relative on the leakage
    // and 2e-10 on the fidelity.
    const REFERENCE: [[f64; 2]; 16] = [
        [-0.09241354409501537, 0.8727471050250205],
        [-0.4704834829878449, -0.09075847941779797],
        [0.011019034492715437, 0.0029045027726422655],
        [-0.006153633569781442, -0.0035631059515619746],
        [-0.4704834829878451, -0.090758479417797],
        [-0.23941836626047852, 0.8437103529528273],
        [-0.01308763543761673, -0.022661058320949005],
        [0.0080296526735693, 0.022149306516092253],
        [0.011062605325827772, -0.0027338655623372395],
        [-0.02234977072729477, -0.013612402947709618],
        [-0.31327665893330126, 0.4090395454035984],
        [0.7122076066366814, 0.47590970287374296],
        [-0.007108565503035023, -0.00017671056069147234],
        [0.01766562637054872, 0.015588224598653555],
        [0.7122076066366833, 0.47590970287374074],
        [-0.25746033891506004, 0.44651855909232147],
    ];
    const LEAK_FROM_ONE: f64 = 2.824_659_026_518_732e-4;
    // Total leakage from |10>, |01> and |11>, levels given as (control, target).
    const LEAK_FROM: [([usize; 2], f64); 3] = [
        ([1, 0], 3.649_351_957_900_827e-4),
        ([0, 1], 1.006_936_401_850_813_1e-6),
        ([1, 1], 3.183_691_757_525_039_5e-4),
    ];
    const FIDELITY: f64 = 0.999_828_892_521_845_1;
    let compiled = cross_resonance();
    let reference = from_pairs(&REFERENCE);
    let error = max_entry_error(&compiled.unitary(), &reference);
    assert!(error < 1e-8, "max entry error {error:e}");

    let leakage = compiled.leakage();
    assert!(
        (leakage[0][1] - LEAK_FROM_ONE).abs() < 1e-5 * LEAK_FROM_ONE,
        "control leakage from |1> {:e}",
        leakage[0][1]
    );
    assert!(leakage[0][0] < 1e-7 && leakage[1].iter().all(|&p| p < 1e-7));
    for (levels, expected) in LEAK_FROM {
        let leaked = compiled.leakage_from(&levels);
        assert!(
            (leaked - expected).abs() < 1e-5 * expected,
            "leakage from {levels:?} {leaked:e}"
        );
    }

    let fidelity = compiled.average_fidelity(&Gate::unitary(reference, 2).unwrap());
    assert!((fidelity - FIDELITY).abs() < 1e-9, "fidelity {fidelity}");
}

#[test]
fn cross_resonance_zx_follows_perturbative_rate() {
    // To first order in g and in the drive, control level n drives the target's X at c_n:
    // c_0 = -amp g / (2 D), c_1 = amp g (1 / (2 D) - 1 / (D + alpha)), D the detuning, so
    // H = 2 pi (c_0 - c_1) / 2 ZX + IX terms and U = exp(-i theta ZX / 2) with
    // theta = 2 pi (c_0 - c_1) times the envelope area. The next order goes as
    // (amp / D)^2: the reference integrator puts the angle 1% under this at 10 MHz, 4%
    // under at 20 MHz and 8% under at 30 MHz, so 10% still catches a factor or a sign.
    let unitary = cross_resonance().unitary();
    let x_angle = |b: [Complex64; 4]| {
        let root = (b[0] * b[3] - b[1] * b[2]).sqrt();
        let cos = ((b[0] + b[3]) / root).re * 0.5;
        let sin = (Complex64::new(0.0, 0.5) * (b[1] + b[2]) / root).re;
        // The square root fixes the angle only up to 2 pi.
        let theta = 2.0 * sin.atan2(cos);
        (theta + PI).rem_euclid(TAU) - PI
    };
    let control_zero = x_angle([unitary[0], unitary[1], unitary[4], unitary[5]]);
    let control_one = x_angle([unitary[10], unitary[11], unitary[14], unitary[15]]);
    assert!(control_zero < 0.0 && control_one > 0.0);
    let zx = 0.5 * (control_zero - control_one);

    let (edge, sigma, h) = (20.0 * NS, 10.0 * NS, NS / 4.0);
    let floor = (-0.5 * (edge / sigma).powi(2)).exp();
    let area: f64 = (0..1200)
        .map(|k| {
            let t = (k as f64 + 0.5) * h;
            let x = (t - edge).min(0.0) + (t - edge - 260.0 * NS).max(0.0);
            ((-0.5 * (x / sigma).powi(2)).exp() - floor) / (1.0 - floor) * h
        })
        .sum();
    let detuning = CR_CONTROL - CR_TARGET;
    let rate = 0.5 * CR_AMP * CR_G * -ALPHA / (detuning * (detuning + ALPHA));
    let predicted = TAU * 2.0 * rate * area;
    assert!(
        (zx / predicted - 1.0).abs() < 0.1,
        "zx {zx}, perturbative {predicted}"
    );
}

#[test]
fn error_after_completes_the_ideal_gate() {
    let compiled = pi_pulse(&Device::new([transmon(5e9, 3)]), drag_beta());
    let ideal = Gate::Rx(PI);
    let channel = compiled.error_after(&ideal);
    channel.validate().unwrap();
    let NoiseChannel::Custom { kraus } = channel else {
        panic!("expected a one-qubit Kraus set");
    };
    assert_eq!(kraus.len(), 1);
    let x = ideal.matrix_2x2();
    let unitary = compiled.unitary();
    for row in 0..2 {
        for column in 0..2 {
            let product = kraus[0][row][0] * x[0][column] + kraus[0][row][1] * x[1][column];
            assert!((product - unitary[row * 2 + column]).norm() < 1e-12);
        }
    }
    let fidelity = compiled.average_fidelity(&ideal);
    assert!(fidelity > 0.99 && fidelity < 1.0, "fidelity {fidelity}");
}

#[test]
fn shift_phase_is_a_virtual_z() {
    let device = Device::new([transmon(5e9, 2)]);
    let mut schedule = Schedule::new(NS);
    schedule
        .play(0, 0.0, NS, Envelope::Constant { amp: 0.0 })
        .shift_phase(0, NS, 0.4);
    let unitary = compile(&device, &schedule, 1).unwrap().unitary();
    assert!((unitary[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    assert!((unitary[3] - Complex64::from_polar(1.0, -0.4)).norm() < 1e-12);
}

fn compile_err(device: &Device, schedule: &Schedule) -> PulseError {
    compile(device, schedule, 1).unwrap_err()
}

fn constant_schedule(site: usize, start: f64, duration: f64) -> Schedule {
    let mut schedule = Schedule::new(NS);
    schedule.play(site, start, duration, Envelope::Constant { amp: 1e6 });
    schedule
}

#[test]
fn invalid_devices_are_rejected() {
    let schedule = constant_schedule(0, 0.0, NS);
    assert_eq!(
        compile_err(&Device::new([]), &schedule),
        PulseError::SiteCount(0)
    );
    assert_eq!(
        compile_err(&Device::new([transmon(5e9, 2); 4]), &schedule),
        PulseError::SiteCount(4)
    );
    assert_eq!(
        compile_err(&Device::new([transmon(5e9, 1)]), &schedule),
        PulseError::TooFewLevels { site: 0, levels: 1 }
    );
    assert_eq!(
        compile_err(&Device::new([transmon(5e9, 9); 3]), &schedule),
        PulseError::DimensionTooLarge(729)
    );
    assert!(matches!(
        compile_err(&Device::new([transmon(f64::NAN, 2)]), &schedule),
        PulseError::NonFinite {
            what: "transmon frequency",
            ..
        }
    ));
    let device = Device::new([transmon(5e9, 2), transmon(5.1e9, 2)]).couple(0, 1, f64::INFINITY);
    assert!(matches!(
        compile_err(&device, &schedule),
        PulseError::NonFinite {
            what: "coupling",
            ..
        }
    ));
}

#[test]
fn invalid_schedules_are_rejected() {
    let device = Device::new([transmon(5e9, 3)]);
    for period in [0.0, -NS, f64::NAN] {
        let mut schedule = Schedule::new(period);
        schedule.play(0, 0.0, NS, Envelope::Constant { amp: 1e6 });
        assert!(matches!(
            compile_err(&device, &schedule),
            PulseError::SamplePeriod(_)
        ));
    }
    assert_eq!(
        compile(&device, &constant_schedule(0, 0.0, NS), 0).unwrap_err(),
        PulseError::ZeroSubsteps
    );
    assert_eq!(
        compile_err(&device, &constant_schedule(1, 0.0, NS)),
        PulseError::MissingSite { site: 1, sites: 1 }
    );
    let mut schedule = Schedule::new(NS);
    schedule.shift_phase(2, 0.0, 0.1);
    assert_eq!(
        compile_err(&device, &schedule),
        PulseError::MissingSite { site: 2, sites: 1 }
    );
    for start in [0.5 * NS, -NS] {
        assert!(matches!(
            compile_err(&device, &constant_schedule(0, start, NS)),
            PulseError::OffGrid { site: 0, .. }
        ));
    }
    let mut schedule = Schedule::new(NS);
    schedule.set_frequency(0, 0.0, f64::INFINITY);
    assert!(matches!(
        compile_err(&device, &schedule),
        PulseError::NonFinite {
            what: "carrier frequency",
            ..
        }
    ));

    let mut schedule = constant_schedule(0, 0.0, 10.0 * NS);
    schedule.play(0, 5.0 * NS, 10.0 * NS, Envelope::Constant { amp: 1e6 });
    assert_eq!(
        compile_err(&device, &schedule),
        PulseError::Overlap {
            site: 0,
            time: 5.0 * NS
        }
    );

    let invalid = [
        (
            NS,
            Envelope::Gaussian {
                amp: 1e6,
                sigma: 0.0,
            },
        ),
        (
            NS,
            Envelope::Gaussian {
                amp: 1e6,
                sigma: 1.0,
            },
        ),
        (
            NS,
            Envelope::Drag {
                amp: 1e6,
                sigma: NS,
                beta: f64::NAN,
            },
        ),
        (
            NS,
            Envelope::GaussianSquare {
                amp: 1e6,
                sigma: NS,
                width: 2.0 * NS,
            },
        ),
        (2.0 * NS, Envelope::Samples(vec![Complex64::new(1e6, 0.0)])),
        (0.0, Envelope::Constant { amp: 1e6 }),
        (1.5 * NS, Envelope::Constant { amp: 1e6 }),
    ];
    for (duration, envelope) in invalid {
        let mut schedule = Schedule::new(NS);
        schedule.play(0, 0.0, duration, envelope);
        assert!(matches!(
            compile_err(&device, &schedule),
            PulseError::InvalidPlay { site: 0, .. }
        ));
    }
}

#[test]
fn full_leak_from_one_has_no_nearest_unitary() {
    // Carrier on the 1-2 transition with the 0-1 transition pushed far off resonance:
    // a pi rotation between |1> and |2> (matrix element sqrt 2) empties |1>.
    let alpha = 1e13;
    let device = Device::new([Transmon {
        frequency: 5e9,
        anharmonicity: alpha,
        levels: 3,
    }]);
    let duration = 50.0 * NS;
    let amp = 1.0 / (2.0 * 2f64.sqrt() * duration);
    let mut schedule = Schedule::new(NS);
    schedule
        .set_frequency(0, 0.0, 5e9 + alpha)
        .play(0, 0.0, duration, Envelope::Constant { amp });
    assert!(matches!(
        compile_err(&device, &schedule),
        PulseError::Leaked(_)
    ));
}
