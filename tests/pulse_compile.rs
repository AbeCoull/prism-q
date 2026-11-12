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
    // References from an independent numpy integrator of the same model (eigh per
    // segment, same midpoint grid, rad/ns units).
    const PLAIN: f64 = 1.342_397_295_046_906_7e-5;
    const DRAG: f64 = 1.821_341_311_093_034_2e-10;
    let device = Device::new([transmon(5e9, 3)]);
    let plain = pi_pulse(&device, 0.0).leakage()[0][1];
    let drag = pi_pulse(&device, drag_beta()).leakage()[0][1];
    assert!(
        (plain - PLAIN).abs() < 1e-8 * PLAIN,
        "plain leakage {plain:e}"
    );
    assert!((drag - DRAG).abs() < 1e-8 * DRAG, "drag leakage {drag:e}");
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
}

#[test]
fn carrier_off_the_integration_frame_converges() {
    // A zero-amplitude play on site 1 comes first, so the integration frame sits on its
    // carrier and site 0's drive rotates at 300 MHz in it. Holding that drive at each
    // segment midpoint costs accuracy that falls as the square of the segment length.
    let drag = || Envelope::Drag {
        amp: pi_pulse_amp(),
        sigma: 5.0 * NS,
        beta: drag_beta(),
    };
    let error = |substeps: usize| {
        let mut alone = Schedule::new(NS);
        alone.play(0, NS, 20.0 * NS, drag());
        let one = compile(&Device::new([transmon(5e9, 3)]), &alone, substeps)
            .unwrap()
            .unitary();

        let mut both = Schedule::new(NS);
        both.play(1, 0.0, NS, Envelope::Constant { amp: 0.0 })
            .play(0, NS, 20.0 * NS, drag());
        let pair = Device::new([transmon(5e9, 3), transmon(5.3e9, 2)]);
        let two = compile(&pair, &both, substeps).unwrap().unitary();

        let mut worst = 0.0f64;
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row % 2 == column % 2 {
                    one[(row / 2) * 2 + column / 2]
                } else {
                    Complex64::new(0.0, 0.0)
                };
                worst = worst.max((two[row * 4 + column] - expected).norm());
            }
        }
        worst
    };
    let (coarse, fine) = (error(16), error(64));
    assert!(fine < coarse / 10.0, "coarse {coarse:e}, fine {fine:e}");
    assert!(fine < 1e-4, "fine {fine:e}");
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
