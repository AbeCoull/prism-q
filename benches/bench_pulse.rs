use std::f64::consts::TAU;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use prism_q::pulse::{Device, Envelope, Schedule, Transmon, compile};

mod common;

const NS: f64 = 1e-9;
const ALPHA: f64 = -300e6;

fn transmon(frequency: f64) -> Transmon {
    Transmon {
        frequency,
        anharmonicity: ALPHA,
        levels: 3,
    }
}

fn bench_pulse_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("pulse_compile");
    common::configure_group(&mut group);

    // Cross resonance: the control driven at the target's frequency, dimension 9,
    // 300 samples of 4 segments.
    let pair = Device::new([transmon(5.0e9), transmon(4.9e9)]).couple(0, 1, 3e6);
    let mut cr = Schedule::new(NS);
    cr.set_frequency(0, 0.0, 4.9e9).play(
        0,
        0.0,
        300.0 * NS,
        Envelope::GaussianSquare {
            amp: 20e6,
            sigma: 10.0 * NS,
            width: 260.0 * NS,
        },
    );
    group.bench_function(BenchmarkId::new("cr_two_transmon_d3", "300ns"), |b| {
        b.iter(|| black_box(compile(&pair, &cr, 4).unwrap()))
    });

    // A DRAG pi pulse on one site, dimension 3, 20 samples of 16 segments.
    let single = Device::new([transmon(5.0e9)]);
    let mut x = Schedule::new(NS);
    x.play(
        0,
        0.0,
        20.0 * NS,
        Envelope::Drag {
            amp: 46.7e6,
            sigma: 5.0 * NS,
            beta: -1.0 / (TAU * ALPHA),
        },
    );
    group.bench_function(BenchmarkId::new("x_transmon_d3", "20ns"), |b| {
        b.iter(|| black_box(compile(&single, &x, 16).unwrap()))
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_pulse_compile
}
criterion_main!(benches);
