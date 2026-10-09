//! Sample a GHZ circuit under rule-based noise and under a device calibration.

use std::collections::HashMap;

use prism_q::{
    BackendKind, CircuitBuilder, DeviceCalibration, GateFilter, NoiseBuilder, NoiseChannel,
    simulate,
};

fn ghz_fraction(counts: &HashMap<Vec<u64>, u64>) -> f64 {
    let shots: u64 = counts.values().sum();
    let good = counts.get(&vec![0b000]).unwrap_or(&0) + counts.get(&vec![0b111]).unwrap_or(&0);
    good as f64 / shots as f64
}

fn main() {
    let circuit = CircuitBuilder::new_with_classical(3, 3)
        .h(0)
        .cx(0, 1)
        .cx(1, 2)
        .measure_all()
        .build();

    let noise = NoiseBuilder::new()
        .after_gates(
            GateFilter::all().arity(1),
            NoiseChannel::Depolarizing { p: 0.001 },
        )
        .after_gates_joint(
            GateFilter::all().named("cx"),
            NoiseChannel::TwoQubitDepolarizing { p: 0.02 },
        )
        .uniform_readout_error(0.01, 0.02)
        .build(&circuit)
        .expect("invalid noise rules");
    let ideal = simulate(&circuit).seed(42).sample_counts(10_000).unwrap();
    let noisy = simulate(&circuit)
        .noise(&noise)
        .seed(42)
        .sample_counts(10_000)
        .unwrap();
    println!("ideal GHZ fraction: {:.4}", ghz_fraction(&ideal.counts));
    println!("noisy GHZ fraction: {:.4}", ghz_fraction(&noisy.counts));

    let calibration = DeviceCalibration::parse(
        "qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03
         qubit 1 t1=95e-6 t2=110e-6 p01=0.01 p10=0.02
         qubit 2 t1=60e-6 t2=50e-6 p01=0.02 p10=0.04
         gate1q time=35e-9 error=3e-4
         gate2q time=300e-9 error=8e-3
         gate2q 1 2 time=450e-9 error=2e-2",
    )
    .expect("invalid calibration");
    let device = calibration.to_noise_model(&circuit).unwrap();
    let sampled = simulate(&circuit)
        .noise(&device)
        .seed(42)
        .sample_counts(10_000)
        .unwrap();
    println!(
        "calibrated GHZ fraction: {:.4}",
        ghz_fraction(&sampled.counts)
    );

    let unmeasured = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).build();
    let device = calibration.to_noise_model(&unmeasured).unwrap();
    let exact = simulate(&unmeasured)
        .backend(BackendKind::DensityMatrix)
        .noise(&device)
        .seed(42)
        .run()
        .unwrap()
        .probabilities
        .expect("no probabilities");
    println!(
        "exact GHZ population before readout: {:.4}",
        exact.get(0b000) + exact.get(0b111)
    );
}
