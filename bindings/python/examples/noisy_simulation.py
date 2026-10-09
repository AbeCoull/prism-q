"""Sample a GHZ circuit under rule-based noise and under a device calibration."""

from prism_q import (
    BackendKind,
    CircuitBuilder,
    DeviceCalibration,
    GateFilter,
    NoiseBuilder,
    NoiseChannel,
    simulate,
)


def ghz_fraction(counts):
    shots = sum(counts.values())
    return (counts.get("000", 0) + counts.get("111", 0)) / shots


circuit = CircuitBuilder(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build()

noise = (
    NoiseBuilder()
    .after_gates(GateFilter.all().arity(1), NoiseChannel.depolarizing(0.001))
    .after_gates_joint(GateFilter.all().named("cx"), NoiseChannel.two_qubit_depolarizing(0.02))
    .uniform_readout_error(0.01, 0.02)
    .build(circuit)
)
ideal = simulate(circuit).seed(42).sample_counts(10_000).counts()
noisy = simulate(circuit).seed(42).noise(noise).sample_counts(10_000).counts()
print(f"ideal GHZ fraction: {ghz_fraction(ideal):.4f}")
print(f"noisy GHZ fraction: {ghz_fraction(noisy):.4f}")

calibration = DeviceCalibration.parse(
    """
    qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03
    qubit 1 t1=95e-6 t2=110e-6 p01=0.01 p10=0.02
    qubit 2 t1=60e-6 t2=50e-6 p01=0.02 p10=0.04
    gate1q time=35e-9 error=3e-4
    gate2q time=300e-9 error=8e-3
    gate2q 1 2 time=450e-9 error=2e-2
    """
)
device = calibration.to_noise_model(circuit)
sampled = simulate(circuit).seed(42).noise(device).sample_counts(10_000).counts()
print(f"calibrated GHZ fraction: {ghz_fraction(sampled):.4f}")

unmeasured = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
exact = (
    simulate(unmeasured)
    .backend(BackendKind.density_matrix())
    .noise(calibration.to_noise_model(unmeasured))
    .seed(42)
    .run()
    .probabilities
)
print(f"exact GHZ population before readout: {exact[0] + exact[7]:.4f}")
