"""GPU reach from Python.

Without the `gpu` feature compiled in, or without a usable device, the device
cases skip, so a green run on such a build means "not tested". Set
`PRISM_REQUIRE_GPU=1` to turn either absence into a failure.

These cases pin reachability and results, not device residency. Routing is soft
by design: below the family crossover, and after a device allocation that
fails, the run degrades to the host and no Python-visible signal says which
happened, so a case here cannot assert that a kernel ran. Per-kernel device
coverage is `tests/golden_gpu.rs` on the Rust side.
"""

import math
import os

import numpy as np
import pytest

from prism_q import (
    BackendKind,
    CircuitBuilder,
    GpuContext,
    NoiseModel,
    PrismError,
    circuits,
    simulate,
)

SUPPORTED = GpuContext.is_supported()
AVAILABLE = GpuContext.is_available()

requires_device = pytest.mark.skipif(not AVAILABLE, reason="no usable CUDA device")


def _entangled_chain(n):
    builder = CircuitBuilder(n).h(0)
    for q in range(n - 1):
        builder = builder.cx(q, q + 1)
    return builder.rz(0.3, 0).build()


def test_prism_require_gpu_is_honoured():
    if os.environ.get("PRISM_REQUIRE_GPU") is None:
        pytest.skip("PRISM_REQUIRE_GPU not set")
    assert SUPPORTED, "PRISM_REQUIRE_GPU is set but this build lacks the `gpu` feature"
    assert AVAILABLE, "PRISM_REQUIRE_GPU is set but no usable CUDA device was found"


def test_gpu_surface_exists_in_every_build():
    for name in ("statevector_gpu", "stabilizer_gpu", "density_matrix_gpu", "auto_gpu"):
        assert hasattr(BackendKind, name)


@pytest.mark.skipif(SUPPORTED, reason="build has the `gpu` feature")
def test_context_without_feature_raises_named_error():
    assert AVAILABLE is False
    with pytest.raises(PrismError, match="built without GPU support"):
        GpuContext()


@pytest.mark.skipif(not SUPPORTED or AVAILABLE, reason="build has the `gpu` feature and a device")
def test_context_without_device_raises_rather_than_falling_back():
    with pytest.raises(PrismError):
        GpuContext()


@requires_device
@pytest.mark.parametrize("name", ["statevector_gpu", "auto_gpu"])
def test_gpu_statevector_above_the_crossover_matches_host(name):
    circuit = _entangled_chain(16)
    context = GpuContext(0)
    device = simulate(circuit).backend(getattr(BackendKind, name)(context)).seed(42).run()
    host = simulate(circuit).backend(BackendKind.statevector()).seed(42).run()
    np.testing.assert_allclose(device.probabilities, host.probabilities, atol=1e-12)


@requires_device
def test_gpu_statevector_below_the_crossover_matches_host():
    circuit = _entangled_chain(4)
    context = GpuContext(0)
    device = simulate(circuit).backend(BackendKind.statevector_gpu(context)).seed(42).run()
    host = simulate(circuit).backend(BackendKind.statevector()).seed(42).run()
    np.testing.assert_allclose(device.probabilities, host.probabilities, atol=1e-12)


@requires_device
def test_gpu_density_matrix_matches_host_mixture():
    circuit = _entangled_chain(6)
    model = NoiseModel.uniform_depolarizing(circuit, 0.05)
    context = GpuContext(0)
    device = simulate(circuit).backend(BackendKind.density_matrix_gpu(context)).noise(model).seed(42)
    host = simulate(circuit).backend(BackendKind.density_matrix()).noise(model).seed(42)
    np.testing.assert_allclose(device.run().probabilities, host.run().probabilities, atol=1e-12)
    observables = [[(0, "Z")], [(1, "X"), (2, "Z")]]
    np.testing.assert_allclose(
        device.expectation_values(observables),
        host.expectation_values(observables),
        atol=1e-12,
    )


@requires_device
def test_gpu_stabilizer_matches_host_tableau():
    ghz = circuits.ghz(6)
    context = GpuContext(0)
    device = simulate(ghz).backend(BackendKind.stabilizer_gpu(context)).seed(42).run()
    host = simulate(ghz).backend(BackendKind.stabilizer()).seed(42).run()
    np.testing.assert_allclose(device.probabilities, host.probabilities, atol=1e-12)
    assert math.isclose(device.probabilities[0], 0.5, abs_tol=1e-12)


@requires_device
def test_gpu_stabilizer_rejects_non_clifford():
    circuit = CircuitBuilder(3).h(0).t(0).cx(0, 1).build()
    context = GpuContext(0)
    with pytest.raises(PrismError, match="non-Clifford"):
        simulate(circuit).backend(BackendKind.stabilizer_gpu(context)).seed(42).run()


@requires_device
def test_one_context_serves_repeated_runs():
    context = GpuContext(0)
    circuit = _entangled_chain(16)
    first = simulate(circuit).backend(BackendKind.auto_gpu(context)).seed(42).run().probabilities
    second = simulate(circuit).backend(BackendKind.auto_gpu(context)).seed(42).run().probabilities
    np.testing.assert_allclose(first, second, atol=1e-12)


@requires_device
def test_gpu_shots_stay_on_the_ghz_support():
    circuit = CircuitBuilder(4, 4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all().build()
    context = GpuContext(0)
    counts = simulate(circuit).backend(BackendKind.auto_gpu(context)).seed(42).shots(512).counts()
    assert set(counts) <= {"0000", "1111"}
    assert sum(counts.values()) == 512


@requires_device
def test_context_repr_names_the_device():
    assert repr(GpuContext(0)) == "GpuContext(device_id=0)"
