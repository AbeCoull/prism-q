import math

import numpy as np

from prism_q import CircuitBuilder, simulate


def test_single_rx_gradient_matches_analytic():
    # Rx(theta)|0>, <Z> = cos theta, d/dtheta = -sin theta.
    theta = 0.6
    builder = CircuitBuilder(1)
    builder.rx_param(0, theta, 0)
    circuit = builder.build()
    links = builder.parameter_links()

    hamiltonian = [(1.0, [(0, "Z")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)

    assert isinstance(grad, np.ndarray)
    assert grad.dtype == np.float64
    assert grad.shape == (1,)
    assert math.isclose(value, math.cos(theta), abs_tol=1e-9)
    assert math.isclose(grad[0], -math.sin(theta), abs_tol=1e-9)


def test_two_parameter_circuit_matches_finite_difference():
    builder = CircuitBuilder(2)
    builder.h(0).rz_param(0, 0.9, 0).cx(0, 1).ry_param(1, 0.4, 1)
    circuit = builder.build()
    links = builder.parameter_links()
    assert len(links) == 2

    ham = [(1.0, [(0, "Z"), (1, "Z")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(ham, links)
    assert grad.shape == (2,)

    # Central finite difference on each slot through the forward run().
    def expval(shifted):
        out = simulate(shifted).seed(42).state_vector()
        # <Z0 Z1> from amplitudes: sign is (-1)^(bit0 xor bit1).
        probs = np.abs(out) ** 2
        signs = np.array([1.0 if bin(i).count("1") % 2 == 0 else -1.0 for i in range(len(out))])
        return float(np.sum(probs * signs))

    eps = 1e-5
    for slot, (theta0, rebuild) in enumerate(
        [
            (0.9, lambda t: _build(t, 0.4)),
            (0.4, lambda t: _build(0.9, t)),
        ]
    ):
        plus = expval(rebuild(theta0 + eps))
        minus = expval(rebuild(theta0 - eps))
        fd = (plus - minus) / (2 * eps)
        assert math.isclose(grad[slot], fd, abs_tol=1e-5)


def _build(rz_theta, ry_theta):
    b = CircuitBuilder(2)
    b.h(0).rz(rz_theta, 0).cx(0, 1).ry(ry_theta, 1)
    return b.build()


def test_invalid_axis_is_rejected():
    builder = CircuitBuilder(1)
    builder.rx_param(0, 0.3, 0)
    circuit = builder.build()
    links = builder.parameter_links()
    try:
        simulate(circuit).seed(42).expectation_gradient([(1.0, [(0, "W")])], links)
        assert False, "expected an error for invalid axis"
    except Exception:
        pass
