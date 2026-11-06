import math
import pickle

import numpy as np
import pytest

from prism_q import CircuitBuilder, ExpectationGradient, Parameters, simulate


def test_single_rx_gradient_matches_analytic():
    # Rx(theta)|0>, <Z> = cos theta, d/dtheta = -sin theta.
    theta = 0.6
    builder = CircuitBuilder(1)
    builder.rx(theta, 0).param(0)
    circuit = builder.build()
    params = builder.parameters()

    hamiltonian = [(1.0, [(0, "Z")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(hamiltonian, params)

    assert isinstance(grad, np.ndarray)
    assert grad.dtype == np.float64
    assert grad.shape == (1,)
    assert math.isclose(value, math.cos(theta), abs_tol=1e-9)
    assert math.isclose(grad[0], -math.sin(theta), abs_tol=1e-9)


def test_two_parameter_circuit_matches_finite_difference():
    builder = CircuitBuilder(2)
    builder.h(0).rz(0.9, 0).param(0).cx(0, 1).ry(0.4, 1).param(1)
    circuit = builder.build()
    params = builder.parameters()
    assert len(params.links()) == 2

    ham = [(1.0, [(0, "Z"), (1, "Z")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(ham, params)
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
    builder.rx(0.3, 0).param(0)
    circuit = builder.build()
    params = builder.parameters()
    try:
        simulate(circuit).seed(42).expectation_gradient([(1.0, [(0, "W")])], params)
        assert False, "expected an error for invalid axis"
    except Exception:
        pass

def test_parameter_shift_matches_the_adjoint_gradient():
    builder = CircuitBuilder(2)
    builder.h(0).rz(0.9, 0).param(0).cx(0, 1).ry(0.4, 1).param(1)
    circuit = builder.build()
    params = builder.parameters()

    hamiltonian = [(1.0, [(0, "Z"), (1, "Z")]), (0.5, [(1, "X")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(hamiltonian, params)
    shift_value, shift_grad = (
        simulate(circuit).seed(42).expectation_gradient_shift(hamiltonian, params)
    )

    assert math.isclose(value, shift_value, abs_tol=1e-9)
    assert np.allclose(grad, shift_grad, atol=1e-9)


def test_parameter_shift_reads_a_noisy_mixture_on_the_density_matrix():
    from prism_q import BackendKind, NoiseModel

    def circuit(theta):
        return CircuitBuilder(2).ry(theta, 0).param(0).cx(0, 1).rx(0.4, 1).build()

    def noisy_value(theta):
        c = circuit(theta)
        noise = NoiseModel.uniform_depolarizing(c, 0.05)
        sim = simulate(c).backend(BackendKind.density_matrix()).noise(noise).seed(42)
        return sim.expectation([[(0, "Z"), (1, "Z")]]).values[0]

    theta = 0.7
    template = circuit(theta)
    noise = NoiseModel.uniform_depolarizing(template, 0.05)
    hamiltonian = [(1.0, [(0, "Z"), (1, "Z")])]
    value, grad = (
        simulate(template)
        .backend(BackendKind.density_matrix())
        .noise(noise)
        .seed(42)
        .expectation_gradient_shift(hamiltonian, Parameters.all_rotations(template))
    )
    step = 1e-5
    finite = (noisy_value(theta + step) - noisy_value(theta - step)) / (2 * step)
    assert math.isclose(value, noisy_value(theta), abs_tol=1e-10)
    assert math.isclose(grad[0], finite, abs_tol=1e-6)


def test_parameter_shift_under_noise_needs_the_mixture():
    import pytest
    from prism_q import NoiseModel, PrismError

    template = CircuitBuilder(1).ry(0.3, 0).param(0).build()
    noise = NoiseModel.uniform_depolarizing(template, 0.05)
    with pytest.raises(PrismError):
        simulate(template).noise(noise).seed(42).expectation_gradient_shift(
            [(1.0, [(0, "Z")])], Parameters.all_rotations(template)
        )


def test_parameter_shift_honours_the_initial_state():
    theta = 0.9
    template = CircuitBuilder(1).ry(theta, 0).param(0).build()
    value, grad = (
        simulate(template)
        .initial_state([0, 1])
        .seed(42)
        .expectation_gradient_shift([(1.0, [(0, "Z")])], Parameters.all_rotations(template))
    )
    assert math.isclose(value, -math.cos(theta), abs_tol=1e-12)
    assert math.isclose(grad[0], math.sin(theta), abs_tol=1e-9)


def test_gradient_result_names_its_fields_and_unpacks():
    theta = 0.6
    builder = CircuitBuilder(1)
    builder.rx(theta, 0).param(0)
    circuit = builder.build()
    params = builder.parameters()

    result = simulate(circuit).seed(42).expectation_gradient([(1.0, "Z")], params)
    assert isinstance(result, ExpectationGradient)
    assert math.isclose(result.value, math.cos(theta), abs_tol=1e-9)
    assert result.gradient.shape == (1,)
    assert math.isclose(result.gradient[0], -math.sin(theta), abs_tol=1e-9)

    value, grad = result
    assert value == result.value
    assert np.array_equal(grad, result.gradient)

    shift = simulate(circuit).seed(42).expectation_gradient_shift([(1.0, "Z")], params)
    assert isinstance(shift, ExpectationGradient)
    assert np.allclose(shift.gradient, result.gradient, atol=1e-9)


def test_gradient_result_indexes_and_pickles_as_the_pair():
    builder = CircuitBuilder(2).ry(0.3, 0).param(0).cx(0, 1).rz(0.7, 1).param(1)
    circuit, params = builder.build_parametric()
    result = simulate(circuit).seed(42).expectation_gradient([(1.0, "ZZ")], params)

    assert len(result) == 2
    assert result[0] == result[-2] == result.value
    assert np.array_equal(result[1], result.gradient)
    assert np.array_equal(result[-1], result.gradient)
    for index in (2, -3):
        with pytest.raises(IndexError):
            result[index]

    restored = pickle.loads(pickle.dumps(result))
    assert isinstance(restored, ExpectationGradient)
    assert restored.value == result.value
    assert np.array_equal(restored.gradient, result.gradient)
    value, grad = restored
    assert value == result.value
    assert np.array_equal(grad, result.gradient)


def test_gradient_result_slices_as_the_pair():
    builder = CircuitBuilder(2).ry(0.3, 0).param(0).cx(0, 1).rz(0.7, 1).param(1)
    circuit, params = builder.build_parametric()
    result = simulate(circuit).seed(42).expectation_gradient([(1.0, "ZZ")], params)

    whole = result[:]
    assert isinstance(whole, tuple) and len(whole) == 2
    assert whole[0] == result.value
    assert np.array_equal(whole[1], result.gradient)
    tail = result[1:]
    assert len(tail) == 1 and np.array_equal(tail[0], result.gradient)
    assert result[:1] == (result.value,)
    assert result[5:] == ()


def _prepared_ansatz():
    builder = CircuitBuilder(3)
    for q in range(3):
        builder.ry(0.2 + 0.1 * q, q).param(q)
    builder.cx(0, 1).cx(1, 2)
    for q in range(3):
        builder.rz(0.3, q).param(3 + q % 2)
    circuit, params = builder.build_parametric()
    return circuit, params


PREPARED_HAMILTONIAN = [(1.0, "ZZI"), (0.5, [(2, "X")]), (-0.3, [(0, "Y"), (2, "Z")])]


def test_prepared_gradient_matches_the_bound_simulation():
    from prism_q import PauliObservable, PreparedCircuit

    circuit, params = _prepared_ansatz()
    prepared = PreparedCircuit(circuit, params)
    values = [0.4, -0.7, 1.1, 0.2, -0.5]
    bound = params.bind(circuit, values)
    for hamiltonian in (PREPARED_HAMILTONIAN, PauliObservable(PREPARED_HAMILTONIAN)):
        for method in ("expectation_gradient", "expectation_gradient_shift"):
            result = getattr(prepared, method)(values, hamiltonian, seed=42)
            assert isinstance(result, ExpectationGradient)
            assert result.gradient.shape == (params.num_slots,)
            want = getattr(simulate(bound).seed(42), method)(hamiltonian, params)
            assert math.isclose(result.value, want.value, abs_tol=1e-10)
            np.testing.assert_allclose(result.gradient, want.gradient, atol=1e-10)


def test_prepared_gradient_many_returns_value_and_gradient_arrays():
    from prism_q import PreparedCircuit

    circuit, params = _prepared_ansatz()
    prepared = PreparedCircuit(circuit, params)
    points = np.array([[0.1 * (p + k) for k in range(5)] for p in range(4)])
    for method in ("expectation_gradient", "expectation_gradient_shift"):
        values, gradients = getattr(prepared, method + "_many")(
            points, PREPARED_HAMILTONIAN, seed=42
        )
        assert values.dtype == gradients.dtype == np.float64
        assert values.shape == (4,)
        assert gradients.shape == (4, params.num_slots)
        for row, point in enumerate(points):
            single = getattr(prepared, method)(list(point), PREPARED_HAMILTONIAN, seed=42)
            assert values[row] == single.value
            assert np.array_equal(gradients[row], single.gradient)


def test_prepared_gradient_rejects_a_short_binding_and_a_wide_observable():
    from prism_q import PreparedCircuit, PrismError

    circuit, params = _prepared_ansatz()
    prepared = PreparedCircuit(circuit, params)
    with pytest.raises(PrismError):
        prepared.expectation_gradient([0.1], PREPARED_HAMILTONIAN)
    with pytest.raises(PrismError, match="3 qubits"):
        prepared.expectation_gradient([0.1] * 5, [(1.0, "ZZZZ")])


def test_gradient_takes_parameters_not_links():
    circuit, params = _prepared_ansatz()
    with pytest.raises(TypeError):
        simulate(circuit).seed(42).expectation_gradient(PREPARED_HAMILTONIAN, params.links())
