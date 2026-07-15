import math
import importlib

import numpy as np
import pytest

import prism_q
from prism_q import CircuitBuilder, circuits, parse_qasm, run_qasm, simulate

BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_version_is_string():
    assert isinstance(prism_q.__version__, str)
    assert prism_q.__version__.count(".") == 2


def test_bell_state_probabilities():
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    out = simulate(bell).seed(42).run()
    probs = out.probabilities
    assert isinstance(probs, np.ndarray)
    assert probs.dtype == np.float64
    assert probs.shape == (4,)
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[3], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[1], 0.0, abs_tol=1e-9)


def test_builder_is_chainable_and_counts_gates():
    circuit = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
    assert circuit.num_qubits == 3
    assert circuit.gate_count() == 3
    assert circuit.is_clifford_only()


def test_build_is_non_destructive():
    builder = CircuitBuilder(2).h(0).cx(0, 1)
    c1 = builder.build()
    c2 = builder.build()
    # Repeated build() must not empty the builder.
    assert c1.num_qubits == 2 and c1.gate_count() == 2
    assert c2.num_qubits == 2 and c2.gate_count() == 2
    # The builder is still usable afterward.
    c3 = builder.t(0).build()
    assert c3.gate_count() == 3


def test_parse_qasm_and_shots_counts():
    circuit = parse_qasm(BELL_QASM)
    result = simulate(circuit).seed(7).shots(1000)
    assert result.num_shots == 1000
    assert result.num_classical_bits == 2
    counts = result.counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000


def test_sample_counts():
    circuit = parse_qasm(BELL_QASM)
    cr = simulate(circuit).seed(3).sample_counts(500)
    counts = cr.counts()
    assert sum(counts.values()) == 500
    assert set(counts) <= {"00", "11"}


def test_run_qasm_helper():
    # No measurement: probabilities reflect the superposition (measuring would
    # collapse the state).
    qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];"
    out = run_qasm(qasm, 42)
    probs = out.probabilities
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[3], 0.5, abs_tol=1e-9)


def test_statevector_is_complex128():
    sv = simulate(circuits.ghz(3)).seed(1).state_vector()
    assert sv.dtype == np.complex128
    assert sv.shape == (8,)
    assert math.isclose(abs(sv[0]) ** 2, 0.5, abs_tol=1e-9)
    assert math.isclose(abs(sv[7]) ** 2, 0.5, abs_tol=1e-9)


def test_marginals():
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    marginals = simulate(bell).seed(1).marginals()
    assert len(marginals) == 2
    for p0, p1 in marginals:
        assert math.isclose(p0, 0.5, abs_tol=1e-9)
        assert math.isclose(p1, 0.5, abs_tol=1e-9)


def test_cu_accepts_nested_list_and_numpy():
    xmat_list = [[0, 1], [1, 0]]
    c1 = CircuitBuilder(2).h(0).cu(xmat_list, 0, 1).build()
    xmat_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    c2 = CircuitBuilder(2).h(0).cu(xmat_np, 0, 1).build()
    p1 = simulate(c1).seed(1).run().probabilities
    p2 = simulate(c2).seed(1).run().probabilities
    np.testing.assert_allclose(p1, p2, atol=1e-12)
    # Behaves like a CNOT: equal weight on |00> and |11>.
    assert math.isclose(p1[0], 0.5, abs_tol=1e-9)
    assert math.isclose(p1[3], 0.5, abs_tol=1e-9)


@pytest.mark.parametrize(
    "prep, expected_index",
    [
        ([0, 1], 0b111),  # both controls set -> target flips to |111>
        ([0], 0b001),  # only one control set -> no flip, stays |001>
    ],
)
def test_mcu_toffoli(prep, expected_index):
    xmat = [[0, 1], [1, 0]]
    builder = CircuitBuilder(3)
    for q in prep:
        builder.x(q)
    circuit = builder.mcu(xmat, [0, 1], 2).build()
    probs = simulate(circuit).seed(1).run().probabilities
    assert math.isclose(probs[expected_index], 1.0, abs_tol=1e-9)


def test_invalid_matrix_shape_raises():
    with pytest.raises(prism_q.PrismError):
        prism_q.Gate.cu([[1, 0, 0], [0, 1, 0]])


def test_prebuilt_circuits_smoke():
    assert circuits.qft(4).num_qubits == 4
    assert circuits.qaoa(4, 2, seed=1).num_qubits == 4
    assert circuits.hardware_efficient_ansatz(4, 2, seed=1).num_qubits == 4
    assert circuits.w_state(3).num_qubits == 3
    assert circuits.clifford_t(4, 4, t_fraction=0.2, seed=1).num_qubits == 4


def test_circuits_submodule_is_importable():
    mod = importlib.import_module("prism_q.circuits")
    assert mod.ghz(2).num_qubits == 2


def test_invalid_programmatic_indices_raise_prism_error():
    with pytest.raises(prism_q.PrismError):
        CircuitBuilder(1).h(2)
    with pytest.raises(prism_q.PrismError):
        prism_q.Circuit(1).add_gate(prism_q.Gate.h(), [2])
    with pytest.raises(prism_q.PrismError):
        CircuitBuilder(1, 0).measure(0, 0)
