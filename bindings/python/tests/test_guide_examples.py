"""Execute the Python examples in docs/guides/python.md.

Blocks fenced as ``python`` run in order in one namespace, so an example may use
what an earlier one defined. ``python,ignore`` marks a block that needs hardware
or a launcher CI does not have. A block whose last line ends in ``# raises`` must
raise ``PrismError``.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import prism_q

GUIDE = Path(__file__).resolve().parents[3] / "docs" / "guides" / "python.md"

# Inputs the guide names without building, keyed by a fragment of the block that
# first reads them and supplied just before it runs.
SETUP = {
    "builder.pauli_rotation(": "builder = CircuitBuilder(4)",
    "parse_qasm(qasm_source)": (
        'qasm_source = "OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];"'
    ),
    "parse_braket(source)": (
        'source = """OPENQASM 3.0;\n'
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "#pragma braket result probability all\n"
        "#pragma braket result expectation z(q[0]) @ z(q[1])\n"
        '"""'
    ),
    "simulate(big_circuit)": "big_circuit = circuits.hardware_efficient_ansatz(40, 2)",
    "simulate(first)": (
        "first = CircuitBuilder(2).h(0).cx(0, 1).build()\n"
        "second = CircuitBuilder(2).rx(0.3, 0).cz(0, 1).build()"
    ),
    "prep.run(clifford_prefix)": (
        "clifford_prefix = CircuitBuilder(4).h(0).cx(0, 1).cx(1, 2).s(3).build()\n"
        "measurement_suffix = CircuitBuilder(4, 8).h(3).measure_all().build()"
    ),
    "NoiseModel.uniform_depolarizing(circuit, 0.01)": (
        "circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()"
    ),
    "values = simulate(circuit).seed(42).expectation_values(": (
        "circuit = CircuitBuilder(2).h(0).cx(0, 1).build()"
    ),
    "run_batch(circuits": (
        "circuits = [prism_q.circuits.hardware_efficient_ansatz(4, 2, seed=s) "
        "for s in range(3)]"
    ),
    "observable_expectation_many(points": (
        "points = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])\n"
        'hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]'
    ),
    "PreparedCircuit(circuit, params": (
        "circuit, params = builder.build(), builder.parameters()"
    ),
    "qp.detector_error_model()": (
        "qp = QecProgram.from_text('''\n"
        "R 0 1 2 3 4\n"
        "X_ERROR(0.01) 0 2 4\n"
        "CX 0 1 2 1 2 3 4 3\n"
        "MR 1 3\n"
        "DETECTOR rec[-2]\n"
        "DETECTOR rec[-1]\n"
        "M 0 2 4\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
        "''')"
    ),
    "simulate(huge)": "huge = prism_q.circuits.ghz(40)",
}


def fenced_blocks(text):
    """Return ``(line, info, code)`` for each fenced block, ``line`` 1-based."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opening = re.match(r"^```(\S*)", lines[i])
        if opening is None:
            i += 1
            continue
        start = i
        i += 1
        while not lines[i].startswith("```"):
            i += 1
        blocks.append((start + 1, opening.group(1), "\n".join(lines[start + 1 : i])))
        i += 1
    return blocks


@pytest.mark.skipif(not GUIDE.exists(), reason="needs the repository checkout")
def test_python_guide_examples_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {name: getattr(prism_q, name) for name in prism_q.__all__}
    namespace["prism_q"] = prism_q
    unused = set(SETUP)
    ran = 0
    for line, info, code in fenced_blocks(GUIDE.read_text(encoding="utf-8")):
        if info != "python":
            continue
        for fragment, setup in SETUP.items():
            if fragment in code and fragment in unused:
                exec(setup, namespace)
                unused.discard(fragment)
        source = compile(code, f"{GUIDE.name}:{line}", "exec")
        if code.rstrip().endswith("# raises"):
            with pytest.raises(prism_q.PrismError):
                exec(source, namespace)
        else:
            try:
                exec(source, namespace)
            except Exception as exc:
                pytest.fail(f"python.md block at line {line} failed: {exc!r}")
        ran += 1
    assert unused == set(), f"setup keys matched no block: {sorted(unused)}"
    assert ran > 0
