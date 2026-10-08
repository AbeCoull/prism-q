import numpy as np
import pytest

import prism_q
from prism_q import CircuitBuilder, SaveSpec, circuits, parse_qasm, simulate


def _rotated():
    return (
        CircuitBuilder(3, 3)
        .h(0)
        .cx(0, 1)
        .rz(0.1234567890123, 2)
        .pauli_rotation(0.3, [(0, "X"), (1, "Y"), (2, "Z")])
        .measure_all()
        .build()
    )


def test_to_qasm_round_trips_through_the_parser():
    circuit = _rotated()
    text = circuit.to_qasm()
    assert text.startswith("OPENQASM 3.0;")
    assert "rxyz(0.3)" in text
    assert "rz(0.1234567890123)" in text
    assert parse_qasm(text).to_qasm() == text


def test_expanded_export_keeps_the_state():
    unitary = CircuitBuilder(3).h(0).pauli_rotation(0.7, [(0, "X"), (2, "Y")]).build()
    text = unitary.to_qasm(expand_pauli_rotations=True)
    assert "rxy" not in text
    before = simulate(unitary).state_vector()
    after = simulate(parse_qasm(text)).state_vector()
    assert abs(np.vdot(before, after)) == pytest.approx(1.0, abs=1e-12)


def test_export_covers_guarded_regions():
    builder = CircuitBuilder(2, 2).h(0).measure(0, 0)
    builder.guarded(
        prism_q.ClassicalCondition.bit(0), lambda t: t.x(1), lambda e: e.h(1).z(1)
    )
    text = builder.build().to_qasm()
    assert "if (c[0]) x q[1];" in text
    assert "if (!c[0]) {" in text
    assert parse_qasm(text).to_qasm() == text


def test_export_rejects_a_save_point():
    circuit = CircuitBuilder(1).h(0).build()
    circuit.add_save(SaveSpec.StateVector, "psi")
    with pytest.raises(prism_q.PrismError) as info:
        circuit.to_qasm()
    assert info.value.kind == "export_unsupported"


def test_str_is_the_text_diagram():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    text = str(circuit)
    assert text == circuit.draw()
    assert "q[0]" in text and "q[1]" in text
    assert repr(circuit).startswith("Circuit(")


def test_draw_options_reach_the_renderer():
    circuit = CircuitBuilder(3).h(0).barrier([0, 1]).cx(0, 1).build()
    assert len(circuit.draw(show_idle_wires=False).splitlines()) < len(
        circuit.draw().splitlines()
    )
    narrow = circuits.random(4, 20, seed=1).draw(fold_width=40)
    assert max(len(line) for line in narrow.splitlines()) <= 40
    assert "Gate density heatmap" in circuit.heatmap()
    assert circuit.summary().startswith("Circuit: 3 qubits")


def test_wide_circuits_draw_as_a_summary():
    wide = circuits.ghz(80)
    assert wide.draw() == wide.summary()


def test_svg_renders_with_options():
    circuit = _rotated()
    light = circuit.to_svg()
    assert light.startswith("<svg")
    assert circuit.to_svg(dark_mode=True) != light
    assert circuit.to_svg(show_legend=True, show_stats_header=True) != light
    assert "<svg" in circuit.to_svg(compact=True, ellipsis=(1, 1), padding=(5, 5, 5, 5))
    assert circuit.to_svg_heatmap().startswith("<svg")
    assert circuit.to_svg_heatmap(dark_mode=True) != circuit.to_svg_heatmap()


def test_jupyter_display_is_a_static_svg():
    svg = _rotated()._repr_svg_()
    assert svg.startswith("<svg")
    assert "prefers-color-scheme" in svg
    assert "@keyframes" not in svg
    assert circuits.ghz(200)._repr_svg_().startswith("<svg")
