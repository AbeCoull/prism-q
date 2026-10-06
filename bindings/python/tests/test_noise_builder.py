import math

import numpy as np
import pytest

from prism_q import (
    Circuit,
    CircuitBuilder,
    Gate,
    GateFilter,
    NoiseBuilder,
    NoiseChannel,
    NoiseModel,
    PrismError,
    simulate,
)


def _counts(circuit, model):
    return simulate(circuit).seed(42).noise(model).shots(1024).counts()


def _expectations(circuit, model):
    return simulate(circuit).seed(42).noise(model).density_matrix_expectation_values(
        [[(qubit, "Z")] for qubit in range(circuit.num_qubits)]
    )


def test_gate_filters_combine_and_capture_their_registration_state():
    circuit = (
        CircuitBuilder(2, 2).h(0).cx(0, 1).cx(1, 0).h(1).measure_all().build()
    )
    channel = NoiseChannel.pauli(0.2, 0.0, 0.0)
    gate_filter = GateFilter()
    assert gate_filter.named("cx") is gate_filter
    assert gate_filter.arity(2) is gate_filter
    assert gate_filter.on_targets((0, 1)) is gate_filter
    assert gate_filter.on_qubits([1, 1]) is gate_filter
    builder = NoiseBuilder()
    assert builder.after_gates(gate_filter, channel) is builder
    gate_filter.named("h")

    manual = NoiseModel.empty(circuit)
    manual.add_event(1, channel, [1])
    first = builder.build(circuit)
    assert _counts(circuit, first) == _counts(circuit, manual)
    assert _counts(circuit, builder.build(circuit)) == _counts(circuit, first)

    extra = NoiseChannel.pauli(0.1, 0.0, 0.0)
    builder.after_gates(GateFilter.all().named("h"), extra)
    manual.add_event(0, extra, [0])
    manual.add_event(3, extra, [1])
    assert _counts(circuit, builder.build(circuit)) == _counts(circuit, manual)
    original = NoiseModel.empty(circuit)
    original.add_event(1, channel, [1])
    assert _counts(circuit, first) == _counts(circuit, original)


def test_joint_channel_matches_only_the_complete_directed_target_list():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).cx(1, 0).measure_all().build()
    channel = NoiseChannel.two_qubit_depolarizing(0.3)
    model = (
        NoiseBuilder()
        .after_gates_joint(GateFilter.all().on_targets([0, 1]), channel)
        .build(circuit)
    )
    manual = NoiseModel.empty(circuit)
    manual.add_event(1, channel, [0, 1])
    assert _counts(circuit, model) == _counts(circuit, manual)


@pytest.mark.parametrize("joint", [False, True])
def test_crosstalk_excludes_gate_targets_and_deduplicates_edges(joint):
    circuit = CircuitBuilder(3, 3).h(0).cx(0, 1).measure_all().build()
    channel = (
        NoiseChannel.two_qubit_depolarizing(0.2)
        if joint
        else NoiseChannel.pauli(0.2, 0.0, 0.0)
    )
    model = (
        NoiseBuilder()
        .crosstalk(
            GateFilter.all().named("cx"),
            [(0, 2), (2, 0), (1, 2), (0, 1)],
            channel,
        )
        .build(circuit)
    )
    manual = NoiseModel.empty(circuit)
    for targets in ([0, 2], [1, 2]) if joint else ([2],):
        manual.add_event(1, channel, targets)
    assert _counts(circuit, model) == _counts(circuit, manual)


def test_idle_damping_matches_a_manual_event_at_the_layer_end():
    circuit = CircuitBuilder(2).x(0).x(1).id(0).build()
    channel = NoiseChannel.amplitude_damping(0.7)
    model = NoiseBuilder().on_idle_qubits(channel).build(circuit)
    manual = NoiseModel.empty(circuit)
    manual.add_event(2, channel, [1])
    np.testing.assert_allclose(_expectations(circuit, model), [-1.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(
        _expectations(circuit, model), _expectations(circuit, manual), atol=1e-12
    )


def test_over_rotation_matches_a_manual_kraus_rotation():
    theta, relative = 0.6, 0.25
    circuit = CircuitBuilder(1).ry(theta, 0).build()
    model = NoiseBuilder().over_rotation(GateFilter.all(), relative).build(circuit)
    angle = theta * relative / 2.0
    matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    manual = NoiseModel.empty(circuit)
    manual.add_event(0, NoiseChannel.custom([matrix]), [0])
    np.testing.assert_allclose(
        _expectations(circuit, model), _expectations(circuit, manual), atol=1e-12
    )


def test_reset_and_pre_measurement_rules_preserve_event_order():
    circuit = Circuit(1, 1)
    circuit.add_reset(0)
    circuit.add_measure(0, 0)
    flip = NoiseChannel.pauli(1.0, 0.0, 0.0)
    damp = NoiseChannel.amplitude_damping(1.0)
    model = NoiseBuilder().after_resets(flip).before_measurements(damp).build(circuit)
    manual = NoiseModel.empty(circuit)
    manual.add_event(0, flip, [0])
    manual.add_event(0, damp, [0])
    assert _counts(circuit, model) == _counts(circuit, manual) == {"0": 1024}
    reversed_model = (
        NoiseBuilder().before_measurements(damp).after_resets(flip).build(circuit)
    )
    assert _counts(circuit, reversed_model) == {"1": 1024}


def test_uniform_readout_matches_manual_rates():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    model = NoiseBuilder().uniform_readout_error(0.1, 0.2).build(circuit)
    manual = NoiseModel.empty(circuit)
    manual.with_readout_error(0.1, 0.2)
    assert _counts(circuit, model) == _counts(circuit, manual)


@pytest.mark.parametrize("uniform_first", [False, True])
def test_per_bit_readout_overrides_uniform_rates_in_either_order(uniform_first):
    circuit = CircuitBuilder(2, 2).measure_all().build()
    builder = NoiseBuilder()
    if uniform_first:
        builder.uniform_readout_error(1.0, 1.0).readout_error(0, 0.0, 0.0)
    else:
        builder.readout_error(0, 0.0, 0.0).uniform_readout_error(1.0, 1.0)
    assert _counts(circuit, builder.build(circuit)) == {"01": 1024}


def test_build_preserves_validation_errors_and_builder_after_failure():
    builder = NoiseBuilder().readout_error(1, 0.1, 0.2)
    with pytest.raises(PrismError) as error:
        builder.build(Circuit(1, 1))
    assert error.value.kind == "invalid_parameter"
    builder.build(Circuit(2, 2)).validate()

    circuit = CircuitBuilder(1).x(0).build()
    invalid = NoiseBuilder().after_gates(GateFilter.all(), NoiseChannel.depolarizing(-0.1))
    with pytest.raises(PrismError) as error:
        invalid.build(circuit)
    assert error.value.kind == "invalid_parameter"

    measured_first = Circuit(1, 1)
    measured_first.add_measure(0, 0)
    with pytest.raises(PrismError) as error:
        NoiseBuilder().before_measurements(NoiseChannel.pauli(0.1, 0.0, 0.0)).build(
            measured_first
        )
    assert error.value.kind == "invalid_parameter"


def test_unknown_gate_filter_emits_no_events():
    circuit = Circuit(1, 1)
    circuit.add_gate(Gate.x(), [0])
    circuit.add_measure(0, 0)
    model = (
        NoiseBuilder()
        .after_gates(GateFilter.all().named("unknown"), NoiseChannel.pauli(1.0, 0.0, 0.0))
        .build(circuit)
    )
    assert _counts(circuit, model) == _counts(circuit, NoiseModel.empty(circuit))
