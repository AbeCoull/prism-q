import numpy as np
import pytest

import prism_q
from prism_q import CircuitBuilder, CompiledSampler, NoiseModel


def _ghz(n=3):
    builder = CircuitBuilder(n, n).h(0)
    for q in range(n - 1):
        builder.cx(q, q + 1)
    return builder.measure_all().build()


def test_samples_are_records_in_measurement_order():
    sampler = CompiledSampler(_ghz(), seed=7)
    assert sampler.num_measurements == 3
    assert sampler.rank == 1
    shots = sampler.sample(500)
    assert shots.dtype == np.bool_ and shots.shape == (500, 3)
    assert (shots == shots[:, :1]).all()
    assert 0 < shots[:, 0].sum() < 500


def test_packed_samples_unpack_to_the_bool_layout():
    first = CompiledSampler(_ghz(10), seed=3)
    second = CompiledSampler(_ghz(10), seed=3)
    packed = first.sample_packed(200)
    assert packed.dtype == np.uint8 and packed.shape == (200, 2)
    unpacked = np.unpackbits(packed, axis=1, count=10, bitorder="little").astype(bool)
    assert (unpacked == second.sample(200)).all()


def test_calls_continue_one_seeded_stream():
    sampler = CompiledSampler(_ghz(), seed=9)
    a, b = sampler.sample(64), sampler.sample(64)
    replay = CompiledSampler(_ghz(), seed=9)
    assert (replay.sample(64) == a).all()
    assert (replay.sample(64) == b).all()


def test_counts_scale_past_memory():
    counts = CompiledSampler(_ghz(), seed=1).sample_counts(10**9)
    assert set(counts) == {"000", "111"}
    assert sum(counts.values()) == 10**9
    assert abs(counts["000"] - 5 * 10**8) < 10**6


def test_exact_counts_enumerate_the_rank():
    assert CompiledSampler(_ghz(), seed=1).exact_counts() == {"000": 1, "111": 1}


def test_streamed_reductions():
    circuit = CircuitBuilder(3, 3).h(0).cx(0, 1).x(2).measure_all().build()
    sampler = CompiledSampler(circuit, seed=2)
    marginals = sampler.marginals(100_000)
    assert marginals.shape == (3,)
    assert marginals[0] == pytest.approx(0.5, abs=0.01)
    assert marginals[2] == 1.0
    parities = sampler.parity_expectations([[0, 1], [2], [0]], 50_000)
    assert parities[0] == 1.0
    assert parities[1] == -1.0
    assert abs(parities[2]) < 0.03
    correlators = sampler.correlators([(0, 1), (0, 2)], 50_000)
    assert correlators[0] == 1.0
    assert abs(correlators[1]) < 0.03


def test_record_indices_are_checked():
    sampler = CompiledSampler(_ghz(), seed=1)
    with pytest.raises(prism_q.PrismError):
        sampler.parity_expectations([[3]], 10)
    with pytest.raises(prism_q.PrismError):
        sampler.correlators([(0, 3)], 10)


def test_noisy_sampler_matches_the_simulation_statistics():
    circuit = _ghz()
    model = NoiseModel.uniform_depolarizing(circuit, 0.05)
    sampler = CompiledSampler(circuit, seed=4, noise=model)
    assert sampler.rank is None
    assert sampler.exact_counts() is None
    counts = sampler.sample_counts(20_000)
    reference = prism_q.simulate(circuit).seed(4).noise(model).sample_counts(20_000).counts()
    for key in ("000", "111"):
        assert counts[key] / 20_000 == pytest.approx(reference[key] / 20_000, abs=0.02)
    assert len(counts) > 2
    assert sampler.sample(10).shape == (10, 3)


def test_rejects_what_the_compiled_path_cannot_run():
    with pytest.raises(prism_q.PrismError):
        CompiledSampler(CircuitBuilder(1, 1).t(0).measure_all().build())
    with pytest.raises(prism_q.PrismError):
        CompiledSampler(CircuitBuilder(1, 2).measure(0, 0).h(0).measure(0, 1).build())
    circuit = _ghz()
    damping = NoiseModel.with_amplitude_damping(circuit, 0.1)
    with pytest.raises(prism_q.PrismError):
        CompiledSampler(circuit, noise=damping)
