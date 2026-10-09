"""Simulate a GHZ circuit, sample it, and compare the backends that can run it."""

from prism_q import BackendKind, CircuitBuilder, simulate

ghz = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
outcome = simulate(ghz).seed(42).run()
print("probabilities:", outcome.probabilities)
print("backend:", outcome.metadata.backend)

measured = CircuitBuilder(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build()
counts = simulate(measured).seed(42).sample_counts(1000).counts()
print("counts:", dict(sorted(counts.items())))

for kind in [BackendKind.statevector(), BackendKind.mps(16), BackendKind.sparse()]:
    result = simulate(ghz).backend(kind).seed(42).run()
    print(result.metadata.backend, result.probabilities[[0, 7]])
