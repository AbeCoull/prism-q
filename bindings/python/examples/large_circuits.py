"""Run circuits too wide for a statevector on the engines built for their structure."""

from prism_q import BackendKind, CircuitBuilder, PrismError, circuits, simulate

n = 1000
builder = CircuitBuilder(n, n).h(0)
for q in range(n - 1):
    builder.cx(q, q + 1)
ghz = builder.measure_all().build()
counts = simulate(ghz).seed(42).sample_counts(1000)
print(f"{n}-qubit GHZ on {counts.metadata.backend}: {len(counts.counts())} distinct outcomes")


def kicked_chain(n, measure):
    builder = CircuitBuilder(n, n if measure else 0)
    for q in range(0, n, 6):
        builder.h(q).t(q).h(q)
    for q in range(n - 1):
        builder.cx(q, q + 1)
    if measure:
        builder.measure_all()
    return builder.build()


chain = kicked_chain(60, measure=False)
values = (
    simulate(chain)
    .backend(BackendKind.deterministic_pauli())
    .seed(42)
    .expectation_values_reported([[(59, "Z")]])
)
print(f"{chain.t_count()} T gates, <Z59> = {values.values[0]:.5f}, exact: {values.metadata.is_exact}")
shots = simulate(kicked_chain(60, measure=True)).seed(42).shots(2000)
print(f"shots on {shots.metadata.backend}: P(q59 = 1) = {shots.shots[:, 59].mean():.4f}")

wide = circuits.hardware_efficient_ansatz(60, 2)
for kind in [BackendKind.auto(), BackendKind.mps(2)]:
    result = simulate(wide).backend(kind).seed(42).expectation_values_reported([[(0, "Z")]])
    meta = result.metadata
    print(
        f"{meta.backend}: <Z0> = {result.values[0]:.4f}, exact {meta.is_exact}, "
        f"fidelity bound {meta.fidelity_lower_bound}, bond peak {meta.bond.peak} "
        f"of cap {meta.bond.cap}, saturated {meta.bond.saturated}"
    )

try:
    simulate(wide).seed(42).require_exact().expectation_values([[(0, "Z")]])
except PrismError as exc:
    print("require_exact:", exc.kind)
