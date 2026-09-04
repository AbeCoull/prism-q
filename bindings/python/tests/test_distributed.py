"""Distributed backend reach from Python.

Nothing here starts or stops MPI: mpi4py owns `MPI_Init` and its atexit
`MPI_Finalize`, and this extension only attaches to the world it finds. Cases
needing a live world skip without mpi4py and skip again on a world of one rank,
so a green run outside `mpiexec` means "not tested". The four-rank case is
gated together with the Rust one by `scripts/test-mpi.ps1`.
"""

import pytest

from prism_q import BackendKind, CircuitBuilder, DistributedContext, PrismError, simulate

SUPPORTED = DistributedContext.is_supported()

requires_build = pytest.mark.skipif(
    not SUPPORTED, reason="built without the distributed-mpi feature"
)


def world():
    """The mpi4py world communicator, skipping the case when it is absent."""
    pytest.importorskip("mpi4py", reason="mpi4py owns MPI_Init; without it there is no world")
    from mpi4py import MPI

    return MPI.COMM_WORLD


def check_circuit(n, start=0):
    """A circuit spanning local and global qubits at every rank count.

    `start` moves the opening Hadamard without changing the width or the
    instruction count, which is what the circuit-agreement case needs: the
    configuration check cannot tell two such circuits apart.
    """
    b = CircuitBuilder(n)
    b.h(start)
    for q in range(n - 1):
        b.cx(q, q + 1)
    b.rx(0.7, n - 1).ry(0.3, 0)
    return b.build()


@pytest.mark.skipif(SUPPORTED, reason="this build has MPI support")
def test_constructor_names_the_missing_feature():
    with pytest.raises(PrismError, match="distributed-mpi"):
        DistributedContext()


@requires_build
def test_context_agrees_with_mpi4py():
    comm = world()
    context = DistributedContext()
    assert context.rank == comm.Get_rank()
    assert context.size == comm.Get_size()


@requires_build
def test_a_world_of_one_rank_is_loud():
    comm = world()
    if comm.Get_size() != 1:
        pytest.skip("this case is the forgotten-mpiexec world")
    context = DistributedContext()
    with pytest.raises(PrismError, match="allow_single_rank"):
        BackendKind.statevector_distributed(context)
    BackendKind.statevector_distributed(context, allow_single_rank=True)


@requires_build
def test_every_rank_sees_the_single_process_probabilities():
    comm = world()
    if comm.Get_size() < 2:
        pytest.skip("needs a world larger than one rank; launch under mpiexec")
    n = 20
    circuit = check_circuit(n)
    context = DistributedContext()
    distributed = (
        simulate(circuit)
        .backend(BackendKind.statevector_distributed(context))
        .seed(42)
        .run()
    )
    local = simulate(circuit).backend(BackendKind.statevector()).seed(42).run()
    assert len(distributed.probabilities) == len(local.probabilities)
    for got, want in zip(distributed.probabilities, local.probabilities):
        assert abs(got - want) < 1e-10


@requires_build
def test_ranks_running_different_circuits_error_rather_than_hang():
    comm = world()
    if comm.Get_size() < 2:
        pytest.skip("needs a world larger than one rank; launch under mpiexec")
    n = 20
    circuit = check_circuit(n, start=1 if comm.Get_rank() == 1 else 0)
    context = DistributedContext()
    with pytest.raises(PrismError, match="same circuit"):
        simulate(circuit).backend(
            BackendKind.statevector_distributed(context)
        ).seed(42).run()
