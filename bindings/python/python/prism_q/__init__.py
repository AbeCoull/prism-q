"""PRISM-Q: performance-first quantum circuit simulator.

Thin Python bindings over the Rust ``prism-q`` crate. The compiled extension
``prism_q._prism_q`` provides the implementation; this package re-exports it.
"""

import sys as _sys

from ._prism_q import (
    __version__,
    BackendKind,
    Circuit,
    CircuitBuilder,
    CountsResult,
    Gate,
    NoiseChannel,
    NoiseModel,
    PrismError,
    QecBasis,
    QecNoise,
    QecProgram,
    QecResult,
    RecordRef,
    RunOutcome,
    ShotsResult,
    Simulation,
    circuits,
    parse_qasm,
    run_qasm,
    simulate,
)

_sys.modules[__name__ + ".circuits"] = circuits

__all__ = [
    "__version__",
    "BackendKind",
    "Circuit",
    "CircuitBuilder",
    "CountsResult",
    "Gate",
    "NoiseChannel",
    "NoiseModel",
    "PrismError",
    "QecBasis",
    "QecNoise",
    "QecProgram",
    "QecResult",
    "RecordRef",
    "RunOutcome",
    "ShotsResult",
    "Simulation",
    "circuits",
    "parse_qasm",
    "run_qasm",
    "simulate",
]
