"""PRISM-Q: performance-first quantum circuit simulator.

Thin Python bindings over the Rust ``prism-q`` crate. The compiled extension
``prism_q._prism_q`` provides the implementation; this package re-exports it.
"""

import os as _os
import sys as _sys


def _add_cuda_dll_directory():
    """Put the CUDA toolkit on the Windows DLL search path.

    Only a build carrying the `gpu` feature links CUDA, and Windows resolves an
    extension's dependencies from the DLL search path rather than from PATH.
    Called on import failure rather than unconditionally, so the CUDA-free
    wheels leave the search path alone and no unrelated package resolves its
    own CUDA libraries against the system toolkit. The returned handle must
    stay alive: dropping it calls ``RemoveDllDirectory``.
    """
    if _sys.platform != "win32":
        return None
    root = _os.environ.get("CUDA_PATH")
    if not root or not _os.path.isdir(_os.path.join(root, "bin")):
        return None
    return _os.add_dll_directory(_os.path.join(root, "bin"))


try:
    from . import _prism_q
except ImportError:
    _cuda_dll_dir = _add_cuda_dll_directory()
    if _cuda_dll_dir is None:
        raise
    from . import _prism_q

from ._prism_q import (
    __version__,
    BackendKind,
    Circuit,
    CircuitBuilder,
    CountsResult,
    DetectorErrorModel,
    Gate,
    GpuContext,
    NoiseChannel,
    NoiseModel,
    PrismError,
    QecBasis,
    QecNoise,
    QecProgram,
    QecResult,
    RecordRef,
    RunMetadata,
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
    "DetectorErrorModel",
    "Gate",
    "GpuContext",
    "NoiseChannel",
    "NoiseModel",
    "PrismError",
    "QecBasis",
    "QecNoise",
    "QecProgram",
    "QecResult",
    "RecordRef",
    "RunMetadata",
    "RunOutcome",
    "ShotsResult",
    "Simulation",
    "circuits",
    "parse_qasm",
    "run_qasm",
    "simulate",
]
