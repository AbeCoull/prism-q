"""Load NVRTC from the ``nvidia-cuda-nvrtc-cu12`` package the ``cuda12`` extra installs.

The extension opens NVRTC by file name when it compiles kernels, through the platform
loader's default search, which never looks inside site-packages. Loading the packaged
copy by full path makes that lookup resolve to it, since both loaders return an already
loaded library whose name matches. ``GpuContext`` calls this only after opening a device
failed for want of NVRTC, so a cached kernel image or a toolkit NVRTC on the loader path
comes first and the packaged copy stays unloaded otherwise.
"""

import ctypes
import functools
import glob
import os
import sys

_handles = []


@functools.cache
def preload_nvrtc():
    """Load the packaged NVRTC once; return its directory, or ``None`` when absent."""
    try:
        import nvidia
    except ImportError:
        return None
    if sys.platform == "win32":
        subdir, library, companions = "bin", "nvrtc64_120_0.dll", "nvrtc-builtins64_*.dll"
    else:
        subdir, library, companions = "lib", "libnvrtc.so.12", None
    for root in nvidia.__path__:
        directory = os.path.join(root, "cuda_nvrtc", subdir)
        path = os.path.join(directory, library)
        if not os.path.isfile(path):
            continue
        if companions:
            _handles.extend(ctypes.CDLL(p) for p in glob.glob(os.path.join(directory, companions)))
        _handles.append(ctypes.CDLL(path))
        return directory
    return None
