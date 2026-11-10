"""Build Stim from its source release with the AVX2 extension and install it into this interpreter.

    comparison/.venv/Scripts/python comparison/build_stim.py

Stim's release enables only its SSE2 and portable extensions; this build carries the AVX2
extension alone, so it needs an AVX2 host. The Windows wheel compiles
`_stim_sse2` with MSVC `/arch:SSE2`, which leaves `__SSE2__` undefined, so Stim's
`simd_word.h` selects 64-bit words. This builds the same source with the `_stim_avx2`
extension and import that Stim ships commented out, so `__AVX2__` selects 256-bit words.
The wheel is tagged `+avx2` so the run's package versions show which build ran.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tarfile
import tempfile

VERSION = "1.16.0"
LOCAL_TAG = "avx2"
# Stim 1.16.0's bindings fail to compile against pybind11 3.x (C3487 in tableau_simulator.pybind.cc).
PYBIND11 = "2.13.6"


def _uncomment(text: str, start: str, end: str) -> str:
    """Strip the leading `# ` from every line from the marker `start` up to the marker `end`."""
    i = text.find(start)
    j = text.find(end, i)
    if i < 0 or j < 0:
        raise SystemExit(f"markers not found: {start!r} .. {end!r}")
    body = "\n".join(line[2:] if line.startswith("# ") else line for line in text[i:j].split("\n"))
    return text[:i] + body + text[j:]


def patch(src: str) -> None:
    setup_py = os.path.join(src, "setup.py")
    with open(setup_py, encoding="utf-8") as handle:
        text = handle.read()
    text = _uncomment(text, "# stim_avx2 = Extension(", "\nwith open(")
    # Only the AVX2 extension is built: the extensions share object paths, so they cannot build in
    # parallel, and the harness needs an AVX2 host. Stim picks its x86 extensions from
    # platform.processor(), which on Windows names the CPU family only.
    text = text.replace(
        "        return [stim_detect_machine_architecture, stim_polyfill,\n                # stim_avx2,\n"
        "                stim_sse2]",
        "        return [stim_detect_machine_architecture, stim_avx2]",
    )
    text = text.replace("platform.processor().lower()", "(platform.processor() + platform.machine()).lower()")
    text = text.replace(
        f"__version__ = '{VERSION}'", f"__version__ = '{VERSION}'\nwheel_version = '{VERSION}+{LOCAL_TAG}'"
    )
    text = text.replace("    version=__version__,", "    version=wheel_version,")
    if "stim_detect_machine_architecture, stim_avx2]" not in text or "wheel_version" not in text:
        raise SystemExit("setup.py did not take the AVX2 patch")
    with open(setup_py, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    init = os.path.join(src, "glue", "python", "src", "stim", "__init__.py")
    with open(init, encoding="utf-8") as handle:
        text = handle.read()
    disabled = (
        "    # if _tmp == 'avx2':\n"
        "    #     from stim._stim_avx2 import _UNSTABLE_raw_format_data, __version__\n"
        "    #     from stim._stim_avx2 import *\n"
        "    if _tmp == 'avx2' or _tmp == 'sse2':\n"
    )
    enabled = (
        "    if _tmp == 'avx2':\n"
        "        from stim._stim_avx2 import _UNSTABLE_raw_format_data, __version__\n"
        "        from stim._stim_avx2 import *\n"
        "    elif _tmp == 'sse2':\n"
    )
    if disabled not in text:
        raise SystemExit("stim/__init__.py did not take the AVX2 patch")
    with open(init, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace(disabled, enabled))


def main() -> int:
    work = tempfile.mkdtemp(prefix="stim-build-")
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(
        [*pip, "download", f"stim=={VERSION}", "--no-binary", ":all:", "--no-deps", "--no-build-isolation", "-d", work],
        check=True,
    )
    with tarfile.open(os.path.join(work, f"stim-{VERSION}.tar.gz")) as archive:
        archive.extractall(work, filter="data")
    src = os.path.join(work, f"stim-{VERSION}")
    patch(src)
    headers = os.path.join(work, "pybind11")
    subprocess.run([*pip, "install", "--target", headers, f"pybind11=={PYBIND11}"], check=True)
    env = {**os.environ, "PYTHONPATH": headers}
    wheels = os.path.join(work, "wheels")
    subprocess.run([*pip, "wheel", src, "--no-build-isolation", "--no-deps", "-w", wheels], check=True, env=env)
    wheel = glob.glob(os.path.join(wheels, "stim-*.whl"))[0]
    subprocess.run([*pip, "install", "--force-reinstall", "--no-deps", wheel], check=True)
    check = "import sys, stim; assert 'stim._stim_avx2' in sys.modules, sorted(sys.modules); print(stim.__version__)"
    subprocess.run([sys.executable, "-c", check], check=True)
    print(f"built in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
