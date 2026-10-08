"""Run each script in bindings/python/examples the way a reader would."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py")) if EXAMPLES.exists() else []


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda script: script.stem)
def test_example_runs(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "the example printed nothing"
