"""Execute the Python examples in docs/guides/python*.md, docs/tutorials and the READMEs.

Blocks fenced as ``python`` in the guide pages run in book order in one namespace,
so an example may use what an earlier one defined. ``python,ignore`` marks a block
that needs hardware or a launcher CI does not have. A block whose last line ends in
``# raises`` must raise ``PrismError``. A tutorial or README page starts from an empty
namespace, so its blocks import what they use, as a reader copying them would.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import prism_q

ROOT = Path(__file__).resolve().parents[3]
GUIDES = [
    ROOT / "docs" / "guides" / f"{name}.md"
    for name in (
        "python",
        "python-circuits",
        "python-running",
        "python-observables",
        "python-noise",
        "python-state",
        "python-qec",
        "python-names",
    )
]
TUTORIALS = ROOT / "docs" / "tutorials"
READMES = [ROOT / "README.md", ROOT / "bindings" / "python" / "README.md"]


def fenced_blocks(text):
    """Return ``(line, info, code)`` for each fenced block, ``line`` 1-based."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opening = re.match(r"^```(\S*)", lines[i])
        if opening is None:
            i += 1
            continue
        start = i
        i += 1
        while not lines[i].startswith("```"):
            i += 1
        blocks.append((start + 1, opening.group(1), "\n".join(lines[start + 1 : i])))
        i += 1
    return blocks


@pytest.mark.skipif(not GUIDES[0].exists(), reason="needs the repository checkout")
def test_python_guide_examples_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {name: getattr(prism_q, name) for name in prism_q.__all__}
    namespace["prism_q"] = prism_q
    ran = 0
    blocks = [
        (guide, line, info, code)
        for guide in GUIDES
        for line, info, code in fenced_blocks(guide.read_text(encoding="utf-8"))
    ]
    for guide, line, info, code in blocks:
        if info != "python":
            continue
        source = compile(code, f"{guide.name}:{line}", "exec")
        if code.rstrip().endswith("# raises"):
            with pytest.raises(prism_q.PrismError):
                exec(source, namespace)
        else:
            try:
                exec(source, namespace)
            except Exception as exc:
                pytest.fail(f"{guide.name} block at line {line} failed: {exc!r}")
        ran += 1
    assert ran > 0


def example_pages():
    pages = [page for page in READMES if page.exists()]
    if TUTORIALS.exists():
        pages += sorted(TUTORIALS.glob("*.md"))
    return [page for page in pages if "```python\n" in page.read_text(encoding="utf-8")]


@pytest.mark.parametrize(
    "page", example_pages(), ids=lambda page: page.relative_to(ROOT).as_posix()
)
def test_page_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    for line, info, code in fenced_blocks(page.read_text(encoding="utf-8")):
        if info != "python":
            continue
        source = compile(code, f"{page.name}:{line}", "exec")
        if code.rstrip().endswith("# raises"):
            with pytest.raises(prism_q.PrismError):
                exec(source, namespace)
        else:
            exec(source, namespace)
