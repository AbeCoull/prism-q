import ast
from pathlib import Path

import prism_q
from prism_q import _prism_q


def extension_names():
    return {n for n in dir(_prism_q) if not n.startswith("_")} | {"__version__"}


def test_every_extension_name_is_reexported():
    assert extension_names() - set(prism_q.__all__) == set()
    for name in prism_q.__all__:
        assert getattr(prism_q, name) is getattr(_prism_q, name), name


def test_every_extension_name_is_in_the_stub():
    stub = ast.parse((Path(prism_q.__file__).parent / "_prism_q.pyi").read_text())
    declared = set()
    for node in stub.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
    assert extension_names() - declared == set()
