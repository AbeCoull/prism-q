"""Rust-to-Python parity: every tracked item of the Rust public API has a decision.

The Rust surface comes from docs/architecture/public-api.txt, which CI regenerates
for every new public item, and bindings/python/parity.toml records what each
tracked item is called in Python or why it stays Rust-only.
"""

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest

import prism_q

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO / "docs" / "architecture" / "public-api.txt"
MANIFEST = REPO / "bindings" / "python" / "parity.toml"
GUIDE = REPO / "docs" / "guides" / "python.md"

DECL = re.compile(r"^(?:#\[[^\]]*\] )?pub (struct|enum|trait|mod|static|use|union) ")


def strip_generics(text):
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def item_key(path):
    """`Type::member` for an associated item, else the module path below the root."""
    segments = path.split("::")[1:]
    owner = next((i for i, s in enumerate(segments) if s[:1].isupper()), None)
    return "::".join(segments if owner is None else segments[owner:])


def rust_items(text):
    """Keys of the inherent functions, fields and variants in a public-api snapshot.

    Methods of a trait definition or a trait impl are skipped, as are fields of
    enum variants. A function re-exported at the crate root keys by its root name.
    """
    keys = set()
    trait_owner = None
    for line in text.splitlines():
        if line.startswith("impl"):
            head = strip_generics(line)
            trait_owner = None
            if " for " in head:
                trait_owner = head.split(" for ", 1)[1].split()[-1].lstrip("&")
            continue
        decl = DECL.match(line)
        if decl:
            trait_owner = None
            if decl.group(1) == "trait":
                trait_owner = strip_generics(line[decl.end():]).split(":", 1)[0].strip()
            continue
        if line.startswith("pub fn "):
            path = strip_generics(line[len("pub fn "):].split("(", 1)[0])
            if trait_owner is not None and path.rsplit("::", 1)[0] == trait_owner:
                continue
        elif line.startswith("pub prism_q::"):
            path = strip_generics(line[len("pub "):].split(": ", 1)[0].split("(", 1)[0])
        else:
            continue
        key = item_key(path)
        if key[:1].isupper() and key.count("::") > 1:
            continue
        keys.add(key)
    return {
        key
        for key in keys
        if key[:1].isupper() or "::" not in key or key.rsplit("::", 1)[1] not in keys
    }


def in_scope(key, scope):
    if any(fnmatch.fnmatchcase(key, pattern) for pattern in scope["exclude"]):
        return False
    if key[:1].isupper():
        return key.split("::", 1)[0] in scope["types"]
    return True


def load_manifest():
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def tracked_items():
    manifest = load_manifest()
    items = rust_items(SNAPSHOT.read_text(encoding="utf-8"))
    return {key for key in items if in_scope(key, manifest["scope"])}


def resolve(dotted):
    target = prism_q
    for part in dotted.split("."):
        target = getattr(target, part)
    return target


pytestmark = pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="needs the repository checkout for the API snapshot"
)


def test_every_tracked_rust_item_has_a_decision():
    entries = load_manifest()["items"]
    missing = sorted(tracked_items() - set(entries))
    assert missing == [], (
        "Rust items without a Python decision; add each to bindings/python/parity.toml "
        "as a Python name or as { rust_only = \"reason\" }"
    )


def test_no_entry_outlives_its_rust_item():
    entries = load_manifest()["items"]
    assert sorted(set(entries) - tracked_items()) == []


def test_every_entry_is_a_name_or_a_reason():
    for key, entry in load_manifest()["items"].items():
        if isinstance(entry, dict):
            assert set(entry) == {"rust_only"}, key
            assert entry["rust_only"].strip(), key
        else:
            assert isinstance(entry, str) and entry, key


def test_every_mapped_python_name_resolves():
    unresolved = []
    for key, entry in load_manifest()["items"].items():
        if isinstance(entry, str):
            try:
                resolve(entry)
            except AttributeError:
                unresolved.append((key, entry))
    assert unresolved == []


def test_guide_table_matches_the_manifest():
    lines = GUIDE.read_text(encoding="utf-8").splitlines()
    start = lines.index("<!-- parity-table:start -->")
    end = lines.index("<!-- parity-table:end -->")
    rows = [line for line in lines[start + 1 : end] if line.startswith("| `")]
    table = {}
    for row in rows:
        rust, python = (cell.strip().strip("`") for cell in row.strip("|").split("|")[:2])
        table[rust] = python
    expected = {}
    for key, entry in load_manifest()["items"].items():
        if isinstance(entry, str) and not key[:1].isupper():
            expected[key] = entry
    assert table == expected
