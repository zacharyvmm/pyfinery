"""The structural checks the JVM engine compiles with the rules (pure Python: no JVM)."""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import Field

from pyfinery import Container, Contains, Metamodel, Ref, RefineryNode
from pyfinery._mapping import Catalogue
from pyfinery.jvm import _cache
from pyfinery.jvm._structure import CONTAINS, INVALID_CONTAINER, MAX_JOINED, Check, structure

from . import graph_models


def _structure(*classes: type):
    schema = Metamodel(*classes).schema
    return structure(schema, Catalogue(schema))


def _lines(text: str) -> dict[str, str]:
    """The predicates of ``text`` by name, each its whole line."""
    found = {}
    for line in text.splitlines():
        assert line.startswith("pred ") and line.endswith("."), line
        found[line.removeprefix("pred ").split("(")[0]] = line
    return found


class Exact(RefineryNode):
    __node_id__ = "name"
    name: str
    one: Annotated[Exact | None, Ref(multiplicity="1")] = None
    pair: Annotated[list[Exact], Ref(multiplicity="2..4")] = Field(default_factory=list)
    many: Annotated[list[Exact], Ref(multiplicity="0..*")] = Field(default_factory=list)
    edge: Annotated[list[Exact], Ref(multiplicity=f"0..{MAX_JOINED - 1}")] = Field(
        default_factory=list
    )
    lowest: Annotated[list[Exact], Ref(multiplicity=f"{MAX_JOINED}..*")] = Field(
        default_factory=list
    )


class Wide(RefineryNode):
    __node_id__ = "name"
    name: str
    upper: Annotated[list[Wide], Ref(multiplicity=f"0..{MAX_JOINED}")] = Field(default_factory=list)
    lower: Annotated[list[Wide], Ref(multiplicity=f"{MAX_JOINED + 1}..*")] = Field(
        default_factory=list
    )


class Box(RefineryNode):
    __node_id__ = "name"
    name: str
    items: Annotated[list[Item], Contains(opposite="box")] = Field(default_factory=list)
    extra: Annotated[list[Item], Contains()] = Field(default_factory=list)


class Item(RefineryNode):
    __node_id__ = "name"
    name: str
    box: Annotated[Box | None, Container(opposite="items")] = None


def test_no_containment_no_containment_checks():
    found = _structure(Exact)
    assert not found.containment
    assert CONTAINS not in found.text and INVALID_CONTAINER not in found.text
    assert {check.kind for check in found.checks} == {"multiplicity"}
    assert found.unchecked == []


def test_multiplicity_checks():
    found = _structure(Exact)
    lines = _lines(found.text)
    checks = {check.constraint: check for check in found.checks}
    # 0..* can't be broken: no check.
    assert set(checks) == {
        f"Exact::{name}::invalidMultiplicity" for name in ("one", "pair", "edge", "lowest")
    }
    for check in checks.values():
        assert check.kind == "multiplicity"
        assert check.predicate in lines
        assert lines[check.predicate].startswith(f"pred {check.predicate}(Exact x) <-> ")

    # Exactly one: none, or two distinct.
    one = lines[checks["Exact::one::invalidMultiplicity"].predicate]
    assert one.endswith("<-> !Exact::one(x, _) ; Exact::one(x, y0), Exact::one(x, y1), y0 != y1.")

    # 2..4: fewer than two distinct (a helper), or five distinct.
    pair = checks["Exact::pair::invalidMultiplicity"].predicate
    helper = lines[f"{pair}AtLeast"]
    assert helper == (
        f"pred {pair}AtLeast(Exact x) <-> Exact::pair(x, y0), Exact::pair(x, y1), y0 != y1."
    )
    body = lines[pair].split(" <-> ")[1].removesuffix(".")
    first, second = body.split(" ; ")
    assert first == f"!{pair}AtLeast(x)"
    assert second.count("Exact::pair(x, ") == 5
    assert second.count(" != ") == 10  # every pair of five targets

    # The largest bounds that are still joined.
    edge = lines[checks["Exact::edge::invalidMultiplicity"].predicate]
    assert edge.count("Exact::edge(x, ") == MAX_JOINED and " ; " not in edge
    lowest = checks["Exact::lowest::invalidMultiplicity"].predicate
    assert lines[f"{lowest}AtLeast"].count("Exact::lowest(x, ") == MAX_JOINED
    assert lines[lowest].endswith(f"<-> !{lowest}AtLeast(x).")


def test_multiplicity_beyond_max_joined_is_unchecked():
    found = _structure(Wide)
    assert found.unchecked == [
        f"Wide::upper [0..{MAX_JOINED}]",
        f"Wide::lower [{MAX_JOINED + 1}..*]",
    ]
    assert found.checks == [] and found.text == ""


def test_containment_checks():
    found = _structure(Box)
    assert found.containment
    lines = _lines(found.text)
    assert lines[CONTAINS] == (
        f"pred {CONTAINS}(node x, node y) <-> Box::items(x, y) ; Box::extra(x, y)."
    )
    invalid = lines[INVALID_CONTAINER]
    assert invalid.startswith(f"pred {INVALID_CONTAINER}(contained y) <-> ")
    # No container; several; on a cycle.
    assert f"!{CONTAINS}(_, y)" in invalid
    assert f"{CONTAINS}(a, y), {CONTAINS}(b, y), a != b" in invalid
    assert f"{CONTAINS}+(y, y)" in invalid

    by_kind: dict[str, list[Check]] = {}
    for check in found.checks:
        by_kind.setdefault(check.kind, []).append(check)
    assert by_kind["containment"] == [
        Check(INVALID_CONTAINER, "containment", "builtin::invalidContainer")
    ]
    # A contradiction for each containment reference, and each container reference.
    contradictions = {c.constraint: c.predicate for c in by_kind["contradiction"]}
    assert set(contradictions) == {"Box::items", "Box::extra", "Item::box"}
    for constraint, predicate in contradictions.items():
        line = lines[predicate]
        assert line.startswith(f"pred {predicate}(node x, node y) <-> {constraint}(x, y), ")
    assert lines[contradictions["Box::items"]].endswith(f"{CONTAINS}+(y, x).")
    assert lines[contradictions["Item::box"]].endswith(f"{CONTAINS}+(x, y).")
    # Containment references have no bounds here, and a container reference is never checked.
    assert "multiplicity" not in by_kind
    assert found.unchecked == []


def test_graph_models_structure():
    schema = graph_models.metamodel().schema
    found = structure(schema, Catalogue(schema))
    constraints = {(c.kind, c.constraint) for c in found.checks}
    assert ("multiplicity", "Filesystem::root::invalidMultiplicity") in constraints
    assert ("multiplicity", "Dir::contents::invalidMultiplicity") in constraints
    assert ("multiplicity", "Link::target::invalidMultiplicity") in constraints
    assert ("contradiction", "FSObject::parent") in constraints
    # Unbounded references and attributes have no check.
    assert not any("backups" in c or "size" in c for _, c in constraints)
    # Every check's predicate is defined, once.
    lines = _lines(found.text)
    assert len(lines) == len(found.text.splitlines())
    assert all(check.predicate in lines for check in found.checks)


# -- the compile cache, with a stand-in for CompiledProblem ---------------------------------------


class _Fake:
    made = 0

    def __init__(self, schema, rules, predicates=()):
        type(self).made += 1
        self.schema = schema
        self.rules = list(rules)


@pytest.fixture
def fake_compile(monkeypatch):
    monkeypatch.setattr(_cache, "CompiledProblem", _Fake)
    _Fake.made = 0
    _cache.clear()
    yield _Fake
    _cache.clear()


def test_cache_reuses_and_evicts(fake_compile, monkeypatch):
    from pyfinery import Predicate

    schema = graph_models.metamodel().schema
    first = _cache.compiled(schema, ["a"])
    assert _cache.compiled(schema, ("a",)) is first  # any sequence of the same rules
    assert _cache.compiled(schema, ["b"]) is not first
    # Predicates count by name and parameters, not by their facts.
    p1 = _cache.compiled(schema, ["a"], [Predicate("p", (graph_models.Dir,), tuples=[])])
    p2 = _cache.compiled(
        schema, ["a"], [Predicate("p", (graph_models.Dir,), evaluate=lambda d: True)]
    )
    assert p1 is p2 and p1 is not first
    # Another schema object compiles anew.
    assert _cache.compiled(graph_models.metamodel().schema, ["a"]) is not first
    assert fake_compile.made == 4

    monkeypatch.setattr(_cache, "SIZE", 2)
    _cache.clear()
    a = _cache.compiled(schema, ["a"])
    _cache.compiled(schema, ["b"])
    assert _cache.compiled(schema, ["a"]) is a  # now the most recently used
    _cache.compiled(schema, ["c"])  # evicts b
    assert _cache.compiled(schema, ["a"]) is a
    made = fake_compile.made
    _cache.compiled(schema, ["b"])
    assert fake_compile.made == made + 1
