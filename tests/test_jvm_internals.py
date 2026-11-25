"""The JVM package's own data types, without a JVM: Java classes, helpers, compiled problems and
model snapshots, with stand-ins for what the JVM would give."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import pyfinery.jvm.runtime as runtime
from pyfinery._instances import build_graph
from pyfinery.jvm import compile as _compile
from pyfinery.jvm.classes import JavaClasses, java
from pyfinery.jvm.compile import CompiledProblem, ErrorQuery
from pyfinery.jvm.mapping import Catalogue
from pyfinery.jvm.model import ModelSnapshot, PlainModel
from pyfinery.jvm.structure import Check

from . import graph_models


class FakeJPype:
    """JClass gives back the name of the class it is asked for."""

    def JClass(self, name: str) -> str:  # noqa: N802 (JPype's name)
        return name


@pytest.fixture
def no_jvm(monkeypatch):
    monkeypatch.setattr(runtime, "start", lambda: None)
    monkeypatch.setattr(runtime, "_jpype", FakeJPype)
    monkeypatch.setattr(runtime, "_helpers", None)


def test_java_classes_are_listed_once_with_their_java_names():
    entries = dataclasses.fields(JavaClasses)
    paths = [entry.metadata["java"] for entry in entries]
    assert len(entries) == 50
    assert len(set(paths)) == len(paths)
    assert all(path.startswith(("tools.refinery.", "java.")) for path in paths)


def test_java_resolves_every_class(no_jvm):
    classes = java.__wrapped__()
    assert isinstance(classes, JavaClasses)
    assert classes.Dnf == "tools.refinery.logic.dnf.Dnf"
    assert classes.ArrayList == "java.util.ArrayList"
    with pytest.raises(dataclasses.FrozenInstanceError):
        classes.Dnf = None  # type: ignore[misc]


def test_helpers_are_resolved_once(no_jvm):
    helpers = runtime.helpers()
    assert isinstance(helpers, runtime.JvmHelpers)
    assert (helpers.Bulk, helpers.Natives) == ("pyfinery.jvm.Bulk", "pyfinery.jvm.Natives")
    assert runtime.helpers() is helpers


def _problem(errors: list[ErrorQuery], symbols: dict[str, Any]) -> CompiledProblem:
    schema = graph_models.metamodel().schema
    return CompiledProblem(
        schema=schema,
        catalogue=Catalogue(schema),
        source="",
        timings={},
        reserved_ids=0,
        constants={},
        refused=[],
        parameters={},
        attributes={},
        queries={},
        unsupported={},
        errors=errors,
        store=None,
        _symbols=symbols,
    )


def test_compiled_problem_is_frozen_and_looks_symbols_up():
    problem = _problem(
        [
            ErrorQuery("pyfinery_m", "multiplicity", "Dir::contents::invalidMultiplicity", 0),
            ErrorQuery("emptyDir", "rule", "emptyDir", 1),
        ],
        {"Dir": "the symbol of Dir"},
    )
    assert problem.symbol("Dir") == "the symbol of Dir"
    with pytest.raises(KeyError, match="'Nothing' has no plain symbol"):
        problem.symbol("Nothing")
    assert problem.rule_errors == ["emptyDir"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.source = "changed"  # type: ignore[misc]
    assert problem == problem and problem != _problem([], {})  # compared by identity
    assert len({problem}) == 1


def test_error_queries_follow_the_relations_and_list_each_check_once():
    multiplicity = Check("pyfinery_m", "multiplicity", "Dir::contents::invalidMultiplicity")
    containment = Check("pyfinery_c", "containment", "builtin::invalidContainer")
    order = {"emptyDir": 3, "Dir::contents::invalidMultiplicity": 1, "invalidContainer": 0}
    errors = _compile._error_queries(["emptyDir"], [multiplicity, containment, multiplicity], order)
    assert [(e.predicate, e.kind, e.order) for e in errors] == [
        ("pyfinery_c", "containment", 0),
        ("pyfinery_m", "multiplicity", 1),
        ("emptyDir", "rule", 3),
    ]


class FakeTrace:
    """A problem trace that strips Refinery quotes from every segment of a relation name."""

    class Partial:
        def __init__(self, name: str) -> None:
            self._name = name

        def name(self) -> str:
            return self._name

    def getPartialRelation(self, name: str) -> Partial:  # noqa: N802 (Refinery's name)
        if "'" not in name:
            raise LookupError(name)  # as a Java exception would be
        return self.Partial("::".join(part.strip("'") for part in name.split("::")))


def test_symbols_are_also_found_under_rendered_names():
    from pydantic import BaseModel

    from pyfinery import Metamodel, Predicate

    class Item(BaseModel):
        pass

    class Box(BaseModel):
        default: Item | None = None

    mm = Metamodel(Box)
    catalogue = Catalogue(mm.schema)
    predicates = [Predicate("odd name", (Box,), lambda _: True)]
    written = _compile._written_relations(catalogue, predicates)
    assert "Box::'default'" in written
    assert "'odd name'" in written

    symbols = {"Box::default": "feature-symbol", "odd name": "pred-symbol"}
    named = _compile._with_rendered_names(symbols, FakeTrace(), written)
    assert named["Box::'default'"] == "feature-symbol"
    assert named["'odd name'"] == "pred-symbol"
    assert symbols == {"Box::default": "feature-symbol", "odd name": "pred-symbol"}


def test_checkpoint_preserves_model_snapshot_accessors():
    from pyfinery.jvm import Checkpoint

    snapshot = ModelSnapshot(
        version=3,
        facts=frozenset({("Dir", (5,), True)}),
        known={"a": 5},
        next_id=6,
        nodes_by_id={},
    )
    checkpoint = Checkpoint(model=snapshot, roots={}, predicates=(), reported={})
    assert checkpoint.version == 3
    assert checkpoint.facts == frozenset({("Dir", (5,), True)})
    assert checkpoint.known == {"a": 5}
    assert checkpoint.next_id == 6


class FakeModel:
    """A Refinery model's versions: each commit is a new one."""

    def __init__(self) -> None:
        self.commits = 0
        self.restored: list[int] = []

    def commit(self) -> int:
        self.commits += 1
        return self.commits

    def restore(self, version: int) -> None:
        self.restored.append(version)


def test_snapshot_and_restore_bring_back_the_model_and_its_nodes():
    schema = graph_models.metamodel().schema
    first = build_graph(schema, [graph_models.Dir(name="first")]).nodes[0]
    second = build_graph(schema, [graph_models.Dir(name="second")]).nodes[0]
    model = object.__new__(PlainModel)  # no store: only what snapshots hold
    model.model = FakeModel()
    model._constant_names = {}
    model.facts = {("Dir", (5,), True)}
    model.known = {first.identity: 5}
    model.next_id = 6
    model._nodes_by_id = {5: first}

    snapshot = model.snapshot()
    assert isinstance(snapshot, ModelSnapshot)
    assert snapshot.version == 1
    model.facts.add(("Dir", (6,), True))
    model.known[second.identity] = 6
    model.next_id = 7
    model._nodes_by_id = {6: second}
    assert snapshot.facts == {("Dir", (5,), True)} and len(snapshot.known) == 1

    model.restore(snapshot)
    assert model.model.restored == [1]
    assert model.facts == {("Dir", (5,), True)}
    assert model.known == {first.identity: 5}
    assert model.next_id == 6
    assert model.label((5,)) == (("first",), (first.obj,))
    model.facts.add(("Dir", (7,), True))  # the model's own copy, not the snapshot's
    assert snapshot.facts == {("Dir", (5,), True)}
