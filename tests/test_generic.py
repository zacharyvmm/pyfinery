# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from pyfinery import ClassOptions, Metamodel, Ref, RefineryNode
from pyfinery._instances import build_graph

from .generic_models import (
    OPTIONS,
    Effect,
    Endpoint,
    File,
    Graph,
    Path,
    Unrelated,
    graph,
    metamodel,
)

EXPECTED_SOURCE = """\
enum EffectOp { read, write, remove }
enum PathAnchor { root, home, 'unknown' }
abstract class Resource.
class Model.
class Effect extends Model {
    EffectOp[1] op
    Resource[0..1] target
    Resource[0..*] sources
    boolean definite
    int index
}
class Endpoint extends Model, Resource {
    string url
}
class File extends Model, Resource {
    contains Path[1] path
}
class Graph extends Model {
    Resource[0..*] resources
    contains Effect[0..*] effects
}
class Path extends Model {
    PathAnchor[1] anchor
}
"""


def test_generic_source():
    mm = metamodel()
    assert mm.source == EXPECTED_SOURCE
    assert Unrelated not in mm.classes  # only reachable through the shared base
    assert [(c.__name__, f) for c, f, _ in mm.skipped] == [("Path", "segments")]


def test_skip_collections_of_none():
    class Holder(BaseModel):
        name: str
        nothing: list[None]
        empty: tuple[None, ...]

    mm = Metamodel(Holder, unsupported="skip")
    assert [field for _, field, _ in mm.skipped] == ["nothing", "empty"]
    with pytest.raises(TypeError, match="Holder.nothing: unsupported field type None"):
        Metamodel(Holder)


def test_keyed_copies_are_one_node():
    g = graph()
    mm = metamodel()
    nodes = mm.collect([g])
    assert sum(isinstance(n, File) for n in nodes) == 1
    assert sum(isinstance(n, Endpoint) for n in nodes) == 1
    facts = mm.problem([g]).removeprefix(EXPECTED_SOURCE).strip().splitlines()
    assert facts == [
        "Graph(graph_1).",
        "Graph::resources(graph_1, 'file:.ssh/id_rsa').",
        "Graph::resources(graph_1, 'endpoint:https://example.com').",
        "Graph::effects(graph_1, effect_1).",
        "Graph::effects(graph_1, effect_2).",
        "File('file:.ssh/id_rsa').",
        "File::path('file:.ssh/id_rsa', path_1).",
        "Endpoint('endpoint:https://example.com').",
        "Endpoint::url('endpoint:https://example.com'): \"https://example.com\".",
        "Effect(effect_1).",
        "Effect::op(effect_1, EffectOp::read).",
        "Effect::target(effect_1, 'file:.ssh/id_rsa').",
        "Effect::definite(effect_1): true.",
        "Effect::index(effect_1): 0.",
        "Effect(effect_2).",
        "Effect::op(effect_2, EffectOp::write).",
        "Effect::target(effect_2, 'endpoint:https://example.com').",
        "Effect::sources(effect_2, 'file:.ssh/id_rsa').",
        "Effect::definite(effect_2): true.",
        "Effect::index(effect_2): 1.",
        "Path(path_1).",
        "Path::anchor(path_1, PathAnchor::home).",
    ]


def test_node_name_and_abstract_options():
    mm = Metamodel(
        Graph,
        options={
            File: ClassOptions(key=lambda r: r.key, node_name=lambda r: r.key),
            Endpoint: ClassOptions(key=lambda r: r.key, node_name=lambda r: r.url),
            Effect: ClassOptions(exclude=["command", "trace"]),
            Path: ClassOptions(abstract=False),
        },
        unsupported="skip",
    )
    assert "abstract class FileOrEndpoint." in mm.source  # default union name
    names = {n.name for n in build_graph(mm.schema, [graph()]).nodes}
    assert {"'file:.ssh/id_rsa'", "'https://example.com'"} <= names


def test_unsupported_fields_raise_by_default():
    with pytest.raises(TypeError, match=r"Path.segments.*ClassOptions\(exclude"):
        Metamodel(Graph, options=OPTIONS)


def test_union_errors():
    class Owned(BaseModel):
        x: int = 0

    class Shared(RefineryNode):
        pass

    class Mixed(RefineryNode):
        items: list[Owned | Shared] = []

    with pytest.raises(TypeError, match="mixes owned value classes with entity classes"):
        Metamodel(Mixed)

    class Opp(RefineryNode):
        back: Annotated[Shared | Opp | None, Ref(opposite="x")] = None

    with pytest.raises(TypeError, match="unions can't have opposites"):
        Metamodel(Opp)

    class Numbers(RefineryNode):
        level: Literal[1, 2] = 1

    with pytest.raises(TypeError, match="Literals of strings"):
        Metamodel(Numbers)


def test_duplicate_class_names():
    def make():
        class File(RefineryNode):
            pass

        return File

    first, second = make(), make()

    class Both(RefineryNode):
        a: first | None = None  # type: ignore[valid-type]
        b: second | None = None  # type: ignore[valid-type]

    with pytest.raises(TypeError, match="both named 'File'"):
        Metamodel(Both)


def test_literal_enums_are_shared_by_value():
    class A(RefineryNode):
        mode: Literal["x", "y"] = "x"

    class B(RefineryNode):
        a: A | None = None
        mode: Literal["x", "y"] = "x"

    source = Metamodel(B).source
    assert source.count("enum ") == 1
    assert "enum BMode { x, y }" in source
    assert "BMode[1] mode" in source and source.count("BMode[1] mode") == 2


def test_report_for_node_matches_copies():
    from pyfinery.graph import ValidationReport, Violation

    mm = metamodel()
    stored = File(path=Path(anchor="home", segments=["a"]))
    report = ValidationReport(
        [Violation("rule", "r", ("'file:.ssh/id_rsa'",), (stored,))], "", None, mm.identity
    )
    copy = File(path=Path(anchor="home", segments=["a"]))
    assert report.for_node(copy) == report.violations
    assert report.for_node(File(path=Path(anchor="home", segments=["b"]))) == []


def test_effect_field_type_check():
    # Effects stay owned values (no key): each one is its own node.
    assert len([n for n in metamodel().collect([graph()]) if isinstance(n, Effect)]) == 2


def test_one_shot_iterable_fields_are_read_once():
    # Pydantic validates Iterable fields into one-shot iterators.
    class Thing(BaseModel):
        name: str

    class Group(BaseModel):
        children: Iterable[Thing]

    Group.model_rebuild(_types_namespace={"Thing": Thing})
    source = Metamodel(Group).problem([Group(children=[Thing(name="a")])])
    assert "Group::children(group_1, thing_1)." in source
    assert 'Thing::name(thing_1): "a".' in source
