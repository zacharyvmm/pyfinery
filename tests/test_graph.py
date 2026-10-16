# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

from __future__ import annotations

import asyncio
from typing import Annotated

import httpx
import pytest

import pyfinery as pf
from pyfinery import Container, Contains, Metamodel, Ref, RefineryNode
from pyfinery.graph import quote_id

from .conftest import Recorder, sse, sse_response
from .graph_models import Dir, File, Filesystem, Link, Permission, metamodel

EXPECTED_SOURCE = """\
enum Permission { read, write }
abstract class FSObject {
    container Dir parent opposite contents
}
class Filesystem {
    contains Dir[1] root
}
class Dir extends FSObject {
    string name
    contains FSObject[0..3] contents opposite parent
}
class File extends FSObject {
    int size
    real ratio
    boolean hidden
    string label
    Permission[0..1] permission
}
class Link extends FSObject {
    FSObject[1] target
    FSObject[0..*] backups
}

error pred emptyDir(Dir d) <-> !contents(d, _).
"""


def test_metamodel_source():
    assert metamodel().source == EXPECTED_SOURCE


def test_facts():
    home = Dir(name="my home")
    report = File(size=-3, ratio=2e-7, hidden=True, label='say "hi"\n', permission=Permission.read)
    home.contents = [report, Link(target=report)]
    problem = metamodel().problem([Filesystem(root=home)])
    facts = problem.removeprefix(EXPECTED_SOURCE).strip().splitlines()
    assert facts == [
        "Filesystem(filesystem_1).",
        "Filesystem::root(filesystem_1, 'my home').",
        "Dir('my home').",
        "Dir::name('my home'): \"my home\".",
        "Dir::contents('my home', file_1).",
        "Dir::contents('my home', link_1).",
        "File(file_1).",
        "File::size(file_1): -3.",
        "File::ratio(file_1): 2e-07.",
        "File::hidden(file_1): true.",
        'File::label(file_1): "say \\"hi\\"\\n".',
        "File::permission(file_1, Permission::read).",
        "Link(link_1).",
        "Link::target(link_1, file_1).",
    ]


def test_quote_id():
    assert quote_id("plain_1") == "plain_1"
    assert quote_id("class") == "'class'"
    assert quote_id("it's") == "'it\\'s'"
    assert quote_id("1st") == "'1st'"


def test_generated_names_avoid_explicit_ids():
    taken = Dir(name="filesystem_1")
    names = metamodel().node_names([taken, Filesystem(root=Dir(name="x"))])
    assert sorted(names.values()) == ["filesystem_1", "filesystem_2"]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate node id"):
        metamodel().problem([Dir(name="a", contents=[Dir(name="a")])])


def test_abstract_class_cannot_be_instantiated():
    from .graph_models import FSObject

    with pytest.raises(TypeError, match="abstract"):
        FSObject()


def test_nodes_use_identity_and_safe_repr():
    a, b = Dir(name="a"), Dir(name="a")
    assert a != b and len({a, b}) == 2
    a.contents = [b]
    b.parent = a
    assert "<Dir at" in repr(a)


def test_collect_follows_references():
    file = File()
    link = Link(target=file)
    assert metamodel().collect([link]) == [link, file]


def test_unknown_class_rejected():
    class Stray(RefineryNode):
        pass

    with pytest.raises(TypeError, match="not part of this metamodel"):
        metamodel().collect([Stray()])


def test_metamodel_definition_errors():
    class A(RefineryNode):
        b: Annotated[list[B], Ref(opposite="a")] = []

    class B(RefineryNode):
        a: A | None = None

    A.model_rebuild(_types_namespace={"B": B})
    with pytest.raises(TypeError, match="must declare opposite='b'"):
        Metamodel(A)

    class C(RefineryNode):
        values: dict[str, int] = {}

    with pytest.raises(TypeError, match="unsupported field type"):
        Metamodel(C)

    class D(RefineryNode):
        children: Annotated[list[E], Contains(opposite="owner")] = []

    class E(RefineryNode):
        owner: Annotated[D | None, Ref(opposite="children")] = None

    D.model_rebuild(_types_namespace={"E": E})
    with pytest.raises(TypeError, match="must use Container"):
        Metamodel(D)

    class F(RefineryNode):
        owners: Annotated[list[F], Container(opposite="x")] = []

    with pytest.raises(TypeError, match="single node"):
        Metamodel(F)


def test_multiplicity_from_field_constraints():
    from pydantic import Field

    class G(RefineryNode):
        exactly_two: list[G] = Field(default_factory=list, min_length=2, max_length=2)
        at_least_one: list[G] = Field(default_factory=list, min_length=1)
        override: Annotated[G | None, Ref(multiplicity="1..*")] = None

    source = Metamodel(G).source
    assert "G[2] exactly_two" in source
    assert "G[1..*] at_least_one" in source
    assert "G[1..*] override" in source


def _model_json(nodes, relations, interpretation):
    return {
        "concreteness": "candidate",
        "nodes": [{"name": n, "simpleName": n, "kind": "default"} for n in nodes],
        "relations": relations,
        "partialInterpretation": interpretation,
    }


def _pred(name, kind="error", arity=1):
    return {
        "name": name,
        "simpleName": name.rsplit("::", 1)[-1],
        "arity": arity,
        "detail": {"type": "pred", "kind": kind},
    }


def test_validate_classifies_violations():
    home, orphan = Dir(name="home"), Dir(name="orphan")
    fs = Filesystem(root=home)
    model = _model_json(
        ["Dir::new", "filesystem_1", "home", "orphan"],
        [
            _pred("builtin::invalidContainer"),
            _pred("builtin::contains", kind="defined", arity=2),
            _pred("Dir::contents::invalidMultiplicity"),
            _pred("emptyDir"),
            {
                "name": "Filesystem::root",
                "simpleName": "root",
                "arity": 2,
                "detail": {"type": "reference", "isContainment": True},
            },
        ],
        {
            "builtin::invalidContainer": [[3, "error"], [2, "false"]],
            "builtin::contains": [[1, 2, "error"]],
            "Dir::contents::invalidMultiplicity": [[2, "error"]],
            "emptyDir": [[0, "error"], [3, "error"]],
            "Filesystem::root": [[1, 2, "error"]],
        },
    )
    body = sse({"result": "success", "value": {"issues": [], "json": model}})
    recorder = Recorder(lambda _: sse_response(body))
    client = pf.Refinery(
        "http://x/api/v1", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )

    report = metamodel().validate([fs, orphan], client)

    assert recorder.requests[0].url.path == "/api/v1/concretize"
    assert recorder.last_json["format"]["source"]["enabled"] is False
    assert [str(v) for v in report.violations] == [
        "containment: builtin::invalidContainer(orphan)",
        "multiplicity: Dir::contents::invalidMultiplicity(home)",
        "rule: emptyDir(Dir::new)",
        "rule: emptyDir(orphan)",
        "contradiction: Filesystem::root(filesystem_1, home)",
    ]
    assert report.violations[0].nodes == (orphan,)
    assert report.violations[2].nodes == (None,)
    assert report.violations[4].nodes == (fs, home)
    assert not report.ok and not report
    assert [v.kind for v in report.for_node(home)] == ["multiplicity", "contradiction"]
    with pytest.raises(pf.ModelValidationError, match="5 Refinery constraint violation"):
        report.raise_for_violations()


def test_validate_timeout_hint():
    response = httpx.Response(408, json={"result": "timeout", "message": "Request timed out"})
    client = pf.Refinery(
        "http://x", http_client=httpx.Client(transport=httpx.MockTransport(lambda _: response))
    )
    with pytest.raises(pf.RefineryTimeoutError, match="semantics_timeout"):
        metamodel().validate([Dir(name="a")], client)


def test_avalidate_ok():
    model = _model_json(["a"], [_pred("emptyDir")], {"emptyDir": [[0, "false"]]})
    body = sse({"result": "success", "value": {"issues": [], "json": model}})

    async def main():
        transport = httpx.MockTransport(lambda _: sse_response(body))
        async with pf.AsyncRefinery(
            "http://x", http_client=httpx.AsyncClient(transport=transport)
        ) as client:
            return await metamodel().avalidate([Dir(name="a")], client)

    report = asyncio.run(main())
    assert report.ok
    report.raise_for_violations()
