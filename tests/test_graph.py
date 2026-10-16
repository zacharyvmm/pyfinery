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

    class Owner(RefineryNode):
        items: Annotated[list[Item], Ref(opposite="owner")] = []

    class Item(RefineryNode):
        owner: Annotated[Owner | None, Container(opposite="items")] = None

    Owner.model_rebuild(_types_namespace={"Item": Item})
    with pytest.raises(TypeError, match="opposite of a container reference must use Contains"):
        Metamodel(Owner)


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


def test_link_opposites_fills_missing_sides():
    home = Dir(name="home")
    doc = File()
    home.contents = [doc]  # only the containment side
    note = File(parent=home)  # only the container side
    mm = metamodel()

    # note is only reachable through its own parent pointer, so pass it explicitly.
    assert mm.link_opposites([Filesystem(root=home), note]) == 2
    assert doc.parent is home
    assert home.contents == [doc, note]
    assert mm.link_opposites([home]) == 0  # idempotent


def test_link_opposites_leaves_conflicts_for_validation():
    a, b = Dir(name="a"), Dir(name="b")
    doc = File(parent=b)
    a.contents = [doc]
    metamodel().link_opposites([a, b])
    assert doc.parent is b  # not overwritten
    assert a.contents == [doc] and b.contents == [doc]


def test_link_opposites_symmetric_and_collections():
    class Person(RefineryNode):
        friends: Annotated[tuple[Person, ...], Ref(opposite="friends")] = ()
        tags: Annotated[set[Tag], Ref(opposite="people")] = set()

    class Tag(RefineryNode):
        people: Annotated[frozenset[Person], Ref(opposite="tags")] = frozenset()

    Person.model_rebuild(_types_namespace={"Tag": Tag})
    alice, bob, tag = Person(), Person(), Tag()
    alice.friends = (bob,)
    tag.people = frozenset({alice})
    mm = Metamodel(Person)

    assert mm.link_opposites([alice, tag]) == 2
    assert bob.friends == (alice,)
    assert alice.tags == {tag}
    assert repr(alice).count("Person(") == 1  # referenced nodes are abbreviated


def test_nested_plain_models_become_contained_classes():
    from .nested_models import people

    source = people().source
    assert "enum Country { CA, FR }" in source
    assert "class Address {\n    string street\n    Country[1] country\n}" in source
    assert "class Contact {\n    string value\n}" in source
    assert "class Phone extends Contact." in source
    assert "class Email extends Contact {\n    boolean verified\n}" in source
    assert (
        "class Person {\n"
        "    string name\n"
        "    contains Address[1] home\n"
        "    contains Address[0..1] work\n"
        "    contains Contact[0..2] contacts\n"
        "    Person[0..1] manager\n"
        "}"
    ) in source


def test_nested_plain_model_facts():
    from .nested_models import Address, Country, Email, Person, people

    shared = Address(street="1 Main", country=Country.CA)
    alice = Person(name="alice", home=shared, contacts=[Email(value="a@x")])
    bob = Person(name="bob", home=Address(street="2 Side", country=Country.FR), work=shared)
    facts = people().problem([alice, bob]).split("\n\n")[-1].splitlines()
    assert "Person::home(alice, address_1)." in facts
    assert "Person::work(bob, address_1)." in facts  # same instance, same node
    assert "Person::contacts(alice, email_1)." in facts
    assert 'Contact::value(email_1): "a@x".' in facts
    assert "Email::verified(email_1): false." in facts
    assert "Address::country(address_2, Country::FR)." in facts


def test_fully_plain_metamodel():
    from .nested_models import Line, Order

    mm = Metamodel(Order)
    assert "class Order {\n    contains Line[1..*] lines\n}" in mm.source
    order = Order(lines=[Line(sku="a", qty=2)])
    assert mm.collect([order]) == [order, order.lines[0]]


def test_plain_models_cannot_use_opposites():
    from pydantic import BaseModel

    class Note(BaseModel):
        text: str

    class Holder(RefineryNode):
        note: Annotated[Note | None, Ref(opposite="holder")] = None

    with pytest.raises(TypeError, match="plain Pydantic model"):
        Metamodel(Holder)

    class Holder2(RefineryNode):
        note: Annotated[Note | None, Container(opposite="x")] = None

    with pytest.raises(TypeError, match="plain Pydantic model"):
        Metamodel(Holder2)
