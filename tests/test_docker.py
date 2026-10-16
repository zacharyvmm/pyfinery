# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""End-to-end tests against a real Refinery server started in Docker.

Opt in with ``PYFINERY_DOCKER_TESTS=1``; the first run pulls the image.
"""

import asyncio
import os

import pytest

import pyfinery as pf

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.environ.get("PYFINERY_DOCKER_TESTS") != "1", reason="set PYFINERY_DOCKER_TESTS=1"
    ),
]


@pytest.fixture(scope="module")
def server():
    with pf.RefineryServer() as server:
        yield server


@pytest.fixture
def refinery(server):
    with server.client() as client:
        yield client


def test_generate(refinery, filesystem_problem):
    result = refinery.generate(filesystem_problem, scopes=["File=2..*"])
    assert result.json_ is not None and result.json_.concreteness == "candidate"
    files = [node for (node,) in result.json_.true_tuples("File")]
    assert len(files) >= 2
    assert result.source is not None and "File(" in result.source


def test_generate_many(refinery, filesystem_problem):
    result = refinery.generate_many(filesystem_problem, 3, random_seed=42)
    assert result.stop_reason == "requestFulfilled"
    assert len(result.models) == 3


def test_semantics_and_concretize(refinery):
    source = "class A.\nA(a).\n"
    semantics = refinery.semantics(source)
    assert semantics.json_ is not None and semantics.json_.concreteness == "partial"
    assert "A(a)." in (refinery.concretize(source).source or "")


def test_errors(refinery):
    with pytest.raises(pf.UnsatisfiableError):
        refinery.generate("class A.\nscope A = 3.\nerror pred p() <-> A(x).")
    with pytest.raises(pf.InvalidProblemError) as info:
        refinery.generate("class A { Missing[1] b }")
    assert info.value.issues


def test_early_close(refinery, filesystem_problem):
    stream = refinery.stream_generate(filesystem_problem, scopes=["node=5000..6000"])
    assert isinstance(next(stream), pf.GenerateStatus)
    stream.close()
    # The server must still serve requests after the client disconnected.
    assert refinery.semantics("class A.").issues == []


def test_async(server, filesystem_problem):
    async def main():
        async with server.async_client() as client:
            return await client.generate(filesystem_problem)

    assert asyncio.run(main()).source


def test_validate_graph(refinery):
    from .graph_models import Dir, File, Filesystem, Link, metamodel

    home = Dir(name="home")
    file = File(size=3)
    home.contents = [file, Link(target=file)]
    filesystem = Filesystem(root=home)
    report = metamodel().validate([filesystem], refinery)
    assert report.ok, report.violations

    orphan = Dir(name="orphan")
    home.contents += [File(), File()]  # 4 > [0..3]
    report = metamodel().validate([filesystem, orphan], refinery)
    assert {(v.kind, v.constraint, v.nodes) for v in report.violations} == {
        ("containment", "builtin::invalidContainer", (orphan,)),
        ("multiplicity", "Dir::contents::invalidMultiplicity", (home,)),
        ("rule", "emptyDir", (orphan,)),
    }


def test_validate_contradiction(refinery):
    from .graph_models import Dir, Filesystem, metamodel

    a, b = Dir(name="a"), Dir(name="b")
    a.contents = [b]
    b.contents = [a]  # containment cycle
    report = metamodel().validate(
        [Filesystem(root=Dir(name="root", contents=[Dir(name="c")])), a], refinery
    )
    assert {v.kind for v in report.for_node(a)} >= {"containment"}
    with pytest.raises(pf.ModelValidationError):
        report.raise_for_violations()


def test_fresh_server_validates_immediately():
    """Without warm-up, the first validation on a fresh server tends to time out."""
    from .graph_models import Dir, File, Filesystem, metamodel

    with pf.RefineryServer() as fresh, fresh.client() as refinery:
        report = metamodel().validate([Filesystem(root=Dir(name="a", contents=[File()]))], refinery)
    assert report.ok, report.violations


def test_validate_nested_plain_models(refinery):
    from .nested_models import Address, Country, Order, Person, Phone, people

    shared = Address(street="1 Main", country=Country.CA)
    alice = Person(name="alice", home=shared, contacts=[Phone(value="1")])
    assert people().validate([alice], refinery).ok

    bob = Person(name="bob", home=Address(street="2", country=Country.FR), work=shared)
    bob.manager = bob
    report = people().validate([alice, bob], refinery)
    found = sorted((v.kind, v.constraint, v.nodes) for v in report.violations)
    # Plain Pydantic models aren't hashable, so compare as an ordered list.
    assert [(kind, constraint) for kind, constraint, _ in found] == [
        ("containment", "builtin::invalidContainer"),  # the shared value object
        ("rule", "selfManaged"),
    ]
    assert found[0][2][0] is shared and found[1][2][0] is bob

    empty = Order(lines=[])
    report = pf.Metamodel(Order).validate([empty], refinery)
    assert [(v.kind, v.nodes) for v in report.violations] == [("multiplicity", (empty,))]
