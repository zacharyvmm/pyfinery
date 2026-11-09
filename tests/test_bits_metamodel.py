# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

""":meth:`Metamodel.query` with :class:`~pyfinery.bits.BitsEngine` instead of a Refinery client.

The scenarios of ``test_docker.py`` run as they are, and with
``PYFINERY_DOCKER_TESTS=1``, both engines answer the same queries over random
graphs of three model families.
"""

from __future__ import annotations

import asyncio
import enum
import json
import os
import random
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel

import pyfinery as pf
from pyfinery import ClassOptions, Metamodel, Predicate
from pyfinery._instances import build_graph
from pyfinery.bits import BitsEngine, UnsupportedRuleError
from pyfinery.bits._metamodel import partial_model

from . import deployment_models as deployment
from . import generic_models as generic
from . import graph_models as fs
from . import test_docker
from .conformance.recording import IMAGE
from .test_query import PREDICATES, RULES, metamodel, uncertain_graph

EXAMPLES = Path(__file__).parent.parent / "examples"


# -- the scenarios checked against Refinery in test_docker.py ------------------------


def test_query_true_and_unknown():
    test_docker.test_query_true_and_unknown(BitsEngine())


def test_query_closed_world_has_no_spurious_errors():
    test_docker.test_query_closed_world_has_no_spurious_errors(BitsEngine())


def test_query_containment_closed():
    doc, tmp = fs.File(), fs.Dir(name="tmp")
    root = fs.Dir(name="root", contents=[fs.Dir(name="home", contents=[doc]), tmp])
    tmp.contents.append(fs.File())
    rules = ["pred elsewhere(Dir d, File f) <-> !contents(d, f)."]
    result = fs.metamodel().query([fs.Filesystem(root=root)], BitsEngine(), rules)
    assert sorted(map(str, result.matches)) == [
        "true: elsewhere(home, file_2)",
        "true: elsewhere(root, file_1)",
        "true: elsewhere(root, file_2)",
        "true: elsewhere(tmp, file_1)",
    ]
    # Refinery reports the orphan as builtin::invalidContainer; the bits engine
    # doesn't check containment (validate() does).
    orphan = fs.File()
    result = fs.metamodel().query([fs.Filesystem(root=root), orphan], BitsEngine(), rules)
    assert len(result.matches) == 7 and all(m.logic_value == "true" for m in result.matches)
    assert result.errors == []


def test_readme_query():
    """The README's query of a deployment whose worker's dependencies were guessed."""
    data = json.loads((EXAMPLES / "deployment.json").read_text())
    guessed = Metamodel(
        deployment.Deployment,
        options={
            deployment.Service: ClassOptions(
                key=lambda s: s.name,
                references={"depends_on": "Component"},
                logic_value=lambda s, field, value: "unknown" if s.name == "worker" else True,
            ),
            deployment.Database: ClassOptions(key=lambda d: d.name),
        },
        unions={"Component": [deployment.Service, deployment.Database]},
    )
    rules = [
        "pred reachesUnencrypted(Service s, Database d) <-> "
        "Service::depends_on+(s, d), !Database::encrypted(d).",
    ]
    objects = [deployment.Deployment.model_validate(data)]
    result = guessed.query(objects, BitsEngine(), rules)
    assert sorted(map(str, result.matches)) == [
        "true: reachesUnencrypted(api, db)",
        "true: reachesUnencrypted(web, db)",
        "unknown: reachesUnencrypted(worker, db)",
    ]


# -- the model the engine evaluates over -------------------------------------------


def test_partial_model_of_a_graph():
    schema = metamodel().schema
    graph = build_graph(schema, [uncertain_graph()], existence=True)
    model, objects = partial_model(schema, graph, PREDICATES)
    names = [node.name for node in model.nodes]
    assert "EffectOp::write" in names and model.atom_named("EffectOp::write") is not None
    assert objects[names.index("EffectOp::write")] is None

    def tuples(relation: str, which: str = "may") -> set[tuple[str, ...]]:
        found = model.relation_named(relation)
        assert found is not None
        return {tuple(names[i] for i in t) for t in getattr(found, which).tolist()}

    assert ("effect_4",) in tuples("Effect") and ("effect_4",) not in tuples("File")
    assert tuples("Resource") == tuples("File") | tuples("Endpoint")
    assert ("effect_4",) not in tuples("exists", "must") and ("effect_4",) in tuples("exists")
    assert ("'file:x'",) in tuples("secretFile") - tuples("secretFile", "must")
    assert ("effect_1", "EffectOp::read") in tuples("Effect::op", "must")


def test_attributes_have_no_value_outside_their_class():
    class Item(BaseModel):
        size: int | None = None
        flag: bool = False

    class Box(BaseModel):
        items: list[Item] = []

    metamodel = Metamodel(Box)
    rules = [
        "pred big(x) <-> exists(x), Item::size(x) > 2.",
        "pred flagged(x) <-> exists(x), Item::flag(x).",
    ]
    result = metamodel.query([Box(items=[Item(size=3), Item(flag=True)])], BitsEngine(), rules)
    found = {(m.predicate, m.node_names): m.logic_value for m in result.matches}
    assert found[("big", ("item_1",))] == "true"
    assert found[("big", ("item_2",))] == "unknown"  # no size: unknown
    assert found[("flagged", ("item_2",))] == "true"
    # The box has neither attribute: an error, reported as a contradiction.
    assert {(v.kind, v.constraint, v.node_names) for v in result.errors} == {
        ("contradiction", "big", ("box_1",)),
        ("contradiction", "flagged", ("box_1",)),
    }


def test_python_enums_opposites_and_predicates_without_a_body():
    home = fs.Dir(name="home", contents=[fs.File(permission=fs.Permission.write, size=7)])
    rules = [
        "pred writable(File f) <-> File::permission(f, Permission::write).",
        'pred inHome(FSObject o) <-> FSObject::parent(o, d), Dir::name(d) == "home".',
        "pred vetted(File f).",
        "pred risky(File f) <-> writable(f), !vetted(f).",
    ]
    result = fs.metamodel().query([fs.Filesystem(root=home)], BitsEngine(), rules)
    assert sorted(map(str, result.matches)) == [
        "true: inHome(file_1)",
        "true: writable(file_1)",
        "unknown: risky(file_1)",
        "unknown: vetted(file_1)",
    ]


def test_error_predicates_are_reported_as_rule_violations():
    result = fs.metamodel().query([fs.Filesystem(root=fs.Dir(name="empty"))], BitsEngine(), [])
    assert [(v.kind, v.constraint, v.node_names) for v in result.errors] == [
        ("rule", "emptyDir", ("empty",))
    ]


def test_select_and_aquery():
    result = metamodel().query(
        [uncertain_graph()], BitsEngine(), RULES, predicates=PREDICATES, select=["removes"]
    )
    assert [m.predicate for m in result.matches] == ["removes"]
    assert result.source.startswith("\n".join(RULES)) and result.model is None
    assert "pred secretFile(File p0)." in result.source
    asynchronous = asyncio.run(
        metamodel().aquery([uncertain_graph()], BitsEngine(), RULES, predicates=PREDICATES)
    )
    synchronous = metamodel().query([uncertain_graph()], BitsEngine(), RULES, predicates=PREDICATES)
    assert asynchronous.matches == synchronous.matches


def test_validate_needs_refinery():
    with pytest.raises(TypeError, match="validate"):
        metamodel().validate([uncertain_graph()], BitsEngine())  # type: ignore[arg-type]


def test_assertions_in_rules_are_refused():
    with pytest.raises(UnsupportedRuleError, match="assertions in rules"):
        metamodel().query([uncertain_graph()], BitsEngine(), ["pred p(File f).", "p(x)."])


# -- where Refinery would refine uncertain facts ------------------------------------


def test_refuses_a_required_reference_that_only_may_hold():
    link = fs.Link(target=fs.File())
    root = fs.Dir(name="root", contents=[link, link.target])
    options = {fs.Link: ClassOptions(logic_value=lambda obj, field, value: "unknown")}
    metamodel = Metamodel(fs.Filesystem, options=options)
    rules = ["pred points(Link l, FSObject o) <-> Link::target(l, o)."]
    with pytest.raises(UnsupportedRuleError, match=r"Link::target.*\[1\.\.1\]"):
        metamodel.query([fs.Filesystem(root=root)], BitsEngine(), rules)


def test_refuses_uncertain_containers_besides_a_certain_one():
    file = fs.File()
    root = fs.Dir(name="root", contents=[file])
    other = fs.Dir(name="other", contents=[file])
    root.contents.append(other)
    options = {
        fs.Dir: ClassOptions(logic_value=lambda d, f, v: "unknown" if d.name == "other" else True)
    }
    metamodel = Metamodel(fs.Filesystem, options=options)
    rules = ["pred contained(FSObject o) <-> FSObject::parent(o, _)."]
    with pytest.raises(UnsupportedRuleError, match="FSObject::parent|uncertain containers"):
        metamodel.query([fs.Filesystem(root=root)], BitsEngine(), rules)
    # A query that doesn't depend on containment is answered.
    result = metamodel.query(
        [fs.Filesystem(root=root)], BitsEngine(), ["pred d(Dir d) <-> Dir(d)."]
    )
    assert len(result.matches) == 2


def test_refuses_error_predicates_that_may_hold():
    with pytest.raises(UnsupportedRuleError, match="error predicate removesSecret"):
        metamodel().query(
            [uncertain_graph()],
            BitsEngine(),
            ["error pred removesSecret(Effect e) <-> Effect::target(e, f), secretFile(f)."],
            predicates=PREDICATES,
        )


# -- edge cases, each checked against Refinery with PYFINERY_DOCKER_TESTS=1 ---------


class Text(BaseModel):
    text: str


class Color(enum.Enum):
    red = "r"
    blue = "b"


class Colored(BaseModel):
    flag: bool
    color: Color


class Flagged(BaseModel):
    flag: bool


class Plain(BaseModel):
    pass


class Pair(BaseModel):
    a: list[Flagged] = []
    b: list[Plain] = []
    texts: list[Text] = []
    colored: list[Colored] = []


def _pair(**fields) -> tuple[Metamodel, list[BaseModel]]:
    return Metamodel(Pair), [Pair(**fields)]


EDGE_CASES = {
    # Strings spelled like logic values are values (review of #29, round 1).
    "strings": lambda: (
        *_pair(texts=[Text(text="unknown"), Text(text="error")]),
        [
            'pred word(Text t) <-> Text::text(t) == "unknown".',
            'pred err(Text t) <-> Text::text(t) == "error".',
        ],
        [],
        None,
    ),
    # A boolean attribute has no value on enum literals either.
    "booleans on enum literals": lambda: (
        *_pair(colored=[Colored(flag=True, color=Color.red)]),
        ["pred p(x) <-> exists(x), Colored::flag(x)."],
        [],
        None,
    ),
    # A Predicate object can be selected.
    "selected predicate": lambda: (
        *_pair(b=[Plain()]),
        [],
        [Predicate("chosen", (Plain,), lambda x: True)],
        ["chosen"],
    ),
    # Contradictions are reported whatever is selected.
    "contradictions not selected": lambda: (
        *_pair(a=[Flagged(flag=True)], b=[Plain()]),
        ["pred p(x) <-> exists(x), Flagged::flag(x)."],
        [],
        [],
    ),
}


def _run(case: str, client):
    metamodel, objects, rules, predicates, select = EDGE_CASES[case]()
    result = metamodel.query(objects, client, rules, predicates=predicates, select=select)
    matches = sorted(
        (m.predicate, m.node_names, m.logic_value, m.parameters) for m in result.matches
    )
    return matches, _errors(result)


def _errors(result) -> list:
    """The errors both engines report. The Refinery engine reports each contradiction
    of a predicate twice, also for its ``::computed`` relation (as on main)."""
    kinds = ("rule", "reference", "contradiction")
    return sorted(
        (v.kind, v.constraint, v.node_names)
        for v in result.errors
        if v.kind in kinds and not v.constraint.endswith("::computed")
    )


def test_edge_cases():
    assert _run("strings", BitsEngine()) == (
        [("err", ("text_2",), "true", ("t",)), ("word", ("text_1",), "true", ("t",))],
        [],
    )
    matches, errors = _run("booleans on enum literals", BitsEngine())
    assert ("contradiction", "p", ("Color::red",)) in errors
    assert ("contradiction", "p", ("Color::blue",)) in errors
    assert _run("selected predicate", BitsEngine())[0] == [
        ("chosen", ("plain_1",), "true", ("p0",))
    ]
    matches, errors = _run("contradictions not selected", BitsEngine())
    assert matches == [] and ("contradiction", "p", ("plain_1",)) in errors


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("PYFINERY_DOCKER_TESTS") != "1", reason="set PYFINERY_DOCKER_TESTS=1"
)
@pytest.mark.parametrize("case", sorted(EDGE_CASES))
def test_edge_cases_agree_with_refinery(refinery, case):
    assert _run(case, BitsEngine()) == _run(case, refinery)


class Box(BaseModel):
    kids: Annotated[list[Box], pf.Contains()] = []


class Required(BaseModel):
    target: Plain


class Holder(BaseModel):
    sources: list[Required] = []


class Shape(BaseModel):
    pass


class Square(Shape):
    pass


class Shapes(BaseModel):
    shapes: list[Shape] = []


def test_refuses_containment_cycles():
    one, two = Box(), Box()
    one.kids.append(two)
    two.kids.append(one)
    # Certain: an uncertain cycle is refused too, as each box's one container is uncertain.
    metamodel = Metamodel(Box)
    with pytest.raises(UnsupportedRuleError, match="containment cycle"):
        metamodel.query([one], BitsEngine(), ["pred p(Box x, Box y) <-> Box::kids(x, y)."])


class Child(BaseModel):
    pass


class Parent(BaseModel):
    kids: list[Child] = []


class Family(BaseModel):
    parents: list[Parent] = []
    children: list[Child] = []


def test_refuses_an_uncertain_only_container():
    """A Child needs a container: its only possible one, uncertain, must hold (review of #29)."""
    options = {Parent: ClassOptions(logic_value=lambda obj, f, v: "unknown")}
    metamodel = Metamodel(Parent, options=options)
    rules = ["pred p(Parent x, Child y) <-> Parent::kids(x, y)."]
    with pytest.raises(UnsupportedRuleError, match="one possible container of child_1"):
        metamodel.query([Parent(kids=[Child()])], BitsEngine(), rules)


def test_refuses_an_uncontainable_object_that_may_not_exist():
    options = {Child: ClassOptions(exists=lambda child: "unknown")}
    metamodel = Metamodel(Parent, options=options)
    with pytest.raises(UnsupportedRuleError, match="no possible container"):
        metamodel.query([Parent(), Child()], BitsEngine(), ["pred p() <-> Child(_)."])


def test_refuses_refinement_an_unselected_error_predicate_depends_on():
    metamodel = Metamodel(
        Holder,
        options={Required: ClassOptions(logic_value=lambda obj, f, v: "unknown")},
        rules=["error pred forbidden(Required s) <-> Required::target(s, _)."],
    )
    with pytest.raises(UnsupportedRuleError, match="Required::target"):
        metamodel.query([Holder(sources=[Required(target=Plain())])], BitsEngine(), [])


def test_refuses_instances_of_abstract_classes():
    metamodel = Metamodel(Shapes, Square, options={Shape: ClassOptions(abstract=True)})
    with pytest.raises(UnsupportedRuleError, match="abstract class Shape"):
        metamodel.query([Shapes(shapes=[Shape()])], BitsEngine(), ["pred p(Shape x) <-> Shape(x)."])


# -- against Refinery, on random graphs -----------------------------------------------


def _generic(rng: random.Random):
    files = [
        generic.File(
            path=generic.Path(
                anchor=rng.choice(["root", "home", "unknown"]),
                segments=[rng.choice(["a", ".ssh"]), rng.choice(["x", "id_rsa"])],
            )
        )
        for _ in range(rng.randint(1, 4))
    ]
    endpoints = [generic.Endpoint(url=f"https://{h}.example") for h in rng.sample("abc", 2)]
    resources = files + endpoints
    effects = [
        generic.Effect(
            op=rng.choice(["read", "write", "remove"]),
            target=rng.choice(resources).model_copy() if rng.random() < 0.9 else None,
            sources=[r.model_copy() for r in rng.sample(resources, rng.randint(0, 2))],
            definite=rng.random() < 0.7,
            index=rng.randint(0, 5),
        )
        for _ in range(rng.randint(1, 6))
    ]
    options = dict(generic.OPTIONS)
    options[generic.Effect] = ClassOptions(
        exclude=["command", "trace"],
        exists=lambda e: "true" if e.definite else "unknown",
        logic_value=lambda e, field, v: "unknown" if field == "sources" and e.index % 2 else True,
    )
    metamodel = Metamodel(
        generic.Graph,
        options=options,
        unions={"Resource": [generic.File, generic.Endpoint]},
        unsupported="skip",
    )
    trusted = [endpoints[0]]
    predicates = [
        *PREDICATES,
        Predicate("trusted", (generic.Endpoint,), tuples=[((e,), "unknown") for e in trusted]),
    ]
    rules = [
        *RULES,
        "pred before(Effect a, Effect b) <-> Effect::index(a) < Effect::index(b).",
        "pred readAfterRemove(Effect r, Effect d) <-> removes(d, f), Effect::target(r, f), "
        "Effect::op(r, EffectOp::read), before(d, r).",
        "pred untouched(File f) <-> !Effect::target(_, f), !Effect::sources(_, f).",
        "pred definiteWrite(Effect e) <-> Effect::definite(e), Effect::op(e, EffectOp::write).",
        "pred leaks(File f, Endpoint n) <-> exfiltration(f, n), !trusted(n).",
        "pred chain(Effect a, Effect b, Resource r) <-> Effect::target(a, r), "
        "Effect::sources(b, r), a != b.",
        "pred lonely(Endpoint n) <-> !writes(_, n).",
    ]
    return metamodel, [generic.Graph(resources=resources, effects=effects)], rules, predicates


def _filesystem(rng: random.Random):
    dirs = [fs.Dir(name=f"d{i}") for i in range(rng.randint(1, 4))]
    for index, d in enumerate(dirs[1:], start=1):
        rng.choice(dirs[:index]).contents.append(d)
    files = [
        fs.File(
            size=rng.randint(0, 9),
            ratio=rng.random(),
            hidden=rng.random() < 0.3,
            label=rng.choice([None, "a", "b"]),
            permission=rng.choice([None, fs.Permission.read, fs.Permission.write]),
        )
        for _ in range(rng.randint(0, 4))
    ]
    for file in files:
        rng.choice(dirs).contents.append(file)
    for _ in range(rng.randint(0, 2)):
        rng.choice(dirs).contents.append(
            fs.Link(target=rng.choice(dirs + files), backups=rng.sample(dirs + files, 1))
        )
    rules = [
        "pred inside(Dir d, FSObject o) <-> contents+(d, o).",
        "pred big(File f) <-> File::size(f) > 4.",
        "pred hiddenBig(File f) <-> big(f), File::hidden(f).",
        'pred labelled(File f) <-> File::label(f) == "a".',
        "pred writable(File f) <-> File::permission(f, Permission::write).",
        "pred dangling(Link l) <-> Link::target(l, t), !inside(_, t).",
        "pred siblings(FSObject a, FSObject b) <-> FSObject::parent(a, d), "
        "FSObject::parent(b, d), a != b.",
        "pred high(File f) <-> File::ratio(f) >= 0.5.",
        "pred top(FSObject o) <-> !FSObject::parent(o, _).",
    ]
    return fs.metamodel(), [fs.Filesystem(root=dirs[0])], rules, []


def _deployment(rng: random.Random):
    names = ["web", "api", "worker", "db", "cache"][: rng.randint(2, 5)]
    components: list = []
    for name in names:
        if rng.random() < 0.6:
            components.append(
                deployment.Service(
                    name=name,
                    image=rng.choice(["x:1", "x:latest"]),
                    tier=rng.choice(["frontend", "backend", "data"]),
                    ports=[
                        deployment.Port(number=1000 * rng.randint(1, 3), public=rng.random() < 0.5)
                        for _ in range(rng.randint(0, 2))
                    ],
                    depends_on=rng.sample([*names, "ghost"], rng.randint(0, 2)),
                    replicas=rng.choice([None, 1, 3]),
                )
            )
        else:
            components.append(
                deployment.Database(
                    name=name,
                    engine=rng.choice(["postgres", "redis"]),
                    encrypted=rng.random() < 0.5,
                )
            )
    metamodel = Metamodel(
        deployment.Deployment,
        options={
            deployment.Service: ClassOptions(
                key=lambda s: s.name,
                references={"depends_on": "Component"},
                # The worker's dependencies were guessed: they only may hold.
                logic_value=lambda s, field, v: (
                    "unknown" if s.name == "worker" and field == "depends_on" else True
                ),
            ),
            deployment.Database: ClassOptions(
                key=lambda d: d.name, exists=lambda d: "unknown" if d.name == "cache" else True
            ),
        },
        unions={"Component": [deployment.Service, deployment.Database]},
    )
    rules = [
        "pred reachesUnencrypted(Service s, Database d) <-> "
        "Service::depends_on+(s, d), !Database::encrypted(d).",
        "pred publicPort(Service s) <-> Service::ports(s, p), Port::public(p).",
        "pred backend(Service s) <-> Service::tier(s, ServiceTier::backend).",
        "pred scaled(Service s) <-> Service::replicas(s) > 1.",
        "pred redisUser(Service s) <-> Service::depends_on(s, d), "
        "Database::engine(d, DatabaseEngine::redis).",
        "pred unused(Component c) <-> !Service::depends_on(_, c).",
        "pred exposed(Service s, Port p, Component c) <-> Service::ports(s, p), "
        "Port::public(p), Service::depends_on(s, c).",
    ]
    return metamodel, [deployment.Deployment(components=components)], rules, []


FAMILIES = {"generic": _generic, "filesystem": _filesystem, "deployment": _deployment}


@pytest.fixture(scope="module")
def refinery():
    with (
        pf.RefineryServer(image=IMAGE, semantics_timeout=60_000) as server,
        server.client(timeout=None) as client,
    ):
        yield client


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("PYFINERY_DOCKER_TESTS") != "1", reason="set PYFINERY_DOCKER_TESTS=1"
)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_engines_agree_on_random_graphs(refinery, family):
    compared = refused = 0
    for seed in range(25):
        metamodel, objects, rules, predicates = FAMILIES[family](random.Random(seed))
        try:
            bits = metamodel.query(objects, BitsEngine(), rules, predicates=predicates)
        except UnsupportedRuleError:
            refused += 1
            continue
        expected = metamodel.query(objects, refinery, rules, predicates=predicates)

        def matches(result):
            return sorted((m.predicate, m.node_names, m.logic_value) for m in result.matches)

        assert matches(bits) == matches(expected), f"seed {seed}"
        assert _errors(bits) == _errors(expected), f"seed {seed}"
        compared += 1
    assert compared >= 15, f"{refused} of 25 refused"
