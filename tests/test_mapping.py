from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from pyfinery import ClassOptions, Metamodel
from pyfinery._instances import Graph, build_graph
from pyfinery._mapping import Catalogue, Fact, NodeIds, facts, objects, read_facts
from pyfinery._render import render_facts

from . import deployment_models, generic_models, graph_models, nested_models

METAMODELS = {
    "deployment": deployment_models.metamodel,
    "graph": graph_models.metamodel,
    "nested": nested_models.people,
    "generic": generic_models.metamodel,
}


def _declared(source: str) -> tuple[set[str], set[str], set[str]]:
    """Classes (and enums), features, and enum literals declared in ``source``."""
    types: set[str] = set()
    features: set[str] = set()
    literals: set[str] = set()
    owner = None
    for line in source.splitlines():
        if match := re.match(r"enum (\S+) \{ (.*) \}$", line):
            types.add(match[1])
            literals |= {f"{match[1]}::{m}" for m in match[2].split(", ")}
        elif match := re.match(r"(?:abstract )?class (\S+?)(?:\.|\s|$)", line):
            types.add(match[1])
            owner = match[1] if line.endswith("{") else None
        elif line == "}":
            owner = None
        elif owner is not None and line.startswith("    "):
            features.add(
                f"{owner}::{line.split()[-1] if 'opposite' not in line else line.split()[-3]}"
            )
    return types, features, literals


def _setup(
    metamodel: Metamodel, objs: list[BaseModel], **kwargs
) -> tuple[Catalogue, Graph, NodeIds]:
    catalogue = Catalogue(metamodel.schema)
    names = catalogue.literal_names()
    ids = NodeIds(first=len(names))
    for i, name in enumerate(names):
        ids.constant(name, i)
    return catalogue, build_graph(metamodel.schema, objs, **kwargs), ids


@pytest.mark.parametrize("which", list(METAMODELS))
def test_catalogue_names_match_metamodel(which):
    mm = METAMODELS[which]()
    catalogue = Catalogue(mm.schema)
    types, features, literals = _declared(mm.source)
    assert set(catalogue.classes) | set(catalogue.enums) == types
    assert set(catalogue.features) == features
    assert set(catalogue.literals) == literals
    for name in [*catalogue.classes, *catalogue.enums, *catalogue.features, *catalogue.literals]:
        assert catalogue.lookup(name) is not None
    for name in [*catalogue.classes, *catalogue.enums]:
        assert re.search(rf"\b(class|enum) {re.escape(name)}\b", mm.source)


def test_catalogue_lookups_both_ways():
    from .deployment_models import Database, Service

    catalogue = Catalogue(deployment_models.metamodel().schema)
    ports = catalogue.feature(Service, "ports")
    assert catalogue.name(ports) == "Service::ports"
    assert catalogue.lookup("Service::ports") is ports
    assert catalogue.name(Service) == "Service"
    assert catalogue.lookup("Service") is Service
    component = catalogue.lookup("Component")
    assert catalogue.name(component) == "Component"  # type: ignore[arg-type]
    tier = catalogue.feature(Service, "tier").target
    assert catalogue.name(tier) == "ServiceTier"  # type: ignore[arg-type]
    assert catalogue.literal(tier, "backend") == "ServiceTier::backend"  # type: ignore[arg-type]
    assert catalogue.lookup("ServiceTier::backend") == (tier, "backend")
    assert catalogue.literal_names()[:3] == [
        "ServiceTier::frontend",
        "ServiceTier::backend",
        "ServiceTier::data",
    ]
    with pytest.raises(KeyError):
        catalogue.name(BaseModel)
    with pytest.raises(KeyError):
        catalogue.feature(Database, "ports")
    with pytest.raises(KeyError):
        catalogue.lookup("Nothing")


def test_catalogue_quotes_names_like_the_metamodel():
    class Thing(BaseModel):
        default: int
        error: bool = False

    mm = Metamodel(Thing)
    catalogue = Catalogue(mm.schema)
    assert set(catalogue.features) == {"Thing::'default'", "Thing::'error'"}
    assert "int 'default'" in mm.source


def test_supertypes_include_superclasses_and_unions():
    from .deployment_models import Service
    from .graph_models import Dir, File
    from .nested_models import Email, Phone

    deployment = Catalogue(deployment_models.metamodel().schema)
    assert deployment.supertypes(Service) == ("Service", "Component")
    fs = Catalogue(graph_models.metamodel().schema)
    assert fs.supertypes(Dir) == ("Dir", "FSObject")
    assert fs.supertypes(File) == ("File", "FSObject")
    people = Catalogue(nested_models.people().schema)
    assert people.supertypes(Email) == ("Email", "Contact")
    assert people.supertypes(Phone) == ("Phone", "Contact")
    generic = Catalogue(generic_models.metamodel().schema)
    assert set(generic.supertypes(generic_models.File)) == {"File", "Model", "Resource"}


def test_opposites():
    from .graph_models import Dir, FSObject

    catalogue = Catalogue(graph_models.metamodel().schema)
    contents = catalogue.feature(Dir, "contents")
    parent = catalogue.feature(FSObject, "parent")
    assert catalogue.opposite(contents) is parent
    assert catalogue.opposite(parent) is contents
    assert catalogue.opposite(catalogue.feature(Dir, "name")) is None


def _rendered_pairs(text: str, names: dict[str, int], literals: NodeIds) -> set[tuple]:
    """The facts of ``render_facts`` output as ``(relation, ids, value)``."""
    out = set()
    for line in text.strip().splitlines():
        if match := re.match(r"(\S+)\((.*)\): (.*)\.$", line):
            out.add((match[1], (names[match[2]],), match[3]))
            continue
        match = re.match(r"(\S+?)\((.*)\)\.$", line)
        assert match, line
        args = tuple(
            names[a] if a in names else literals.constant_id(a) for a in match[2].split(", ")
        )
        out.add((match[1], args, None))
    return out


def test_facts_match_render_facts():
    from .graph_models import Dir, File, Filesystem, Link, Permission

    mm = graph_models.metamodel()
    home = Dir(name="home")
    report = File(size=-3, hidden=True, label="x", permission=Permission.read)
    link = Link(target=report)
    home.contents = [report, link]
    report.parent = home  # both sides set: written once
    catalogue, graph, ids = _setup(mm, [Filesystem(root=home)])
    written = facts(catalogue, graph, ids)
    names = {node.name: ids.id_of(node) for node in graph.nodes}
    rendered = _rendered_pairs(render_facts(mm.schema, graph), names, ids)

    ours = {(f.relation, f.nodes) for f in written}
    theirs = {(r, n) for r, n, _ in rendered}
    # The same relations and tuples, plus the supertypes and opposites below.
    assert theirs <= ours
    assert len(written) == len({(f.relation, f.nodes) for f in written})
    home_id, report_id, link_id = (ids.id_of(graph.node_of(o)) for o in (home, report, link))
    by_relation = {(f.relation, f.nodes): f.value for f in written}
    # Supertype memberships, which render_facts leaves to Refinery.
    assert by_relation[("FSObject", (report_id,))] is True
    assert by_relation[("FSObject", (home_id,))] is True
    assert ("FSObject", (link_id,)) in by_relation
    # Opposite directions.
    assert by_relation[("FSObject::parent", (report_id, home_id))] is True
    assert by_relation[("FSObject::parent", (link_id, home_id))] is True
    # Attributes carry the Python value, enums refer to the literal constant.
    assert by_relation[("File::size", (report_id,))] == -3
    assert by_relation[("File::hidden", (report_id,))] is True
    assert by_relation[("File::label", (report_id,))] == "x"
    assert by_relation[("Dir::name", (home_id,))] == "home"
    assert ("File::permission", (report_id, ids.constant_id("Permission::read"))) in by_relation
    assert ours - theirs == {
        ("FSObject", (home_id,)),
        ("FSObject", (report_id,)),
        ("FSObject", (link_id,)),
        ("FSObject::parent", (link_id, home_id)),  # report.parent is set, so rendered
    }


def test_facts_by_key_and_unions():
    mm = deployment_models.metamodel()
    catalogue, graph, ids = _setup(mm, [deployment_models.deployment()])
    written = {(f.relation, f.nodes): f.value for f in facts(catalogue, graph, ids)}
    node = {n.name: ids.id_of(n) for n in graph.nodes}
    assert written[("Component", (node["web"],))] is True
    assert written[("Component", (node["db"],))] is True
    assert written[("Service::depends_on", (node["web"], node["api"]))] is True
    assert written[("Service::depends_on", (node["web"], node["cache"]))] is True
    assert written[("Service::depends_on", (node["api"], node["db"]))] is True
    # "billing" resolves to no object: skipped.
    depends = [n for r, n in written if r == "Service::depends_on"]
    assert len(depends) == 3
    tier = ids.constant_id("ServiceTier::backend")
    assert written[("Service::tier", (node["api"], tier))] is True
    assert written[("Database::encrypted", (node["db"],))] is False


def test_facts_holds_value_is_abstract():
    catalogue, graph, ids = _setup(deployment_models.metamodel(), [deployment_models.deployment()])
    written = facts(catalogue, graph, ids, holds="yes")
    assert {f.value for f in written if f.relation == "Component"} == {"yes"}
    assert {f.value for f in written if f.relation == "Database::encrypted"} == {False, True}


class _Item(BaseModel):
    name: str
    size: int = 0
    friends: list[_Item] = []


def _logic(item: _Item, field: str, value: object) -> str:
    return "unknown" if field in ("size", "friends") and item.name == "b" else "true"


def test_uncertain_values_and_existence():
    mm = Metamodel(
        _Item,
        options={
            _Item: ClassOptions(
                key=lambda i: i.name,
                logic_value=_logic,
                exists=lambda i: "unknown" if i.name == "c" else True,
            )
        },
    )
    c = _Item(name="c")
    a = _Item(name="a", size=1)
    b = _Item(name="b", size=2, friends=[a])
    a.friends = [b, c]
    catalogue, graph, ids = _setup(mm, [a], existence=True)
    names = {"a": 0, "b": 1, "c": 2}

    def relations(written: list[Fact]) -> set[tuple]:
        return {(f.relation, f.nodes, f.value) for f in written}

    omitted = relations(facts(catalogue, graph, ids))
    assert ("_Item::size", (names["a"],), 1) in omitted
    assert not any(r == "_Item::size" and n == (names["b"],) for r, n, _ in omitted)
    assert ("_Item::friends", (names["b"], names["a"]), True) not in omitted
    assert ("_Item::friends", (names["a"], names["c"]), True) in omitted

    included = relations(facts(catalogue, graph, ids, uncertain="include"))
    assert ("_Item::size", (names["b"],), 2) in included
    assert ("_Item::friends", (names["b"], names["a"]), True) in included

    fresh = NodeIds()
    dropped = facts(catalogue, graph, fresh, unknown_existence="omit")
    assert all(names["c"] not in f.nodes for f in dropped)
    assert len(fresh) == 2
    assert graph.nodes[2] not in fresh


def test_node_ids():
    graph = build_graph(
        nested_models.people().schema,
        [
            nested_models.Person(
                name="p", home=nested_models.Address(street="s", country=nested_models.Country.CA)
            )
        ],
    )
    person, address = graph.nodes
    ids = NodeIds(first=10)
    ids.constant("Country::CA", 11)
    assert ids.add(person) == 10
    assert ids.add(address) == 12  # 11 is a constant
    assert ids.add(person) == 10
    assert ids.node(12) is address
    assert ids.nodes() == {10: person, 12: address}
    assert ids.remove(person) == 10
    assert person not in ids and ids.get(person) is None
    with pytest.raises(KeyError):
        ids.node(10)
    with pytest.raises(KeyError):
        ids.id_of(person)
    assert ids.add(person) == 13  # ids are not reused
    assert ids.id_of(address) == 12
    assert ids.constant_id("Country::CA") == 11
    assert ids.constant_name(11) == "Country::CA"
    assert ids.constant_name(12) is None
    ids.constant("Country::CA", 11)  # the same registration again is fine
    with pytest.raises(ValueError):
        ids.constant("Country::FR", 11)
    with pytest.raises(ValueError):
        ids.constant("Country::FR", 12)
    with pytest.raises(ValueError):
        ids.constant("Country::CA", 3)
    with pytest.raises(KeyError):
        ids.constant_id("Country::FR")
    assert len(ids) == 2


def test_read_back_round_trip():
    from .graph_models import Dir, File, Filesystem, Permission

    mm = graph_models.metamodel()
    home = Dir(name="home")
    report = File(size=5, permission=Permission.write)
    home.contents = [report]
    catalogue, graph, ids = _setup(mm, [Filesystem(root=home)])
    written = facts(catalogue, graph, ids)
    rows = read_facts(catalogue, ids, written)
    assert sum(len(r) for r in rows.values()) == len(written)
    home_node, report_node = graph.node_of(home), graph.node_of(report)
    assert ((home_node, report_node), True) in rows["Dir::contents"]
    assert ((report_node, home_node), True) in rows["FSObject::parent"]
    assert ((report_node, Permission.write), True) in rows["File::permission"]
    assert ((report_node,), 5) in rows["File::size"]
    assert list(objects(rows["File::size"])) == [((report,), 5)]
    assert ((home,), True) in list(objects(rows["FSObject"]))
    # Literal enums read back as their strings; values are kept as given.
    dmm = deployment_models.metamodel()
    catalogue, graph, ids = _setup(dmm, [deployment_models.deployment()])
    facts(catalogue, graph, ids)
    api = next(n for n in graph.nodes if n.name == "api")
    tier = Fact(
        "Service::tier", (ids.id_of(api), ids.constant_id("ServiceTier::backend")), "unknown"
    )
    assert read_facts(catalogue, ids, [tier]) == {"Service::tier": [((api, "backend"), "unknown")]}


def test_read_back_unknown_ids():
    catalogue, graph, ids = _setup(deployment_models.metamodel(), [deployment_models.deployment()])
    facts(catalogue, graph, ids)
    generated = Fact("Service", (999,), True)
    with pytest.raises(KeyError):
        read_facts(catalogue, ids, [generated])
    assert read_facts(catalogue, ids, [generated], strict=False) == {"Service": [((999,), True)]}
    ids.constant("atom", 500)
    assert read_facts(catalogue, ids, [Fact("p", (500,), True)]) == {"p": [(("atom",), True)]}


def test_node_ids_put_keeps_ids():
    from pyfinery._instances import Node

    a, b, c, d = (Node(obj=None, identity=i, name=n) for i, n in enumerate("abcd"))  # type: ignore[arg-type]
    ids = NodeIds(first=10)
    ids.constant("E::x", 12)
    ids.put(a, 20)
    assert ids.id_of(a) == 20 and ids.node(20) is a
    ids.put(a, 20)  # again, the same id: nothing changes
    with pytest.raises(ValueError, match="already has id 20"):
        ids.put(a, 21)
    with pytest.raises(ValueError, match="already taken"):
        ids.put(b, 20)  # another node's
    with pytest.raises(ValueError, match="already taken"):
        ids.put(b, 12)  # a constant's
    assert ids.get(b) is None
    # add() after put() gives ids past it, never one put() gave.
    assert ids.add(b) == 21
    ids.put(c, 15)  # below the next id: add() keeps going from 22
    assert ids.add(d) == 22
    assert ids.node(15) is c
