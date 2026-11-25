"""The relations of a plain model and a graph's facts in them (pure Python: no JVM)."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from pyfinery import ClassOptions, Metamodel
from pyfinery._instances import Graph, Node, build_graph
from pyfinery._options import LogicValue
from pyfinery._render import render_facts
from pyfinery._schema import EnumFeature, Feature
from pyfinery.jvm.mapping import Catalogue, Fact, graph_facts

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


def _feature(metamodel: Metamodel, cls: type, name: str) -> Feature:
    found = metamodel.schema.find_feature(cls, name)
    assert found is not None
    return found


class _Facts:
    """A graph's facts, with the node and constant ids they were written with."""

    def __init__(self, metamodel: Metamodel, objs: list[BaseModel]) -> None:
        self.catalogue = Catalogue(metamodel.schema)
        names = self.catalogue.literal_names()
        self.constants = {name: i for i, name in enumerate(names)}
        self.graph: Graph = build_graph(metamodel.schema, objs)
        self.ids: dict[Node, int] = {
            node: len(names) + i for i, node in enumerate(self.graph.nodes)
        }
        self.facts = graph_facts(self.catalogue, self.graph, self.ids, self.constants)
        self.by_key = {(f.relation, f.nodes): f.value for f in self.facts}

    def id_of(self, obj: BaseModel) -> int:
        node = self.graph.node_of(obj)
        assert node is not None
        return self.ids[node]

    def named(self) -> dict[str, int]:
        return {node.name: node_id for node, node_id in self.ids.items()}


@pytest.mark.parametrize("which", list(METAMODELS))
def test_catalogue_names_match_metamodel(which):
    mm = METAMODELS[which]()
    catalogue = Catalogue(mm.schema)
    types, features, literals = _declared(mm.source)
    assert set(catalogue.classes) | set(catalogue.enums) == types
    assert set(catalogue.features) == features
    assert set(catalogue.literals) == literals
    for name in [*catalogue.classes, *catalogue.enums]:
        assert re.search(rf"\b(class|enum) {re.escape(name)}\b", mm.source)
    for name, item in [*catalogue.classes.items(), *catalogue.enums.items()]:
        assert catalogue.name(item) == name
    for name, feature in catalogue.features.items():
        assert catalogue.name(feature) == name
    for name, (enum_type, value) in catalogue.literals.items():
        assert catalogue.literal(enum_type, value) == name


def test_catalogue_names_and_literals():
    from .deployment_models import Service

    mm = deployment_models.metamodel()
    catalogue = Catalogue(mm.schema)
    ports = _feature(mm, Service, "ports")
    assert catalogue.name(ports) == "Service::ports"
    assert catalogue.features["Service::ports"] is ports
    assert catalogue.name(Service) == "Service"
    assert catalogue.classes["Service"] is Service
    assert catalogue.name(catalogue.classes["Component"]) == "Component"  # type: ignore[arg-type]
    tier_feature = _feature(mm, Service, "tier")
    assert isinstance(tier_feature, EnumFeature)
    tier = tier_feature.target
    assert catalogue.name(tier) == "ServiceTier"
    assert catalogue.enums["ServiceTier"] is tier
    assert catalogue.literal(tier, "backend") == "ServiceTier::backend"
    assert catalogue.literals["ServiceTier::backend"] == (tier, "backend")
    assert catalogue.literal_names()[:3] == [
        "ServiceTier::frontend",
        "ServiceTier::backend",
        "ServiceTier::data",
    ]
    with pytest.raises(KeyError):
        catalogue.name(BaseModel)
    with pytest.raises(KeyError):
        catalogue.literal(tier, "nothing")


def test_enum_class_literals():
    from .graph_models import Permission

    catalogue = Catalogue(graph_models.metamodel().schema)
    assert catalogue.literal(Permission, Permission.read) == "Permission::read"
    assert catalogue.literals["Permission::read"] == (Permission, Permission.read)


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
    with pytest.raises(KeyError):
        fs.supertypes(Service)


def test_opposites():
    from .graph_models import Dir, File, FSObject

    mm = graph_models.metamodel()
    catalogue = Catalogue(mm.schema)
    contents = _feature(mm, Dir, "contents")
    parent = _feature(mm, FSObject, "parent")
    assert catalogue.opposite(contents) is parent
    assert catalogue.opposite(parent) is contents
    assert catalogue.opposite(contents, File) is parent  # inherited by the subclass
    assert catalogue.opposite(_feature(mm, Dir, "name")) is None


def _rendered_pairs(text: str, names: dict[str, int], constants: dict[str, int]) -> set[tuple]:
    """The facts of ``render_facts`` output as ``(relation, ids)``."""
    out = set()
    for line in text.strip().splitlines():
        if match := re.match(r"(\S+)\((.*)\): (.*)\.$", line):
            out.add((match[1], (names[match[2]],)))
            continue
        match = re.match(r"(\S+?)\((.*)\)\.$", line)
        assert match, line
        args = tuple(names[a] if a in names else constants[a] for a in match[2].split(", "))
        out.add((match[1], args))
    return out


def test_graph_facts_match_render_facts():
    from .graph_models import Dir, File, Filesystem, Link, Permission

    mm = graph_models.metamodel()
    home = Dir(name="home")
    report = File(size=-3, hidden=True, label="x", permission=Permission.read)
    link = Link(target=report)
    home.contents = [report, link]
    report.parent = home  # both sides set: written once
    written = _Facts(mm, [Filesystem(root=home)])
    rendered = _rendered_pairs(
        render_facts(mm.schema, written.graph), written.named(), written.constants
    )

    ours = set(written.by_key)
    # The same relations and tuples, plus the supertypes and opposites below.
    assert rendered <= ours
    assert len(written.facts) == len(ours)  # unique by relation and nodes
    home_id, report_id, link_id = (written.id_of(o) for o in (home, report, link))
    by_key = written.by_key
    # Supertype memberships, which render_facts leaves to Refinery.
    assert by_key[("FSObject", (report_id,))] is True
    assert by_key[("FSObject", (home_id,))] is True
    assert by_key[("FSObject", (link_id,))] is True
    # Opposite directions.
    assert by_key[("Dir::contents", (home_id, report_id))] is True
    assert by_key[("FSObject::parent", (report_id, home_id))] is True
    assert by_key[("FSObject::parent", (link_id, home_id))] is True
    # Attributes carry the Python value, enums refer to the literal constant.
    assert by_key[("File::size", (report_id,))] == -3
    assert by_key[("File::hidden", (report_id,))] is True
    assert by_key[("File::label", (report_id,))] == "x"
    assert by_key[("Dir::name", (home_id,))] == "home"
    permission = written.constants["Permission::read"]
    assert by_key[("File::permission", (report_id, permission))] is True
    assert ours - rendered == {
        ("FSObject", (home_id,)),
        ("FSObject", (report_id,)),
        ("FSObject", (link_id,)),
        ("FSObject::parent", (link_id, home_id)),  # report.parent is set, so rendered
    }
    assert {f.value for f in written.facts if f.relation == "FSObject"} == {True}


def test_graph_facts_by_key_and_unions():
    written = _Facts(deployment_models.metamodel(), [deployment_models.deployment()])
    by_key, node = written.by_key, written.named()
    assert by_key[("Component", (node["web"],))] is True
    assert by_key[("Component", (node["db"],))] is True
    assert by_key[("Service::depends_on", (node["web"], node["api"]))] is True
    assert by_key[("Service::depends_on", (node["web"], node["cache"]))] is True
    assert by_key[("Service::depends_on", (node["api"], node["db"]))] is True
    # "billing" resolves to no object: skipped.
    depends = [n for r, n in by_key if r == "Service::depends_on"]
    assert len(depends) == 3
    tier = written.constants["ServiceTier::backend"]
    assert by_key[("Service::tier", (node["api"], tier))] is True
    assert by_key[("Database::encrypted", (node["db"],))] is False


class _Item(BaseModel):
    name: str
    size: int = 0
    friends: list[_Item] = []


def _logic(item: _Item, field: str, value: object) -> LogicValue:
    return "unknown" if field in ("size", "friends") and item.name == "b" else "true"


def test_uncertain_values_are_omitted():
    mm = Metamodel(_Item, options={_Item: ClassOptions(key=lambda i: i.name, logic_value=_logic)})
    c = _Item(name="c")
    a = _Item(name="a", size=1)
    b = _Item(name="b", size=2, friends=[a])
    a.friends = [b, c]
    written = _Facts(mm, [a])
    ids = {name: written.id_of(obj) for name, obj in (("a", a), ("b", b), ("c", c))}
    facts = {(f.relation, f.nodes, f.value) for f in written.facts}
    assert ("_Item::size", (ids["a"],), 1) in facts
    assert not any(r == "_Item::size" and n == (ids["b"],) for r, n, _ in facts)
    assert ("_Item::friends", (ids["b"], ids["a"]), True) not in facts
    assert ("_Item::friends", (ids["a"], ids["c"]), True) in facts
    assert ("_Item", (ids["b"],), True) in facts
    assert all(isinstance(f, Fact) for f in written.facts)
