# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

""":meth:`Metamodel.query` by the bits engine: the partial model Refinery would build,
built in Python from the instance graph.

:meth:`Metamodel.query` sends Refinery the graph's facts in a closed world
(``render_query_problem``): there is nothing beyond what is asserted, and
what is uncertain stays unknown. The partial model that follows has:

- the graph's nodes, which exist or only may, and the enums' literals, which
  are atoms; no ``C::new`` objects, which don't exist;
- each class (and union, and enum) holding exactly its own and its
  subclasses' nodes;
- each reference holding the values asserted, ``true`` or ``unknown``, on
  both sides of an opposite pair, and no others;
- each attribute a value (or unknown) on the nodes of its class, and none,
  an error, on others;
- each :class:`~pyfinery.Predicate` its facts, and nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..._instances import Graph, Value
from ..._options import LogicValue, Predicate
from ..._render import predicate_facts
from ..._schema import EnumLike, Feature, LiteralEnum, Schema
from ..matrices.kernels import NodeIndex
from ..matrices.model import Attribute, DataType, Node, PartialModel, Relation
from ..matrices.values import NodeTuple
from ..names import refinery_name

if TYPE_CHECKING:
    from pydantic import BaseModel


def opposite_name(schema: Schema, feature: Feature) -> str:
    other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
    assert other is not None
    return feature_name(other)


def partial_model(
    schema: Schema, graph: Graph, predicates: Sequence[Predicate]
) -> tuple[PartialModel, list[BaseModel | None]]:
    """The partial model of ``graph`` in a closed world, and the object of each node
    (``None`` for enum literals)."""
    model = ModelBuilder(graph)
    for enum_type in schema.enums:
        model.enum(enum_type)
    for cls in schema.classes:
        model.class_(refinery_name(cls.__name__), cls)
    for union in schema.unions:
        model.class_(refinery_name(union.name), union.members)
    for cls in schema.classes:
        for feature in schema.features[cls]:
            model.feature(feature)
    for i, node in enumerate(graph.nodes):
        for value in node.values:
            model.value(schema, i, value)
    for predicate in predicates:
        model.relation(refinery_name(predicate.name), len(predicate.parameters))
    for predicate, arguments, answer in predicate_facts(schema, graph, predicates):
        key = tuple(model.index[id(node)] for node in arguments)
        model.holds(refinery_name(predicate.name), key, answer)
    return model.build(), model.objects


type Facts = dict[NodeTuple, LogicValue]
"""What holds of the tuples of a relation; those not given are false."""


class ModelBuilder:
    """The partial model of a graph, built up from its classes, features and values."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        names = [output_name(node.name) for node in graph.nodes]
        self.nodes = [Node(name, name) for name in names]
        self.objects: list[BaseModel | None] = [node.obj for node in graph.nodes]
        self.index: dict[int, NodeIndex] = {id(node): i for i, node in enumerate(graph.nodes)}
        """Each of the graph's nodes, by its ``id()``."""
        self.literals: dict[tuple[str, str], NodeIndex] = {}
        """Each enum literal, by the enum's name and the member's."""
        self.relations: dict[str, tuple[int, Facts]] = {}
        """Each relation's arity and facts, by name."""
        self.attributes: dict[str, tuple[DataType, dict[NodeIndex, object]]] = {}
        """Each attribute's type and values (``None`` if unknown), by name."""
        exists = self.relation("builtin::exists", 1)
        for i, node in enumerate(graph.nodes):
            exists[(i,)] = node.exists

    def relation(self, name: str, arity: int) -> Facts:
        return self.relations.setdefault(name, (arity, {}))[1]

    def holds(self, name: str, key: NodeTuple, value: LogicValue) -> None:
        tuples = self.relations[name][1]
        if tuples.get(key) != "true":  # of a fact given twice, the surer one
            tuples[key] = value

    def enum(self, enum_type: EnumLike) -> None:
        """The enum's literals are atoms, and its class holds them."""
        if isinstance(enum_type, LiteralEnum):
            members: tuple[str, ...] = enum_type.values
        else:
            members = tuple(member.name for member in enum_type)
        name = type_name(enum_type)
        literals = self.relation(name, 1)
        for member in members:
            index = len(self.nodes)
            self.literals[(name, member)] = index
            literal = refinery_name(member)
            self.nodes.append(Node(f"{name}::{literal}", literal, "atom"))
            self.objects.append(None)
            literals[(index,)] = "true"
            self.holds("builtin::exists", (index,), "true")

    def class_(self, name: str, classes: type | tuple[type, ...]) -> None:
        members = self.relation(name, 1)
        for i, node in enumerate(self.graph.nodes):
            if isinstance(node.obj, classes):
                members[(i,)] = "true"

    def feature(self, feature: Feature) -> None:
        name = feature_name(feature)
        if not feature.is_attribute:
            self.relation(name, 2)
            return
        # Unknown on the nodes of the class, until a value is given; none elsewhere.
        members = [
            i for i, node in enumerate(self.graph.nodes) if isinstance(node.obj, feature.owner)
        ]
        self.attributes[name] = (feature.target, {i: None for i in members})  # type: ignore[assignment]
        if feature.target == "boolean":
            values = self.relation(name, 1)
            for i in range(len(self.nodes)):  # enum literals too
                values[(i,)] = "error"  # no value outside the class
            for i in members:
                values[(i,)] = "unknown"

    def value(self, schema: Schema, i: NodeIndex, value: Value) -> None:
        feature = value.feature
        name = feature_name(feature)
        if feature.is_attribute:
            if value.certain:  # an uncertain value is simply unknown
                self.attributes[name][1][i] = value.value
                if feature.target == "boolean":
                    self.relations[name][1][(i,)] = "true" if value.value else "false"
            return
        certainty: LogicValue = "true" if value.certain else "unknown"
        if feature.is_enum:
            member = value.value if isinstance(feature.target, LiteralEnum) else value.value.name
            target = self.literals[(type_name(feature.target), member)]
        elif value.target is not None:
            target = self.index[id(value.target)]
        else:
            return  # a key that resolves to no node: reported by reference_errors
        self.holds(name, (i, target), certainty)
        if feature.opposite is not None:
            other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
            assert other is not None
            self.holds(feature_name(other), (target, i), certainty)

    def build(self) -> PartialModel:
        n = len(self.nodes)
        relations = [
            Relation.of(name, arity, tuples.items())
            for name, (arity, tuples) in self.relations.items()
        ]
        attributes = [
            Attribute.of(name, data_type, n, values.items())
            for name, (data_type, values) in self.attributes.items()
        ]
        return PartialModel(self.nodes, relations, attributes)


def type_name(node_class: object) -> str:
    """The name of a class, union or enum, as Refinery's output writes it."""
    if isinstance(node_class, LiteralEnum):
        return refinery_name(node_class.name)
    return refinery_name(node_class.__name__)  # type: ignore[attr-defined]


def feature_name(feature: Feature) -> str:
    return f"{refinery_name(feature.owner.__name__)}::{refinery_name(feature.name)}"


def output_name(rendered: str) -> str:
    """A node's name as Refinery's output writes it, from the name pyfinery gives it in
    the problem (``quote_id``, which quotes some names Refinery's output doesn't)."""
    if len(rendered) >= 2 and rendered[0] == rendered[-1] == "'":
        return refinery_name(re.sub(r"\\(.)", r"\1", rendered[1:-1]))
    return refinery_name(rendered)
