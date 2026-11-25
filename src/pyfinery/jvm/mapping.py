"""The relations of a plain Refinery model of a :class:`Schema`, and a graph's facts in them.

A plain model holds relations rather than problem text. Each relation (a class such as
``Service``, a feature such as ``Service::ports``, a predicate) holds tuples of integer node
ids, each with a value. :class:`Catalogue` names the relations as
:func:`~pyfinery._render.render_metamodel` does, and :func:`graph_facts` writes a graph as
:class:`Fact` s.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .._instances import Graph, Node
from .._render import enum_literal, feature_symbol
from .._schema import (
    AttributeFeature,
    ClassLike,
    EnumFeature,
    EnumLike,
    Feature,
    LiteralEnum,
    ReferenceFeature,
    Schema,
    UnionClass,
    class_name,
)


@dataclass(frozen=True, slots=True)
class Fact:
    """A tuple of node ids in a relation, and its value: ``True`` for a class or reference
    (it holds), the Python value for an attribute."""

    relation: str
    nodes: tuple[int, ...]
    value: object


class Catalogue:
    """The Refinery names of the classes, unions, enums, enum literals and features of a
    :class:`Schema`, spelled as in its rendered metamodel."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.classes: dict[str, ClassLike] = {}
        """Classes and union superclasses by Refinery name."""
        self.enums: dict[str, EnumLike] = {}
        self.literals: dict[str, tuple[EnumLike, Any]] = {}
        """Enum literals by Refinery name (``Color::red``): the enum, and the Python value a
        field holds for it (the member, or the string of a ``Literal``)."""
        self.features: dict[str, Feature] = {}
        """Features by Refinery name (``Owner::field``)."""
        self._names: dict[Any, str] = {}
        self._supertypes: dict[type[BaseModel], tuple[str, ...]] = {}
        self._opposites: dict[Feature, Feature] = {}

        for enum_type in schema.enums:
            self._add(self.enums, enum_type, class_name(enum_type))
            for value in _enum_values(enum_type):
                name = enum_literal(enum_type, value)
                self.literals[name] = (enum_type, value)
                self._names[(enum_type, value)] = name
        for union in schema.unions:
            self._add(self.classes, union, class_name(union))
        for cls in schema.classes:
            self._add(self.classes, cls, class_name(cls))
            for feature in schema.features[cls]:
                self._add(self.features, feature, feature_symbol(feature))
        for cls in schema.classes:
            supers = [c for c in cls.__mro__ if c in schema.features]
            unions = [u for u in schema.unions if any(c in u.members for c in supers)]
            self._supertypes[cls] = tuple(self._names[t] for t in [*supers, *unions])
            for feature in schema.features[cls]:
                if not isinstance(feature, ReferenceFeature) or feature.opposite is None:
                    continue
                # check() rejects opposites of unions and opposites that aren't fields.
                assert not isinstance(feature.target, UnionClass)
                other = schema.find_feature(feature.target, feature.opposite)
                assert other is not None
                self._opposites[feature] = other

    def _add(self, table: dict[str, Any], item: Any, name: str) -> None:
        table[name] = item
        self._names[item] = name

    def name(self, item: type | UnionClass | LiteralEnum | Feature) -> str:
        """The Refinery name of a class, union, enum, or feature of the schema."""
        try:
            return self._names[item]
        except (KeyError, TypeError):
            raise KeyError(f"{item!r} is not part of this metamodel") from None

    def literal(self, enum_type: EnumLike, value: Any) -> str:
        """The Refinery name of an enum value, e.g. ``Color::red``."""
        try:
            return self._names[(enum_type, value)]
        except (KeyError, TypeError):
            raise KeyError(f"{value!r} is not a literal of {enum_type!r}") from None

    def supertypes(self, cls: type[BaseModel]) -> tuple[str, ...]:
        """The names of ``cls`` and of every class it is an instance of: its superclasses in
        the schema, then the unions of any of them.

        A plain model has no reasoning layer, so a node must be put in each of these
        relations explicitly.
        """
        try:
            return self._supertypes[cls]
        except KeyError:
            raise KeyError(f"{cls.__name__} is not part of this metamodel") from None

    def opposite(self, feature: Feature, target: type | None = None) -> Feature | None:
        """The feature on the other side of ``feature``, if it has an opposite: as declared on
        ``feature``'s target class, or on ``target``, the class of the object referred to, which
        may be a subclass (as ``render_facts`` looks it up)."""
        if not isinstance(feature, ReferenceFeature) or feature.opposite is None:
            return None
        if target is not None:
            return self.schema.find_feature(target, feature.opposite)
        return self._opposites.get(feature)

    def literal_names(self) -> list[str]:
        """The names of every enum literal, in declaration order: the constants a model needs
        before :func:`graph_facts` can refer to them."""
        return list(self.literals)


def _enum_values(enum_type: EnumLike) -> tuple[Any, ...]:
    """The values a field of ``enum_type`` holds: its members, or its strings."""
    if isinstance(enum_type, LiteralEnum):
        return enum_type.values
    assert isinstance(enum_type, type) and issubclass(enum_type, enum.Enum)
    return tuple(enum_type)


def graph_facts(
    catalogue: Catalogue,
    graph: Graph,
    node_ids: Mapping[Node, int],
    constants: Mapping[str, int],
) -> list[Fact]:
    """The facts of ``graph`` in a two-valued model, unique by relation and nodes.

    For each node: its membership in its class and in every supertype
    (:meth:`Catalogue.supertypes`); a fact for each reference, and its reverse for a feature
    with an opposite; a reference to the enum literal's constant id in ``constants`` for each
    enum value; and the value of each attribute. Every node of ``graph`` must have an id in
    ``node_ids``.

    A value that only may hold (``Value.certain`` false) is left out, as
    :meth:`Metamodel.validate` treats it as not holding; an uncertain attribute has no value,
    as in :func:`~pyfinery._render.render_facts`. So is a key that resolves to no object.
    """
    written: dict[tuple[str, tuple[int, ...]], Fact] = {}

    def write(relation: str, nodes: tuple[int, ...], value: object) -> None:
        written.setdefault((relation, nodes), Fact(relation, nodes, value))

    for node in graph.nodes:
        node_id = node_ids[node]
        for relation in catalogue.supertypes(type(node.obj)):
            write(relation, (node_id,), True)
        for value in node.values:
            if not value.certain:
                continue
            feature = value.feature
            relation = catalogue.name(feature)
            if isinstance(feature, AttributeFeature):
                write(relation, (node_id,), value.value)
            elif isinstance(feature, EnumFeature):
                literal = catalogue.literal(feature.target, value.value)
                write(relation, (node_id, constants[literal]), True)
            else:
                target = value.target
                if target is None:
                    continue  # an unresolved key (see reference_errors)
                target_id = node_ids[target]
                write(relation, (node_id, target_id), True)
                other = catalogue.opposite(feature, type(target.obj))
                if other is not None:
                    write(catalogue.name(other), (target_id, node_id), True)
    return list(written.values())
