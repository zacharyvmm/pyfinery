"""Facts as tuples of node ids: the mapping between a :class:`Schema` and its
:class:`Graph` on one side and the relations of a Refinery model on the other.

An engine running Refinery in-process reads and writes relations directly
rather than problem text. Each relation (a class such as ``Service``, a
feature such as ``Service::ports``, a predicate) holds tuples of integer node
ids, each with a value. :class:`Catalogue` names the relations as
:func:`~pyfinery._render.render_metamodel` does, :class:`NodeIds` numbers the
nodes, :func:`facts` writes a graph as :class:`Fact` s, and :func:`read_facts`
turns facts back into nodes and Python values.

The value of a fact is kept abstract: :func:`facts` writes the value it is
given for what holds, and :func:`read_facts` keeps whatever value it reads.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from ._instances import Graph, Node
from ._render import _enum_literal, feature_symbol
from ._schema import ClassLike, EnumLike, Feature, LiteralEnum, Schema, UnionClass, class_name


@dataclass(frozen=True, slots=True)
class Fact:
    """A tuple of node ids in a relation, and its value.

    For a class or reference, the value says whether the tuple holds (``True``
    in a two-valued model); for an attribute, it is the attribute's value.
    """

    relation: str
    nodes: tuple[int, ...]
    value: object


type AttributeType = Literal["boolean", "int", "real", "string"]
"""The Refinery data type of an attribute."""


def attribute_type(feature: Feature) -> AttributeType:
    """The data type of the values of ``feature``, an attribute: what an engine converts the
    Python value of its :class:`Fact` by."""
    if not feature.is_attribute:
        raise ValueError(f"{feature_symbol(feature)} is not an attribute")
    return feature.target  # type: ignore[return-value]


# -- names ---------------------------------------------------------------------


class Catalogue:
    """The Refinery names of the classes, unions, enums, enum literals and
    features of a :class:`Schema`, spelled as in its rendered metamodel."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.classes: dict[str, ClassLike] = {}
        """Classes and union superclasses by Refinery name."""
        self.enums: dict[str, EnumLike] = {}
        self.literals: dict[str, tuple[EnumLike, Any]] = {}
        """Enum literals by Refinery name (``Color::red``): the enum, and the Python
        value a field holds for it (the member, or the string of a ``Literal``)."""
        self.features: dict[str, Feature] = {}
        """Features by Refinery name (``Owner::field``)."""
        self._names: dict[Any, str] = {}
        self._supertypes: dict[type[BaseModel], tuple[str, ...]] = {}
        self._opposites: dict[Feature, Feature] = {}

        for enum_type in schema.enums:
            self._add(self.enums, enum_type, class_name(enum_type))
            for value in enum_values(enum_type):
                name = _enum_literal(enum_type, value)
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
                if feature.opposite is not None:
                    other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
                    assert other is not None  # checked by _schema.check
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

    def feature(self, cls: type[BaseModel], field: str) -> Feature:
        """The feature for ``cls.field``, declared in ``cls`` or inherited."""
        found = self.schema.find_feature(cls, field)
        if found is None:
            raise KeyError(f"{cls.__name__}.{field} is not a feature of this metamodel")
        return found

    def lookup(self, name: str) -> ClassLike | EnumLike | Feature | tuple[EnumLike, Any]:
        """What a Refinery name stands for: a class or union, an enum, a feature,
        or an enum literal as ``(enum, value)``."""
        for table in (self.classes, self.features, self.enums, self.literals):
            if name in table:
                return table[name]  # type: ignore[return-value]
        raise KeyError(f"{name!r} names nothing in this metamodel")

    def supertypes(self, cls: type[BaseModel]) -> tuple[str, ...]:
        """The names of ``cls`` and of every class it is an instance of: its
        superclasses in the schema, then the unions of any of them.

        A two-valued model has no reasoning layer, so a node must be put in
        each of these relations explicitly.
        """
        try:
            return self._supertypes[cls]
        except KeyError:
            raise KeyError(f"{cls.__name__} is not part of this metamodel") from None

    def opposite(self, feature: Feature, target: type | None = None) -> Feature | None:
        """The feature on the other side of ``feature``, if it has an opposite: as declared on
        ``feature``'s target class, or on ``target``, the class of the object referred to, which
        may be a subclass (as ``render_facts`` looks it up)."""
        if feature.opposite is None:
            return None
        if target is not None:
            return self.schema.find_feature(target, feature.opposite)
        return self._opposites.get(feature)

    def literal_names(self) -> list[str]:
        """The names of every enum literal, in declaration order: the constants a
        model needs before :func:`facts` can refer to them."""
        return list(self.literals)


def enum_values(enum_type: EnumLike) -> tuple[Any, ...]:
    """The values a field of ``enum_type`` holds: its members, or its strings."""
    if isinstance(enum_type, LiteralEnum):
        return enum_type.values
    assert isinstance(enum_type, type) and issubclass(enum_type, enum.Enum)
    return tuple(enum_type)


# -- node ids ------------------------------------------------------------------


class NodeIds:
    """Integer ids for graph nodes, and for named constants such as enum literals.

    Nodes get ids in the order they are added, from ``first`` on, skipping ids
    taken by constants. An id stays with its node until the node is removed,
    and is never given to another node afterwards.
    """

    def __init__(self, first: int = 0) -> None:
        self._next = first
        self._ids: dict[Node, int] = {}
        self._nodes: dict[int, Node] = {}
        self._constant_ids: dict[str, int] = {}
        self._constant_names: dict[int, str] = {}

    def add(self, node: Node) -> int:
        """The id of ``node``, giving it the next free id if it has none."""
        existing = self._ids.get(node)
        if existing is not None:
            return existing
        while self._next in self._constant_names:
            self._next += 1
        node_id = self._next
        self._next += 1
        self._ids[node] = node_id
        self._nodes[node_id] = node
        return node_id

    def put(self, node: Node, node_id: int) -> None:
        """Give ``node`` the id ``node_id``, which no other node or constant has: to keep
        the ids an engine gave the nodes of an earlier graph of the same objects."""
        if self._ids.get(node, node_id) != node_id:
            raise ValueError(f"Node {node.name or node.obj!r} already has id {self._ids[node]}")
        if self._nodes.get(node_id, node) is not node or node_id in self._constant_names:
            raise ValueError(f"Id {node_id} is already taken")
        self._ids[node] = node_id
        self._nodes[node_id] = node
        self._next = max(self._next, node_id + 1)

    def remove(self, node: Node) -> int:
        """Forget ``node``; return the id it had."""
        node_id = self._ids.pop(node)
        del self._nodes[node_id]
        return node_id

    def id_of(self, node: Node) -> int:
        try:
            return self._ids[node]
        except KeyError:
            raise KeyError(f"No id for node {node.name or node.obj!r}") from None

    def get(self, node: Node) -> int | None:
        return self._ids.get(node)

    def node(self, node_id: int) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"No node has id {node_id}") from None

    def constant(self, name: str, node_id: int) -> None:
        """Register the constant ``name`` (e.g. ``Color::red``) under the id the
        engine gave it."""
        taken = self._constant_names.get(node_id, name)
        if taken != name or node_id in self._nodes:
            raise ValueError(f"Id {node_id} is already taken")
        if self._constant_ids.get(name, node_id) != node_id:
            raise ValueError(f"Constant {name!r} already has id {self._constant_ids[name]}")
        self._constant_ids[name] = node_id
        self._constant_names[node_id] = name

    def constant_id(self, name: str) -> int:
        try:
            return self._constant_ids[name]
        except KeyError:
            raise KeyError(f"No constant named {name!r}; register it with constant()") from None

    def constant_name(self, node_id: int) -> str | None:
        return self._constant_names.get(node_id)

    def nodes(self) -> dict[int, Node]:
        """Nodes by id, in the order they were added."""
        return dict(self._nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._ids

    def __len__(self) -> int:
        return len(self._ids)


# -- writing -------------------------------------------------------------------


def facts(
    catalogue: Catalogue,
    graph: Graph,
    ids: NodeIds,
    *,
    holds: object = True,
    uncertain: Literal["omit", "include"] = "omit",
    unknown_existence: Literal["exists", "omit"] = "exists",
) -> list[Fact]:
    """The facts of a concrete, two-valued model of ``graph``.

    Every node is added to ``ids``. The facts are, for each node: its
    membership in its class and in every supertype
    (:meth:`Catalogue.supertypes`); a fact for each reference, and also its
    reverse for a feature with an opposite; a reference to the enum literal
    constant for each enum value (the literals must be registered in ``ids``
    with :meth:`NodeIds.constant`); and the value of each attribute.
    Memberships and references have the value ``holds``; an attribute fact's
    value is the Python value. A key that resolves to no object is left out,
    as in :func:`~pyfinery._render.render_facts`. Facts are unique by
    relation and nodes.

    Two-valued models have no ``unknown``, so:

    - ``uncertain`` says what to do with values that only may hold
      (``Value.certain`` false): ``"omit"`` them, as :meth:`Metamodel.validate`
      does by treating them as not holding, or ``"include"`` them as if they
      held. Under ``"omit"`` an uncertain attribute has no value, as in
      ``render_facts``.
    - ``unknown_existence`` says what to do with nodes that may not exist
      (``Node.exists == "unknown"``, only in graphs built with ``existence``):
      ``"exists"`` keeps them, as ``validate`` does by ignoring
      ``ClassOptions.exists``; ``"omit"`` leaves them out, with every fact that
      mentions them, and doesn't add them to ``ids``.
    """
    written: dict[tuple[str, tuple[int, ...]], Fact] = {}

    def write(relation: str, nodes: tuple[int, ...], value: object) -> None:
        written.setdefault((relation, nodes), Fact(relation, nodes, value))

    present = [
        node
        for node in graph.nodes
        if node.exists == "true" or (node.exists == "unknown" and unknown_existence == "exists")
    ]
    kept = set(present)
    for node in present:
        ids.add(node)
    for node in present:
        node_id = ids.id_of(node)
        for relation in catalogue.supertypes(type(node.obj)):
            write(relation, (node_id,), holds)
        for value in node.values:
            if not value.certain and uncertain == "omit":
                continue
            feature = value.feature
            relation = catalogue.name(feature)
            if feature.is_attribute:
                write(relation, (node_id,), value.value)
            elif feature.is_enum:
                literal = catalogue.literal(feature.target, value.value)  # type: ignore[arg-type]
                write(relation, (node_id, ids.constant_id(literal)), holds)
            else:
                target = value.target
                if target is None or target not in kept:
                    continue  # an unresolved key (see reference_errors), or omitted
                target_id = ids.id_of(target)
                write(relation, (node_id, target_id), holds)
                other = catalogue.opposite(feature, type(target.obj))
                if other is not None:
                    write(catalogue.name(other), (target_id, node_id), holds)
    return list(written.values())


# -- reading -------------------------------------------------------------------


type Row = tuple[tuple[Any, ...], object]
"""The arguments of a fact, as nodes and Python values, and its value."""


def read_facts(
    catalogue: Catalogue,
    ids: NodeIds,
    facts: Iterable[Fact],
    *,
    strict: bool = True,
) -> dict[str, list[Row]]:
    """The rows of ``facts`` by relation, in the order read, with ids resolved.

    An id resolves to its :class:`Node` (whose ``obj`` is the Python object),
    an enum literal constant to the Python value a field holds for it (the
    enum member, or the string of a ``Literal``), and another constant to its
    name. Each value is kept as given. An unknown id (e.g. of a node an engine
    generated) raises ``KeyError``, or without ``strict`` is kept as the int.
    """
    rows: dict[str, list[Row]] = {}
    for fact in facts:
        arguments = tuple(_resolve(catalogue, ids, n, strict) for n in fact.nodes)
        rows.setdefault(fact.relation, []).append((arguments, fact.value))
    return rows


def _resolve(catalogue: Catalogue, ids: NodeIds, node_id: int, strict: bool) -> Any:
    name = ids.constant_name(node_id)
    if name is not None:
        literal = catalogue.literals.get(name)
        return name if literal is None else literal[1]
    try:
        return ids.node(node_id)
    except KeyError:
        if strict:
            raise
        return node_id


def objects(rows: Iterable[Row]) -> Iterator[tuple[tuple[Any, ...], object]]:
    """``rows`` with each :class:`Node` replaced by its Python object."""
    for arguments, value in rows:
        yield tuple(a.obj if isinstance(a, Node) else a for a in arguments), value
