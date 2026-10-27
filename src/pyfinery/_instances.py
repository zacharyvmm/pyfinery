# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The graph of Refinery nodes built from Pydantic objects, which facts are
written from.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from ._options import LogicValue
from ._schema import Feature, Schema, UnionClass
from ._syntax import RESERVED_NODE_NAMES, encodable, quote_id


@dataclass(eq=False, slots=True)
class Value:
    """One value of a field of a node: an attribute or enum value, or a reference."""

    feature: Feature
    value: Any
    """The attribute or enum value, the key of a reference by key, or the referenced object."""
    certain: bool = True
    """Whether the value certainly holds (``ClassOptions.logic_value``), or only may."""
    target: Node | None = None
    """The node a reference resolves to."""
    unresolved: Literal["unresolved", "ambiguous"] | None = None
    """Why a key of a reference by key resolves to no node."""


@dataclass(eq=False, slots=True)
class Node:
    """One Refinery node: a Python object, or the instances sharing a key."""

    obj: BaseModel
    """The object representing the node: of instances sharing a key, the first one reached."""
    identity: Hashable
    name: str = ""
    exists: LogicValue = "true"
    values: list[Value] = field(default_factory=list)


@dataclass(slots=True)
class Graph:
    """The nodes reachable from some objects, in the order they were reached."""

    identity: Callable[[BaseModel], Hashable]
    """:meth:`Metamodel.identity`, which reports use to find a node's instances."""
    nodes: list[Node] = field(default_factory=list)
    by_identity: dict[Hashable, Node] = field(default_factory=dict)
    identities: dict[int, tuple[BaseModel, Hashable]] = field(default_factory=dict)
    """:attr:`identity` by object, keeping the objects alive so that their ``id``
    isn't reused."""

    def identity_of(self, obj: BaseModel) -> Hashable:
        """:attr:`identity` of ``obj``, computed once per object."""
        entry = self.identities.get(id(obj))
        if entry is None:
            entry = self.identities[id(obj)] = (obj, self.identity(obj))
        return entry[1]

    def node_of(self, obj: BaseModel) -> Node | None:
        return self.by_identity.get(self.identity_of(obj))

    def objects_by_name(self) -> dict[str, BaseModel]:
        return {node.name: node.obj for node in self.nodes}


type _Fields = list[tuple[Feature, list[Any]]]


def build_graph(schema: Schema, objects: Iterable[BaseModel], *, existence: bool = False) -> Graph:
    """The Refinery nodes of the graph reachable from ``objects``, with their values.

    The ``key``, ``node_name``, ``logic_value`` and (with ``existence``)
    ``exists`` options are evaluated here, once; everything sent to
    Refinery is rendered from the result. Without ``existence``, every
    node exists, as :meth:`Metamodel.validate` assumes.
    """
    graph = Graph(schema.identity)
    reached = _reach(schema, objects, graph.identity_of)
    graph.nodes = [node for node, _ in reached]
    graph.by_identity = {node.identity: node for node in graph.nodes}
    _name(schema, graph.nodes)
    _add_values(schema, graph, reached)
    if existence:
        for node in graph.nodes:
            node.exists = _exists(schema, node.obj)
    return graph


def collect(schema: Schema, objects: Iterable[BaseModel]) -> list[BaseModel]:
    """The objects of the nodes reachable from ``objects``."""
    return [node.obj for node, _ in _reach(schema, objects, Graph(schema.identity).identity_of)]


def link_opposites(schema: Schema, objects: Iterable[BaseModel]) -> int:
    """Fill in the missing side of every ``opposite`` pair; return how many were added."""
    added = 0
    for node in collect(schema, objects):
        for feature, values in _field_values(schema, node):
            if feature.opposite is None:
                continue
            for target in values:
                other = schema.find_feature(type(target), feature.opposite)
                if other is not None and _add_reference(target, other, node, schema.identity):
                    added += 1
    return added


def _reach(
    schema: Schema, objects: Iterable[BaseModel], identity_of: Callable[[BaseModel], Hashable]
) -> list[tuple[Node, _Fields]]:
    """The nodes reachable from ``objects``, each with the values of its fields."""
    reached = []
    seen: set[Hashable] = set()
    pending = deque(objects)
    while pending:
        obj = pending.popleft()
        if type(obj) not in schema.features:
            raise TypeError(f"{type(obj).__name__} is not part of this metamodel")
        identity = identity_of(obj)
        if identity in seen:
            continue
        seen.add(identity)
        fields = list(_field_values(schema, obj))
        reached.append((Node(obj, identity), fields))
        for feature, values in fields:
            if not (feature.is_attribute or feature.is_enum or feature.by_key):
                pending.extend(values)
    return reached


def _field_values(schema: Schema, obj: BaseModel) -> Iterator[tuple[Feature, list[Any]]]:
    for cls in type(obj).__mro__:
        for feature in schema.features.get(cls, []):  # type: ignore[call-overload]
            value = getattr(obj, feature.name)
            if value is None:
                continue
            yield feature, list(value) if feature.many else [value]


def _name(schema: Schema, nodes: Sequence[Node]) -> None:
    """Give every node the name it has in the problem."""
    used: set[str] = set()
    counters: dict[type, int] = {}
    for node in nodes:
        obj = node.obj
        id_field: str | None = getattr(type(obj), "__node_id__", None)
        naming = schema.options_for(type(obj)).node_name
        if naming is not None or id_field is not None:
            raw = naming(obj) if naming is not None else getattr(obj, id_field)  # type: ignore[arg-type]
            name = quote_id(str(raw))
            if name in used:
                source = "node_name" if naming is not None else f"field {id_field!r}"
                raise ValueError(f"Duplicate node id {name} (from {source})")
        elif (key := _name_from_key(schema, node)) is not None and key not in used:
            name = key
        else:
            while True:
                counters[type(obj)] = counters.get(type(obj), 0) + 1
                name = quote_id(f"{type(obj).__name__.lower()}_{counters[type(obj)]}")
                if name not in used:
                    break
        used.add(name)
        node.name = name


def _name_from_key(schema: Schema, node: Node) -> str | None:
    """The key as a node name, if it is a string or integer that isn't a type's name."""
    if schema.options_for(type(node.obj)).key is None:
        return None
    key = node.identity[1]  # type: ignore[index]
    if isinstance(key, bool) or not isinstance(key, str | int) or str(key) == "":
        return None
    if not encodable(str(key)):
        # Escaped, Refinery would name the node with "?" for each lone
        # surrogate, so the name couldn't be mapped back to the node.
        return None
    name = quote_id(str(key))
    return None if name in schema.type_names or str(key) in RESERVED_NODE_NAMES else name


def _add_values(schema: Schema, graph: Graph, reached: list[tuple[Node, _Fields]]) -> None:
    """Resolve the values of every node's fields: references to nodes, and certainty."""
    by_key: dict[Hashable, list[Node]] = {}
    for node in graph.nodes:
        if schema.options_for(type(node.obj)).key is not None:
            by_key.setdefault(node.identity[1], []).append(node)  # type: ignore[index]
    for node, fields in reached:
        for feature, values in fields:
            for value in values:
                node.values.append(_value(schema, graph, node, feature, value, by_key))


def _value(
    schema: Schema,
    graph: Graph,
    node: Node,
    feature: Feature,
    value: Any,
    by_key: Mapping[Hashable, list[Node]],
) -> Value:
    if feature.by_key:
        target = feature.target
        members = target.members if isinstance(target, UnionClass) else (target,)
        found = [n for n in by_key.get(value, []) if isinstance(n.obj, members)]  # type: ignore[arg-type]
        if len(found) != 1:
            problem = "unresolved" if not found else "ambiguous"
            return Value(feature, value, unresolved=problem)
        return Value(feature, value, _certain(schema, node.obj, feature, value), found[0])
    certain = _certain(schema, node.obj, feature, value)
    if feature.is_attribute or feature.is_enum:
        return Value(feature, value, certain)
    # Every followed reference was reached, by the same identity.
    return Value(feature, value, certain, graph.by_identity[graph.identity_of(value)])


def _certain(schema: Schema, node: BaseModel, feature: Feature, value: Any) -> bool:
    """Whether ``value`` certainly holds, as ``ClassOptions.logic_value`` says."""
    decide = schema.options_for(type(node)).logic_value
    if decide is None:
        return True
    answer = decide(node, feature.name, value)
    if answer in ("true", True):
        return True
    if answer == "unknown":
        return False
    raise ValueError(
        f"ClassOptions.logic_value of {type(node).__name__} must return 'true', 'unknown' or "
        f"True, "
        f"not {answer!r} (leave values that don't hold out of the object)"
    )


def _exists(schema: Schema, node: BaseModel) -> LogicValue:
    decide = schema.options_for(type(node)).exists
    if decide is None:
        return "true"
    answer = decide(node)
    if answer in ("true", True):
        return "true"
    if answer == "unknown":
        return "unknown"
    raise ValueError(
        f"ClassOptions.exists of {type(node).__name__} must return 'true', 'unknown' or True, "
        f"not {answer!r} (leave out objects that don't exist)"
    )


def _add_reference(
    node: BaseModel,
    feature: Feature,
    value: BaseModel,
    identity: Callable[[BaseModel], Hashable],
) -> bool:
    """Add ``value`` to ``node.<feature>`` unless already present or the slot is taken."""
    current = getattr(node, feature.name)
    if not feature.many:
        if current is not None:
            return False
        setattr(node, feature.name, value)
        return True
    if current is None:
        setattr(node, feature.name, [value])
        return True
    if any(identity(item) == identity(value) for item in current):
        return False
    if isinstance(current, list):
        current.append(value)
    elif isinstance(current, set):
        current.add(value)
    elif isinstance(current, frozenset):
        setattr(node, feature.name, frozenset((*current, value)))
    else:
        setattr(node, feature.name, (*current, value))
    return True
