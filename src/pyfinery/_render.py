# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Writing the problems sent to Refinery: the metamodel of a :class:`Schema`, and
the facts of a :class:`Graph` for validation and queries.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from ._instances import Graph, Node
from ._options import Predicate
from ._schema import (
    AttributeFeature,
    EnumFeature,
    EnumLike,
    Feature,
    LiteralEnum,
    ReferenceFeature,
    Schema,
    UnionClass,
    class_name,
    is_model_class,
)
from ._syntax import attribute_literal, multiplicity, quote_id


def render_metamodel(schema: Schema, rules: Sequence[str]) -> str:
    """Declarations of the enums, unions and classes of ``schema``, then ``rules``."""
    lines: list[str] = []
    for enum_type in schema.enums:
        if isinstance(enum_type, LiteralEnum):
            members = enum_type.values
        else:
            members = tuple(member.name for member in enum_type)
        literals = ", ".join(quote_id(member) for member in members)
        lines.append(f"enum {class_name(enum_type)} {{ {literals} }}")
    for union in schema.unions:
        lines.append(f"abstract class {class_name(union)}.")
    for cls in schema.classes:
        header = f"class {class_name(cls)}"
        if schema.is_abstract(cls):
            header = f"abstract {header}"
        supers: list[type | UnionClass] = [b for b in cls.__bases__ if is_model_class(b)]
        supers += [u for u in schema.unions if cls in u.members]
        if supers:
            header += " extends " + ", ".join(class_name(b) for b in supers)
        features = schema.features[cls]
        if not features:
            lines.append(f"{header}.")
            continue
        lines.append(f"{header} {{")
        lines.extend(f"    {_declare_feature(f)}" for f in features)
        lines.append("}")
    if rules:
        lines.append("")
        lines.extend(rules)
    return "\n".join(lines) + "\n"


def render_validation_problem(
    schema: Schema,
    metamodel: str,
    graph: Graph,
    rules: Iterable[str],
    predicates: Sequence[Predicate],
) -> str:
    """The metamodel, the rules, and the facts of ``graph``: what certainly holds."""
    lines = [metamodel, *rules, *(_declare_predicate(schema, p) for p in predicates), ""]
    lines.append(render_facts(schema, graph))
    lines += _predicate_facts(schema, graph, predicates)
    return "\n".join(lines) + "\n"


def render_query_problem(
    schema: Schema,
    metamodel: str,
    graph: Graph,
    rules: Iterable[str],
    predicates: Sequence[Predicate],
) -> str:
    """The metamodel, the rules, and the facts of ``graph`` in a closed world that
    keeps what is unknown: no other objects, references, or subclass memberships."""
    lines = [metamodel, *rules, ""]
    lines += [_declare_predicate(schema, p) for p in predicates]
    lines.append("")
    lines += _closed_world(schema, graph, predicates)
    lines.append("")
    lines.append(render_facts(schema, graph, canonical_opposites=True))
    for node in graph.nodes:
        if node.exists == "unknown":
            lines.append(f"?exists({node.name}).")
        subclasses = schema.subclasses(type(node.obj))
        lines += [f"!{class_name(sub)}({node.name})." for sub in subclasses]
    lines += _predicate_facts(schema, graph, predicates)
    return "\n".join(lines) + "\n"


def render_facts(schema: Schema, graph: Graph, *, canonical_opposites: bool = False) -> str:
    """Assertions describing the nodes of ``graph`` and their values.

    With ``canonical_opposites``, a reference on one side of an opposite
    pair is written as the reverse reference on the other side, so that
    closed-world defaults on that side don't contradict it.
    """
    lines = []
    for node in graph.nodes:
        name = node.name
        lines.append(f"{class_name(type(node.obj))}({name}).")
        for value in node.values:
            feature = value.feature
            symbol = feature_symbol(feature)
            prefix = "" if value.certain else "?"
            target = value.target
            match feature:
                case AttributeFeature():
                    if value.certain:  # an uncertain attribute value is simply unknown
                        literal = attribute_literal(value.value, feature.data_type)
                        lines.append(f"{symbol}({name}): {literal}.")
                case EnumFeature():
                    literal = _enum_literal(feature.target, value.value)
                    lines.append(f"{prefix}{symbol}({name}, {literal}).")
                case ReferenceFeature():
                    if target is None:
                        continue  # a key matching no object: reported by reference_errors
                    opposite = feature.opposite
                    if (
                        canonical_opposites
                        and opposite is not None
                        and not _canonical(schema, feature)
                    ):
                        other = schema.find_feature(type(target.obj), opposite)
                        assert other is not None
                        lines.append(f"{prefix}{feature_symbol(other)}({target.name}, {name}).")
                    else:
                        lines.append(f"{prefix}{symbol}({name}, {target.name}).")
    return "\n".join(lines) + "\n"


def feature_symbol(feature: Feature) -> str:
    """The Refinery name of a feature, e.g. ``Service::ports``."""
    return f"{class_name(feature.owner)}::{quote_id(feature.name)}"


def _declare_feature(feature: Feature) -> str:
    name = quote_id(feature.name)
    bounds = multiplicity(feature.lower, feature.upper)
    if isinstance(feature, AttributeFeature):
        return f"{feature.data_type} {name}"
    target = class_name(feature.target)
    if isinstance(feature, EnumFeature):
        return f"{target}{bounds} {name}"
    opposite = f" opposite {quote_id(feature.opposite)}" if feature.opposite else ""
    if feature.kind == "container":
        return f"container {target} {name}{opposite}"
    prefix = "contains " if feature.kind == "contains" else ""
    return f"{prefix}{target}{bounds} {name}{opposite}"


def _enum_literal(enum_type: EnumLike, value: Any) -> str:
    """The Refinery name of an enum value, e.g. ``Color::red``."""
    member = value if isinstance(enum_type, LiteralEnum) else value.name
    return f"{class_name(enum_type)}::{quote_id(member)}"


def _canonical(schema: Schema, feature: ReferenceFeature) -> bool:
    """Of an opposite pair, the side facts and closed-world defaults are written on."""
    if feature.opposite is None:
        return True
    if feature.kind != "refers":
        return feature.kind == "contains"
    # check() rejects opposites of unions, so the target is one class.
    assert not isinstance(feature.target, UnionClass)
    other = schema.find_feature(feature.target, feature.opposite)
    assert other is not None
    return (feature.owner.__name__, feature.name) <= (other.owner.__name__, other.name)


def _closed_world(schema: Schema, graph: Graph, predicates: Sequence[Predicate]) -> list[str]:
    lines = [
        f"!exists({class_name(cls)}::new)." for cls in schema.classes if not schema.is_abstract(cls)
    ]
    orphans = [node.obj for node in _orphans(graph)]
    for cls in schema.classes:
        for feature in schema.features[cls]:
            if _needs_default(schema, feature, orphans):
                lines.append(f"default !{feature_symbol(feature)}(*, *).")
    for predicate in predicates:
        wildcards = ", ".join("*" for _ in predicate.parameters)
        lines.append(f"default !{quote_id(predicate.name)}({wildcards}).")
    return lines


def _needs_default(schema: Schema, feature: Feature, orphans: Sequence[BaseModel]) -> bool:
    """Whether ``feature`` needs a closed-world default: references and enums do, on the
    canonical side of an opposite pair; attributes have one value each and need none."""
    if isinstance(feature, AttributeFeature):
        return False
    if isinstance(feature, EnumFeature):
        return True
    if not _canonical(schema, feature):
        return False
    if feature.kind != "contains":
        return True
    # Refinery already rules out every other container of a node that certainly has
    # one. A default would only cost time: Refinery refines it for every node pair.
    target = feature.target
    members = target.members if isinstance(target, UnionClass) else (target,)
    return any(isinstance(node, members) for node in orphans)


def _orphans(graph: Graph) -> list[Node]:
    """Nodes without a container that certainly exists and certainly contains them."""
    owned: set[Node] = set()
    for node in graph.nodes:
        for value in node.values:
            feature = value.feature
            if not isinstance(feature, ReferenceFeature):
                continue
            kind = feature.kind
            if kind not in ("contains", "container"):
                continue
            assert value.target is not None
            owner, child = (node, value.target) if kind == "contains" else (value.target, node)
            if value.certain and owner.exists == "true":
                owned.add(child)
    return [node for node in graph.nodes if node not in owned]


def _declare_predicate(schema: Schema, predicate: Predicate) -> str:
    parameters = ", ".join(
        f"{_parameter_type(schema, t)} p{i}" for i, t in enumerate(predicate.parameters)
    )
    return f"pred {quote_id(predicate.name)}({parameters})."


def _parameter_type(schema: Schema, parameter: type[BaseModel] | str) -> str:
    if isinstance(parameter, str):
        if schema.union_named(parameter) is None:
            raise ValueError(f"No union named {parameter!r} in this metamodel")
        return quote_id(parameter)
    if parameter not in schema.features:
        raise ValueError(f"{parameter.__name__} is not part of this metamodel")
    return class_name(parameter)


def _matches_type(schema: Schema, node: BaseModel, parameter: type[BaseModel] | str) -> bool:
    if isinstance(parameter, str):
        union = schema.union_named(parameter)
        assert union is not None  # checked by _parameter_type
        return isinstance(node, union.members)
    return isinstance(node, parameter)


def _predicate_facts(schema: Schema, graph: Graph, predicates: Sequence[Predicate]) -> list[str]:
    lines = []
    for predicate, nodes, answer in predicate_facts(schema, graph, predicates):
        prefix = "?" if answer == "unknown" else ""
        names = ", ".join(node.name for node in nodes)
        lines.append(f"{prefix}{quote_id(predicate.name)}({names}).")
    return lines


def predicate_facts(
    schema: Schema, graph: Graph, predicates: Sequence[Predicate]
) -> Iterator[tuple[Predicate, tuple[Node, ...], Literal["true", "unknown"]]]:
    """The tuples of nodes each predicate holds (``"true"``) or may hold (``"unknown"``) for."""
    objects = [node.obj for node in graph.nodes]
    for predicate in predicates:
        for arguments, answer in _predicate_tuples(schema, predicate, objects):
            if answer in (None, False, "false"):
                continue
            if answer not in (True, "true", "unknown"):
                raise ValueError(
                    f"Predicate {predicate.name!r} returned {answer!r}; expected True, "
                    "False, None, 'true', 'false' or 'unknown'"
                )
            nodes = []
            for argument, parameter in zip(arguments, predicate.parameters, strict=True):
                node = graph.node_of(argument)
                if node is None or not _matches_type(schema, argument, parameter):
                    raise ValueError(
                        f"Predicate {predicate.name!r}: {argument!r} is not a node of type "
                        f"{getattr(parameter, '__name__', parameter)} in this graph"
                    )
                nodes.append(node)
            yield predicate, tuple(nodes), "unknown" if answer == "unknown" else "true"


def _predicate_tuples(
    schema: Schema, predicate: Predicate, nodes: Sequence[BaseModel]
) -> Iterator[tuple[tuple[BaseModel, ...], Any]]:
    arity = len(predicate.parameters)
    if predicate.evaluate is not None:
        domains = [[n for n in nodes if _matches_type(schema, n, t)] for t in predicate.parameters]
        for arguments in itertools.product(*domains):
            yield arguments, predicate.evaluate(*arguments)
        return
    for item in predicate.tuples or ():
        # ((a, b), "unknown") or (a, b); for arity 1, ((a,), "unknown"), (a,) or a.
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], tuple)
            and (arity != 2 or isinstance(item[1], str | bool))
        ):
            arguments, answer = item
        else:
            arguments, answer = (item if isinstance(item, tuple) else (item,)), True
        if len(arguments) != arity:
            raise ValueError(
                f"Predicate {predicate.name!r} takes {arity} argument(s), got {arguments!r}"
            )
        yield arguments, answer
