# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Writing the problems sent to Refinery: the metamodel of a :class:`Schema`, and
the facts of a :class:`Graph` for validation and queries.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from pydantic import BaseModel

from ._instances import Graph, Node
from ._options import Predicate
from ._schema import (
    ClassLike,
    EnumLike,
    Feature,
    LiteralEnum,
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
            symbol = f"{class_name(feature.owner)}::{quote_id(feature.name)}"
            prefix = "" if value.certain else "?"
            if feature.by_key:
                if value.target is not None:  # others are reported by reference_errors
                    lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
            elif canonical_opposites and not _canonical(schema, feature):
                assert value.target is not None
                other = schema.find_feature(type(value.target.obj), feature.opposite)  # type: ignore[arg-type]
                assert other is not None
                reverse = f"{class_name(other.owner)}::{quote_id(other.name)}"
                lines.append(f"{prefix}{reverse}({value.target.name}, {name}).")
            elif feature.is_attribute:
                if value.certain:  # an uncertain attribute value is simply unknown
                    literal = attribute_literal(value.value, feature.target)  # type: ignore[arg-type]
                    lines.append(f"{symbol}({name}): {literal}.")
            elif feature.is_enum:
                literal = _enum_literal(feature.target, value.value)  # type: ignore[arg-type]
                lines.append(f"{prefix}{symbol}({name}, {literal}).")
            else:
                assert value.target is not None
                lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
    return "\n".join(lines) + "\n"


# -- the metamodel ------------------------------------------------------------


def _declare_feature(feature: Feature) -> str:
    name = quote_id(feature.name)
    if feature.is_attribute:
        return f"{feature.target} {name}"
    target = class_name(feature.target)  # type: ignore[arg-type]
    opposite = f" opposite {quote_id(feature.opposite)}" if feature.opposite else ""
    if feature.kind == "container":
        return f"container {target} {name}{opposite}"
    prefix = "contains " if feature.kind == "contains" else ""
    return f"{prefix}{target}{multiplicity(feature.lower, feature.upper)} {name}{opposite}"


def _enum_literal(enum_type: EnumLike, value: Any) -> str:
    """The Refinery name of an enum value, e.g. ``Color::red``."""
    member = value if isinstance(enum_type, LiteralEnum) else value.name
    return f"{class_name(enum_type)}::{quote_id(member)}"


# -- the closed world ---------------------------------------------------------


def _canonical(schema: Schema, feature: Feature) -> bool:
    """Of an opposite pair, the side facts and closed-world defaults are written on."""
    if feature.opposite is None:
        return True
    if feature.kind != "refers":
        return feature.kind == "contains"
    other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
    assert other is not None
    return (feature.owner.__name__, feature.name) <= (other.owner.__name__, other.name)


def _closed_world(schema: Schema, graph: Graph, predicates: Sequence[Predicate]) -> list[str]:
    lines = [
        f"!exists({class_name(cls)}::new)." for cls in schema.classes if not schema.is_abstract(cls)
    ]
    orphans = [node.obj for node in _orphans(graph)]
    for cls in schema.classes:
        for feature in schema.features[cls]:
            if feature.is_attribute or not _canonical(schema, feature):
                continue
            if feature.kind == "contains":
                target: ClassLike = feature.target  # type: ignore[assignment]
                members = target.members if isinstance(target, UnionClass) else (target,)
                if not any(isinstance(node, members) for node in orphans):
                    # Refinery already rules out every other container of
                    # a node that certainly has one. A default would only
                    # cost time: Refinery refines it for every node pair.
                    continue
            symbol = f"{class_name(cls)}::{quote_id(feature.name)}"
            lines.append(f"default !{symbol}(*, *).")
    for predicate in predicates:
        wildcards = ", ".join("*" for _ in predicate.parameters)
        lines.append(f"default !{quote_id(predicate.name)}({wildcards}).")
    return lines


def _orphans(graph: Graph) -> list[Node]:
    """Nodes without a container that certainly exists and certainly contains them."""
    owned: set[Node] = set()
    for node in graph.nodes:
        for value in node.values:
            kind = value.feature.kind
            if kind not in ("contains", "container"):
                continue
            assert value.target is not None
            owner, child = (node, value.target) if kind == "contains" else (value.target, node)
            if value.certain and owner.exists == "true":
                owned.add(child)
    return [node for node in graph.nodes if node not in owned]


# -- Python predicates --------------------------------------------------------


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
            prefix = "?" if answer == "unknown" else ""
            rendered = []
            for argument, parameter in zip(arguments, predicate.parameters, strict=True):
                node = graph.node_of(argument)
                if node is None or not _matches_type(schema, argument, parameter):
                    raise ValueError(
                        f"Predicate {predicate.name!r}: {argument!r} is not a node of type "
                        f"{getattr(parameter, '__name__', parameter)} in this graph"
                    )
                rendered.append(node.name)
            lines.append(f"{prefix}{quote_id(predicate.name)}({', '.join(rendered)}).")
    return lines


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
