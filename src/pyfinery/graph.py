# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Validate and query Pydantic object graphs with Refinery.

A :class:`Metamodel` derives a Refinery metamodel from Pydantic model classes,
and turns instances of them into Refinery facts:

- :meth:`Metamodel.validate` checks a graph against the metamodel and
  ``error pred`` rules, closing the world (anything not asserted is false)
  with Refinery's ``concretize`` operation, and reports each constraint that
  evaluates to ``error`` as a :class:`Violation`.
- :meth:`Metamodel.query` evaluates graph patterns written as Refinery
  predicates over a closed world that may hold *uncertain* facts, and
  reports each :class:`Match` with its value in Refinery's four-valued
  logic: ``"true"``, or ``"unknown"`` if it only may hold.

How fields map to Refinery:

========================================  =========================================
Python                                    Refinery
========================================  =========================================
``int`` / ``float`` / ``str`` / ``bool``  ``int`` / ``real`` / ``string`` / ``boolean``
``Literal["a", "b"]``                     an enum (single-value tags are skipped)
an :class:`enum.Enum` subclass            an enum
``Model`` / ``Model | None`` / ``list``   ``Model[1]`` / ``Model[0..1]`` / ``Model[0..*]``
``A | B``                                 an abstract superclass of ``A`` and ``B``
subclassing                               ``extends``
========================================  =========================================

Whether a field typed as a model class is containment or a plain reference
depends on the identity of that class:

- plain Pydantic models are *owned values*: each instance is a node
  contained in the object holding it;
- classes with ``ClassOptions(key=...)`` are *entities*: instances with equal
  keys are one node, and fields typed as them are references. Fields that
  hold keys rather than objects are mapped with
  ``ClassOptions(references=...)``;
- :class:`RefineryNode` subclasses are entities by object identity, for
  hand-built graphs, with :func:`Contains`, :func:`Container` and
  :class:`Ref` markers for containment, opposites, and multiplicities.

:class:`ClassOptions` configure classes from outside, so generated or
third-party models can be used unchanged; :class:`Predicate` provides
relations computed in Python (e.g. regular expressions) to rules.

This module holds :class:`Metamodel` and re-exports the rest, which lives in
private modules: the options in ``_options``, the types and fields in
``_schema``, the graph of nodes built from objects in ``_instances``,
Refinery syntax in ``_syntax``, and reports in ``_results``.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ._instances import Graph, Node, Value
from ._options import (
    ClassOptions,
    Container,
    Contains,
    LogicValue,
    Predicate,
    Ref,
    ReferenceKind,
    RefineryNode,
)
from ._results import (
    Match,
    ModelValidationError,
    QueryResult,
    ValidationReport,
    Violation,
    ViolationKind,
    declared_predicates,
    read_matches,
    read_violations,
)
from ._schema import (
    ClassLike,
    EnumLike,
    Feature,
    LiteralEnum,
    UnionClass,
    check,
    class_name,
    discover,
    is_model_class,
)
from ._syntax import (
    RESERVED_NODE_NAMES,
    attribute_literal,
    encodable,
    multiplicity,
    quote_id,
)
from .errors import RefineryTimeoutError
from .models import JsonOutputFormat, OutputFormats, SourceOutputFormat

if TYPE_CHECKING:
    from .client import AsyncRefinery, Refinery

__all__ = [
    "ClassLike",
    "ClassOptions",
    "Container",
    "Contains",
    "EnumLike",
    "Feature",
    "LiteralEnum",
    "LogicValue",
    "Match",
    "Metamodel",
    "ModelValidationError",
    "Predicate",
    "QueryResult",
    "Ref",
    "ReferenceKind",
    "RefineryNode",
    "UnionClass",
    "ValidationReport",
    "Violation",
    "ViolationKind",
    "quote_id",
]


class Metamodel:
    """A Refinery metamodel derived from :class:`RefineryNode` subclasses
    (and the plain Pydantic models nested in them).

    >>> metamodel = Metamodel(Filesystem, rules=[
    ...     "error pred emptyDir(Dir d) <-> !contents(d, _).",
    ... ])
    >>> print(metamodel.source)
    >>> report = metamodel.validate([my_filesystem], refinery)

    Classes reachable from the given ones are included automatically: their
    superclasses, the classes their fields refer to, and the already-imported
    subclasses of those (but not every subclass of a shared base class).

    :param options: :class:`ClassOptions` by class.
    :param unions: Names for the abstract superclasses standing for unions,
        e.g. ``{"Component": [Service, Database]}``.
    :param unsupported: ``"skip"`` to leave out fields with no Refinery
        counterpart instead of raising ``TypeError``.
    """

    def __init__(
        self,
        *classes: type[BaseModel],
        rules: Iterable[str] = (),
        options: Mapping[type, ClassOptions] | None = None,
        unions: Mapping[str, Iterable[type[BaseModel]]] | None = None,
        unsupported: Literal["error", "skip"] = "error",
    ) -> None:
        if not classes:
            raise ValueError("Metamodel needs at least one model class")
        self.rules = list(rules)
        self.schema = discover(classes, options, unions, unsupported)
        check(self.schema)
        self.source = self._render()

    @property
    def options(self) -> dict[type, ClassOptions]:
        return self.schema.options

    @property
    def classes(self) -> list[type[BaseModel]]:
        return self.schema.classes

    @property
    def unions(self) -> list[UnionClass]:
        return self.schema.unions

    @property
    def enums(self) -> list[EnumLike]:
        return self.schema.enums

    @property
    def features(self) -> dict[type[BaseModel], list[Feature]]:
        return self.schema.features

    @property
    def skipped(self) -> list[tuple[type[BaseModel], str, str]]:
        """Fields left out of the metamodel: ``(class, field, reason)``."""
        return self.schema.skipped

    # -- rendering ----------------------------------------------------------

    @staticmethod
    def class_name(node_class: type | UnionClass | LiteralEnum) -> str:
        """The Refinery name of a class, union, or enum."""
        return class_name(node_class)

    def _literal(self, enum_type: EnumLike, value: Any) -> str:
        """The Refinery name of an enum value, e.g. ``Color::red``."""
        member = value if isinstance(enum_type, LiteralEnum) else value.name
        return f"{class_name(enum_type)}::{quote_id(member)}"

    def _render_feature(self, feature: Feature) -> str:
        name = quote_id(feature.name)
        if feature.is_attribute:
            return f"{feature.target} {name}"
        target = class_name(feature.target)  # type: ignore[arg-type]
        opposite = f" opposite {quote_id(feature.opposite)}" if feature.opposite else ""
        if feature.kind == "container":
            return f"container {target} {name}{opposite}"
        prefix = "contains " if feature.kind == "contains" else ""
        return f"{prefix}{target}{multiplicity(feature.lower, feature.upper)} {name}{opposite}"

    def _render(self) -> str:
        lines: list[str] = []
        for enum_type in self.schema.enums:
            if isinstance(enum_type, LiteralEnum):
                members = enum_type.values
            else:
                members = tuple(member.name for member in enum_type)
            literals = ", ".join(quote_id(member) for member in members)
            lines.append(f"enum {class_name(enum_type)} {{ {literals} }}")
        for union in self.schema.unions:
            lines.append(f"abstract class {class_name(union)}.")
        for cls in self.schema.classes:
            header = f"class {class_name(cls)}"
            if self.schema.is_abstract(cls):
                header = f"abstract {header}"
            supers: list[type | UnionClass] = [b for b in cls.__bases__ if is_model_class(b)]
            supers += [u for u in self.schema.unions if cls in u.members]
            if supers:
                header += " extends " + ", ".join(class_name(b) for b in supers)
            features = self.schema.features[cls]
            if not features:
                lines.append(f"{header}.")
                continue
            lines.append(f"{header} {{")
            lines.extend(f"    {self._render_feature(f)}" for f in features)
            lines.append("}")
        if self.rules:
            lines.append("")
            lines.extend(self.rules)
        return "\n".join(lines) + "\n"

    # -- instances ----------------------------------------------------------

    def collect(self, objects: Iterable[BaseModel]) -> list[BaseModel]:
        """All nodes reachable from ``objects`` through reference fields.

        Only outgoing references are followed: a node that is referenced by
        nothing (e.g., a child that only sets its own ``parent``) must be
        passed in ``objects`` itself.
        """
        return [node.obj for node, _ in self._reach(objects, {})]

    def identity(self, node: BaseModel) -> Hashable:
        """What makes ``node`` one Refinery node: its key if its class has one, else itself.

        Of several instances with the same key, the first one reached
        represents them all (its fields are the ones asserted).
        """
        return self.schema.identity(node)

    def link_opposites(self, objects: Iterable[BaseModel]) -> int:
        """Fill in the missing side of every ``opposite`` pair in the graph.

        For example, after ``home.contents.append(doc)`` this sets
        ``doc.parent = home``, and after ``doc.parent = home`` it appends
        ``doc`` to ``home.contents``. Refinery infers missing sides on its own,
        so this is only needed to keep the Python objects consistent.

        Single-valued fields that already point at a different node are left
        unchanged; :meth:`validate` reports such conflicts. Returns the number
        of references added.
        """
        added = 0
        for node in self.collect(objects):
            for _, feature, values in self._values(node):
                if feature.opposite is None:
                    continue
                for target in values:
                    other = self.schema.find_feature(type(target), feature.opposite)
                    if other is not None and _add_reference(
                        target, other, node, self.schema.identity
                    ):
                        added += 1
        return added

    def _values(self, node: BaseModel) -> Iterator[tuple[type, Feature, list[Any]]]:
        for cls in type(node).__mro__:
            for feature in self.schema.features.get(cls, []):  # type: ignore[call-overload]
                value = getattr(node, feature.name)
                if value is None:
                    continue
                values = list(value) if feature.many else [value]
                yield cls, feature, values

    # -- the instance graph -------------------------------------------------

    def _graph(self, objects: Iterable[BaseModel], *, existence: bool = False) -> Graph:
        """The Refinery nodes of the graph reachable from ``objects``, with their values.

        The ``key``, ``node_name``, ``logic_value`` and (with ``existence``)
        ``exists`` options are evaluated here, once; everything sent to
        Refinery is rendered from the result. Without ``existence``, every
        node exists, as :meth:`validate` assumes.
        """
        identities: dict[int, tuple[BaseModel, Hashable]] = {}
        reached = self._reach(objects, identities)
        nodes = [node for node, _ in reached]
        graph = Graph(
            nodes, {node.identity: node for node in nodes}, identities, self.schema.identity
        )
        self._name(graph.nodes)
        index: dict[Hashable, list[Node]] = {}
        for node in graph.nodes:
            if self.schema.options_for(type(node.obj)).key is not None:
                index.setdefault(node.identity[1], []).append(node)  # type: ignore[index]

        def node_of(obj: BaseModel) -> Node:
            # Every followed reference was reached, by the same identity.
            return graph.by_identity[self._identity(obj, graph.identities)]

        for node, fields in reached:
            for _, feature, values in fields:
                for value in values:
                    node.values.append(self._value(node, feature, value, index, node_of))
        if existence:
            for node in graph.nodes:
                node.exists = self._exists(node.obj)
        return graph

    def _reach(
        self, objects: Iterable[BaseModel], identities: dict[int, tuple[BaseModel, Hashable]]
    ) -> list[tuple[Node, list[tuple[type, Feature, list[Any]]]]]:
        """The nodes reachable from ``objects``, each with the values of its fields.

        ``identities`` memoizes :meth:`identity` by object, keeping the
        objects alive so that their ``id`` isn't reused.
        """
        reached = []
        seen: set[Hashable] = set()
        pending = deque(objects)
        while pending:
            obj = pending.popleft()
            if type(obj) not in self.schema.features:
                raise TypeError(f"{type(obj).__name__} is not part of this metamodel")
            identity = self._identity(obj, identities)
            if identity in seen:
                continue
            seen.add(identity)
            fields = list(self._values(obj))
            reached.append((Node(obj, identity), fields))
            for _, feature, values in fields:
                if not (feature.is_attribute or feature.is_enum or feature.by_key):
                    pending.extend(values)
        return reached

    def _identity(
        self, obj: BaseModel, identities: dict[int, tuple[BaseModel, Hashable]]
    ) -> Hashable:
        entry = identities.get(id(obj))
        if entry is None:
            entry = identities[id(obj)] = (obj, self.identity(obj))
        return entry[1]

    def _name(self, nodes: Sequence[Node]) -> None:
        """Give every node the name it has in the problem."""
        used: set[str] = set()
        counters: dict[type, int] = {}
        for node in nodes:
            obj = node.obj
            id_field: str | None = getattr(type(obj), "__node_id__", None)
            naming = self.schema.options_for(type(obj)).node_name
            if naming is not None or id_field is not None:
                raw = naming(obj) if naming is not None else getattr(obj, id_field)  # type: ignore[arg-type]
                name = quote_id(str(raw))
                if name in used:
                    source = "node_name" if naming is not None else f"field {id_field!r}"
                    raise ValueError(f"Duplicate node id {name} (from {source})")
            elif (key := self._name_from_key(node)) is not None and key not in used:
                name = key
            else:
                while True:
                    counters[type(obj)] = counters.get(type(obj), 0) + 1
                    name = quote_id(f"{type(obj).__name__.lower()}_{counters[type(obj)]}")
                    if name not in used:
                        break
            used.add(name)
            node.name = name

    def _name_from_key(self, node: Node) -> str | None:
        """The key as a node name, if it is a string or integer that isn't a type's name."""
        if self.schema.options_for(type(node.obj)).key is None:
            return None
        key = node.identity[1]  # type: ignore[index]
        if isinstance(key, bool) or not isinstance(key, str | int) or str(key) == "":
            return None
        if not encodable(str(key)):
            # Escaped, Refinery would name the node with "?" for each lone
            # surrogate, so the name couldn't be mapped back to the node.
            return None
        name = quote_id(str(key))
        return None if name in self.schema.type_names or str(key) in RESERVED_NODE_NAMES else name

    def _value(
        self,
        node: Node,
        feature: Feature,
        value: Any,
        index: Mapping[Hashable, list[Node]],
        node_of: Callable[[BaseModel], Node],
    ) -> Value:
        if feature.by_key:
            target = feature.target
            members = target.members if isinstance(target, UnionClass) else (target,)
            found = [n for n in index.get(value, []) if isinstance(n.obj, members)]  # type: ignore[arg-type]
            if len(found) != 1:
                problem = "unresolved" if not found else "ambiguous"
                return Value(feature, value, unresolved=problem)
            return Value(feature, value, self._certain(node.obj, feature, value), found[0])
        certain = self._certain(node.obj, feature, value)
        if feature.is_attribute or feature.is_enum:
            return Value(feature, value, certain)
        return Value(feature, value, certain, node_of(value))

    def _certain(self, node: BaseModel, feature: Feature, value: Any) -> bool:
        """Whether ``value`` certainly holds, as ``ClassOptions.logic_value`` says."""
        decide = self.schema.options_for(type(node)).logic_value
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

    # -- rendering instances ------------------------------------------------

    def _facts(self, graph: Graph, *, canonical_opposites: bool = False) -> str:
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
                    if value.target is not None:  # others are reported by _reference_errors
                        lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
                elif canonical_opposites and not self._canonical(feature):
                    assert value.target is not None
                    other = self.schema.find_feature(type(value.target.obj), feature.opposite)  # type: ignore[arg-type]
                    assert other is not None
                    reverse = f"{class_name(other.owner)}::{quote_id(other.name)}"
                    lines.append(f"{prefix}{reverse}({value.target.name}, {name}).")
                elif feature.is_attribute:
                    if value.certain:  # an uncertain attribute value is simply unknown
                        literal = attribute_literal(value.value, feature.target)  # type: ignore[arg-type]
                        lines.append(f"{symbol}({name}): {literal}.")
                elif feature.is_enum:
                    literal = self._literal(feature.target, value.value)  # type: ignore[arg-type]
                    lines.append(f"{prefix}{symbol}({name}, {literal}).")
                else:
                    assert value.target is not None
                    lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
        return "\n".join(lines) + "\n"

    def problem(self, objects: Iterable[BaseModel]) -> str:
        """The metamodel, rules, and the facts of the graph reachable from ``objects``."""
        return self.source + "\n" + self._facts(self._graph(objects))

    # -- validation ---------------------------------------------------------

    def validate(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Check the graph reachable from ``objects`` against the metamodel and rules.

        ``rules`` (Refinery source, typically ``error pred`` declarations) are added to
        those given to the :class:`Metamodel`, and ``predicates`` provide
        relations computed in Python for them to use.

        Validation is two-valued: it checks what certainly holds. Facts that
        are only ``"unknown"`` (from :class:`Predicate` or
        ``ClassOptions(logic_value=...)``) count as not holding, and every
        object exists (``ClassOptions(exists=...)`` is ignored). Use
        :meth:`query` to reason about what may hold.

        Only nodes reachable from ``objects`` through outgoing references are
        checked (see :meth:`collect`). Raises
        :class:`~pyfinery.InvalidProblemError` if the rules don't compile;
        constraint violations are returned in the report.
        """
        predicates = list(predicates)
        graph = self._graph(objects)
        source = self._validation_problem(graph, rules, predicates)
        with _explain_timeout():
            result = refinery.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    async def avalidate(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Asynchronous variant of :meth:`validate`."""
        predicates = list(predicates)
        graph = self._graph(objects)
        source = self._validation_problem(graph, rules, predicates)
        with _explain_timeout():
            result = await refinery.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    def _validation_problem(
        self, graph: Graph, rules: Iterable[str], predicates: Sequence[Predicate]
    ) -> str:
        lines = [self.source, *rules, *(self._declare_predicate(p) for p in predicates), ""]
        lines.append(self._facts(graph))
        lines += self._predicate_facts(predicates, graph)
        return "\n".join(lines) + "\n"

    # -- queries ------------------------------------------------------------

    def query(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Evaluate graph patterns over the graph reachable from ``objects``.

        ``rules`` are Refinery predicates, e.g.
        ``"pred reaches(Service s, Database d) <-> Service::depends_on+(s, d)."``;
        every ``pred`` they declare is reported unless ``select`` names the
        ones to report. Unlike :meth:`validate`, the world is closed by
        asserting that there is nothing else (no other objects, references,
        or subclass memberships) while keeping uncertainty: objects whose
        ``exists`` is ``"unknown"`` and :class:`Predicate` facts that are
        ``"unknown"`` make the matches that depend on them ``"unknown"``.
        """
        rules, predicates = list(rules), list(predicates)
        graph = self._graph(objects, existence=True)
        source = self._query_problem(graph, rules, predicates)
        selected = list(select) if select is not None else declared_predicates(rules)
        with _explain_timeout():
            result = refinery.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, selected)

    async def aquery(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Asynchronous variant of :meth:`query`."""
        rules, predicates = list(rules), list(predicates)
        graph = self._graph(objects, existence=True)
        source = self._query_problem(graph, rules, predicates)
        selected = list(select) if select is not None else declared_predicates(rules)
        with _explain_timeout():
            result = await refinery.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, selected)

    def query_problem(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> str:
        """The problem :meth:`query` would send, e.g. to debug it in the web editor."""
        rules, predicates = list(rules), list(predicates)
        return self._query_problem(self._graph(objects, existence=True), rules, predicates)

    def _query_problem(
        self, graph: Graph, rules: Iterable[str], predicates: Sequence[Predicate]
    ) -> str:
        lines = [self.source, *rules, ""]
        lines += [self._declare_predicate(p) for p in predicates]
        lines.append("")
        lines += self._closed_world(predicates, graph)
        lines.append("")
        lines.append(self._facts(graph, canonical_opposites=True))
        for node in graph.nodes:
            if node.exists == "unknown":
                lines.append(f"?exists({node.name}).")
            subclasses = self.schema.subclasses(type(node.obj))
            lines += [f"!{class_name(sub)}({node.name})." for sub in subclasses]
        lines += self._predicate_facts(predicates, graph)
        return "\n".join(lines) + "\n"

    def _exists(self, node: BaseModel) -> LogicValue:
        decide = self.schema.options_for(type(node)).exists
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

    def _parameter_type(self, parameter: type[BaseModel] | str) -> str:
        if isinstance(parameter, str):
            if self.schema.union_named(parameter) is None:
                raise ValueError(f"No union named {parameter!r} in this metamodel")
            return quote_id(parameter)
        if parameter not in self.schema.features:
            raise ValueError(f"{parameter.__name__} is not part of this metamodel")
        return class_name(parameter)

    def _declare_predicate(self, predicate: Predicate) -> str:
        parameters = ", ".join(
            f"{self._parameter_type(t)} p{i}" for i, t in enumerate(predicate.parameters)
        )
        return f"pred {quote_id(predicate.name)}({parameters})."

    def _matches_type(self, node: BaseModel, parameter: type[BaseModel] | str) -> bool:
        if isinstance(parameter, str):
            union = self.schema.union_named(parameter)
            assert union is not None  # checked by _parameter_type
            return isinstance(node, union.members)
        return isinstance(node, parameter)

    def _predicate_facts(self, predicates: Sequence[Predicate], graph: Graph) -> list[str]:
        lines = []
        objects = [node.obj for node in graph.nodes]
        for predicate in predicates:
            for arguments, answer in self._predicate_tuples(predicate, objects):
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
                    node = graph.by_identity.get(self._identity(argument, graph.identities))
                    if node is None or not self._matches_type(argument, parameter):
                        raise ValueError(
                            f"Predicate {predicate.name!r}: {argument!r} is not a node of type "
                            f"{getattr(parameter, '__name__', parameter)} in this graph"
                        )
                    rendered.append(node.name)
                lines.append(f"{prefix}{quote_id(predicate.name)}({', '.join(rendered)}).")
        return lines

    def _predicate_tuples(
        self, predicate: Predicate, nodes: Sequence[BaseModel]
    ) -> Iterator[tuple[tuple[BaseModel, ...], Any]]:
        arity = len(predicate.parameters)
        if predicate.evaluate is not None:
            domains = [[n for n in nodes if self._matches_type(n, t)] for t in predicate.parameters]
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

    def _canonical(self, feature: Feature) -> bool:
        """Of an opposite pair, the side facts and closed-world defaults are written on."""
        if feature.opposite is None:
            return True
        if feature.kind != "refers":
            return feature.kind == "contains"
        other = self.schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
        assert other is not None
        return (feature.owner.__name__, feature.name) <= (other.owner.__name__, other.name)

    def _closed_world(self, predicates: Sequence[Predicate], graph: Graph) -> list[str]:
        lines = [
            f"!exists({class_name(cls)}::new)."
            for cls in self.schema.classes
            if not self.schema.is_abstract(cls)
        ]
        orphans = [node.obj for node in self._orphans(graph)]
        for cls in self.schema.classes:
            for feature in self.schema.features[cls]:
                if feature.is_attribute or not self._canonical(feature):
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

    @staticmethod
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


_JSON_ONLY = OutputFormats(json_=JsonOutputFormat(), source=SourceOutputFormat(enabled=False))


@contextmanager
def _explain_timeout() -> Iterator[None]:
    """Add what to do about it to a timeout of validation or a query."""
    try:
        yield
    except RefineryTimeoutError as e:
        raise RefineryTimeoutError(
            f"{e.message}. Validation runs under the server's semantics timeout (1 second by "
            "default); raise it with RefineryServer(semantics_timeout=<ms>) or the "
            "REFINERY_SEMANTICS_TIMEOUT_MS environment variable of the server. Refinery "
            "computes every relation in full, so helper predicates with many tuples (e.g., "
            "with three parameters) are often what makes it slow, and the upstream Refinery "
            "image is much slower than pyfinery's default on large graphs; see Performance in "
            "pyfinery's README."
        ) from e


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
