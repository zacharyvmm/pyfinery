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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .._instances import Graph, Value
from .._options import LogicValue, Predicate
from .._render import _declare_predicate, predicate_facts
from .._results import Match, QueryResult, Violation, ViolationKind, reference_errors
from .._schema import EnumLike, Feature, LiteralEnum, Schema, UnionClass
from . import kernels as k
from .errors import UnsupportedRuleError
from .evaluation import PredicateValues
from .kernels import NodeIndex
from .model import Attribute, DataType, Node, PartialModel, Relation
from .packed import PackedModel
from .rules import Rules
from .syntax import Atom, Call, Source, parse, refinery_name
from .syntax import Predicate as RulePredicate
from .values import NodeTuple, Tuples

if TYPE_CHECKING:
    from pydantic import BaseModel


class BitsEngine:
    """Evaluates :meth:`~pyfinery.Metamodel.query` in Python, without a Refinery server.

    Pass it where :meth:`~pyfinery.Metamodel.query` takes a client::

        result = metamodel.query(objects, BitsEngine(), rules)

    The matches are those Refinery's ``semantics`` gives, but:

    - the rules may use what :mod:`pyfinery.bits` evaluates; others raise
      :class:`~pyfinery.bits.UnsupportedRuleError`;
    - Refinery refines uncertain facts from what must hold: multiplicities,
      one container per object, and ``error`` predicates. The bits engine
      doesn't, so it raises :class:`~pyfinery.bits.UnsupportedRuleError` where
      Refinery would refine; without uncertainty, only where Refinery finds the
      graph contradictory (an object in two containers, a containment cycle, an
      instance of an abstract class);
    - ``result.errors`` holds the keys that resolve to no object, and the
      tuples of the rules' predicates that are ``error``; it doesn't check
      multiplicities or containment (:meth:`~pyfinery.Metamodel.validate`
      does);
    - ``result.source`` is what was evaluated: the rules, and the declarations
      of the :class:`~pyfinery.Predicate` objects; ``result.model`` is ``None``.
    """

    def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        source = query_source(schema, rules, predicates)
        parsed = parse(source)
        refuse_assertions(parsed)
        refuse_abstract_instances(schema, graph)
        model, objects = partial_model(schema, graph, predicates)
        rule_predicates = {p.name: p for p in parsed.predicates}
        selected, evaluated = predicates_to_evaluate(parsed, select)
        used = dependencies(rule_predicates, evaluated, model)
        refuse_containment_refinement(schema, graph, model, used)
        packed = PackedModel.build(model)
        values = Rules.of(parsed).evaluate(packed, evaluated)
        refuse_error_refinement(graph, model, rule_predicates, values, used)
        tuples = {name: result.tuples() for name, result in values.items()}
        labels = Labels([node.name for node in model.nodes], objects)
        violations = [*reference_errors(graph), *error_violations(values, tuples, labels)]
        matches = selected_matches(graph, rule_predicates, selected, tuples, labels)
        return QueryResult(matches, violations, source, None)


def query_source(schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate]) -> str:
    """The rules, and the :class:`~pyfinery.Predicate` objects declared, as in the
    problem :meth:`~pyfinery.Metamodel.query` sends."""
    declarations = [_declare_predicate(schema, p) for p in predicates]
    return "\n".join([*rules, *declarations])


def predicates_to_evaluate(
    parsed: Source, select: Sequence[str]
) -> tuple[dict[str, str], list[str]]:
    """The selected predicates (their names in the rules, e.g. ``'odd name'``, to the names
    as selected), and the predicates to evaluate: those, and every one with a body, as
    Refinery evaluates them all and reports their error tuples, whatever is selected."""
    names = {p.name for p in parsed.predicates}
    selected = {(n if n in names else refinery_name(n)): n for n in select}
    with_body = [p.name for p in parsed.predicates if p.body]
    return selected, list(dict.fromkeys([*selected, *with_body]))


# -- results ------------------------------------------------------------------------

type TuplesByValue = dict[LogicValue, Tuples]
"""The tuples of a predicate that are ``true``, ``unknown`` and ``error``."""


@dataclass(frozen=True)
class Labels:
    """The name and object (``None`` for an enum literal) of each node of the model."""

    names: list[str]
    objects: list[BaseModel | None]

    def of(self, key: Sequence[NodeIndex]) -> tuple[tuple[str, ...], tuple[BaseModel | None, ...]]:
        return tuple(self.names[i] for i in key), tuple(self.objects[i] for i in key)


def error_violations(
    values: dict[str, PredicateValues],
    tuples: dict[str, TuplesByValue],
    labels: Labels,
) -> list[Violation]:
    """The tuples of each predicate that are ``error``: of an ``error`` predicate, rule
    violations; of another, contradictions."""
    violations = []
    for name, result in values.items():
        kind: ViolationKind = "rule" if result.kind == "error" else "contradiction"
        for key in sorted_tuples(tuples[name]["error"]).tolist():
            violations.append(Violation(kind, name, *labels.of(key)))
    return violations


def selected_matches(
    graph: Graph,
    rule_predicates: dict[str, RulePredicate],
    selected: dict[str, str],
    tuples: dict[str, TuplesByValue],
    labels: Labels,
) -> list[Match]:
    """The ``true`` and ``unknown`` tuples of the selected predicates, ordered by tuple.

    As ``read_matches`` does: Refinery doesn't weaken a predicate for the existence of
    its arguments, only of nodes quantified in its body; a match whose arguments only
    may exist is ``unknown``.
    """
    maybe = np.zeros(len(labels.names), bool)
    maybe[[i for i, node in enumerate(graph.nodes) if node.exists == "unknown"]] = True
    matches = []
    for name, by_value in tuples.items():
        if name not in selected:
            continue
        parameters = tuple(p.name for p in rule_predicates[name].parameters)
        true, unknown = by_value["true"], by_value["unknown"]
        weakened = maybe[true].any(axis=1) if len(true) and true.shape[1] else None
        keys = np.concatenate([true, unknown])
        logic_values = np.array(["true"] * len(true) + ["unknown"] * len(unknown), dtype=object)
        if weakened is not None:
            logic_values[: len(true)][weakened] = "unknown"
        order = np.lexsort(keys.T[::-1]) if keys.shape[1] else np.arange(len(keys))
        for key, value in zip(keys[order].tolist(), logic_values[order].tolist(), strict=True):
            matches.append(Match(selected[name], *labels.of(key), value, parameters))
    return matches


def sorted_tuples(tuples: Tuples) -> Tuples:
    return tuples[np.lexsort(tuples.T[::-1])] if tuples.shape[1] else tuples


# -- what Refinery would refine -------------------------------------------------------
#
# Refinery refines uncertain facts from what must hold; the bits engine evaluates
# the facts as given. These find where the two would differ, to refuse rather
# than answer differently. A graph without uncertainty is refined only where it is
# contradictory (an object in two containers, a containment cycle).


class Refinement(NamedTuple):
    """Where Refinery would refine the graph's facts."""

    reason: str
    changes: set[str] | None
    """The relations it would change; ``None`` if it would change what exists."""


@dataclass
class ValueCount:
    """How many values a reference has on an object, certain and uncertain."""

    certain: int = 0
    uncertain: int = 0


type Counts = dict[tuple[str, NodeIndex], ValueCount]
"""The values of each reference (by name) on each object."""


class PossibleContainer(NamedTuple):
    parent: NodeIndex
    certain: bool


type Containers = dict[NodeIndex, list[PossibleContainer]]
"""The possible containers of each contained object."""


def refuse_assertions(parsed: Source) -> None:
    if parsed.asserted:
        name, position = next(iter(parsed.asserted.items()))
        raise UnsupportedRuleError(
            f"assertions in rules ({name}): give facts as Predicate objects", position
        )


def refuse_abstract_instances(schema: Schema, graph: Graph) -> None:
    """An object of an abstract class: Refinery finds its class memberships contradictory."""
    for node in graph.nodes:
        if schema.is_abstract(type(node.obj)):
            raise UnsupportedRuleError(
                f"{node.name}, an instance of the abstract class {type(node.obj).__name__}: "
                "Refinery reports its memberships as contradictions"
            )


def refuse_containment_refinement(
    schema: Schema, graph: Graph, model: PartialModel, used: set[str]
) -> None:
    """Refuse where Refinery would refine a reference the query uses (``used``), or what
    exists, from multiplicities or containment."""
    references = [f for fs in schema.features.values() for f in fs if not f.is_attribute]
    contains = {feature_name(f) for f in references if f.kind == "contains"}
    every_containment = contains | {
        opposite_name(schema, f) for f in references if f.kind == "contains" and f.opposite
    }
    counts, containers = tally(model, contains)
    refinements = chain(
        refined_by_multiplicity(schema, graph, references, counts),
        refined_by_containers(graph, references, containers, every_containment),
        refined_by_cycles(graph, model, contains, every_containment),
    )
    for refinement in refinements:
        if refinement.changes is None or refinement.changes & used:
            raise UnsupportedRuleError(refinement.reason)


def tally(model: PartialModel, contains: set[str]) -> tuple[Counts, Containers]:
    """The values of each reference on each object, and the possible containers of each
    object (``contains`` names the containment references)."""
    counts: Counts = {}
    containers: Containers = {}
    for relation in model.relations:
        if relation.arity != 2:
            continue
        must = {tuple(t) for t in relation.must.tolist()}
        for key in {tuple(t) for t in relation.may.tolist()} | must:
            certain = key in must
            count = counts.setdefault((relation.name, key[0]), ValueCount())
            if certain:
                count.certain += 1
            else:
                count.uncertain += 1
            if relation.name in contains:
                containers.setdefault(key[1], []).append(PossibleContainer(key[0], certain))
    return counts, containers


def refined_by_multiplicity(
    schema: Schema,
    graph: Graph,
    references: list[Feature],
    counts: Counts,
) -> Iterator[Refinement]:
    """With ``c`` values certain and ``u`` uncertain, against ``[lower..upper]``:
    the uncertain ones must hold if ``c + u <= lower``, and can't if ``c >= upper``;
    an object that only may exist can't if ``c + u < lower`` or ``c > upper``."""
    for feature in references:
        name = feature_name(feature)
        changed = {name} | ({opposite_name(schema, feature)} if feature.opposite else set())
        for i, node in enumerate(graph.nodes):
            if not isinstance(node.obj, feature.owner):
                continue
            count = counts.get((name, i), ValueCount())
            certain, uncertain = count.certain, count.uncertain
            upper = feature.upper
            if uncertain and (
                certain + uncertain <= feature.lower or (upper is not None and certain >= upper)
            ):
                yield Refinement(
                    f"uncertain values of {name} on {node.name}, which Refinery would refine "
                    f"from its multiplicity [{feature.lower}..{'*' if upper is None else upper}]",
                    changed,
                )
            if certain + uncertain < feature.lower and node.exists == "unknown":
                yield Refinement(
                    f"{node.name}, which only may exist and lacks the values {name} needs: "
                    "Refinery would conclude it doesn't exist",
                    None,
                )
            if upper is not None and certain > upper and node.exists == "unknown":
                yield Refinement(
                    f"{node.name}, which only may exist and has more values of {name} than "
                    f"[{feature.lower}..{upper}] allows: Refinery would conclude it doesn't exist",
                    None,
                )


def refined_by_containers(
    graph: Graph,
    references: list[Feature],
    containers: Containers,
    every_containment: set[str],
) -> Iterator[Refinement]:
    """An object has one container at most, and exactly one if a containment can hold
    its class."""
    held = tuple(target_classes(f.target) for f in references if f.kind == "contains")
    contained = tuple(c for classes in held for c in classes)
    for child, node in enumerate(graph.nodes):
        found = containers.get(child, [])
        holders = [container.parent for container in found if container.certain]
        certain, uncertain = len(holders), len(found) - len(holders)
        if certain > 1:
            # A contradiction: Refinery makes the containment links errors, and rules
            # out the objects involved that only may exist.
            involved = [child, *holders]
            uncertain_objects = any(only_may_exist(graph, i) for i in involved)
            yield Refinement(
                f"{node.name}, in {certain} containers at once: Refinery contradicts it",
                None if uncertain_objects else every_containment,
            )
        if uncertain and certain:
            yield Refinement(
                f"uncertain containers of {node.name}, which has a certain one: "
                "Refinery would rule them out",
                every_containment,
            )
        if not contained or not isinstance(node.obj, contained):
            continue
        if uncertain == 1 and not certain:
            yield Refinement(
                f"the one possible container of {node.name}, uncertain: Refinery would "
                "conclude it holds, as the object needs a container",
                every_containment,
            )
        if not certain and not uncertain and node.exists == "unknown":
            yield Refinement(
                f"{node.name}, which only may exist and has no possible container: "
                "Refinery would conclude it doesn't exist",
                None,
            )


def refined_by_cycles(
    graph: Graph, model: PartialModel, contains: set[str], every_containment: set[str]
) -> Iterator[Refinement]:
    """Containment has no cycles: Refinery rules out the objects of one that only may
    exist, or else contradicts it."""
    n = len(model.nodes)
    edges = [r for r in model.relations if r.name in contains]
    if not edges:
        return
    rows = np.concatenate([r.may[:, 0] for r in edges] + [r.must[:, 0] for r in edges])
    cols = np.concatenate([r.may[:, 1] for r in edges] + [r.must[:, 1] for r in edges])
    cyclic = k.indices(k.diagonal(k.closure(k.matrix_of(n, rows, cols), n), n), n)
    if len(cyclic):
        uncertain = any(only_may_exist(graph, int(i)) for i in cyclic)
        names = ", ".join(model.nodes[int(i)].name for i in cyclic)
        yield Refinement(
            f"a containment cycle through {names}: Refinery rules out or contradicts it",
            None if uncertain else every_containment,
        )


def only_may_exist(graph: Graph, i: NodeIndex) -> bool:
    """Whether node ``i`` is an object that only may exist (enum literals do exist)."""
    return i < len(graph.nodes) and graph.nodes[i].exists == "unknown"


def target_classes(target: object) -> tuple[type, ...]:
    """The model classes a reference's target stands for."""
    if isinstance(target, UnionClass):
        return target.members
    return (target,) if isinstance(target, type) else ()


def refuse_error_refinement(
    graph: Graph,
    model: PartialModel,
    rule_predicates: dict[str, RulePredicate],
    values: dict[str, PredicateValues],
    used: set[str],
) -> None:
    """Refuse where Refinery would refine facts from an error predicate in a way that
    could change the query: to make it false where it may hold, it refines the facts
    it depends on (those the query depends on too matter), or rules out objects that
    only may exist."""
    for name, result in values.items():
        if rule_predicates[name].kind != "error":
            continue
        shared = dependencies(rule_predicates, [name], model) & used
        for key, value in result.items():
            if value not in ("unknown", "error"):
                continue
            if any(only_may_exist(graph, i) for i in key) or (value == "unknown" and shared):
                names = ", ".join(model.nodes[i].name for i in key)
                raise UnsupportedRuleError(
                    f"error predicate {result.name}({names}) is {value} where facts are "
                    "uncertain: Refinery would refine them to rule it out"
                )


def dependencies(
    predicates: dict[str, RulePredicate], names: Iterable[str], model: PartialModel
) -> set[str]:
    """The relations and attributes the predicates ``names`` depend on, through the
    predicates they call, by their names in the model."""
    found: set[str] = set()
    seen: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        predicate = predicates.get(name)
        if predicate is None or predicate.body is None:
            found.add(model_name(model, name))
            continue
        for parameter in predicate.parameters:
            if parameter.type is not None:
                pending.append(parameter.type)
        for clause in predicate.body:
            for literal in clause:
                if isinstance(literal, Atom):
                    pending.append(literal.relation)
                    continue
                for side in (literal.left, literal.right):
                    if isinstance(side, Call):
                        found.add(model_name(model, side.relation))
    return found


def model_name(model: PartialModel, name: str) -> str:
    try:
        found = model.relation_named(name) or model.attribute_named(name)
    except ValueError:
        return name
    return found.name if found is not None else name


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
