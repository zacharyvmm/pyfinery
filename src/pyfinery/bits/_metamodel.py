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

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._instances import Graph, Value
from .._options import LogicValue, Predicate
from .._render import _declare_predicate, predicate_facts
from .._results import Match, QueryResult, Violation, reference_errors
from .._schema import EnumLike, Feature, LiteralEnum, Schema, class_name
from .._syntax import quote_id
from . import _kernels as k
from ._errors import UnsupportedRuleError
from ._evaluate import PredicateValues, evaluate
from ._model import Attribute, DataType, Node, PartialModel, Relation
from ._syntax import Atom, Call, parse
from ._syntax import Predicate as Predicate_

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
      Refinery would refine (never for a graph without uncertainty);
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
        # The Predicate objects are declared, as in the problem query() sends.
        declarations = [_declare_predicate(schema, p) for p in predicates]
        source = "\n".join([*rules, *declarations])
        parsed = parse(source)
        if parsed.asserted:
            name, position = next(iter(parsed.asserted.items()))
            raise UnsupportedRuleError(
                f"assertions in rules ({name}): give facts as Predicate objects", position
            )
        abstract = _abstract_instance(schema, graph)
        if abstract is not None:
            raise UnsupportedRuleError(abstract)
        model, objects = partial_model(schema, graph, predicates)
        kinds = {p.name: p for p in parsed.predicates}
        # Every predicate with a body is evaluated, as Refinery does: its error
        # tuples are reported, whatever is selected.
        names = list(dict.fromkeys([*select, *(p.name for p in parsed.predicates if p.body)]))
        used = _dependencies(kinds, names, model)
        for reason, relations in _refined_by_containment(schema, graph, model):
            if relations is None or relations & used:
                raise UnsupportedRuleError(reason)
        values = evaluate(source, model, select=names)
        errors = [name for name in names if kinds[name].kind == "error"]
        for name in errors:
            reason = _refined_by_error(
                graph, model, values[name], _dependencies(kinds, [name], model) & used
            )
            if reason is not None:
                raise UnsupportedRuleError(reason)
        exists = [node.exists for node in graph.nodes]
        matches, violations = [], reference_errors(graph)
        for name, result in values.items():
            parameters = tuple(p.name for p in kinds[name].parameters)
            for key, value in result.items():
                node_names = tuple(model.nodes[i].name for i in key)
                nodes = tuple(objects[i] for i in key)
                if value == "error":
                    kind = "rule" if result.kind == "error" else "contradiction"
                    violations.append(Violation(kind, name, node_names, nodes))
                    continue
                if name not in select:
                    continue
                # As read_matches does: Refinery doesn't weaken a predicate for the
                # existence of its arguments, only of nodes quantified in its body.
                if any(i < len(exists) and exists[i] == "unknown" for i in key):
                    value = "unknown"
                matches.append(Match(name, node_names, nodes, value, parameters))
        return QueryResult(matches, violations, source, None)


# -- what Refinery would propagate ------------------------------------------------
#
# Refinery refines uncertain facts from what must hold; the bits engine evaluates
# the facts as given. These find where the two would differ, to refuse rather
# than answer differently. A graph without uncertainty is never refined.


def _abstract_instance(schema: Schema, graph: Graph) -> str | None:
    """An object of an abstract class: Refinery finds its class memberships contradictory."""
    for node in graph.nodes:
        if schema.is_abstract(type(node.obj)):
            return (
                f"{node.name}, an instance of the abstract class {type(node.obj).__name__}: "
                "Refinery reports its memberships as contradictions"
            )
    return None


def _refined_by_containment(
    schema: Schema, graph: Graph, model: PartialModel
) -> Iterator[tuple[str, set[str] | None]]:
    """Where Refinery would refine a reference from multiplicities or containment: why,
    and the relations it would change (``None`` if it would change what exists).

    With ``c`` values certain and ``u`` uncertain, against ``[lower..upper]``:
    the uncertain ones must hold if ``c + u <= lower``, and can't if
    ``c >= upper``; an object that only may exist can't if ``c + u < lower``.
    An object has one container at most, and containment has no cycles.
    """
    exists = {i: node.exists for i, node in enumerate(graph.nodes)}
    counts: dict[tuple[str, int], list[int]] = {}
    containers: dict[int, list[int]] = {}
    features = [f for fs in schema.features.values() for f in fs if not f.is_attribute]
    contains = {_feature_name(f) for f in features if f.kind == "contains"}
    for relation in model.relations:
        if relation.arity != 2:
            continue
        must = {tuple(t) for t in relation.must.tolist()}
        for key in {tuple(t) for t in relation.may.tolist()} | must:
            certain = key in must
            counts.setdefault((relation.name, key[0]), [0, 0])[0 if certain else 1] += 1
            if relation.name in contains:
                containers.setdefault(key[1], [0, 0])[0 if certain else 1] += 1
    for feature in features:
        name = _feature_name(feature)
        changed = {name} | ({_opposite_name(schema, feature)} if feature.opposite else set())
        for i, node in enumerate(graph.nodes):
            if not isinstance(node.obj, feature.owner):
                continue
            certain, uncertain = counts.get((name, i), [0, 0])
            upper = feature.upper
            if uncertain and (
                certain + uncertain <= feature.lower or (upper is not None and certain >= upper)
            ):
                yield (
                    f"uncertain values of {name} on {node.name}, which Refinery would refine "
                    f"from its multiplicity [{feature.lower}..{'*' if upper is None else upper}]",
                    changed,
                )
            if certain + uncertain < feature.lower and exists[i] == "unknown":
                yield (
                    f"{node.name}, which only may exist and lacks the values {name} needs: "
                    "Refinery would conclude it doesn't exist",
                    None,
                )
    every_containment = contains | {
        _opposite_name(schema, f) for f in features if f.kind == "contains" and f.opposite
    }
    for child, (certain, uncertain) in containers.items():
        if uncertain and certain:
            yield (
                f"uncertain containers of {model.nodes[child].name}, which has a certain one: "
                "Refinery would rule them out",
                every_containment,
            )
    n = len(model.nodes)
    edges = [r for r in model.relations if r.name in contains]
    if edges:
        rows = np.concatenate([r.may[:, 0] for r in edges] + [r.must[:, 0] for r in edges])
        cols = np.concatenate([r.may[:, 1] for r in edges] + [r.must[:, 1] for r in edges])
        cyclic = k.indices(k.diagonal(k.closure(k.matrix_of(n, rows, cols), n), n), n)
        if len(cyclic):
            uncertain = any(exists.get(int(i)) == "unknown" for i in cyclic)
            names = ", ".join(model.nodes[int(i)].name for i in cyclic)
            yield (
                f"a containment cycle through {names}: Refinery rules out or contradicts it",
                None if uncertain else every_containment,
            )


def _refined_by_error(
    graph: Graph, model: PartialModel, result: PredicateValues, shared: set[str]
) -> str | None:
    """Why Refinery would refine facts from an error predicate, if it would and that
    could change the query: to make it false where it may hold, it refines the facts
    it depends on (``shared`` are those the query depends on too), or rules out
    objects that only may exist."""
    for key, value in result.items():
        if value not in ("unknown", "error"):
            continue
        uncertain_objects = any(
            graph.nodes[i].exists == "unknown" for i in key if i < len(graph.nodes)
        )
        if uncertain_objects or (value == "unknown" and shared):
            names = ", ".join(model.nodes[i].name for i in key)
            return (
                f"error predicate {result.name}({names}) is {value} where facts are "
                "uncertain: Refinery would refine them to rule it out"
            )
    return None


def _dependencies(
    predicates: dict[str, Predicate_], names: Iterable[str], model: PartialModel
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
            found.add(_model_name(model, name))
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
                        found.add(_model_name(model, side.relation))
    return found


def _model_name(model: PartialModel, name: str) -> str:
    try:
        found = model.relation_named(name) or model.attribute_named(name)
    except ValueError:
        return name
    return found.name if found is not None else name


def _opposite_name(schema: Schema, feature: Feature) -> str:
    other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
    assert other is not None
    return _feature_name(other)


def partial_model(
    schema: Schema, graph: Graph, predicates: Sequence[Predicate]
) -> tuple[PartialModel, list[BaseModel | None]]:
    """The partial model of ``graph`` in a closed world, and the object of each node
    (``None`` for enum literals)."""
    model = _Builder(graph)
    for enum_type in schema.enums:
        model.enum(enum_type)
    for cls in schema.classes:
        model.class_(cls.__name__, cls)
    for union in schema.unions:
        model.class_(union.name, union.members)
    for cls in schema.classes:
        for feature in schema.features[cls]:
            model.feature(feature)
    for i, node in enumerate(graph.nodes):
        for value in node.values:
            model.value(schema, i, value)
    for predicate in predicates:
        model.relation(predicate.name, len(predicate.parameters))
    for predicate, arguments, answer in predicate_facts(schema, graph, predicates):
        model.holds(predicate.name, tuple(model.index[id(node)] for node in arguments), answer)
    return model.build(), model.objects


class _Builder:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.nodes = [Node(node.name, node.name) for node in graph.nodes]
        self.objects: list[BaseModel | None] = [node.obj for node in graph.nodes]
        self.index = {id(node): i for i, node in enumerate(graph.nodes)}
        self.literals: dict[tuple[str, str], int] = {}
        self.relations: dict[str, tuple[int, dict[tuple[int, ...], LogicValue]]] = {}
        self.attributes: dict[str, tuple[DataType, dict[int, object]]] = {}
        exists = self.relation("builtin::exists", 1)
        for i, node in enumerate(graph.nodes):
            exists[(i,)] = node.exists

    def relation(self, name: str, arity: int) -> dict[tuple[int, ...], LogicValue]:
        return self.relations.setdefault(name, (arity, {}))[1]

    def holds(self, name: str, key: tuple[int, ...], value: LogicValue) -> None:
        tuples = self.relations[name][1]
        if tuples.get(key) != "true":  # of a fact given twice, the surer one
            tuples[key] = value

    def enum(self, enum_type: EnumLike) -> None:
        """The enum's literals are atoms, and its class holds them."""
        if isinstance(enum_type, LiteralEnum):
            members: tuple[str, ...] = enum_type.values
        else:
            members = tuple(member.name for member in enum_type)
        name = _type_name(enum_type)
        literals = self.relation(name, 1)
        for member in members:
            index = len(self.nodes)
            self.literals[(name, member)] = index
            self.nodes.append(Node(f"{class_name(enum_type)}::{quote_id(member)}", member, "atom"))
            self.objects.append(None)
            literals[(index,)] = "true"
            self.holds("builtin::exists", (index,), "true")

    def class_(self, name: str, classes: type | tuple[type, ...]) -> None:
        members = self.relation(name, 1)
        for i, node in enumerate(self.graph.nodes):
            if isinstance(node.obj, classes):
                members[(i,)] = "true"

    def feature(self, feature: Feature) -> None:
        name = _feature_name(feature)
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

    def value(self, schema: Schema, i: int, value: Value) -> None:
        feature = value.feature
        name = _feature_name(feature)
        if feature.is_attribute:
            if value.certain:  # an uncertain value is simply unknown
                self.attributes[name][1][i] = value.value
                if feature.target == "boolean":
                    self.relations[name][1][(i,)] = "true" if value.value else "false"
            return
        certainty: LogicValue = "true" if value.certain else "unknown"
        if feature.is_enum:
            member = value.value if isinstance(feature.target, LiteralEnum) else value.value.name
            target = self.literals[(_type_name(feature.target), member)]
        elif value.target is not None:
            target = self.index[id(value.target)]
        else:
            return  # a key that resolves to no node: reported by reference_errors
        self.holds(name, (i, target), certainty)
        if feature.opposite is not None:
            other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
            assert other is not None
            self.holds(_feature_name(other), (target, i), certainty)

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


def _type_name(node_class: object) -> str:
    """The name of a class, union or enum, as rules name it (unquoted)."""
    if isinstance(node_class, LiteralEnum):
        return node_class.name
    return node_class.__name__  # type: ignore[attr-defined]


def _feature_name(feature: Feature) -> str:
    return f"{feature.owner.__name__}::{feature.name}"
