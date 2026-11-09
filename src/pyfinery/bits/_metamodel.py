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
from typing import TYPE_CHECKING

import numpy as np

from .._instances import Graph, Value
from .._options import LogicValue, Predicate
from .._render import _declare_predicate, predicate_facts
from .._results import Match, QueryResult, Violation, reference_errors
from .._schema import EnumLike, Feature, LiteralEnum, Schema, UnionClass
from . import _kernels as k
from ._errors import UnsupportedRuleError
from ._evaluate import PredicateValues, evaluate
from ._model import Attribute, DataType, Node, PartialModel, Relation
from ._syntax import Atom, Call, parse, refinery_name
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
        # The names as selected, by the names the rules give them (e.g. 'odd name').
        selected = {(n if n in kinds else refinery_name(n)): n for n in select}
        # Every predicate with a body is evaluated, as Refinery does: its error
        # tuples are reported, whatever is selected.
        names = list(dict.fromkeys([*selected, *(p.name for p in parsed.predicates if p.body)]))
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
        node_names = [node.name for node in model.nodes]
        # As read_matches does: Refinery doesn't weaken a predicate for the existence
        # of its arguments, only of nodes quantified in its body.
        maybe = np.zeros(len(model.nodes), bool)
        maybe[[i for i, node in enumerate(graph.nodes) if node.exists == "unknown"]] = True
        matches, violations = [], reference_errors(graph)
        for name, result in values.items():
            by_value = result.tuples()
            kind = "rule" if result.kind == "error" else "contradiction"
            for key in _sorted(by_value["error"]).tolist():
                names_, nodes = _named(key, node_names, objects)
                violations.append(Violation(kind, name, names_, nodes))
            if name not in selected:
                continue
            parameters = tuple(p.name for p in kinds[name].parameters)
            true, unknown = by_value["true"], by_value["unknown"]
            weakened = maybe[true].any(axis=1) if len(true) and true.shape[1] else None
            keys = np.concatenate([true, unknown])
            logic_values = np.array(["true"] * len(true) + ["unknown"] * len(unknown), dtype=object)
            if weakened is not None:
                logic_values[: len(true)][weakened] = "unknown"
            order = np.lexsort(keys.T[::-1]) if keys.shape[1] else np.arange(len(keys))
            for key, value in zip(keys[order].tolist(), logic_values[order].tolist(), strict=True):
                names_, nodes = _named(key, node_names, objects)
                matches.append(Match(selected[name], names_, nodes, value, parameters))
        return QueryResult(matches, violations, source, None)


def _sorted(tuples: np.ndarray) -> np.ndarray:
    return tuples[np.lexsort(tuples.T[::-1])] if tuples.shape[1] else tuples


def _named(
    key: list[int], names: list[str], objects: list[BaseModel | None]
) -> tuple[tuple[str, ...], tuple[BaseModel | None, ...]]:
    return tuple(names[i] for i in key), tuple(objects[i] for i in key)


# -- what Refinery would propagate ------------------------------------------------
#
# Refinery refines uncertain facts from what must hold; the bits engine evaluates
# the facts as given. These find where the two would differ, to refuse rather
# than answer differently. A graph without uncertainty is refined only where it is
# contradictory (an object in two containers, a containment cycle).


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
    ``c >= upper``; an object that only may exist can't if ``c + u < lower`` or
    ``c > upper``. An object has one container at most, and exactly one if a
    containment can hold its class; containment has no cycles.
    """
    exists = {i: node.exists for i, node in enumerate(graph.nodes)}
    counts: dict[tuple[str, int], list[int]] = {}
    containers: dict[int, list[tuple[int, bool]]] = {}  # child: (container, certain)
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
                containers.setdefault(key[1], []).append((key[0], certain))
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
            if upper is not None and certain > upper and exists[i] == "unknown":
                yield (
                    f"{node.name}, which only may exist and has more values of {name} than "
                    f"[{feature.lower}..{upper}] allows: Refinery would conclude it doesn't exist",
                    None,
                )
    every_containment = contains | {
        _opposite_name(schema, f) for f in features if f.kind == "contains" and f.opposite
    }
    # An object of a class a containment can hold has exactly one container.
    held = tuple(_classes(f.target) for f in features if f.kind == "contains")
    contained = tuple(c for classes in held for c in classes)
    for child, node in enumerate(graph.nodes):
        found = containers.get(child, [])
        holders = [parent for parent, sure in found if sure]
        certain, uncertain = len(holders), len(found) - len(holders)
        if certain > 1:
            # A contradiction: Refinery makes the containment links errors, and rules
            # out the objects involved that only may exist.
            involved = [child, *holders]
            uncertain_objects = any(exists.get(i) == "unknown" for i in involved)
            yield (
                f"{node.name}, in {certain} containers at once: Refinery contradicts it",
                None if uncertain_objects else every_containment,
            )
        if uncertain and certain:
            yield (
                f"uncertain containers of {node.name}, which has a certain one: "
                "Refinery would rule them out",
                every_containment,
            )
        if not contained or not isinstance(node.obj, contained):
            continue
        if uncertain == 1 and not certain:
            yield (
                f"the one possible container of {node.name}, uncertain: Refinery would "
                "conclude it holds, as the object needs a container",
                every_containment,
            )
        if not certain and not uncertain and exists[child] == "unknown":
            yield (
                f"{node.name}, which only may exist and has no possible container: "
                "Refinery would conclude it doesn't exist",
                None,
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


def _classes(target: object) -> tuple[type, ...]:
    """The model classes a reference's target stands for."""
    if isinstance(target, UnionClass):
        return target.members
    return (target,) if isinstance(target, type) else ()


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


class _Builder:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        names = [_output_name(node.name) for node in graph.nodes]
        self.nodes = [Node(name, name) for name in names]
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
    """The name of a class, union or enum, as Refinery's output writes it."""
    if isinstance(node_class, LiteralEnum):
        return refinery_name(node_class.name)
    return refinery_name(node_class.__name__)  # type: ignore[attr-defined]


def _feature_name(feature: Feature) -> str:
    return f"{refinery_name(feature.owner.__name__)}::{refinery_name(feature.name)}"


def _output_name(rendered: str) -> str:
    """A node's name as Refinery's output writes it, from the name pyfinery gives it in
    the problem (``quote_id``, which quotes some names Refinery's output doesn't)."""
    if len(rendered) >= 2 and rendered[0] == rendered[-1] == "'":
        return refinery_name(re.sub(r"\\(.)", r"\1", rendered[1:-1]))
    return refinery_name(rendered)
