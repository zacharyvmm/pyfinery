"""One plain Refinery model of a :class:`CompiledProblem`: a graph's facts written into it as
the difference from what it holds, its violations and matches read back.

Refinery keeps the value of an attribute in an abstract domain: an interval
of integers or of reals, a four-valued truth value, a string that may be
unknown. A plain model holds each value as the exact element of its domain,
``7`` as the interval ``[7, 7]``, so the comparisons and arithmetic Refinery's
compiler emits work on it unchanged, and a check holds when it is certainly
true. An attribute with no value reads as the domain's ``unknown``, of which
no check is certain.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from .._instances import Graph, Node
from .._options import Predicate
from .._results import Match, Violation, ViolationKind
from .._schema import AttributeFeature, AttributeType, ReferenceFeature
from .._syntax import quote_id
from .classes import java
from .compile import CONTAINED, CONTAINER, NODE, CompiledProblem
from .mapping import graph_facts

type PredicateFact = tuple[Predicate, tuple[Node, ...], str]
"""A tuple of a :class:`~pyfinery.Predicate` and its logic value, as
:func:`~pyfinery._render.predicate_facts` gives them."""

type Key = tuple[str, tuple[int, ...], object]
"""A fact of a plain model: a relation, a tuple of node ids, and its value there: ``True`` (it
holds), or the Python value of an attribute."""

NO_VALUE = object()
"""The value of an attribute an object of its class has no value for: unknown."""

_DOMAINS = {"boolean": "TruthValue", "int": "IntInterval", "real": "RealInterval"}


def to_java(data_type: AttributeType, value: Any) -> Any:
    """The exact Refinery value of ``value``, an attribute of type ``data_type``."""
    j = java()
    if value is NO_VALUE:
        return getattr(j, _DOMAINS.get(data_type, "StringValue")).UNKNOWN
    match data_type:
        case "boolean":
            return j.TruthValue.TRUE if value else j.TruthValue.FALSE
        case "int":
            return j.IntInterval.of(j.BigInteger(str(int(value))))
        case "real":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Refinery cannot represent the real value {number!r}")
            return j.RealInterval.of(repr(number))
        case _:
            return j.StringValue.of(str(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSnapshot:
    """A state of a :class:`PlainModel` to :meth:`~PlainModel.restore`: the store's version, and
    what the Python side knows of it."""

    version: Any
    """The version :meth:`PlainModel.snapshot` committed."""
    facts: frozenset[Key]
    known: dict[Hashable, int]
    next_id: int
    nodes_by_id: dict[int, Node]


class PlainModel:
    """A model of ``problem``'s store, which :meth:`sync` keeps equal to a graph."""

    def __init__(self, problem: CompiledProblem) -> None:
        from .runtime import _jpype, helpers

        j = java()
        self.problem = problem
        self._bulk = helpers().Bulk
        self._jpype = _jpype()
        self.model = problem.store.createEmptyModel()
        self._queries = self.model.getAdapter(j.ModelQueryAdapter)
        self._interpretations: dict[str, Any] = {}
        self._results = {
            name: (self._queries.getResultSet(query), int(query.arity()))
            for name, query in problem.queries.items()
        }
        self._class_names = _class_memberships(problem)
        self.facts: set[Key] = set()
        """What the model holds."""
        self.known: dict[Hashable, int] = {}
        """Node ids by node identity, of every node the model has held: an id is never reused."""
        self.next_id = problem.reserved_ids
        """The id the next new node gets: past the ids of the constants."""
        self._constant_names = {node_id: name for name, node_id in problem.constants.items()}
        self._nodes_by_id: dict[int, Node] = {}
        """The nodes of the graph last synchronized, by id."""
        constants: set[Key] = set()
        for name, node_id in problem.constants.items():
            enum_type, _ = problem.catalogue.literals[name]
            constants.add((NODE, (node_id,), True))
            constants.add((problem.catalogue.name(enum_type), (node_id,), True))
        self._constants = constants
        self._write(constants, set())
        self.facts = set(constants)

    def sync(self, graph: Graph, predicate_facts: Iterable[PredicateFact] = ()) -> int:
        """Make the model hold the facts of ``graph`` and ``predicate_facts`` (those that
        certainly hold); return how many facts changed."""
        node_ids: dict[Node, int] = {}
        for node in graph.nodes:
            node_id = self.known.get(node.identity)
            if node_id is None:
                node_id = self.known[node.identity] = self.next_id
                self.next_id += 1
            node_ids[node] = node_id
        wanted = set(self._constants)
        wanted.update(self._graph_facts(graph, node_ids))
        for predicate, nodes, answer in predicate_facts:
            if answer == "true":
                wanted.add((quote_id(predicate.name), tuple(node_ids[n] for n in nodes), True))
        added, removed = wanted - self.facts, self.facts - wanted
        self._write(added, removed)
        self.facts = wanted
        self._nodes_by_id = {node_id: node for node, node_id in node_ids.items()}
        return len(added) + len(removed)

    def _graph_facts(self, graph: Graph, node_ids: dict[Node, int]) -> Iterable[Key]:
        catalogue, read = self.problem.catalogue, self.problem.attributes
        valued = set()
        for fact in graph_facts(catalogue, graph, node_ids, self.problem.constants):
            feature = catalogue.features.get(fact.relation)
            if not isinstance(feature, AttributeFeature) or fact.relation in read:
                yield fact.relation, fact.nodes, fact.value  # not an attribute no rule reads
                valued.add((fact.relation, fact.nodes))
        for name in read:  # unknown for the objects of the attribute's class without a value
            owner = catalogue.features[name].owner
            for node in graph.nodes:
                key = (node_ids[node],)
                if isinstance(node.obj, owner) and (name, key) not in valued:
                    yield name, key, NO_VALUE
        for node in graph.nodes:
            node_id = (node_ids[node],)
            yield NODE, node_id, True
            for name in self._class_names.get(type(node.obj), ()):
                yield name, node_id, True

    def _write(self, added: set[Key], removed: set[Key]) -> None:
        """Take out ``removed``, then put ``added``: a changed value is in both."""
        attributes = self.problem.attributes
        for holds, keys in ((False, removed), (True, added)):
            grouped: dict[tuple[str, object], list[int]] = {}
            for relation, nodes, value in keys:
                grouped.setdefault((relation, value), []).extend(nodes)
            for (relation, value), flat in grouped.items():
                interpretation, arity = self._interpretation(relation)
                array = self._jpype.JArray(self._jpype.JInt)(flat)
                if relation not in attributes:
                    stored = self._jpype.JBoolean(holds)
                else:  # no value is the symbol's default
                    stored = to_java(attributes[relation], value) if holds else None
                self._bulk.put(interpretation, array, arity, stored)

    def _interpretation(self, relation: str) -> tuple[Any, int]:
        found = self._interpretations.get(relation)
        if found is None:
            symbol = self.problem.symbol(relation)
            found = self._interpretations[relation] = (
                self.model.getInterpretation(symbol),
                int(symbol.arity()),
            )
        return found

    def tuples(self, predicate: str) -> list[tuple[int, ...]]:
        """The tuples of ``predicate``, as node ids, sorted."""
        self._queries.flushChanges()
        result_set, arity = self._results[predicate]
        flat = list(self._bulk.keys(result_set, arity))
        if arity == 0:
            return [()] if result_set.size() else []
        return sorted(tuple(flat[i : i + arity]) for i in range(0, len(flat), arity))

    def violations(self) -> list[Violation]:
        """The tuples of the error predicates and structural checks, as a server lists them."""
        # Checks that report the same relation (a link on a cycle, a link twice) are listed as
        # one: each tuple once, in order.
        reported: dict[tuple[ViolationKind, str], set[tuple[int, ...]]] = {}
        for error in self.problem.errors:
            reported.setdefault((error.kind, error.constraint), set()).update(
                self.tuples(error.predicate)
            )
        found = []
        for (kind, constraint), keys in reported.items():
            for key in sorted(keys):
                names, nodes = self.label(key)
                found.append(Violation(kind, constraint, names, nodes))
        return found

    def matches(self, predicate: str, reported: str) -> list[Match]:
        """The tuples of ``predicate`` as :class:`Match` es named ``reported``."""
        parameters = self.problem.parameters.get(predicate, ())
        return [
            Match(reported, *self.label(key), "true", parameters) for key in self.tuples(predicate)
        ]

    def label(self, key: tuple[int, ...]) -> tuple[tuple[str, ...], tuple[Any, ...]]:
        """The names and Python objects of the nodes of ``key`` (``None`` for a constant)."""
        names, nodes = [], []
        for node_id in key:
            constant = self._constant_names.get(node_id)
            if constant is not None:
                names.append(constant)
                nodes.append(None)
            else:
                node = self._nodes_by_id[node_id]
                names.append(node.name)
                nodes.append(node.obj)
        return tuple(names), tuple(nodes)

    def snapshot(self) -> ModelSnapshot:
        """Commit the model, and return its state to :meth:`restore` later."""
        return ModelSnapshot(
            version=self.model.commit(),
            facts=frozenset(self.facts),
            known=dict(self.known),
            next_id=self.next_id,
            nodes_by_id=dict(self._nodes_by_id),
        )

    def restore(self, snapshot: ModelSnapshot) -> None:
        """Go back to ``snapshot``: the store's version, the facts, the node ids, and the nodes
        :meth:`label` names."""
        self.model.restore(snapshot.version)
        self.facts = set(snapshot.facts)
        self.known = dict(snapshot.known)
        self.next_id = snapshot.next_id
        self._nodes_by_id = dict(snapshot.nodes_by_id)

    def close(self) -> None:
        self.model.close()


def _class_memberships(problem: CompiledProblem) -> dict[type, tuple[str, ...]]:
    """``container`` and ``contained`` for each class whose instances are: those with an
    outgoing, or incoming, containment reference."""
    schema, catalogue = problem.schema, problem.catalogue
    containment = [
        f
        for cls in schema.classes
        for f in schema.features[cls]
        if isinstance(f, ReferenceFeature) and f.kind == "contains"
    ]
    owners = {catalogue.name(f.owner) for f in containment}
    targets = {catalogue.name(f.target) for f in containment}
    memberships = {}
    for cls in schema.classes:
        supertypes = set(catalogue.supertypes(cls))
        names = []
        if supertypes & owners:
            names.append(CONTAINER)
        if supertypes & targets:
            names.append(CONTAINED)
        memberships[cls] = tuple(names)
    return memberships
