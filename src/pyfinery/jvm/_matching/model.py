"""One plain Refinery model of a :class:`CompiledProblem`: a graph's facts written into it as
the difference from what it holds, its violations and matches read back."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

from ..._instances import Graph
from ..._mapping import NodeIds, facts
from ..._options import Predicate
from ..._render import predicate_facts
from ..._results import Match, Violation
from ..._syntax import quote_id
from .classes import java
from .compile import CONTAINED, CONTAINER, NODE, CompiledProblem
from .values import NO_VALUE, to_java

type Key = tuple[str, tuple[int, ...], object]
"""A fact of a plain model: a relation, a tuple of node ids, and its value there: ``True`` (it
holds), or the Python value of an attribute."""


class PlainModel:
    """A model of ``problem``'s store, which :meth:`sync` keeps equal to a graph."""

    def __init__(self, problem: CompiledProblem) -> None:
        from .._runtime import _jpype, helpers

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
        self.ids = self._new_ids()
        """The ids of the nodes of the graph last synchronized."""
        self.known: dict[Hashable, int] = {}
        """Node ids by node identity, of every node the model has held: an id is never reused."""
        self.next_id = problem.reserved_ids
        self.graph: Graph | None = None
        constants: set[Key] = set()
        for name, node_id in problem.constants.items():
            enum_type, _ = problem.catalogue.literals[name]
            constants.add((NODE, (node_id,), True))
            constants.add((problem.catalogue.name(enum_type), (node_id,), True))
        self._constants = constants
        self._write(constants, set())
        self.facts = set(constants)

    # -- writing ---------------------------------------------------------------------------------

    def sync(self, graph: Graph, predicates: Sequence[Predicate] = ()) -> int:
        """Make the model hold the facts of ``graph`` and of ``predicates`` over it (what
        certainly holds); return how many facts changed."""
        ids = self._new_ids()
        for node in graph.nodes:
            node_id = self.known.get(node.identity)
            if node_id is None:
                node_id = self.known[node.identity] = self.next_id
                self.next_id += 1
            ids.put(node, node_id)
        wanted = set(self._constants)
        wanted.update(self._graph_facts(graph, ids))
        for predicate, nodes, answer in predicate_facts(self.problem.schema, graph, predicates):
            if answer == "true":
                wanted.add((quote_id(predicate.name), tuple(ids.id_of(n) for n in nodes), True))
        added, removed = wanted - self.facts, self.facts - wanted
        self._write(added, removed)
        self.facts = wanted
        self.ids = ids
        self.graph = graph
        return len(added) + len(removed)

    def _graph_facts(self, graph: Graph, ids: NodeIds) -> Iterable[Key]:
        catalogue, read = self.problem.catalogue, self.problem.attributes
        valued = set()
        for fact in facts(catalogue, graph, ids):
            feature = catalogue.features.get(fact.relation)
            if feature is None or not feature.is_attribute or fact.relation in read:
                yield fact.relation, fact.nodes, fact.value  # not an attribute no rule reads
                valued.add((fact.relation, fact.nodes))
        for name in read:  # unknown for the objects of the attribute's class without a value
            owner = catalogue.features[name].owner
            for node in graph.nodes:
                key = (ids.id_of(node),)
                if isinstance(node.obj, owner) and (name, key) not in valued:
                    yield name, key, NO_VALUE
        for node in graph.nodes:
            node_id = (ids.id_of(node),)
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

    def _new_ids(self) -> NodeIds:
        ids = NodeIds(self.problem.reserved_ids)
        for name, node_id in self.problem.constants.items():
            ids.constant(name, node_id)
        return ids

    # -- reading ---------------------------------------------------------------------------------

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
        found = []
        for error in self.problem.errors:
            for key in self.tuples(error.predicate):
                names, nodes = self.label(key)
                found.append(Violation(error.kind, error.constraint, names, nodes))
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
            constant = self.ids.constant_name(node_id)
            if constant is not None:
                names.append(constant)
                nodes.append(None)
            else:
                node = self.ids.node(node_id)
                names.append(node.name)
                nodes.append(node.obj)
        return tuple(names), tuple(nodes)

    # -- versions ----------------------------------------------------------------------------------

    def commit(self) -> Any:
        return self.model.commit()

    def restore(self, version: Any) -> None:
        self.model.restore(version)

    def close(self) -> None:
        self.model.close()


def _class_memberships(problem: CompiledProblem) -> dict[type, tuple[str, ...]]:
    """``container`` and ``contained`` for each class whose instances are: those with an
    outgoing, or incoming, containment reference."""
    schema, catalogue = problem.schema, problem.catalogue
    containment = [
        f for cls in schema.classes for f in schema.features[cls] if f.kind == "contains"
    ]
    owners = {catalogue.name(f.owner) for f in containment}
    targets = {catalogue.name(f.target) for f in containment}  # type: ignore[arg-type]
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
