# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The bits engine's pipeline for :meth:`~pyfinery.Metamodel.query`: the graph to a
partial model, built into matrices, the rules evaluated over them, read back as matches.

:meth:`~pyfinery.Metamodel.query_batch` runs the same pipeline on many graphs: the
rules are read once, and evaluated once on the graphs of each size together
(:class:`~..matrices.batch.PackedBatch`); building each graph's partial model,
checking it, and reading its matches back are done per graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pydantic import BaseModel

from ..._engines import BatchQueryEngine
from ..._instances import Graph
from ..._options import Predicate
from ..._render import _declare_predicate
from ..._results import QueryResult, reference_errors
from ..._schema import Schema
from ..evaluation.pipeline import BatchValues, evaluate_batch
from ..matrices.batch import PackedBatch, group_by_size
from ..matrices.model import PartialModel
from ..names import refinery_name
from ..rules.compiled import Rules
from ..rules.syntax import Predicate as RulePredicate
from ..rules.syntax import PredicateKind, Source, parse
from .build import partial_model
from .matches import Labels, TuplesByValue, error_violations, selected_matches
from .refinement import (
    dependencies,
    refuse_abstract_instances,
    refuse_assertions,
    refuse_containment_refinement,
    refuse_error_refinement,
)


class BitsEngine(BatchQueryEngine):
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

    :meth:`~pyfinery.Metamodel.query_batch` evaluates the rules on many graphs at
    once (:meth:`query_graphs`).
    """

    def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        (result,) = self.query_graphs(schema, [graph], rules, [predicates], select)
        if isinstance(result, Exception):
            raise result
        return result

    def query_graphs(
        self,
        schema: Schema,
        graphs: Sequence[Graph],
        rules: Sequence[str],
        predicates: Sequence[Sequence[Predicate]],
        select: Sequence[str],
    ) -> list[QueryResult | Exception]:
        """What :meth:`query_graph` gives each graph (with its own ``predicates``), or the
        exception it raises, evaluating the rules on the graphs together.

        Graphs whose predicates are declared alike (the same names and parameter
        types) share one reading of the rules; of those, graphs of about the same
        size are evaluated as one batch.
        """
        results: list[QueryResult | Exception | None] = [None] * len(graphs)
        groups: dict[str, list[int]] = {}
        for i, given in enumerate(predicates):
            groups.setdefault(query_source(schema, rules, given), []).append(i)
        for source, members in groups.items():
            try:
                query = Query.of(source, tuple(select))
            except Exception as e:
                for i in members:
                    results[i] = e
                continue
            prepared: list[Prepared] = []
            for i in members:
                try:
                    prepared.append(query.prepare(i, schema, graphs[i], predicates[i]))
                except Exception as e:
                    results[i] = e
            for bucket in group_by_size([p.model.size for p in prepared]):
                ready = [prepared[j] for j in bucket]
                try:
                    batch = PackedBatch.build([p.model for p in ready])
                    values = evaluate_batch(query.rules, batch, query.evaluated)
                    tuples = tuples_by_graph(values, len(batch))
                except Exception as e:
                    for p in ready:
                        results[p.index] = e
                    continue
                for graph, p in enumerate(ready):
                    try:
                        results[p.index] = query.finish(p, tuples[graph])
                    except Exception as e:
                        results[p.index] = e
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]


@dataclass(frozen=True)
class Prepared:
    """A graph ready to evaluate: its partial model, and what reading back needs."""

    index: int
    graph: Graph
    model: PartialModel
    objects: list[BaseModel | None]
    used: set[str]
    """The relations the evaluated predicates depend on, by their names in the model."""


@dataclass(frozen=True)
class Query:
    """Rules read once, to evaluate on every graph whose predicates are declared alike."""

    source: str
    rules: Rules
    rule_predicates: dict[str, RulePredicate]
    selected: dict[str, str]
    evaluated: list[str]

    @staticmethod
    @lru_cache(maxsize=32)
    def of(source: str, select: tuple[str, ...]) -> Query:
        """The query of ``source`` (rules, and the declarations of the predicates), read
        once for as many queries as use it."""
        parsed = parse(source)
        refuse_assertions(parsed)
        selected, evaluated = predicates_to_evaluate(parsed, select)
        rule_predicates = {p.name: p for p in parsed.predicates}
        return Query(source, Rules.of(parsed), rule_predicates, selected, evaluated)

    def prepare(
        self, index: int, schema: Schema, graph: Graph, predicates: Sequence[Predicate]
    ) -> Prepared:
        """The partial model of ``graph``, refused where Refinery would refine it."""
        refuse_abstract_instances(schema, graph)
        model, objects = partial_model(schema, graph, predicates)
        used = dependencies(self.rule_predicates, self.evaluated, model)
        refuse_containment_refinement(schema, graph, model, used)
        return Prepared(index, graph, model, objects, used)

    def finish(self, prepared: Prepared, tuples: dict[str, TuplesByValue]) -> QueryResult:
        """The matches and errors of a graph, from the tuples of its predicates."""
        graph, model = prepared.graph, prepared.model
        rule_predicates = self.rule_predicates
        refuse_error_refinement(graph, model, rule_predicates, tuples, prepared.used)
        kinds: dict[str, PredicateKind] = {name: rule_predicates[name].kind for name in tuples}
        labels = Labels([node.name for node in model.nodes], prepared.objects)
        violations = [*reference_errors(graph), *error_violations(kinds, tuples, labels)]
        matches = selected_matches(graph, rule_predicates, self.selected, tuples, labels)
        return QueryResult(matches, violations, self.source, None)


def tuples_by_graph(values: dict[str, BatchValues], graphs: int) -> list[dict[str, TuplesByValue]]:
    """The ``true``, ``unknown`` and ``error`` tuples of each predicate, of each of the
    batch's ``graphs``, in the graph's own node indices."""
    found: list[dict[str, TuplesByValue]] = [{} for _ in range(graphs)]
    for name, result in values.items():
        for logic_value, tuples in result.tuples().items():
            bounds = np.searchsorted(tuples[:, 0], np.arange(graphs + 1))
            for graph in range(graphs):
                own = tuples[bounds[graph] : bounds[graph + 1], 1:]
                found[graph].setdefault(name, {})[logic_value] = own
    return found


def query_source(schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate]) -> str:
    """The rules, and the :class:`~pyfinery.Predicate` objects declared, as in the
    problem :meth:`Metamodel.query` sends."""
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
