# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The bits engine's pipeline for :meth:`~pyfinery.Metamodel.query`: the graph to a
partial model, built into matrices, the rules evaluated over them, read back as matches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..._instances import Graph
from ..._options import Predicate
from ..._render import _declare_predicate
from ..._results import QueryResult, reference_errors
from ..._schema import Schema
from ..evaluation.pipeline import evaluate
from ..matrices.packed import PackedModel
from ..names import refinery_name
from ..rules.compiled import Rules
from ..rules.syntax import Source, parse

if TYPE_CHECKING:
    pass
from .build import partial_model
from .matches import Labels, error_violations, selected_matches
from .refinement import (
    dependencies,
    refuse_abstract_instances,
    refuse_assertions,
    refuse_containment_refinement,
    refuse_error_refinement,
)


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
        values = evaluate(Rules.of(parsed), packed, evaluated)
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
