"""Validation and queries on Refinery's own matcher, in this process.

:class:`JvmEngine` is an :class:`~pyfinery._engines.Engine`: pass it where
:meth:`~pyfinery.Metamodel.validate` and :meth:`~pyfinery.Metamodel.query`
take a client::

    from pyfinery.jvm import JvmEngine

    report = metamodel.validate(objects, JvmEngine())

It compiles the rules with Refinery's compiler and matches them directly on a
plain, two-valued model of the graph with Refinery's incremental (Rete)
matcher: no server, no partial-model reasoning, no concretization. The
violations are those a server's ``concretize`` reports for a complete model.
To validate a graph as it changes, use a :class:`~pyfinery.jvm.Session`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..._engines import Engine
from ..._instances import Graph
from ..._options import Predicate
from ..._render import predicate_facts
from ..._results import QueryResult, ValidationReport, reference_errors, resolve_predicate
from ..._schema import Schema
from ..._syntax import unquote_id
from ..errors import UnsupportedRuleError
from .cache import compiled
from .compile import CompiledProblem
from .model import PlainModel, PredicateFact

__all__ = ["JvmEngine"]


class JvmEngine(Engine):
    """Validates and queries with Refinery's matcher in this process (see the module).

    :meth:`query_graph` matches complete graphs only: an object that may not
    exist, or a fact that only may hold, raises
    :class:`~pyfinery.jvm.UnsupportedRuleError`; use a Refinery client to
    reason about what may hold.
    """

    def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        problem = compiled(schema, rules, predicates)
        require_supported(problem, problem.rule_errors)
        model = PlainModel(problem)
        try:
            model.sync(graph, predicate_facts(schema, graph, predicates))
            violations = reference_errors(graph) + model.violations()
        finally:
            model.close()
        return ValidationReport(violations, problem.source, None, graph.identity)

    def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        problem = compiled(schema, rules, predicates)
        require_supported(problem)  # before the names: a refused assertion declares a relation
        selected = {name: problem_name(problem, name) for name in select}
        require_supported(problem, [*problem.rule_errors, *selected.values()])
        # Each predicate is evaluated once: what is checked for uncertainty is what is matched.
        facts = list(predicate_facts(schema, graph, predicates))
        refuse_uncertainty(graph, facts)
        model = PlainModel(problem)
        try:
            model.sync(graph, facts)
            matches = [m for name, p in selected.items() for m in model.matches(p, str(name))]
            errors = reference_errors(graph) + model.violations()
        finally:
            model.close()
        return QueryResult(matches, errors, problem.source, None)


def require_supported(problem: CompiledProblem, predicates: Sequence[str] = ()) -> None:
    """Raise :class:`UnsupportedRuleError` if the rules state what the direct path can't hold
    (:attr:`~.compile.CompiledProblem.refused`), or any of ``predicates`` can't be matched on it."""
    reasons = [*problem.refused]
    reasons += [
        f"{name}: uses {problem.unsupported[name]}"
        for name in predicates
        if name in problem.unsupported
    ]
    if reasons:
        raise UnsupportedRuleError("The JVM engine can't match these:\n  " + "\n  ".join(reasons))


def problem_name(problem: CompiledProblem, name: str) -> str:
    """The name of the predicate ``name`` (as declared, or as selected) in ``problem``."""
    by_decoded = {unquote_id(key): key for key in (*problem.queries, *problem.unsupported)}
    found = resolve_predicate(name, by_decoded)
    if found is not None:
        return found
    raise ValueError(f"No predicate named {str(name)!r} to report")


def refuse_uncertainty(graph: Graph, predicate_facts: Iterable[PredicateFact]) -> None:
    """Raise :class:`UnsupportedRuleError` if ``graph`` (built with existence) or
    ``predicate_facts`` have something that only may hold."""
    for node in graph.nodes:
        if node.exists != "true":
            raise UnsupportedRuleError(
                f"{node.name} only may exist: the JVM engine matches complete graphs"
            )
        for value in node.values:
            if not value.certain:
                raise UnsupportedRuleError(
                    f"{node.name}.{value.feature.name} only may hold: "
                    "the JVM engine matches complete graphs"
                )
    for predicate, nodes, answer in predicate_facts:
        if answer != "true":
            names = ", ".join(n.name for n in nodes)
            raise UnsupportedRuleError(
                f"{predicate.name}({names}) only may hold: the JVM engine matches complete graphs"
            )
