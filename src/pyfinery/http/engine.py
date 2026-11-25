# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Refinery clients as engines: render the problem, send it, read the report.

:class:`ClientEngine` and :class:`AsyncClientEngine` send the problem
:class:`Problems` renders to the server's ``concretize`` (to validate) or
``semantics`` (to query), and read the violations or matches from the JSON
model it returns.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from pydantic import BaseModel

from .._instances import Graph
from .._options import LogicValue, Predicate
from .._render import render_metamodel, render_query_problem, render_validation_problem
from .._results import (
    Match,
    QueryResult,
    ValidationReport,
    Violation,
    ViolationKind,
    reference_errors,
)
from .._schema import Schema
from .._syntax import quote_id
from ..errors import RefineryError, RefineryTimeoutError
from .client import AsyncRefinery, Refinery
from .models import JsonOutput, JsonOutputFormat, OutputFormats, SourceOutputFormat

__all__ = [
    "AsyncClientEngine",
    "ClientEngine",
    "Problems",
    "read_matches",
    "read_violations",
]


class Problems:
    """Renders the problems sent to a Refinery server for a metamodel.

    ``source`` is the metamodel as :class:`~pyfinery.Metamodel` rendered it, with
    its own ``rules``; when the rules an engine is given start with those, the
    source is reused, so the problem is the one :meth:`~pyfinery.Metamodel.problem`
    and :meth:`~pyfinery.Metamodel.query_problem` show. Other rules are rendered
    with the metamodel in full.
    """

    def __init__(self, source: str, rules: Sequence[str]) -> None:
        self.source = source
        self.rules = list(rules)

    def validation(
        self, schema: Schema, graph: Graph, rules: Sequence[str], predicates: Sequence[Predicate]
    ) -> str:
        """The metamodel, the rules, and the facts of ``graph``: what certainly holds."""
        metamodel, rest = self._split(schema, rules)
        return render_validation_problem(schema, metamodel, graph, rest, predicates)

    def query(
        self, schema: Schema, graph: Graph, rules: Sequence[str], predicates: Sequence[Predicate]
    ) -> str:
        """The metamodel, the rules, and the facts of ``graph`` in a closed world that
        keeps what is unknown."""
        metamodel, rest = self._split(schema, rules)
        return render_query_problem(schema, metamodel, graph, rest, predicates)

    def _split(self, schema: Schema, rules: Sequence[str]) -> tuple[str, list[str]]:
        """The metamodel with the rules it holds, and the rules left to add."""
        mine = len(self.rules)
        if list(rules[:mine]) == self.rules:
            return self.source, list(rules[mine:])
        return render_metamodel(schema, list(rules)), []


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


class ClientEngine:
    """A :class:`Refinery` client as an :class:`~pyfinery._engines.Engine`: validates
    with the server's ``concretize``, and queries with its ``semantics``."""

    def __init__(self, client: Refinery, problems: Problems) -> None:
        self.client = client
        self.problems = problems

    def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        source = self.problems.validation(schema, graph, rules, predicates)
        with _explain_timeout():
            result = self.client.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        source = self.problems.query(schema, graph, rules, predicates)
        with _explain_timeout():
            result = self.client.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, select)


class AsyncClientEngine:
    """An :class:`AsyncRefinery` client as an :class:`~pyfinery._engines.AsyncEngine`."""

    def __init__(self, client: AsyncRefinery, problems: Problems) -> None:
        self.client = client
        self.problems = problems

    async def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        source = self.problems.validation(schema, graph, rules, predicates)
        with _explain_timeout():
            result = await self.client.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    async def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        source = self.problems.query(schema, graph, rules, predicates)
        with _explain_timeout():
            result = await self.client.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, select)


def read_violations(model: JsonOutput | None, graph: Graph, source: str) -> ValidationReport:
    """The constraints ``graph`` violates, from Refinery's output for ``source``."""
    if model is None:
        raise RefineryError("Refinery did not return a model to validate")
    violations = reference_errors(graph) + _errors(model, graph.objects_by_name())
    return ValidationReport(violations, source, model, graph.identity)


def read_matches(
    model: JsonOutput | None, graph: Graph, source: str, predicates: Sequence[str]
) -> QueryResult:
    """The matches of ``predicates``, and the constraints ``graph`` violates, from
    Refinery's output for ``source``."""
    if model is None:
        raise RefineryError("Refinery did not return a model to query")
    by_name = graph.objects_by_name()
    node_names = _node_names(model, by_name)
    exists = {node.name: node.exists for node in graph.nodes}
    relations = {r.name: r for r in model.relations}
    matches = []
    for predicate in predicates:
        relation = relations.get(predicate)
        if relation is None:
            raise ValueError(f"Refinery returned no relation named {predicate!r}")
        for raw in model.partial_interpretation.get(relation.name, []):
            value: LogicValue = raw[-1]  # type: ignore[assignment]
            if value not in ("true", "unknown"):
                continue  # errors are reported as contradictions
            names = tuple(node_names[i] for i in raw[: relation.arity])  # type: ignore[index]
            if any(name.endswith("::new") for name in names):
                continue  # objects that don't exist in a closed world
            # Refinery doesn't weaken a predicate for the existence of
            # its arguments (only of nodes quantified in its body).
            if any(exists.get(name) == "unknown" for name in names):
                value = "unknown"
            nodes = tuple(by_name.get(name) for name in names)
            parameters = tuple(relation.parameter_names or ())
            matches.append(Match(predicate, names, nodes, value, parameters))
    errors = [
        error
        for error in reference_errors(graph) + _errors(model, by_name)
        # Refinery still evaluates constraints on the C::new objects the
        # closed world says don't exist.
        if not any(name.endswith("::new") for name in error.node_names)
    ]
    return QueryResult(matches, errors, source, model)


_MULTIPLICITY_SUFFIX = "::invalidMultiplicity"


def _classify(relation: str, is_error_predicate: bool) -> ViolationKind | None:
    if relation == "builtin::invalidContainer":
        return "containment"
    if relation.startswith("builtin::"):
        # Consequences of other errors (e.g., builtin::contains for a cycle).
        return None
    if is_error_predicate:
        return "multiplicity" if relation.endswith(_MULTIPLICITY_SUFFIX) else "rule"
    return "contradiction"


def _errors(model: JsonOutput, by_name: dict[str, BaseModel]) -> list[Violation]:
    """The tuples of error predicates and builtin constraints that are ``error``."""
    node_names = _node_names(model, by_name)
    violations = []
    for relation in model.relations:
        is_error_predicate = relation.detail.type == "pred" and relation.detail.kind == "error"
        kind = _classify(relation.name, is_error_predicate)
        if kind is None:
            continue
        for raw in model.partial_interpretation.get(relation.name, []):
            if raw[-1] != "error":
                continue
            names = tuple(node_names[i] for i in raw[: relation.arity])  # type: ignore[index]
            nodes = tuple(by_name.get(name) for name in names)
            violations.append(Violation(kind, relation.name, names, nodes))
    return violations


def _node_names(model: JsonOutput, by_name: dict[str, BaseModel]) -> list[str]:
    """The name of each node of ``model``, as pyfinery named it in the problem.

    Refinery returns a name without the quotes pyfinery wrote it with when the name
    doesn't need them in Refinery's syntax, e.g. ``e`` for ``'e'`` (pyfinery quotes
    more names than Refinery has to, to be safe)."""
    names = []
    for node in model.nodes:
        name = node.name
        if name not in by_name and quote_id(name) in by_name:
            name = quote_id(name)
        names.append(name)
    return names
