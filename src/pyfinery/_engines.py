"""The engines that evaluate :meth:`~pyfinery.Metamodel.validate` and
:meth:`~pyfinery.Metamodel.query`, and how what the user passes becomes one.

An engine is given what the metamodel knows, never a rendered problem: the
:class:`~pyfinery._schema.Schema`, the :class:`~pyfinery._instances.Graph` built
from the objects, the rules (the metamodel's own, then those of the call), the
:class:`~pyfinery.Predicate` objects, and, to query, the names of the predicates
to report. :class:`~pyfinery.Metamodel` combines its rules with those of the call
and builds the graph; what an engine does with them is its own business:

- a :class:`QueryEngine` queries, e.g. :class:`~pyfinery.bits.BitsEngine`;
- an :class:`Engine` also validates;
- :class:`AsyncQueryEngine` and :class:`AsyncEngine` are their asynchronous variants.

:class:`ClientEngine` and :class:`AsyncClientEngine` make a Refinery client one:
they render the problem, send it to the server's ``concretize`` (to validate) or
``semantics`` (to query), and read the report from what it returns.
:func:`engine`, :func:`query_engine`, :func:`async_engine` and
:func:`async_query_engine` turn what the user passed into the engine a method
needs, or raise :class:`TypeError`.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from ._instances import Graph
from ._options import Predicate
from ._render import render_metamodel, render_query_problem, render_validation_problem
from ._results import QueryResult, ValidationReport, read_matches, read_violations
from ._schema import Schema
from .client import AsyncRefinery, Refinery
from .errors import RefineryTimeoutError
from .models import JsonOutputFormat, OutputFormats, SourceOutputFormat

__all__ = [
    "AsyncClientEngine",
    "AsyncEngine",
    "AsyncQueryEngine",
    "ClientEngine",
    "Engine",
    "Problems",
    "QueryEngine",
    "async_engine",
    "async_query_engine",
    "engine",
    "query_engine",
]


# -- protocols ----------------------------------------------------------------


@runtime_checkable
class QueryEngine(Protocol):
    """Evaluates :meth:`~pyfinery.Metamodel.query`."""

    def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        """The matches of the predicates named in ``select``, and the errors, over
        ``graph`` (built with ``existence=True``) in a closed world that keeps what
        is unknown. ``rules`` are the metamodel's, then those of the call."""
        ...


@runtime_checkable
class Engine(QueryEngine, Protocol):
    """Evaluates :meth:`~pyfinery.Metamodel.validate` as well as
    :meth:`~pyfinery.Metamodel.query`."""

    def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        """The constraints ``graph`` violates, of the metamodel and ``rules`` (the
        metamodel's, then those of the call), counting only what certainly holds."""
        ...


@runtime_checkable
class AsyncQueryEngine(Protocol):
    """Asynchronous variant of :class:`QueryEngine`."""

    async def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult: ...


@runtime_checkable
class AsyncEngine(AsyncQueryEngine, Protocol):
    """Asynchronous variant of :class:`Engine`."""

    async def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport: ...


# -- the problems sent to a server --------------------------------------------


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


# -- clients as engines -------------------------------------------------------


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
    """A :class:`Refinery` client as an :class:`Engine`: validates with the server's
    ``concretize``, and queries with its ``semantics``."""

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
    """An :class:`AsyncRefinery` client as an :class:`AsyncEngine`."""

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


class _Awaited:
    """A synchronous engine as an asynchronous one: it runs in the caller's thread,
    as :class:`~pyfinery.bits.BitsEngine` did before :meth:`~pyfinery.Metamodel.aquery`
    awaited engines."""

    def __init__(self, engine: QueryEngine) -> None:
        self.engine = engine

    async def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        return self.engine.query_graph(schema, graph, rules, predicates, select)

    async def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        assert isinstance(self.engine, Engine)
        return self.engine.validate_graph(schema, graph, rules, predicates)


# -- what the user passed, as an engine ---------------------------------------


def query_engine(refinery: object, problems: Problems, method: str = "query") -> QueryEngine:
    """``refinery`` as the engine :meth:`~pyfinery.Metamodel.query` needs: a
    :class:`Refinery` client wrapped in a :class:`ClientEngine`, or an object with a
    synchronous ``query_graph`` (e.g. :class:`~pyfinery.bits.BitsEngine`) as it is."""
    if isinstance(refinery, Refinery) or _is_client(refinery, asynchronous=False):
        return ClientEngine(refinery, problems)  # type: ignore[arg-type]
    if (
        isinstance(refinery, AsyncRefinery)
        or _is_async(refinery, "query_graph")
        or _is_client(refinery, asynchronous=True)
    ):
        raise TypeError(
            f"{method}() needs a synchronous engine, not {_name(refinery)}; use a{method}()"
        )
    if not isinstance(refinery, QueryEngine):
        raise TypeError(
            f"{method}() needs a Refinery client or an engine with query_graph(), "
            f"not {_name(refinery)}"
        )
    return refinery


def engine(refinery: object, problems: Problems, method: str = "validate") -> Engine:
    """``refinery`` as the engine :meth:`~pyfinery.Metamodel.validate` needs: as
    :func:`query_engine`, but it must also have ``validate_graph``."""
    found = query_engine(refinery, problems, method)
    if not isinstance(found, Engine):
        raise _only_queries(method, found)
    return found


def async_query_engine(
    refinery: object, problems: Problems, method: str = "aquery"
) -> AsyncQueryEngine:
    """``refinery`` as the engine :meth:`~pyfinery.Metamodel.aquery` needs: an
    :class:`AsyncRefinery` client wrapped in an :class:`AsyncClientEngine`, an object
    with an asynchronous ``query_graph`` as it is, or one with a synchronous one
    awaited in the caller's thread."""
    if isinstance(refinery, AsyncRefinery) or _is_client(refinery, asynchronous=True):
        return AsyncClientEngine(refinery, problems)  # type: ignore[arg-type]
    if isinstance(refinery, Refinery) or _is_client(refinery, asynchronous=False):
        raise TypeError(
            f"{method}() needs an AsyncRefinery client, not a Refinery; use {method[1:]}()"
        )
    if not isinstance(refinery, AsyncQueryEngine):
        raise TypeError(
            f"{method}() needs an AsyncRefinery client or an engine with query_graph(), "
            f"not {_name(refinery)}"
        )
    if _is_async(refinery, "query_graph"):
        return refinery
    return _Awaited(refinery)  # type: ignore[arg-type]


def async_engine(refinery: object, problems: Problems, method: str = "avalidate") -> AsyncEngine:
    """``refinery`` as the engine :meth:`~pyfinery.Metamodel.avalidate` needs: as
    :func:`async_query_engine`, but it must also have ``validate_graph``."""
    found = async_query_engine(refinery, problems, method)
    target = found.engine if isinstance(found, _Awaited) else found
    if not isinstance(target, AsyncEngine):
        raise _only_queries(method, target)
    return found  # type: ignore[return-value]


def _only_queries(method: str, found: object) -> TypeError:
    return TypeError(
        f"{method}() needs an engine that validates, such as a Refinery client; "
        f"{_name(found)} only queries"
    )


def _name(obj: object) -> str:
    """How an error names what was passed: the bits engine as it always has."""
    name = type(obj).__name__
    return "the bits engine" if name == "BitsEngine" else name


def _is_client(obj: object, *, asynchronous: bool) -> bool:
    """Whether ``obj`` is not an engine but has a client's ``concretize`` and ``semantics``, as
    coroutine functions or not: such objects were accepted as clients before engines existed."""
    if hasattr(obj, "query_graph"):
        return False
    methods = [getattr(obj, name, None) for name in ("concretize", "semantics")]
    if not all(callable(method) for method in methods):
        return False
    return all(inspect.iscoroutinefunction(method) == asynchronous for method in methods)


def _is_async(obj: object, method: str) -> bool:
    return inspect.iscoroutinefunction(getattr(obj, method, None))
