"""The engines that evaluate :meth:`~pyfinery.Metamodel.validate` and
:meth:`~pyfinery.Metamodel.query`, and how what the user passes becomes one.

An engine is given what the metamodel knows, never a rendered problem: the
:class:`~pyfinery._schema.Schema`, the :class:`~pyfinery._instances.Graph` built
from the objects, the rules (the metamodel's own, then those of the call), the
:class:`~pyfinery.Predicate` objects, and, to query, the names of the predicates
to report. An :class:`Engine` validates and queries; an :class:`AsyncEngine` is
its asynchronous variant.

:func:`sync_engine` and :func:`async_engine` turn what the user passed into the
engine a method needs, or raise :class:`TypeError`: a Refinery client becomes a
:class:`~pyfinery.http.ClientEngine` or :class:`~pyfinery.http.AsyncClientEngine`,
an engine is used as it is, and a synchronous engine runs in asynchronous methods.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ._instances import Graph
from ._options import Predicate
from ._results import QueryResult, ValidationReport
from ._schema import Schema
from .http.client import AsyncRefinery, Refinery
from .http.engine import AsyncClientEngine, ClientEngine, Problems

__all__ = ["AsyncEngine", "Engine", "async_engine", "sync_engine"]

_METHODS = ("validate_graph", "query_graph")


@runtime_checkable
class Engine(Protocol):
    """Evaluates :meth:`~pyfinery.Metamodel.validate` and :meth:`~pyfinery.Metamodel.query`."""

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
class AsyncEngine(Protocol):
    """Asynchronous variant of :class:`Engine`."""

    async def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport: ...

    async def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult: ...


class _Awaited:
    """A synchronous engine as an asynchronous one: it runs in the caller's thread."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    async def validate_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
    ) -> ValidationReport:
        return self.engine.validate_graph(schema, graph, rules, predicates)

    async def query_graph(
        self,
        schema: Schema,
        graph: Graph,
        rules: Sequence[str],
        predicates: Sequence[Predicate],
        select: Sequence[str],
    ) -> QueryResult:
        return self.engine.query_graph(schema, graph, rules, predicates, select)


def sync_engine(backend: object, source: str, rules: Sequence[str], method: str) -> Engine:
    """``backend`` as the engine :meth:`~pyfinery.Metamodel.validate` and
    :meth:`~pyfinery.Metamodel.query` need: a :class:`Refinery` client wrapped in a
    :class:`ClientEngine` that renders problems from the metamodel's ``source`` and
    ``rules``, or a synchronous :class:`Engine` as it is."""
    if isinstance(backend, Refinery) or _is_client(backend, asynchronous=False):
        return ClientEngine(backend, Problems(source, rules))  # type: ignore[arg-type]
    if isinstance(backend, AsyncRefinery) or _is_client(backend, asynchronous=True):
        raise TypeError(
            f"{method}() needs a synchronous engine, not {_name(backend)}; use a{method}()"
        )
    if not _is_engine(backend):
        raise TypeError(
            f"{method}() needs a Refinery client or an engine with validate_graph() and "
            f"query_graph(), not {_name(backend)}"
        )
    if _is_async(backend, _operation(method)):
        raise TypeError(
            f"{method}() needs a synchronous engine, not {_name(backend)}; use a{method}()"
        )
    return backend  # type: ignore[return-value]


def async_engine(backend: object, source: str, rules: Sequence[str], method: str) -> AsyncEngine:
    """``backend`` as the engine :meth:`~pyfinery.Metamodel.avalidate` and
    :meth:`~pyfinery.Metamodel.aquery` need: an :class:`AsyncRefinery` client wrapped
    in an :class:`AsyncClientEngine`, an :class:`AsyncEngine` as it is, or a
    synchronous :class:`Engine` awaited in the caller's thread."""
    if isinstance(backend, AsyncRefinery) or _is_client(backend, asynchronous=True):
        return AsyncClientEngine(backend, Problems(source, rules))  # type: ignore[arg-type]
    if isinstance(backend, Refinery) or _is_client(backend, asynchronous=False):
        raise TypeError(
            f"{method}() needs an AsyncRefinery client, not a Refinery; use {method[1:]}()"
        )
    if not _is_engine(backend):
        raise TypeError(
            f"{method}() needs an AsyncRefinery client or an engine with validate_graph() and "
            f"query_graph(), not {_name(backend)}"
        )
    if _is_async(backend, _operation(method)):
        return backend  # type: ignore[return-value]
    return _Awaited(backend)  # type: ignore[arg-type]


def _name(obj: object) -> str:
    """How an error names what was passed."""
    return type(obj).__name__


def _operation(method: str) -> str:
    return "validate_graph" if method in ("validate", "avalidate") else "query_graph"


def _is_engine(obj: object) -> bool:
    return all(callable(getattr(obj, name, None)) for name in _METHODS)


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
