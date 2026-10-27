# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Validate and query Pydantic object graphs with Refinery.

A :class:`Metamodel` derives a Refinery metamodel from Pydantic model classes,
and turns instances of them into Refinery facts:

- :meth:`Metamodel.validate` checks a graph against the metamodel and
  ``error pred`` rules, closing the world (anything not asserted is false)
  with Refinery's ``concretize`` operation, and reports each constraint that
  evaluates to ``error`` as a :class:`Violation`.
- :meth:`Metamodel.query` evaluates graph patterns written as Refinery
  predicates over a closed world that may hold *uncertain* facts, and
  reports each :class:`Match` with its value in Refinery's four-valued
  logic: ``"true"``, or ``"unknown"`` if it only may hold.

How fields map to Refinery:

========================================  =========================================
Python                                    Refinery
========================================  =========================================
``int`` / ``float`` / ``str`` / ``bool``  ``int`` / ``real`` / ``string`` / ``boolean``
``Literal["a", "b"]``                     an enum (single-value tags are skipped)
an :class:`enum.Enum` subclass            an enum
``Model`` / ``Model | None`` / ``list``   ``Model[1]`` / ``Model[0..1]`` / ``Model[0..*]``
``A | B``                                 an abstract superclass of ``A`` and ``B``
subclassing                               ``extends``
========================================  =========================================

Whether a field typed as a model class is containment or a plain reference
depends on the identity of that class:

- plain Pydantic models are *owned values*: each instance is a node
  contained in the object holding it;
- classes with ``ClassOptions(key=...)`` are *entities*: instances with equal
  keys are one node, and fields typed as them are references. Fields that
  hold keys rather than objects are mapped with
  ``ClassOptions(references=...)``;
- :class:`RefineryNode` subclasses are entities by object identity, for
  hand-built graphs, with :func:`Contains`, :func:`Container` and
  :class:`Ref` markers for containment, opposites, and multiplicities.

:class:`ClassOptions` configure classes from outside, so generated or
third-party models can be used unchanged; :class:`Predicate` provides
relations computed in Python (e.g. regular expressions) to rules.

This module holds :class:`Metamodel` and re-exports the rest, which lives in
private modules: the options in ``_options``, the types and fields in
``_schema``, the graph of nodes built from objects in ``_instances``, the
problems sent to Refinery in ``_render``, Refinery syntax in ``_syntax``, and
reports in ``_results``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from ._instances import build_graph, collect, link_opposites
from ._options import (
    ClassOptions,
    Container,
    Contains,
    LogicValue,
    Predicate,
    Ref,
    ReferenceKind,
    RefineryNode,
)
from ._render import (
    render_facts,
    render_metamodel,
    render_query_problem,
    render_validation_problem,
)
from ._results import (
    Match,
    ModelValidationError,
    QueryResult,
    ValidationReport,
    Violation,
    ViolationKind,
    declared_predicates,
    read_matches,
    read_violations,
)
from ._schema import (
    ClassLike,
    EnumLike,
    Feature,
    LiteralEnum,
    UnionClass,
    check,
    class_name,
    discover,
)
from ._syntax import (
    quote_id,
)
from .errors import RefineryTimeoutError
from .models import JsonOutputFormat, OutputFormats, SourceOutputFormat

if TYPE_CHECKING:
    from .client import AsyncRefinery, Refinery

__all__ = [
    "ClassLike",
    "ClassOptions",
    "Container",
    "Contains",
    "EnumLike",
    "Feature",
    "LiteralEnum",
    "LogicValue",
    "Match",
    "Metamodel",
    "ModelValidationError",
    "Predicate",
    "QueryResult",
    "Ref",
    "ReferenceKind",
    "RefineryNode",
    "UnionClass",
    "ValidationReport",
    "Violation",
    "ViolationKind",
    "quote_id",
]


class Metamodel:
    """A Refinery metamodel derived from :class:`RefineryNode` subclasses
    (and the plain Pydantic models nested in them).

    >>> metamodel = Metamodel(Filesystem, rules=[
    ...     "error pred emptyDir(Dir d) <-> !contents(d, _).",
    ... ])
    >>> print(metamodel.source)
    >>> report = metamodel.validate([my_filesystem], refinery)

    Classes reachable from the given ones are included automatically: their
    superclasses, the classes their fields refer to, and the already-imported
    subclasses of those (but not every subclass of a shared base class).

    :param options: :class:`ClassOptions` by class.
    :param unions: Names for the abstract superclasses standing for unions,
        e.g. ``{"Component": [Service, Database]}``.
    :param unsupported: ``"skip"`` to leave out fields with no Refinery
        counterpart instead of raising ``TypeError``.
    """

    def __init__(
        self,
        *classes: type[BaseModel],
        rules: Iterable[str] = (),
        options: Mapping[type, ClassOptions] | None = None,
        unions: Mapping[str, Iterable[type[BaseModel]]] | None = None,
        unsupported: Literal["error", "skip"] = "error",
    ) -> None:
        if not classes:
            raise ValueError("Metamodel needs at least one model class")
        self.rules = list(rules)
        self.schema = discover(classes, options, unions, unsupported)
        check(self.schema)
        self.source = render_metamodel(self.schema, self.rules)

    @property
    def options(self) -> dict[type, ClassOptions]:
        return self.schema.options

    @property
    def classes(self) -> list[type[BaseModel]]:
        return self.schema.classes

    @property
    def unions(self) -> list[UnionClass]:
        return self.schema.unions

    @property
    def enums(self) -> list[EnumLike]:
        return self.schema.enums

    @property
    def features(self) -> dict[type[BaseModel], list[Feature]]:
        return self.schema.features

    @property
    def skipped(self) -> list[tuple[type[BaseModel], str, str]]:
        """Fields left out of the metamodel: ``(class, field, reason)``."""
        return self.schema.skipped

    @staticmethod
    def class_name(node_class: type | UnionClass | LiteralEnum) -> str:
        """The Refinery name of a class, union, or enum."""
        return class_name(node_class)

    # -- instances ----------------------------------------------------------

    def collect(self, objects: Iterable[BaseModel]) -> list[BaseModel]:
        """All nodes reachable from ``objects`` through reference fields.

        Only outgoing references are followed: a node that is referenced by
        nothing (e.g., a child that only sets its own ``parent``) must be
        passed in ``objects`` itself.
        """
        return collect(self.schema, objects)

    def identity(self, node: BaseModel) -> Hashable:
        """What makes ``node`` one Refinery node: its key if its class has one, else itself.

        Of several instances with the same key, the first one reached
        represents them all (its fields are the ones asserted).
        """
        return self.schema.identity(node)

    def link_opposites(self, objects: Iterable[BaseModel]) -> int:
        """Fill in the missing side of every ``opposite`` pair in the graph.

        For example, after ``home.contents.append(doc)`` this sets
        ``doc.parent = home``, and after ``doc.parent = home`` it appends
        ``doc`` to ``home.contents``. Refinery infers missing sides on its own,
        so this is only needed to keep the Python objects consistent.

        Single-valued fields that already point at a different node are left
        unchanged; :meth:`validate` reports such conflicts. Returns the number
        of references added.
        """
        return link_opposites(self.schema, objects)

    def problem(self, objects: Iterable[BaseModel]) -> str:
        """The metamodel, rules, and the facts of the graph reachable from ``objects``."""
        return self.source + "\n" + render_facts(self.schema, build_graph(self.schema, objects))

    # -- validation ---------------------------------------------------------

    def validate(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Check the graph reachable from ``objects`` against the metamodel and rules.

        ``rules`` (Refinery source, typically ``error pred`` declarations) are added to
        those given to the :class:`Metamodel`, and ``predicates`` provide
        relations computed in Python for them to use.

        Validation is two-valued: it checks what certainly holds. Facts that
        are only ``"unknown"`` (from :class:`Predicate` or
        ``ClassOptions(logic_value=...)``) count as not holding, and every
        object exists (``ClassOptions(exists=...)`` is ignored). Use
        :meth:`query` to reason about what may hold.

        Only nodes reachable from ``objects`` through outgoing references are
        checked (see :meth:`collect`). Raises
        :class:`~pyfinery.InvalidProblemError` if the rules don't compile;
        constraint violations are returned in the report.
        """
        predicates = list(predicates)
        graph = build_graph(self.schema, objects)
        source = render_validation_problem(self.schema, self.source, graph, rules, predicates)
        with _explain_timeout():
            result = refinery.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    async def avalidate(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Asynchronous variant of :meth:`validate`."""
        predicates = list(predicates)
        graph = build_graph(self.schema, objects)
        source = render_validation_problem(self.schema, self.source, graph, rules, predicates)
        with _explain_timeout():
            result = await refinery.concretize(source, format=_JSON_ONLY)
        return read_violations(result.json_, graph, source)

    # -- queries ------------------------------------------------------------

    def query(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Evaluate graph patterns over the graph reachable from ``objects``.

        ``rules`` are Refinery predicates, e.g.
        ``"pred reaches(Service s, Database d) <-> Service::depends_on+(s, d)."``;
        every ``pred`` they declare is reported unless ``select`` names the
        ones to report. Unlike :meth:`validate`, the world is closed by
        asserting that there is nothing else (no other objects, references,
        or subclass memberships) while keeping uncertainty: objects whose
        ``exists`` is ``"unknown"`` and :class:`Predicate` facts that are
        ``"unknown"`` make the matches that depend on them ``"unknown"``.
        """
        rules, predicates = list(rules), list(predicates)
        graph = build_graph(self.schema, objects, existence=True)
        source = render_query_problem(self.schema, self.source, graph, rules, predicates)
        selected = list(select) if select is not None else declared_predicates(rules)
        with _explain_timeout():
            result = refinery.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, selected)

    async def aquery(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Asynchronous variant of :meth:`query`."""
        rules, predicates = list(rules), list(predicates)
        graph = build_graph(self.schema, objects, existence=True)
        source = render_query_problem(self.schema, self.source, graph, rules, predicates)
        selected = list(select) if select is not None else declared_predicates(rules)
        with _explain_timeout():
            result = await refinery.semantics(source, format=_JSON_ONLY)
        return read_matches(result.json_, graph, source, selected)

    def query_problem(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> str:
        """The problem :meth:`query` would send, e.g. to debug it in the web editor."""
        rules, predicates = list(rules), list(predicates)
        graph = build_graph(self.schema, objects, existence=True)
        return render_query_problem(self.schema, self.source, graph, rules, predicates)


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
