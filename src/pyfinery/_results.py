# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""What validation and queries report: violations and matches."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from ._instances import Graph
from ._options import LogicValue
from ._syntax import is_single_id, quote_id, unquote_id
from .errors import RefineryError
from .http.models import JsonOutput

ViolationKind = Literal["rule", "multiplicity", "containment", "contradiction", "reference"]


@dataclass(frozen=True, slots=True)
class Violation:
    """A constraint that the graph violates.

    ``kind`` is ``"rule"`` for user-defined error predicates,
    ``"multiplicity"`` for reference multiplicities, ``"containment"`` for
    containment hierarchy errors (missing or multiple containers, cycles),
    ``"contradiction"`` for facts that contradict the metamodel (e.g., a
    reference to a node of the wrong class), and ``"reference"`` for keys in
    reference-by-key fields that match no object (``...::unresolved``, with
    the key as second name) or several (``...::ambiguous``).
    """

    kind: ViolationKind
    constraint: str
    """Refinery relation that reported the error, e.g. ``Dir::contents::invalidMultiplicity``."""
    node_names: tuple[str, ...]
    nodes: tuple[BaseModel | None, ...] = field(compare=False)
    """The Python objects, or ``None`` for nodes Refinery added (e.g., enum literals)."""

    def __str__(self) -> str:
        return f"{self.kind}: {self.constraint}({', '.join(self.node_names)})"


class ModelValidationError(RefineryError):
    """Raised by :meth:`ValidationReport.raise_for_violations`."""

    result = "modelValidation"

    def __init__(self, violations: Sequence[Violation]) -> None:
        lines = [f"{len(violations)} Refinery constraint violation(s):"]
        lines += [f"  {violation}" for violation in violations]
        super().__init__("\n".join(lines))
        self.violations = list(violations)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    violations: list[Violation]
    source: str
    """The problem sent to Refinery (metamodel, rules, and facts)."""
    model: JsonOutput | None
    identity: Callable[[BaseModel], Hashable] = field(default=id, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok

    def for_node(self, node: BaseModel) -> list[Violation]:
        """Violations that involve ``node`` (or another instance with the same key)."""
        wanted = self.identity(node)
        return [
            v
            for v in self.violations
            if any(n is not None and self.identity(n) == wanted for n in v.nodes)
        ]

    def raise_for_violations(self) -> None:
        if self.violations:
            raise ModelValidationError(self.violations)


@dataclass(frozen=True, slots=True)
class Match:
    """A way a query predicate holds: its arguments, and its logic value.

    ``"true"`` if it certainly holds, ``"unknown"`` if it only may.
    """

    predicate: str
    node_names: tuple[str, ...]
    nodes: tuple[BaseModel | None, ...] = field(compare=False)
    """The Python objects, or ``None`` for nodes Refinery added (e.g., enum literals)."""
    logic_value: LogicValue = "true"
    parameters: tuple[str, ...] = field(default=(), compare=False)
    """The names of the predicate's parameters, as declared."""

    def __str__(self) -> str:
        return f"{self.logic_value}: {self.predicate}({', '.join(self.node_names)})"

    @property
    def arguments(self) -> dict[str, BaseModel | None]:
        """The Python objects by the name of the parameter they are bound to.

        Empty if Refinery didn't report the parameters' names.
        """
        return dict(zip(self.parameters, self.nodes, strict=True)) if self.parameters else {}

    def __getitem__(self, parameter: str) -> BaseModel | None:
        """The Python object bound to ``parameter``."""
        try:
            return self.nodes[self.parameters.index(parameter)]
        except ValueError:
            raise KeyError(f"{self.predicate} has no parameter named {parameter!r}") from None


@dataclass(frozen=True, slots=True)
class QueryResult:
    matches: list[Match]
    errors: list[Violation]
    """Constraints of the metamodel the graph violates (as reported by :meth:`Metamodel.validate`),
    and contradictions between facts."""
    source: str
    """The problem sent to Refinery."""
    model: JsonOutput | None

    def __getitem__(self, predicate: str) -> list[Match]:
        """The matches of ``predicate``."""
        return [m for m in self.matches if m.predicate == predicate]

    def true(self, predicate: str | None = None) -> list[Match]:
        """Matches that are true, that certainly hold (of ``predicate``, or of any)."""
        return [
            m for m in self.matches if m.logic_value == "true" and predicate in (None, m.predicate)
        ]

    def unknown(self, predicate: str | None = None) -> list[Match]:
        """Matches that are unknown, that only may hold (of ``predicate``, or of any)."""
        return [
            m
            for m in self.matches
            if m.logic_value == "unknown" and predicate in (None, m.predicate)
        ]


class _DeclaredName(str):
    """A predicate name already decoded from a ``pred`` declaration, never unquoted again."""


_PRED_DECLARATION = re.compile(r"\bpred\s+([A-Za-z_][A-Za-z0-9_]*|'(?:[^'\\]|\\.)*')")


def declared_predicates(rules: Iterable[str]) -> list[str]:
    """Names of the predicates declared in ``rules`` (``error`` predicates included)."""
    names: list[str] = []
    for rule in rules:
        for match in _PRED_DECLARATION.finditer(rule):
            name = _DeclaredName(unquote_id(match.group(1)))
            if name not in names:
                names.append(name)
    return names


class PredicateIndex[T]:
    """O(1) predicate lookup built once per query, keeping qualified and single-segment names
    separate.

    Single-segment relations (plain identifiers and single quoted tokens such as
    ``'builtin::exists'`` or ``'he said \\"yes\\"'``) are indexed by their decoded name
    (``unquote_id``). Qualified names (``builtin::exists``, ``'My Class'::'my field'``) stay
    only in ``exact`` so they never collide with single-segment quoted predicates.
    """

    __slots__ = ("exact", "single")

    def __init__(self, relations: Mapping[str, T]) -> None:
        self.exact = relations
        self.single: dict[str, T] = {}
        for key, value in relations.items():
            if is_single_id(key):
                self.single[unquote_id(key)] = value

    def resolve(self, name: str) -> T | None:
        """Resolve ``name``: decoded ``_DeclaredName``s match only ``single``; explicit selectors
        match verbatim first, then decoded single-segment names, then decoded Xtext escapes."""
        if isinstance(name, _DeclaredName):
            return self.single.get(name)
        found = self.exact.get(name)
        if found is not None:
            return found
        found = self.single.get(name)
        if found is not None:
            return found
        if is_single_id(name) and name.startswith("'"):
            return self.single.get(unquote_id(name))
        return None


def reference_errors(graph: Graph) -> list[Violation]:
    """Keys in reference-by-key fields that match no object, or several."""
    errors = []
    for node in graph.nodes:
        for value in node.values:
            if value.unresolved is None:
                continue
            feature = f"{quote_id(value.feature.owner.__name__)}::{quote_id(value.feature.name)}"
            constraint = f"{feature}::{value.unresolved}"
            node_names = (node.name, repr(value.value))
            errors.append(Violation("reference", constraint, node_names, (node.obj, None)))
    return errors
