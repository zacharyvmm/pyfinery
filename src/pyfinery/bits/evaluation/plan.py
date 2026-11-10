# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Which relations evaluating predicates computes, and in what order.

Evaluation computes the ``must`` and ``may`` values of each relation the
selected predicates read, once, in steps that each come after the values
they read:

The relations of the model are already packed (:class:`~.packed.PackedModel`).
Beyond them, there are:

- :class:`Undetermined`: a predicate without a body or values, unknown
  wherever its parameter types may hold;
- :class:`Closure`: the transitive closure ``R+`` of a relation;
- :class:`Derivation`: a predicate with a body, from its clauses.

Planning checks the rules on the way: each clause (:func:`.clauses.normalize`),
and that no predicate depends on itself, which Refinery doesn't allow.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import RuleError
from ..matrices.model import EXISTS
from ..rules.syntax import Position, Predicate
from .clauses import CallLiteral, Clause, Target, normalize, resolve

if TYPE_CHECKING:
    from ..matrices.packed import PackedModel

EXISTS_TARGET = Target("base", EXISTS)
"""Which nodes exist: what variables that aren't parameters range over."""


@dataclass(frozen=True)
class Undetermined:
    """A predicate without a body or values: unknown for the nodes of its parameter
    types, false for others, as Refinery leaves one nothing is asserted about."""

    target: Target
    predicate: Predicate
    domains: tuple[Target | None, ...]
    """Each parameter's type (``None`` for any node), computed in an earlier step."""


@dataclass(frozen=True)
class Closure:
    """``R+``: steps of ``R``, each but the last to a node that exists."""

    target: Target
    written: str
    """``R`` as the rules write it."""
    position: Position | None


@dataclass(frozen=True)
class Derivation:
    """A predicate with a body: the union of its clauses."""

    target: Target
    predicate: Predicate
    clauses: tuple[Clause, ...]
    """Its clauses as Refinery rewrites them, without those always false."""


type Step = Undetermined | Closure | Derivation


def plan(
    targets: Sequence[Target],
    model: PackedModel,
    defined: Mapping[str, Predicate],
    declared: Mapping[str, Predicate],
) -> list[Step]:
    """The steps that compute ``targets``: what they read, depth first, then themselves."""
    steps: list[Step] = []
    planned: set[Target] = set()
    active: list[str] = []  # the predicates being planned, innermost last

    def visit(target: Target, written: str, position: Position | None) -> None:
        if target in planned:
            return
        if target.transitive:
            visit(target.steps, written, position)
            steps.append(Closure(target, written, position))
        elif target.kind == "base":
            pass  # packed already
        else:
            predicate = (defined if target.kind == "derived" else declared)[target.name]
            refuse_recursion(predicate, active)
            active.append(predicate.name)
            if target.kind == "derived":
                clauses = normalized(predicate, model, defined, declared)
                for call in reading_order(clauses):
                    visit(call.target, call.atom.relation, call.atom.position)
                steps.append(Derivation(target, predicate, clauses))
            else:
                domains = parameter_types(predicate, model, defined, declared)
                for parameter, domain in zip(predicate.parameters, domains, strict=True):
                    if domain is not None:
                        visit(domain, parameter.type or "", predicate.position)
                steps.append(Undetermined(target, predicate, domains))
            active.pop()
        planned.add(target)

    for target in targets:
        visit(target, target.name, None)
    return steps


def refuse_recursion(predicate: Predicate, active: list[str]) -> None:
    if predicate.name in active:
        cycle = " -> ".join([*active[active.index(predicate.name) :], predicate.name])
        raise RuleError(
            f"recursive predicates ({cycle}): Refinery allows none; use + for transitive closure",
            predicate.position,
        )


def normalized(
    predicate: Predicate,
    model: PackedModel,
    defined: Mapping[str, Predicate],
    declared: Mapping[str, Predicate],
) -> tuple[Clause, ...]:
    """The clauses of ``predicate`` as Refinery rewrites them, without those always false."""
    assert predicate.body is not None
    clauses = [
        normalize(predicate, literals, model, defined, declared) for literals in predicate.body
    ]
    return tuple(c for c in clauses if c is not None)


def reading_order(clauses: Iterable[Clause]) -> Iterator[CallLiteral]:
    """The atoms of each clause as evaluation reads them: positive ones, then negated."""
    for clause in clauses:
        yield from (call for call in clause.calls if not call.atom.negated)
        yield from (call for call in clause.calls if call.atom.negated)


def parameter_types(
    predicate: Predicate,
    model: PackedModel,
    defined: Mapping[str, Predicate],
    declared: Mapping[str, Predicate],
) -> tuple[Target | None, ...]:
    return tuple(
        None
        if parameter.type is None
        else resolve(parameter.type, predicate.position, model, defined, declared)
        for parameter in predicate.parameters
    )
