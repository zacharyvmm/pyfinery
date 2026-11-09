# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Rules compiled once, to evaluate on as many models as there are.

::

    rules = Rules.compile("pred reach(a, b) <-> edge+(a, b).")
    for model in models:  # PackedModel.build(...) of each
        rules.evaluate(model)

Compiling reads the rules and checks what doesn't depend on a model. What
does (the relations and atoms their names resolve to, and so the steps of the
evaluation) is planned for each model, which takes little next to evaluating.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import UnsupportedRuleError
from .evaluation import PredicateValues, evaluate_rules
from .plan import split_predicates
from .syntax import Predicate, Source, parse

if TYPE_CHECKING:
    from .packed import PackedModel


@dataclass(frozen=True)
class Rules:
    """The predicates of Refinery rules, read and checked."""

    source: Source
    defined: Mapping[str, Predicate]
    """The predicates with a body, by name."""
    declared: Mapping[str, Predicate]
    """The predicates without a body, by name: their values are a model's, or unknown."""

    @classmethod
    def compile(cls, rules: str | Sequence[str]) -> Rules:
        """Read Refinery source: statements other than ``pred`` declarations are skipped."""
        return cls.of(parse(rules if isinstance(rules, str) else "\n".join(rules)))

    @classmethod
    def of(cls, source: Source) -> Rules:
        check_not_refined(source)
        defined, declared = split_predicates(source.predicates)
        return cls(source, defined, declared)

    def evaluate(
        self, model: PackedModel, select: Sequence[str] | None = None
    ) -> dict[str, PredicateValues]:
        """The values over ``model`` of the predicates named in ``select``, or else of
        every one with a body."""
        return evaluate_rules(self, model, select)


def check_not_refined(source: Source) -> None:
    """Refuse sources where Refinery would refine a predicate beyond what its body says.

    A predicate with a body may be named in predicate bodies, as the parameter type
    of a predicate with a body, and in rules' premises; anywhere else (assertions,
    ``subsets``, reference types, parameter types of predicates without a body,
    rules' conclusions, ``scope``...) Refinery may refine it, so the source is refused.
    """
    defined = {p.name: p for p in source.predicates if p.body is not None}
    for predicate in source.predicates:
        if predicate.subsets is None:
            continue
        if predicate.body is not None:
            raise UnsupportedRuleError(
                f"subsets on a predicate with a body ({predicate.name})", predicate.position
            )
        if predicate.subsets in defined:
            raise UnsupportedRuleError(
                f"subsets of a predicate with a body ({predicate.subsets})", predicate.position
            )
    for predicate in source.predicates:
        if predicate.body is not None:
            continue  # its parameter types are literals of its body
        for parameter in predicate.parameters:
            if parameter.type in defined:
                raise UnsupportedRuleError(
                    f"a predicate with a body as a parameter type of {predicate.name}, "
                    f"which has none ({parameter.type})",
                    predicate.position,
                )
    for name in defined:
        if name in source.asserted:
            raise UnsupportedRuleError(
                f"assertions about a predicate with a body ({name})", source.asserted[name]
            )
        if name in source.in_classes:
            raise UnsupportedRuleError(
                f"a predicate with a body as the type of a reference ({name})",
                source.in_classes[name],
            )
        if name in source.mentions:
            raise UnsupportedRuleError(
                f"a predicate with a body named outside predicates and rules' premises ({name})",
                source.mentions[name],
            )
