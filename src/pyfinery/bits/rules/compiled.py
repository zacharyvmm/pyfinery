# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Rules compiled once, to evaluate on as many models as there are.

::

    rules = Rules.compile("pred reach(a, b) <-> edge+(a, b).")
    for model in models:  # PackedModel.build(...) of each
        evaluate(rules, model)

Compiling reads the rules and checks what doesn't depend on a model. What
does (the relations and atoms their names resolve to, and so the steps of the
evaluation) is planned for each model, which takes little next to evaluating.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..errors import RuleError, UnsupportedRuleError
from ..names import refinery_name
from .syntax import Predicate, Source, parse


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

    def select(self, names: Sequence[str] | None) -> list[Predicate]:
        """The predicates named, or else every one with a body."""
        if names is None:
            return list(self.defined.values())
        return [self.find(name) for name in names]

    def find(self, name: str) -> Predicate:
        """A predicate by name, as Refinery writes it or (if it needs quotes) unquoted,
        e.g. ``'odd name'`` or ``odd name``."""
        for candidate in (name, refinery_name(name)):
            found = self.defined.get(candidate) or self.declared.get(candidate)
            if found is not None:
                return found
        raise RuleError(f"no predicate named {name!r} in the rules")


def split_predicates(
    predicates: Iterable[Predicate],
) -> tuple[dict[str, Predicate], dict[str, Predicate]]:
    """The predicates with a body (``defined``), and without (``declared``), by name."""
    defined: dict[str, Predicate] = {}
    declared: dict[str, Predicate] = {}
    for predicate in predicates:
        if predicate.body is None:
            declared.setdefault(predicate.name, predicate)
        elif predicate.name in defined:
            raise RuleError(f"{predicate.name} is defined twice", predicate.position)
        else:
            defined[predicate.name] = predicate
    return defined, declared


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
