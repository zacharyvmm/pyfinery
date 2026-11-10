# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Predicate values read back as :class:`~pyfinery.Match` and :class:`~pyfinery.Violation`
objects: tuples of nodes named, and the objects they stand for."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..._instances import Graph
from ..._options import LogicValue
from ..._results import Match, Violation, ViolationKind
from ..evaluation.pipeline import PredicateValues
from ..matrices.kernels import NodeIndex
from ..matrices.values import Tuples
from ..rules.syntax import Predicate as RulePredicate

if TYPE_CHECKING:
    from pydantic import BaseModel


type TuplesByValue = dict[LogicValue, Tuples]
"""The tuples of a predicate that are ``true``, ``unknown`` and ``error``."""


@dataclass(frozen=True)
class Labels:
    """The name and object (``None`` for an enum literal) of each node of the model."""

    names: list[str]
    objects: list[BaseModel | None]

    def of(self, key: Sequence[NodeIndex]) -> tuple[tuple[str, ...], tuple[BaseModel | None, ...]]:
        return tuple(self.names[i] for i in key), tuple(self.objects[i] for i in key)


def error_violations(
    values: dict[str, PredicateValues],
    tuples: dict[str, TuplesByValue],
    labels: Labels,
) -> list[Violation]:
    """The tuples of each predicate that are ``error``: of an ``error`` predicate, rule
    violations; of another, contradictions."""
    violations = []
    for name, result in values.items():
        kind: ViolationKind = "rule" if result.kind == "error" else "contradiction"
        for key in sorted_tuples(tuples[name]["error"]).tolist():
            violations.append(Violation(kind, name, *labels.of(key)))
    return violations


def selected_matches(
    graph: Graph,
    rule_predicates: dict[str, RulePredicate],
    selected: dict[str, str],
    tuples: dict[str, TuplesByValue],
    labels: Labels,
) -> list[Match]:
    """The ``true`` and ``unknown`` tuples of the selected predicates, ordered by tuple.

    As ``read_matches`` does: Refinery doesn't weaken a predicate for the existence of
    its arguments, only of nodes quantified in its body; a match whose arguments only
    may exist is ``unknown``.
    """
    maybe = np.zeros(len(labels.names), bool)
    maybe[[i for i, node in enumerate(graph.nodes) if node.exists == "unknown"]] = True
    matches = []
    for name, by_value in tuples.items():
        if name not in selected:
            continue
        parameters = tuple(p.name for p in rule_predicates[name].parameters)
        true, unknown = by_value["true"], by_value["unknown"]
        weakened = maybe[true].any(axis=1) if len(true) and true.shape[1] else None
        keys = np.concatenate([true, unknown])
        logic_values = np.array(["true"] * len(true) + ["unknown"] * len(unknown), dtype=object)
        if weakened is not None:
            logic_values[: len(true)][weakened] = "unknown"
        order = np.lexsort(keys.T[::-1]) if keys.shape[1] else np.arange(len(keys))
        for key, value in zip(keys[order].tolist(), logic_values[order].tolist(), strict=True):
            matches.append(Match(selected[name], *labels.of(key), value, parameters))
    return matches


def sorted_tuples(tuples: Tuples) -> Tuples:
    return tuples[np.lexsort(tuples.T[::-1])] if tuples.shape[1] else tuples
