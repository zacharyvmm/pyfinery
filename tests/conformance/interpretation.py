# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Reading values from a partial model, and checking expectations against them.

The checks follow Refinery's ``SemanticsExpectation`` and
``FilteredInterpretation`` (commit ``7470710``): a tuple with an object that
doesn't exist has its relation's default value, and an actual value passes if
it refines the expected one (or, for ``EXPECT EXACTLY``, equals it).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction

from pyfinery.models import ErrorValue, JsonOutput, RelationMetadata, TupleElement, UnknownValue

from .problems import Assertion, Expectation

_TRUTH_VALUES = ("true", "false", "unknown", "error")


class Interpretation:
    """The values of a partial (or candidate) model, by node and relation name."""

    def __init__(self, model: JsonOutput) -> None:
        self.model = model
        self._nodes: dict[str, int] = {}
        simple: dict[str, list[int]] = {}
        for index, node in enumerate(model.nodes):
            self._nodes[node.name] = index
            simple.setdefault(node.simple_name, []).append(index)
        for name, indices in simple.items():
            if len(indices) == 1:
                self._nodes.setdefault(name, indices[0])
        self._values: dict[str, dict[tuple[int, ...], TupleElement]] = {}
        for relation in model.relations:
            self._values[relation.name] = {
                tuple(raw[: relation.arity]): raw[relation.arity]  # type: ignore[misc]
                for raw in model.partial_interpretation.get(relation.name, [])
            }
        # Nodes that don't exist: ``builtin::exists`` lists the others.
        exists = self._values.get("builtin::exists")
        self._missing = (
            set()
            if exists is None
            else {i for i in range(len(model.nodes)) if exists.get((i,), "false") == "false"}
        )

    def node(self, name: str) -> int:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(f"no node named {name!r}") from None

    def relation(self, name: str) -> RelationMetadata:
        if name in ("exists", "equals"):
            name = f"builtin::{name}"
        return self.model.relation(name)

    def exists(self, node: int) -> bool:
        """Whether the node may exist: Refinery filters out tuples of nodes that don't."""
        return node not in self._missing

    def value(self, relation: RelationMetadata, key: tuple[int, ...]) -> TupleElement:
        if not all(self.exists(node) for node in key):
            return default(relation)
        return self._values[relation.name].get(key, default(relation))

    def items(self, relation: RelationMetadata) -> Iterator[tuple[tuple[int, ...], TupleElement]]:
        """The tuples the output lists, of nodes that may exist."""
        for key, value in self._values[relation.name].items():
            if all(self.exists(node) for node in key):
                yield key, value


def default(relation: RelationMetadata) -> TupleElement:
    """What a tuple the output doesn't list has: false, or for an attribute, error.

    Refinery lists an attribute's value for the objects of its class (unknown if
    nothing is asserted); on other objects, the attribute has no value, and
    comparing it is an error.
    """
    if relation.detail.type == "attribute":
        return "error"
    return "false"


# -- refinement ---------------------------------------------------------------


Number = int | Fraction | float  # exact, as Refinery's BigInteger and BigDecimal; or infinite


@dataclass(frozen=True)
class _Interval:
    lower: Number
    upper: Number

    @property
    def empty(self) -> bool:
        return self.lower > self.upper


def _number(text: str, real: bool, rounding: str) -> Number:
    text = text.strip()
    if real:
        # RealInterval.of rounds bounds to 16 significant digits, outward.
        return Fraction(Context(prec=16, rounding=rounding).plus(Decimal(text)))
    try:
        return int(text)
    except ValueError:
        return Fraction(text)


def _interval(value: str, real: bool = False) -> _Interval | None:
    """An ``int`` or ``real`` value as an exact interval; ``None`` if it isn't a number.

    Reals are rounded as Refinery's ``RealInterval.of`` rounds them.
    """
    text = value.strip()
    if text == "unknown":
        return _Interval(-math.inf, math.inf)
    if text == "error":
        return _Interval(math.inf, -math.inf)
    lower, separator, upper = text.partition("..")
    if not separator:
        lower = upper = text
    try:
        low = -math.inf if lower.strip() == "*" else _number(lower, real, ROUND_FLOOR)
        high = math.inf if upper.strip() == "*" else _number(upper, real, ROUND_CEILING)
    except (ValueError, ArithmeticError):
        return None
    return _Interval(low, high)


def _string(value: str) -> str | None:
    """The value of a string literal as Refinery writes it, e.g. ``"it\\'s"``; ``None``
    if it isn't one."""
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return None
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}

    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped.startswith("u"):
            return chr(int(escaped[1:], 16))
        return escapes.get(escaped, escaped)

    return re.sub(r"\\(u[0-9a-fA-F]{4}|.)", replace, value[1:-1])


def _text(value: TupleElement) -> str:
    match value:
        case UnknownValue():
            return value.unknown
        case ErrorValue():
            return "error"
        case _:
            return str(value)


def refines(actual: TupleElement, expected: str, *, exact: bool, real: bool = False) -> bool:
    """Whether the actual value is at least as precise as the expected one (or equal to it).

    ``error`` refines every value, and ``unknown`` is refined by every value.
    """
    text = _text(actual)
    if expected in _TRUTH_VALUES and text in _TRUTH_VALUES:
        if exact:
            return text == expected
        return expected == "unknown" or text == expected or text == "error"
    if not exact and (text == "error" or expected == "unknown"):
        return True
    actual_interval, expected_interval = _interval(text, real), _interval(expected, real)
    if actual_interval is not None and expected_interval is not None:
        if exact:
            return actual_interval == expected_interval or (
                actual_interval.empty and expected_interval.empty
            )
        return (
            expected_interval.lower <= actual_interval.lower
            and actual_interval.upper <= expected_interval.upper
        )
    actual_string, expected_string = _string(text), _string(expected)
    if actual_string is not None and expected_string is not None:
        return actual_string == expected_string
    return text == expected


# -- expectations -------------------------------------------------------------


def check(expectation: Expectation, interpretation: Interpretation) -> list[str]:
    """The assertions of ``expectation`` that don't hold, as messages."""
    return [
        message
        for assertion in expectation.assertions
        if (message := _check(assertion, expectation, interpretation)) is not None
    ]


def _check(
    assertion: Assertion, expectation: Expectation, interpretation: Interpretation
) -> str | None:
    relation = interpretation.relation(assertion.relation)
    if len(assertion.arguments) != relation.arity:
        return f"line {assertion.line}: {assertion.source} has the wrong arity"
    fixed = [None if a is None else interpretation.node(a) for a in assertion.arguments]
    if None not in fixed:
        key = tuple(fixed)  # type: ignore[arg-type]
        actual = interpretation.value(relation, key)  # type: ignore[arg-type]
        if refines(actual, assertion.value, exact=expectation.exact, real=_real(relation)):
            return None
        return _message(assertion, expectation, f"was {_text(actual)} instead")
    # A wildcard: every listed tuple that matches the given nodes.
    for key, actual in interpretation.items(relation):
        matches = all(f is None or f == k for f, k in zip(fixed, key, strict=True))
        real = _real(relation)
        if matches and not refines(actual, assertion.value, exact=expectation.exact, real=real):
            names = ", ".join(interpretation.model.nodes[i].name for i in key)
            return _message(assertion, expectation, f"failed for ({names}): {_text(actual)}")
    return None


def _real(relation: RelationMetadata) -> bool:
    return relation.data_type == "builtin::real"


def _message(assertion: Assertion, expectation: Expectation, detail: str) -> str:
    header = "EXPECT"
    if expectation.concreteness == "candidate":
        header += " CANDIDATE"
    if expectation.exact:
        header += " EXACTLY"
    description = f" ({expectation.description})" if expectation.description else ""
    return f"{header} {assertion.source}{description} {detail} in line {assertion.line}"


def errors(interpretation: Interpretation) -> list[str]:
    """Where the model is inconsistent: tuples whose value is ``error``."""
    found = []
    for relation in interpretation.model.relations:
        for key, value in interpretation.items(relation):
            if _text(value) == "error":
                names = ", ".join(interpretation.model.nodes[i].name for i in key)
                found.append(f"{relation.name}({names})")
    return found
