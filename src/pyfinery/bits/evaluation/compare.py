# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Comparisons of attribute values and constants, as Refinery's ``logic`` compares them.

A number is an interval (``IntInterval``, ``RealInterval``), whose error is
the empty interval ``[inf, -inf]``; a string is known, unknown or an error
(``StringValue``). A comparison must hold if it holds for every pair of
values the operands may take, and may hold if it holds for some.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..errors import RuleError, UnsupportedRuleError
from ..matrices import kernels as k
from ..matrices.batch import ERROR_STRING, BatchAttribute
from ..matrices.kernels import NodeIndex
from ..matrices.model import Numbers, real_interval
from ..matrices.values import Size, Value
from ..rules.syntax import Comparison, Modality
from .solve import Factor, VarId

type Truths = NDArray[np.bool_]
"""Whether something holds, in each graph, for each value of the compared variables
(broadcast over them): shape ``(graphs, n, ...)``, a value per variable."""


class Holds(NamedTuple):
    """Where a comparison must hold, and where it may."""

    must: Truths
    may: Truths

    def negated(self) -> Holds:
        """``not``: must hold where this may not, and may where this must not."""
        return Holds(~self.may, ~self.must)


class Interval(NamedTuple):
    """The bounds of the numbers a side may be, broadcast over the compared variables."""

    low: Numbers
    high: Numbers


@dataclass(frozen=True)
class Side:
    """One side of a comparison: an attribute's value (of a variable or a node), or a constant."""

    variable: VarId | None
    node: NodeIndex | None
    constant: object
    attribute: BatchAttribute | None

    @property
    def is_string(self) -> bool | None:
        if self.attribute is not None:
            return self.attribute.data_type == "string"
        return isinstance(self.constant, str)


def compare(
    left: Side,
    right: Side,
    comparison: Comparison,
    modality: Modality,
    size: Size,
    strings: Mapping[str, int],
) -> Factor:
    """The tuples of the compared variables for which ``comparison`` holds in ``modality``,
    in each graph; ``strings`` are the codes of the batch's strings."""
    variables = tuple(dict.fromkeys(v for v in (left.variable, right.variable) if v is not None))
    kinds = {left.is_string, right.is_string} - {None}
    if len(kinds) > 1:
        raise RuleError("a comparison of a string and a number", comparison.position)
    booleans = any(
        s.attribute is not None and s.attribute.data_type == "boolean" for s in (left, right)
    )
    if booleans and comparison.operator not in ("==", "!="):
        raise UnsupportedRuleError("ordering booleans", comparison.position)
    if kinds == {True}:
        holds = compare_strings(left, right, variables, comparison, size, strings)
    else:
        holds = compare_numbers(left, right, variables, comparison, size)
    if comparison.negated:
        holds = holds.negated()
    truths = holds.must if modality == "must" else holds.may
    return Factor(variables, as_value(truths, len(variables), size))


# -- numbers --------------------------------------------------------------------


def compare_numbers(
    left: Side, right: Side, variables: tuple[VarId, ...], comparison: Comparison, size: Size
) -> Holds:
    """``a <= b`` may hold if ``lo(a) <= hi(b)``, and must if ``hi(a) <= lo(b)``;
    ``a == b`` is ``a <= b`` and ``b <= a``; ``a < b`` is ``not (b <= a)``.
    """
    a_low, a_high, b_low, b_high = exact_together(
        *interval(left, variables, size), *interval(right, variables, size)
    )
    a, b = Interval(a_low, a_high), Interval(b_low, b_high)

    def less_eq(x: Interval, y: Interval) -> Holds:
        return Holds(must=np.asarray(x.high <= y.low, bool), may=np.asarray(x.low <= y.high, bool))

    def equal() -> Holds:
        ab, ba = less_eq(a, b), less_eq(b, a)
        return Holds(ab.must & ba.must, ab.may & ba.may)

    match comparison.operator:
        case "<=":
            return less_eq(a, b)
        case ">=":
            return less_eq(b, a)
        case "<":
            return less_eq(b, a).negated()
        case ">":
            return less_eq(a, b).negated()
        case "==":
            return equal()
        case "!=":
            return equal().negated()
    raise RuleError(f"unknown operator {comparison.operator}", comparison.position)


def interval(side: Side, variables: tuple[VarId, ...], size: Size) -> Interval:
    if side.attribute is None:
        value = side.constant
        if isinstance(value, bool):
            value = int(value)
        low: int | Decimal
        high: int | Decimal
        if isinstance(value, Decimal):
            low, high = real_interval(value)  # a real constant, as RealInterval.of rounds it
        elif isinstance(value, int):
            low = high = value
        else:
            raise RuleError(f"not a number: {value!r}")
        # Floats, unless they can't hold the numbers exactly.
        if float(low) == low and float(high) == high:
            return Interval(np.asarray(float(low)), np.asarray(float(high)))
        return Interval(np.asarray(low, dtype=object), np.asarray(high, dtype=object))
    lower, upper = side.attribute.lower, side.attribute.upper
    shape = broadcast_shape(side, variables, size)
    if side.node is not None:
        return Interval(lower[:, side.node].reshape(shape), upper[:, side.node].reshape(shape))
    return Interval(lower.reshape(shape), upper.reshape(shape))


def exact_together(*arrays: Numbers) -> tuple[Numbers, ...]:
    """The arrays, as exact Python numbers if any of them needs to be."""
    if any(array.dtype == object for array in arrays):
        return tuple(array.astype(object) for array in arrays)
    return arrays


# -- strings --------------------------------------------------------------------

type Codes = NDArray[np.intp]
"""Strings as numbers: equal if the strings are, else negative: unknown
(:data:`~..matrices.batch.UNKNOWN_STRING`) or an error (``ERROR_STRING``)."""


def compare_strings(
    left: Side,
    right: Side,
    variables: tuple[VarId, ...],
    comparison: Comparison,
    size: Size,
    strings: Mapping[str, int],
) -> Holds:
    """A comparison of strings is an error if either is (``StringValue.checkEquals``)."""
    if comparison.operator not in ("==", "!="):
        raise RuleError("strings can only be compared with == and !=", comparison.position)
    others: dict[str, int] = {}  # constants no attribute has: codes of their own

    def encode(side: Side) -> Codes:
        attribute = side.attribute
        if attribute is None:
            text = str(side.constant)
            code = strings.get(text)
            if code is None:
                code = others.setdefault(text, len(strings) + len(others))
            return np.asarray(code)
        assert attribute.codes is not None
        shape = broadcast_shape(side, variables, size)
        if side.node is not None:
            return attribute.codes[:, side.node].reshape(shape)
        return attribute.codes.reshape(shape)

    a, b = encode(left), encode(right)
    known = (a >= 0) & (b >= 0)
    equal = known & (a == b)
    if comparison.operator == "==":
        holds = Holds(must=equal, may=~known | (a == b))
    else:
        holds = Holds(must=known & (a != b), may=~equal)
    error = (a == ERROR_STRING) | (b == ERROR_STRING)
    return Holds(holds.must | error, holds.may & ~error)


# -- shapes ---------------------------------------------------------------------


def broadcast_shape(side: Side, variables: tuple[VarId, ...], size: Size) -> tuple[int, ...]:
    """Where a side's values go, to broadcast over the graphs and the compared variables."""
    shape = [size.graphs] + [1] * len(variables)
    if side.variable is not None:
        shape[1 + variables.index(side.variable)] = size.nodes
    return tuple(shape)


def as_value(truths: Truths, arity: int, size: Size) -> Value:
    """The tuples of nodes where ``truths`` holds, broadcast to every graph and every tuple
    of ``arity``."""
    shape = (size.graphs,) + (size.nodes,) * arity
    truths = k.asarray(np.ascontiguousarray(np.broadcast_to(truths, shape)))  # to the device
    if arity == 0:
        return Value(0, size, truths)
    return Value(arity, size, k.pack_padded(truths, size.nodes))
