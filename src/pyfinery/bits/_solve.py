# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Joining the literals of a clause, and projecting out its existential variables.

A clause is a conjunction of *factors*, each a set of tuples over some of its
variables. Solving it computes the tuples of the head variables for which
the other variables have values that satisfy every factor.

Factors over at most two variables are packed bits. Variables are eliminated
one at a time, fewest neighbours first: those of a variable with neighbours
``a`` and ``b`` combine into one matrix product over ``(a, b)``; with one
neighbour, into "has a pair in this row". That needs no relation over three
variables. When one would be needed, or a factor has three or more
variables, or the head does, the solver loops over the values one variable
can take, and solves the rest for each.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import _kernels as k
from ._values import (
    Value,
    complement,
    empty,
    from_tuples,
    full,
    intersection,
    is_empty,
    to_tuples,
    transpose,
    union,
)


@dataclass(frozen=True)
class Factor:
    """A set of tuples over ``variables`` (distinct), or with ``negated``, its complement.

    Only factors over three or more variables are ``negated``: the others are
    complemented when made.
    """

    variables: tuple[int, ...]
    value: Value
    negated: bool = False


def make_factor(variables: tuple[int, ...], value: Value, negated: bool = False) -> Factor:
    if negated and value.arity <= 2:
        return Factor(variables, complement(value))
    return Factor(variables, value, negated)


@dataclass(frozen=True)
class Var:
    id: int


@dataclass(frozen=True)
class Node:
    index: int


Argument = Var | Node


def bind(value: Value, arguments: Sequence[Argument], negated: bool = False) -> Factor:
    """The factor of a relation's ``value`` at ``arguments``: nodes select tuples,
    a repeated variable keeps the tuples whose places agree."""
    assert len(arguments) == value.arity
    variables: list[int] = []
    for argument in arguments:
        if isinstance(argument, Var) and argument.id not in variables:
            variables.append(argument.id)
    if (
        value.arity <= 2
        and all(isinstance(a, Var) for a in arguments)
        and len(variables) == len(arguments)
    ):
        return make_factor(tuple(variables), value, negated)
    if value.arity == 2:
        return make_factor(tuple(variables), _bind_pair(value, arguments), negated)
    if value.arity == 1:  # a node
        assert isinstance(arguments[0], Node)
        node = arguments[0].index
        return make_factor((), Value(0, value.n, k.contains(value.data, node)), negated)
    tuples = to_tuples(value)
    keep = np.ones(len(tuples), bool)
    first: dict[int, int] = {}
    for place, argument in enumerate(arguments):
        if isinstance(argument, Node):
            keep &= tuples[:, place] == argument.index
        elif argument.id in first:
            keep &= tuples[:, place] == tuples[:, first[argument.id]]
        else:
            first[argument.id] = place
    projected = tuples[keep][:, [first[v] for v in variables]]
    return make_factor(tuple(variables), from_tuples(len(variables), value.n, projected), negated)


def _bind_pair(value: Value, arguments: Sequence[Argument]) -> Value:
    left, right = arguments
    matrix, n = value.data, value.n
    if isinstance(left, Node) and isinstance(right, Node):
        return Value(0, n, k.contains(matrix[left.index], right.index))
    if isinstance(left, Node):
        return Value(1, n, k.row(matrix, left.index))
    if isinstance(right, Node):
        return Value(1, n, k.column(matrix, right.index, n))
    return Value(1, n, k.diagonal(matrix, n))  # the same variable twice


def solve(factors: Sequence[Factor], head: tuple[int, ...], n: int) -> Value:
    """The tuples of ``head`` (distinct variables) that some values of the other
    variables extend to a tuple of every factor."""
    kept: list[Factor] = []
    for factor in factors:
        if factor.value.arity == 0:
            if bool(factor.value.data) == factor.negated:
                return empty(len(head), n)
            continue
        if not factor.negated and is_empty(factor.value):
            return empty(len(head), n)
        kept.append(factor)
    if len(head) > 2 or any(f.value.arity > 2 for f in kept):
        return _split(kept, head, n)
    return _eliminate(kept, head, n)


# -- splitting on a variable's values -------------------------------------------


def _split(factors: list[Factor], head: tuple[int, ...], n: int) -> Value:
    """Loop over the values of one variable: in the head beyond two, or in a
    factor of three or more variables; and solve the rest for each."""
    if len(head) > 2:
        choices = list(head)
    else:
        choices = sorted({v for f in factors if f.value.arity > 2 for v in f.variables})
    variable = min(choices, key=lambda v: k.count(_domain(factors, v, n)))
    return _split_on(factors, head, variable, n)


def _split_on(factors: list[Factor], head: tuple[int, ...], variable: int, n: int) -> Value:
    rest = tuple(v for v in head if v != variable)
    found: list[np.ndarray] = []
    result = empty(len(rest), n)
    for node in k.indices(_domain(factors, variable, n), n):
        value = solve([_select(f, variable, int(node)) for f in factors], rest, n)
        if variable not in head:
            result = union(result, value)
        elif not is_empty(value):
            tuples = to_tuples(value)
            place = head.index(variable)
            found.append(np.insert(tuples, place, int(node), axis=1))
    if variable not in head:
        return result
    tuples = np.concatenate(found) if found else np.empty((0, len(head)), np.intp)
    return from_tuples(len(head), n, tuples)


def _domain(factors: list[Factor], variable: int, n: int) -> k.Bits:
    """The values ``variable`` can take: those every (positive) factor on it allows."""
    domain = k.full_vector(n)
    for factor in factors:
        if factor.negated or variable not in factor.variables:
            continue
        place = factor.variables.index(variable)
        match factor.value.arity:
            case 1:
                domain = domain & factor.value.data
            case 2:
                matrix = factor.value.data
                domain = domain & (k.row_any(matrix, n) if place == 0 else k.column_any(matrix))
            case _:
                domain = domain & k.vector_of(n, np.unique(factor.value.data[:, place]))
    return domain


def _select(factor: Factor, variable: int, node: int) -> Factor:
    """The factor with ``variable`` set to ``node``."""
    if variable not in factor.variables:
        return factor
    place = factor.variables.index(variable)
    variables = factor.variables[:place] + factor.variables[place + 1 :]
    value, n = factor.value, factor.value.n
    match value.arity:
        case 1:
            return Factor((), Value(0, n, k.contains(value.data, node)))
        case 2:
            vector = k.row(value.data, node) if place == 0 else k.column(value.data, node, n)
            return Factor(variables, Value(1, n, vector))
        case _:
            tuples = value.data[value.data[:, place] == node]
            rest = np.delete(tuples, place, axis=1)
            return make_factor(variables, from_tuples(len(variables), n, rest), factor.negated)


# -- eliminating variables --------------------------------------------------------


def _eliminate(factors: list[Factor], head: tuple[int, ...], n: int) -> Value:
    """Solve factors of at most two variables, for a head of at most two."""
    factors = list(factors)
    while True:
        others = {v for f in factors for v in f.variables} - set(head)
        if not others:
            return _combine(factors, head, n)
        variable = min(others, key=lambda v: (len(_neighbours(factors, v)), v))
        neighbours = _neighbours(factors, variable)
        if len(neighbours) > 2:
            return _split_on(factors, head, variable, n)
        factors = _eliminate_one(factors, variable, neighbours, n)


def _neighbours(factors: list[Factor], variable: int) -> list[int]:
    return sorted({v for f in factors if variable in f.variables for v in f.variables} - {variable})


def _eliminate_one(
    factors: list[Factor], variable: int, neighbours: list[int], n: int
) -> list[Factor]:
    touching = [f for f in factors if variable in f.variables]
    rest = [f for f in factors if variable not in f.variables]
    domain = k.full_vector(n)
    for factor in touching:
        if factor.variables == (variable,):
            domain = domain & factor.value.data
    if not neighbours:
        return [*rest, Factor((), Value(0, n, bool(np.any(domain))))]
    if len(neighbours) == 1:
        (a,) = neighbours
        pairs = _pairs(touching, a, variable, n)
        return [*rest, Factor((a,), Value(1, n, k.row_any(k.mask_columns(pairs, domain), n)))]
    a, b = neighbours
    left = k.mask_columns(_pairs(touching, a, variable, n), domain)
    right = _pairs(touching, variable, b, n)
    return [*rest, Factor((a, b), Value(2, n, k.product(left, right, n)))]


def _pairs(factors: list[Factor], row: int, column: int, n: int) -> k.Bits:
    """The intersection of the factors over exactly ``{row, column}``, as a matrix."""
    matrix = full(2, n)
    for factor in factors:
        if factor.variables == (row, column):
            matrix = intersection(matrix, factor.value)
        elif factor.variables == (column, row):
            matrix = intersection(matrix, transpose(factor.value))
    return matrix.data


def _combine(factors: list[Factor], head: tuple[int, ...], n: int) -> Value:
    """The intersection of factors over head variables only."""
    result = full(len(head), n)
    for factor in factors:
        match len(factor.variables):
            case 0:
                if not factor.value.data:
                    return empty(len(head), n)
            case 1 if len(head) == 1:
                result = intersection(result, factor.value)
            case 1:
                (variable,) = factor.variables
                if head.index(variable) == 0:
                    data = k.mask_rows(result.data, factor.value.data, n)
                else:
                    data = k.mask_columns(result.data, factor.value.data)
                result = Value(2, n, data)
            case _:
                value = factor.value
                if factor.variables != head:
                    value = transpose(value)
                result = intersection(result, value)
    return result
