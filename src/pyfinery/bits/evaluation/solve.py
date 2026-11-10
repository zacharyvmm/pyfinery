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
variables, or the head does, :func:`join` takes over. A variable outside the
head that several factors read is split on: the rest is solved for each of its
values. Otherwise the solver joins in bulk: it builds the rows of values of
every variable at once, one variable (or wide factor) at a time, a bounded
block at a time, and keeps those every factor allows.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..matrices import kernels as k
from ..matrices.kernels import Indices, Matrix, NodeIndex, Vector
from ..matrices.values import (
    Tuples,
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

type VarId = int
"""A variable of a clause: its parameters are ``0`` to ``arity - 1``, then the others."""


@dataclass(frozen=True)
class Factor:
    """A set of tuples over ``variables`` (distinct), or with ``negated``, its complement.

    Only factors over three or more variables are ``negated``: the others are
    complemented when made.
    """

    variables: tuple[VarId, ...]
    value: Value
    negated: bool = False


def make_factor(variables: tuple[VarId, ...], value: Value, negated: bool = False) -> Factor:
    if negated and value.arity <= 2:
        return Factor(variables, complement(value))
    return Factor(variables, value, negated)


@dataclass(frozen=True)
class Var:
    id: VarId


@dataclass(frozen=True)
class Node:
    index: NodeIndex


Argument = Var | Node


def bind(value: Value, arguments: Sequence[Argument], negated: bool = False) -> Factor:
    """The factor of a relation's ``value`` at ``arguments``: nodes select tuples,
    a repeated variable keeps the tuples whose places agree."""
    assert len(arguments) == value.arity
    variables: list[VarId] = []
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
        return make_factor(tuple(variables), bind_pair(value, arguments), negated)
    if value.arity == 1:  # a node
        assert isinstance(arguments[0], Node)
        node = arguments[0].index
        return make_factor((), Value(0, value.n, k.contains(value.vector, node)), negated)
    tuples = to_tuples(value)
    keep = np.ones(len(tuples), bool)
    first: dict[VarId, int] = {}  # where each variable is first
    for place, argument in enumerate(arguments):
        if isinstance(argument, Node):
            keep &= tuples[:, place] == argument.index
        elif argument.id in first:
            keep &= tuples[:, place] == tuples[:, first[argument.id]]
        else:
            first[argument.id] = place
    projected = tuples[keep][:, [first[v] for v in variables]]
    return make_factor(tuple(variables), from_tuples(len(variables), value.n, projected), negated)


def bind_pair(value: Value, arguments: Sequence[Argument]) -> Value:
    left, right = arguments
    matrix, n = value.matrix, value.n
    if isinstance(left, Node) and isinstance(right, Node):
        return Value(0, n, k.contains(matrix[left.index], right.index))
    if isinstance(left, Node):
        return Value(1, n, k.row(matrix, left.index))
    if isinstance(right, Node):
        return Value(1, n, k.column(matrix, right.index, n))
    return Value(1, n, k.diagonal(matrix, n))  # the same variable twice


def solve(factors: Sequence[Factor], head: tuple[VarId, ...], n: int) -> Value:
    """The tuples of ``head`` (distinct variables) that some values of the other
    variables extend to a tuple of every factor."""
    kept: list[Factor] = []
    for factor in factors:
        if factor.value.arity == 0:
            if factor.value.truth == factor.negated:
                return empty(len(head), n)
            continue
        if not factor.negated and is_empty(factor.value):
            return empty(len(head), n)
        kept.append(factor)
    if len(head) > 2 or any(f.value.arity > 2 for f in kept):
        return join(kept, head, n)
    return eliminate(kept, head, n)


# -- eliminating variables --------------------------------------------------------


def eliminate(factors: list[Factor], head: tuple[VarId, ...], n: int) -> Value:
    """Solve factors of at most two variables, for a head of at most two."""
    factors = list(factors)
    while True:
        others = {v for f in factors for v in f.variables} - set(head)
        if not others:
            return combine(factors, head, n)
        variable = min(others, key=lambda v: (len(neighbours_of(factors, v)), v))
        neighbours = neighbours_of(factors, variable)
        if len(neighbours) > 2:
            return join(factors, head, n)
        factors = eliminate_one(factors, variable, neighbours, n)


def neighbours_of(factors: list[Factor], variable: VarId) -> list[VarId]:
    return sorted({v for f in factors if variable in f.variables for v in f.variables} - {variable})


def eliminate_one(
    factors: list[Factor], variable: VarId, neighbours: list[VarId], n: int
) -> list[Factor]:
    touching = [f for f in factors if variable in f.variables]
    rest = [f for f in factors if variable not in f.variables]
    domain = k.full_vector(n)
    for factor in touching:
        if factor.variables == (variable,):
            domain = domain & factor.value.vector
    if not neighbours:
        return [*rest, Factor((), Value(0, n, bool(np.any(domain))))]
    if len(neighbours) == 1:
        (a,) = neighbours
        pairs = pairs_between(touching, a, variable, n)
        return [*rest, Factor((a,), Value(1, n, k.row_any(k.mask_columns(pairs, domain), n)))]
    a, b = neighbours
    left = k.mask_columns(pairs_between(touching, a, variable, n), domain)
    right = pairs_between(touching, variable, b, n)
    return [*rest, Factor((a, b), Value(2, n, k.product(left, right, n)))]


def pairs_between(factors: list[Factor], row: VarId, column: VarId, n: int) -> Matrix:
    """The intersection of the factors over exactly ``{row, column}``, as a matrix."""
    matrix = full(2, n)
    for factor in factors:
        if factor.variables == (row, column):
            matrix = intersection(matrix, factor.value)
        elif factor.variables == (column, row):
            matrix = intersection(matrix, transpose(factor.value))
    return matrix.matrix


def combine(factors: list[Factor], head: tuple[VarId, ...], n: int) -> Value:
    """The intersection of factors over head variables only."""
    result = full(len(head), n)
    for factor in factors:
        match len(factor.variables):
            case 0:
                if not factor.value.truth:
                    return empty(len(head), n)
            case 1 if len(head) == 1:
                result = intersection(result, factor.value)
            case 1:
                (variable,) = factor.variables
                if head.index(variable) == 0:
                    data = k.mask_rows(result.matrix, factor.value.vector, n)
                else:
                    data = k.mask_columns(result.matrix, factor.value.vector)
                result = Value(2, n, data)
            case _:
                value = factor.value
                if factor.variables != head:
                    value = transpose(value)
                result = intersection(result, value)
    return result


def domain_of(factors: list[Factor], variable: VarId, n: int) -> Vector:
    """The values ``variable`` can take: those every (positive) factor on it allows."""
    domain = k.full_vector(n)
    for factor in factors:
        if factor.negated or variable not in factor.variables:
            continue
        place = factor.variables.index(variable)
        match factor.value.arity:
            case 1:
                domain = domain & factor.value.vector
            case 2:
                matrix = factor.value.matrix
                domain = domain & (k.row_any(matrix, n) if place == 0 else k.column_any(matrix))
            case _:
                domain = domain & k.vector_of(n, np.unique(factor.value.array[:, place]))
    return domain


# -- joining in bulk ----------------------------------------------------------------

BLOCK_ROWS = 1 << 16
"""How many rows :func:`join` builds at a time, at each step: it joins depth first."""

CHUNK_WORDS = 1 << 20
"""How many words of candidates :func:`extend` holds at a time: 8 MiB."""


@dataclass(frozen=True)
class Rows:
    """Values of some variables that agree with the factors so far: a row per tuple,
    a column per variable."""

    variables: tuple[VarId, ...]
    tuples: Tuples

    def column(self, variable: VarId) -> Indices:
        return self.tuples[:, self.variables.index(variable)]


def join(factors: list[Factor], head: tuple[VarId, ...], n: int) -> Value:
    """Solve factors of any width, for a head of any size, by building the rows of every
    variable in bulk: one variable or wide factor at a time (:func:`grow`), keeping the
    rows the factors over bound variables allow; then project onto the head.

    Variables outside the head that matrix products can eliminate are eliminated
    first. If one that several factors read remains, rows would keep each of its
    values until all those factors are applied: the solver splits on its values
    instead (:func:`split_on`), and solves the rest for each, which matrix products
    usually can. Otherwise the rows are built depth first, a block of at most
    :data:`BLOCK_ROWS` at a time (:func:`complete`), so memory stays bounded; those
    variables are dropped once their factor is applied; and a head of no variables
    holds as soon as one row is complete.
    """
    remaining = eliminate_existentials(factors, head, n)
    if remaining is None:
        return empty(len(head), n)
    shared = shared_existentials(remaining, head)
    if shared:
        variable = min(shared, key=lambda v: (k.count(domain_of(remaining, v, n)), v))
        return split_on(remaining, head, variable, n)
    variables = sorted({v for f in remaining for v in f.variables} | set(head))
    start = Rows((), np.empty((1, 0), np.intp))  # one row, of no variables
    found: list[Tuples] = []
    size = 0
    for block in complete(start, remaining, variables, head, n):
        if not head:
            return Value(0, n, True)
        found.append(block.tuples[:, [block.variables.index(v) for v in head]])
        size += len(found[-1])
        if size > BLOCK_ROWS:  # drop the head tuples found twice
            found = [np.unique(np.concatenate(found), axis=0)]
            size = len(found[0])
    tuples = np.concatenate(found) if found else np.empty((0, len(head)), np.intp)
    return from_tuples(len(head), n, tuples)


def shared_existentials(factors: list[Factor], head: tuple[VarId, ...]) -> list[VarId]:
    """The variables outside the head that several factors read."""
    readers: dict[VarId, int] = {}
    for factor in factors:
        for variable in factor.variables:
            readers[variable] = readers.get(variable, 0) + 1
    return sorted(v for v, count in readers.items() if count > 1 and v not in head)


def split_on(factors: list[Factor], head: tuple[VarId, ...], variable: VarId, n: int) -> Value:
    """The union, over each value ``variable`` (not in the head) can take, of the rest
    solved with ``variable`` set to it."""
    result = empty(len(head), n)
    for node in k.indices(domain_of(factors, variable, n), n):
        result = union(result, solve([select(f, variable, int(node)) for f in factors], head, n))
        if not head and result.truth:
            break
    return result


def select(factor: Factor, variable: VarId, node: NodeIndex) -> Factor:
    """The factor with ``variable`` set to ``node``."""
    if variable not in factor.variables:
        return factor
    place = factor.variables.index(variable)
    variables = factor.variables[:place] + factor.variables[place + 1 :]
    value, n = factor.value, factor.value.n
    match value.arity:
        case 1:
            return Factor((), Value(0, n, k.contains(value.vector, node)))
        case 2:
            matrix = value.matrix
            vector = k.row(matrix, node) if place == 0 else k.column(matrix, node, n)
            return Factor(variables, Value(1, n, vector))
        case _:
            tuples = value.array[value.array[:, place] == node]
            rest = np.delete(tuples, place, axis=1)
            return make_factor(variables, from_tuples(len(variables), n, rest), factor.negated)


def complete(
    rows: Rows, pending: list[Factor], unbound: list[VarId], head: tuple[VarId, ...], n: int
) -> Iterator[Rows]:
    """Blocks of complete rows from ``rows``: bind one more variable or wide factor
    (:func:`grow`), and complete each block of the result before the next."""
    if not unbound:
        yield rows
        return
    blocks, used = grow(rows, pending, unbound, n)
    left = [f for i, f in enumerate(pending) if i not in used]
    for block in blocks:
        block, rest = keep(block, left)
        if not len(block.tuples):
            continue
        still_unbound = [v for v in unbound if v not in block.variables]
        yield from complete(forget(block, rest, head), rest, still_unbound, head, n)


def forget(rows: Rows, pending: list[Factor], head: tuple[VarId, ...]) -> Rows:
    """The rows without the variables outside the head that no pending factor reads,
    and without the rows that then repeat."""
    needed = set(head) | {v for f in pending for v in f.variables}
    kept = [i for i, v in enumerate(rows.variables) if v in needed]
    if len(kept) == len(rows.variables):
        return rows
    if not kept:
        return Rows((), np.empty((1, 0), np.intp))  # some row was complete: that's all
    tuples = np.unique(rows.tuples[:, kept], axis=0)
    return Rows(tuple(rows.variables[i] for i in kept), tuples)


def blocks_of(counts: Indices, limit: int) -> Iterator[tuple[int, int]]:
    """Consecutive ranges of rows whose ``counts`` add up to at most ``limit``, but for a
    single row over it; ranges that add up to nothing are skipped."""
    total = np.cumsum(counts)
    first = 0
    while first < len(counts):
        before = int(total[first - 1]) if first else 0
        last = max(int(np.searchsorted(total, before + limit, "right")), first + 1)
        if int(total[last - 1]) > before:
            yield first, last
        first = last


def eliminate_existentials(
    factors: list[Factor], head: tuple[VarId, ...], n: int
) -> list[Factor] | None:
    """The factors, with the variables outside the head and wide factors that have at most
    two neighbours eliminated; ``None`` if that shows nothing satisfies them."""
    while True:
        wide = {v for f in factors if f.value.arity > 2 for v in f.variables}
        others = {v for f in factors for v in f.variables} - set(head) - wide
        candidates = [v for v in others if len(neighbours_of(factors, v)) <= 2]
        if not candidates:
            break
        variable = min(candidates, key=lambda v: (len(neighbours_of(factors, v)), v))
        factors = eliminate_one(factors, variable, neighbours_of(factors, variable), n)
    if any(not f.variables and not f.value.truth for f in factors):
        return None
    return [f for f in factors if f.variables]


def grow(
    rows: Rows, pending: list[Factor], unbound: list[VarId], n: int
) -> tuple[Iterator[Rows], set[int]]:
    """The rows with more variables bound, in blocks, and which pending factors that used up.

    In order of preference: join a wide factor on a variable already bound; bind
    the variable that the most binary factors link to bound ones, from the rows of
    those factors; join a wide factor; bind the variable with the fewest values.
    """
    bound = set(rows.variables)
    wide = [i for i, f in enumerate(pending) if f.value.arity > 2 and not f.negated]
    linked_wide = [i for i in wide if bound & set(pending[i].variables)]
    if linked_wide:
        i = min(linked_wide, key=lambda i: len(pending[i].value.array))
        return join_tuples(rows, pending[i]), {i}
    unary = {v: [i for i, f in enumerate(pending) if f.variables == (v,)] for v in unbound}
    links = {
        v: [
            i
            for i, f in enumerate(pending)
            if f.value.arity == 2 and v in f.variables and set(f.variables) - {v} <= bound
        ]
        for v in unbound
    }
    variable = max(unbound, key=lambda v: len(links[v]))
    if links[variable]:
        used = {*links[variable], *unary[variable]}
        return extend(rows, variable, [pending[i] for i in sorted(used)], k.full_vector(n), n), used
    if wide:
        i = min(wide, key=lambda i: len(pending[i].value.array))
        return join_tuples(rows, pending[i]), {i}
    variable = min(unbound, key=lambda v: k.count(domain_of(pending, v, n)))
    domain = domain_of(pending, variable, n)  # the unary factors on it, and more
    return extend(rows, variable, [], domain, n), set(unary[variable])


def extend(
    rows: Rows, variable: VarId, factors: list[Factor], domain: Vector, n: int
) -> Iterator[Rows]:
    """Each row, with each value of ``variable`` in ``domain`` that ``factors`` (unary on
    it, or binary from a bound variable to it) allow with that row: in blocks of at most
    :data:`BLOCK_ROWS` rows (but for a single row with more values)."""
    links: list[tuple[Indices, Matrix]] = []  # a bound variable's values, rows of what it allows
    for factor in factors:
        if factor.value.arity == 1:
            domain = domain & factor.value.vector
            continue
        matrix = factor.value.matrix
        if factor.variables[1] != variable:
            matrix = k.transpose(matrix, n)
        (other,) = set(factor.variables) - {variable}
        links.append((rows.column(other), matrix))
    variables = (*rows.variables, variable)
    words = max(k.words(n), 1)
    chunk = max(1, min(BLOCK_ROWS, CHUNK_WORDS // words))
    for start in range(0, len(rows.tuples), chunk):
        stop = min(start + chunk, len(rows.tuples))
        candidates = np.tile(domain, (stop - start, 1))
        for values, matrix in links:
            candidates &= matrix[values[start:stop]]
        counts = np.bitwise_count(candidates).sum(axis=1)  # values per row
        for first, last in blocks_of(counts, BLOCK_ROWS):
            at, value = k.pairs(candidates[first:last], n)
            yield Rows(variables, np.column_stack([rows.tuples[start + first + at], value]))


def join_tuples(rows: Rows, factor: Factor) -> Iterator[Rows]:
    """Each row, with each tuple of a wide ``factor`` that agrees with it: in blocks of at
    most :data:`BLOCK_ROWS` rows (but for a single row with more tuples)."""
    tuples = factor.value.array
    shared = [v for v in factor.variables if v in rows.variables]
    added = [factor.variables.index(v) for v in factor.variables if v not in rows.variables]
    count = len(rows.tuples)
    # The tuples agreeing with row i: order[low[i] : low[i] + matches[i]].
    if not shared:
        order = np.arange(len(tuples))
        low = np.zeros(count, np.intp)
        matches = np.full(count, len(tuples), np.intp)
    else:
        mine, keys = codes(
            rows.tuples[:, [rows.variables.index(v) for v in shared]],
            tuples[:, [factor.variables.index(v) for v in shared]],
        )
        order = np.argsort(keys, kind="stable")
        low = np.searchsorted(keys[order], mine, "left")
        matches = np.searchsorted(keys[order], mine, "right") - low
    variables = (*rows.variables, *(factor.variables[i] for i in added))
    for first, last in blocks_of(matches, BLOCK_ROWS):
        some = matches[first:last]
        at = np.repeat(np.arange(first, last), some)
        within = np.arange(some.sum()) - np.repeat(np.cumsum(some) - some, some)
        theirs = order[np.repeat(low[first:last], some) + within]
        yield Rows(variables, np.column_stack([rows.tuples[at], tuples[theirs][:, added]]))


def keep(rows: Rows, pending: list[Factor]) -> tuple[Rows, list[Factor]]:
    """The rows every factor over bound variables allows, and the factors left."""
    bound = set(rows.variables)
    keeping = np.ones(len(rows.tuples), bool)
    left = []
    for factor in pending:
        if set(factor.variables) <= bound:
            keeping &= allows(rows, factor)
        else:
            left.append(factor)
    return Rows(rows.variables, rows.tuples[keeping]), left


def allows(rows: Rows, factor: Factor) -> NDArray[np.bool_]:
    """Which rows a factor over bound variables allows."""
    columns = [rows.column(v) for v in factor.variables]
    match factor.value.arity:
        case 1:
            (c,) = columns
            found = (factor.value.vector[c >> 6] >> (c & 63).astype(np.uint64)) & k.ONE
        case 2:
            r, c = columns
            found = (factor.value.matrix[r, c >> 6] >> (c & 63).astype(np.uint64)) & k.ONE
        case _:
            mine, theirs = codes(np.column_stack(columns), factor.value.array)
            found = np.isin(mine, theirs)
    allowed: NDArray[np.bool_] = found.astype(bool) != factor.negated
    return allowed


def codes(left: Tuples, right: Tuples) -> tuple[Indices, Indices]:
    """A number for each row of ``left`` and of ``right``: the same where the rows are."""
    if left.shape[1] == 1:
        return left[:, 0], right[:, 0]
    _, numbers = np.unique(np.concatenate([left, right]), axis=0, return_inverse=True)
    numbers = numbers.reshape(-1)
    return numbers[: len(left)], numbers[len(left) :]
