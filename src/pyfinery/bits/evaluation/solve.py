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
variables, or the head does, :func:`join` takes over. If every variable left
is in the head, it joins in bulk: it builds the rows of values of every
variable at once, one variable (or wide factor) at a time, a bounded block at a
time, and keeps those every factor allows. Otherwise it splits on one
variable's values, and solves the rest for each.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

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
"""How many rows :func:`complete` builds at a time, at each step: it joins depth first."""

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
    """Solve factors of any width, for a head of any size.

    Variables outside the head that matrix products can eliminate are eliminated
    first. If every variable left is in the head, each row of values that agrees
    with every factor is an answer: the rows are built in bulk (:func:`complete`),
    one variable or wide factor at a time, depth first, a block of at most
    :data:`BLOCK_ROWS` rows at a time. Otherwise the solver splits on one variable
    (:func:`split_variable`) and solves the rest for each of its values, so that
    what's left can be eliminated by matrix products.
    """
    remaining = eliminate_existentials(factors, head, n)
    if remaining is None:
        return empty(len(head), n)
    variables = sorted({v for f in remaining for v in f.variables} | set(head))
    if not set(variables) <= set(head):
        return split_on(remaining, head, split_variable(remaining, head, n), n)
    start = Rows((), np.empty((1, 0), np.intp))  # one row, of no variables
    found = [
        block.tuples[:, [block.variables.index(v) for v in head]]
        for block in complete(start, remaining, variables, n, {})
    ]  # distinct rows: each binds every variable
    tuples = np.concatenate(found) if found else np.empty((0, len(head)), np.intp)
    return from_tuples(len(head), n, tuples)


def split_variable(factors: list[Factor], head: tuple[VarId, ...], n: int) -> VarId:
    """The variable to split on: of the head, if it has more than two, the one with the
    fewest values; else of the wide factors, the one with the fewest values; else the
    variable outside the head with the fewest neighbours."""
    if len(head) > 2:
        choices = list(head)
    else:
        choices = sorted({v for f in factors if f.value.arity > 2 for v in f.variables})
    if choices:
        return min(choices, key=lambda v: (k.count(domain_of(factors, v, n)), v))
    others = {v for f in factors for v in f.variables} - set(head)
    return min(others, key=lambda v: (len(neighbours_of(factors, v)), v))


def split_on(factors: list[Factor], head: tuple[VarId, ...], variable: VarId, n: int) -> Value:
    """The rest solved for each value ``variable`` can take: with the value put in
    place if ``variable`` is in the head, else their union."""
    rest = tuple(v for v in head if v != variable)
    union_of_rest = empty(len(rest), n)
    found: list[Tuples] = []
    for node in k.indices(domain_of(factors, variable, n), n):
        value = solve([select(f, variable, int(node)) for f in factors], rest, n)
        if variable in head:
            if not is_empty(value):
                found.append(np.insert(to_tuples(value), head.index(variable), node, axis=1))
            continue
        union_of_rest = union(union_of_rest, value)
        if not head and union_of_rest.truth:
            break  # holds: no need for more values
    if variable not in head:
        return union_of_rest
    tuples = np.concatenate(found) if found else np.empty((0, len(head)), np.intp)
    return from_tuples(len(head), n, tuples)


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
    rows: Rows, pending: list[Factor], unbound: list[VarId], n: int, lookups: Lookups
) -> Iterator[Rows]:
    """Blocks of complete rows from ``rows``: bind one more variable or wide factor
    (:func:`grow`), and complete each block of the result before the next."""
    if not unbound:
        yield rows
        return
    blocks, used = grow(rows, pending, unbound, n, lookups)
    left = [f for i, f in enumerate(pending) if i not in used]
    for block in blocks:
        block, rest = keep(block, left, lookups)
        if len(block.tuples):
            still_unbound = [v for v in unbound if v not in block.variables]
            yield from complete(block, rest, still_unbound, n, lookups)


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
    rows: Rows, pending: list[Factor], unbound: list[VarId], n: int, lookups: Lookups
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
        return join_tuples(rows, pending[i], lookups), {i}
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
        return join_tuples(rows, pending[i], lookups), {i}
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


def join_tuples(rows: Rows, factor: Factor, lookups: Lookups) -> Iterator[Rows]:
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
        found = lookup(lookups, factor, [factor.variables.index(v) for v in shared])
        order = found.order
        low, matches = found.matching(rows.tuples[:, [rows.variables.index(v) for v in shared]])
    variables = (*rows.variables, *(factor.variables[i] for i in added))
    for first, last in blocks_of(matches, BLOCK_ROWS):
        some = matches[first:last]
        at = np.repeat(np.arange(first, last), some)
        within = np.arange(some.sum()) - np.repeat(np.cumsum(some) - some, some)
        theirs = order[np.repeat(low[first:last], some) + within]
        yield Rows(variables, np.column_stack([rows.tuples[at], tuples[theirs][:, added]]))


def keep(rows: Rows, pending: list[Factor], lookups: Lookups) -> tuple[Rows, list[Factor]]:
    """The rows every factor over bound variables allows, and the factors left."""
    bound = set(rows.variables)
    keeping = np.ones(len(rows.tuples), bool)
    left = []
    for factor in pending:
        if set(factor.variables) <= bound:
            keeping &= allows(rows, factor, lookups)
        else:
            left.append(factor)
    return Rows(rows.variables, rows.tuples[keeping]), left


def allows(rows: Rows, factor: Factor, lookups: Lookups) -> NDArray[np.bool_]:
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
            every = lookup(lookups, factor, list(range(len(factor.variables))))
            found = every.matching(np.column_stack(columns))[1] > 0
    allowed: NDArray[np.bool_] = found.astype(bool) != factor.negated
    return allowed


@dataclass(frozen=True)
class Lookup:
    """A wide factor's tuples sorted once by some of their columns, to find in bulk the
    tuples that agree with each row on those columns."""

    order: Indices
    """The tuples, by index, sorted by the columns."""
    keys: NDArray[Any]
    """Those columns of the tuples, in that order: one key per tuple."""

    @classmethod
    def of(cls, tuples: Tuples, columns: list[int]) -> Lookup:
        keys = row_keys(tuples[:, columns])
        order = np.argsort(keys, kind="stable")
        return cls(order, keys[order])

    def matching(self, values: Tuples) -> tuple[Indices, Indices]:
        """For each row of ``values`` (of the columns): where its tuples start in
        :attr:`order`, and how many there are."""
        keys = row_keys(values)
        low = np.searchsorted(self.keys, keys, "left")
        return low, np.searchsorted(self.keys, keys, "right") - low


type Lookups = dict[tuple[int, tuple[int, ...]], Lookup]
"""A join's lookups, by the ``id`` of the factor and the columns: each is made once,
and every block of rows uses it."""


def lookup(lookups: Lookups, factor: Factor, columns: list[int]) -> Lookup:
    key = (id(factor), tuple(columns))
    if key not in lookups:
        lookups[key] = Lookup.of(factor.value.array, columns)
    return lookups[key]


def row_keys(columns: Tuples) -> NDArray[Any]:
    """A key per row, equal where the rows are: the value for one column, else the row's
    bytes (whose order isn't numeric, but is the same for every array of rows)."""
    if columns.shape[1] == 1:
        return np.ascontiguousarray(columns[:, 0], np.intp)
    rows = np.ascontiguousarray(columns, np.intp)
    return rows.view(np.dtype((np.void, rows.itemsize * rows.shape[1]))).ravel()
