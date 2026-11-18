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

Every factor holds the tuples of each graph of a batch (:mod:`..matrices.values`),
and every step works on all graphs at once: the graph is a variable every factor
has, which no step eliminates or splits on. Choices (which variable, which
factor first) are made for the batch as a whole; they change how much work a
step does, not what it computes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..matrices import kernels as k
from ..matrices.kernels import Indices, Matrix, NodeIndex, Truths, Vector
from ..matrices.values import (
    Size,
    Tuples,
    Value,
    complement,
    empty,
    from_tuples,
    full,
    intersection,
    is_empty,
    key_base,
    only_in,
    row_keys,
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
        return make_factor((), Value(0, value.size, k.contains(value.vector, node)), negated)
    tuples = to_tuples(value)  # each row: its graph, then its nodes
    keep = k.xp().ones(len(tuples), bool)
    first: dict[VarId, int] = {}  # the column where each variable is first
    for place, argument in enumerate(arguments):
        column = 1 + place
        if isinstance(argument, Node):
            keep &= tuples[:, column] == argument.index
        elif argument.id in first:
            keep &= tuples[:, column] == tuples[:, first[argument.id]]
        else:
            first[argument.id] = column
    projected = tuples[keep][:, [0, *(first[v] for v in variables)]]
    bound = from_tuples(len(variables), value.size, projected)
    return make_factor(tuple(variables), bound, negated)


def bind_pair(value: Value, arguments: Sequence[Argument]) -> Value:
    left, right = arguments
    matrix, size = value.matrix, value.size
    if isinstance(left, Node) and isinstance(right, Node):
        return Value(0, size, k.contains(matrix[:, left.index], right.index))
    if isinstance(left, Node):
        return Value(1, size, k.row(matrix, left.index))
    if isinstance(right, Node):
        return Value(1, size, k.column(matrix, right.index, size.nodes))
    return Value(1, size, k.diagonal(matrix, size.nodes))  # the same variable twice


def solve(factors: Sequence[Factor], head: tuple[VarId, ...], size: Size) -> Value:
    """The tuples of ``head`` (distinct variables), in each graph, that some values of the
    other variables extend to a tuple of every factor."""
    graphs = k.xp().ones(size.graphs, bool)  # where every factor of no variables holds
    kept: list[Factor] = []
    for factor in factors:
        if factor.value.arity == 0:
            graphs &= factor.value.truth != factor.negated
            continue
        if not factor.negated and is_empty(factor.value):
            return empty(len(head), size)
        kept.append(factor)
    if not graphs.any():
        return empty(len(head), size)
    if len(head) > 2 or any(f.value.arity > 2 for f in kept):
        result = join(kept, head, size)
    else:
        result = eliminate(kept, head, size)
    return result if graphs.all() else only_in(result, graphs)


# -- eliminating variables --------------------------------------------------------


def eliminate(factors: list[Factor], head: tuple[VarId, ...], size: Size) -> Value:
    """Solve factors of at most two variables, for a head of at most two."""
    factors = list(factors)
    while True:
        others = {v for f in factors for v in f.variables} - set(head)
        if not others:
            return combine(factors, head, size)
        variable = min(others, key=lambda v: (len(neighbours_of(factors, v)), v))
        neighbours = neighbours_of(factors, variable)
        if len(neighbours) > 2:
            return join(factors, head, size)
        factors = eliminate_one(factors, variable, neighbours, size)


def neighbours_of(factors: list[Factor], variable: VarId) -> list[VarId]:
    return sorted({v for f in factors if variable in f.variables for v in f.variables} - {variable})


def eliminate_one(
    factors: list[Factor], variable: VarId, neighbours: list[VarId], size: Size
) -> list[Factor]:
    touching = [f for f in factors if variable in f.variables]
    rest = [f for f in factors if variable not in f.variables]
    domain = unary_domain(touching, variable, size)
    n = size.nodes
    if not neighbours:
        return [*rest, Factor((), Value(0, size, domain.any(axis=1)))]
    if len(neighbours) == 1:
        (a,) = neighbours
        pairs = pairs_between(touching, a, variable, size)
        return [*rest, Factor((a,), Value(1, size, k.row_any(k.mask_columns(pairs, domain), n)))]
    a, b = neighbours
    left = k.mask_columns(pairs_between(touching, a, variable, size), domain)
    right = pairs_between(touching, variable, b, size)
    return [*rest, Factor((a, b), Value(2, size, k.product(left, right, n)))]


def unary_domain(factors: list[Factor], variable: VarId, size: Size) -> Vector:
    """The intersection of the factors over exactly ``variable``."""
    domain = None
    for factor in factors:
        if factor.variables == (variable,):
            vector = factor.value.vector
            domain = vector if domain is None else domain & vector
    return k.full_vector(*size) if domain is None else domain


def pairs_between(factors: list[Factor], row: VarId, column: VarId, size: Size) -> Matrix:
    """The intersection of the factors over exactly ``{row, column}``, as a matrix."""
    matrix = None
    for factor in factors:
        if factor.variables == (row, column):
            value = factor.value
        elif factor.variables == (column, row):
            value = transpose(factor.value)
        else:
            continue
        matrix = value if matrix is None else intersection(matrix, value)
    return full(2, size).matrix if matrix is None else matrix.matrix


def combine(factors: list[Factor], head: tuple[VarId, ...], size: Size) -> Value:
    """The intersection of factors over head variables only."""
    result = full(len(head), size)
    graphs = k.xp().ones(size.graphs, bool)
    n = size.nodes
    for factor in factors:
        match len(factor.variables):
            case 0:
                graphs &= factor.value.truth
            case 1 if len(head) == 1:
                result = intersection(result, factor.value)
            case 1:
                (variable,) = factor.variables
                if head.index(variable) == 0:
                    data = k.mask_rows(result.matrix, factor.value.vector, n)
                else:
                    data = k.mask_columns(result.matrix, factor.value.vector)
                result = Value(2, size, data)
            case _:
                value = factor.value
                if factor.variables != head:
                    value = transpose(value)
                result = intersection(result, value)
    return result if graphs.all() else only_in(result, graphs)


def domain_of(factors: list[Factor], variable: VarId, size: Size) -> Vector:
    """The values ``variable`` can take in each graph: those every (positive) factor on it
    allows."""
    domain = k.full_vector(*size)
    for factor in factors:
        if factor.negated or variable not in factor.variables:
            continue
        place = factor.variables.index(variable)
        match factor.value.arity:
            case 1:
                domain = domain & factor.value.vector
            case 2:
                matrix = factor.value.matrix
                if place == 0:
                    domain = domain & k.row_any(matrix, size.nodes)
                else:
                    domain = domain & k.column_any(matrix)
            case _:
                tuples = factor.value.array
                domain = domain & k.vector_of(*size, tuples[:, 0], tuples[:, 1 + place])
    return domain


# -- joining in bulk ----------------------------------------------------------------

BLOCK_ROWS = 1 << 16
"""How many rows :func:`complete` builds at a time, at each step: it joins depth first."""

CHUNK_WORDS = 1 << 20
"""How many words of candidates :func:`extend` holds at a time: 8 MiB."""


@dataclass(frozen=True)
class Rows:
    """Values of some variables that agree with the factors so far: a row per tuple,
    its graph and then a column per variable."""

    variables: tuple[VarId, ...]
    tuples: Tuples

    @property
    def graphs(self) -> Indices:
        return self.tuples[:, 0]

    def column(self, variable: VarId) -> Indices:
        return self.tuples[:, 1 + self.variables.index(variable)]

    def columns(self, variables: Sequence[VarId]) -> Tuples:
        """The graph's column, then those of ``variables``."""
        return self.tuples[:, [0, *(1 + self.variables.index(v) for v in variables)]]


def join(factors: list[Factor], head: tuple[VarId, ...], size: Size) -> Value:
    """Solve factors of any width, for a head of any size.

    Variables outside the head that matrix products can eliminate are eliminated
    first. If every variable left is in the head, each row of values that agrees
    with every factor is an answer: the rows are built in bulk (:func:`complete`),
    one variable or wide factor at a time, depth first, a block of at most
    :data:`BLOCK_ROWS` rows at a time. Otherwise the solver splits on one variable
    (:func:`split_variable`) and solves the rest for each of its values, so that
    what's left can be eliminated by matrix products.
    """
    remaining, graphs = eliminate_existentials(factors, head, size)
    if not graphs.any():
        return empty(len(head), size)
    variables = sorted({v for f in remaining for v in f.variables} | set(head))
    if not set(variables) <= set(head):
        holds = Factor((), Value(0, size, graphs))
        return split_on([*remaining, holds], head, split_variable(remaining, head, size), size)
    start = Rows((), k.xp().flatnonzero(graphs).astype(np.intp)[:, None])  # a row per graph
    found = [
        block.columns(head) for block in complete(start, remaining, variables, size, {})
    ]  # distinct rows: each binds every variable
    tuples = k.xp().concatenate(found) if found else k.xp().empty((0, 1 + len(head)), np.intp)
    return from_tuples(len(head), size, tuples)


def split_variable(factors: list[Factor], head: tuple[VarId, ...], size: Size) -> VarId:
    """The variable to split on: of the head, if it has more than two, the one with the
    fewest values; else of the wide factors, the one with the fewest values; else the
    variable outside the head with the fewest neighbours."""
    if len(head) > 2:
        choices = list(head)
    else:
        choices = sorted({v for f in factors if f.value.arity > 2 for v in f.variables})
    if choices:
        return min(choices, key=lambda v: (k.count(domain_of(factors, v, size)), v))
    others = {v for f in factors for v in f.variables} - set(head)
    return min(others, key=lambda v: (len(neighbours_of(factors, v)), v))


def split_on(factors: list[Factor], head: tuple[VarId, ...], variable: VarId, size: Size) -> Value:
    """The rest solved for each value ``variable`` can take: with the value put in
    place if ``variable`` is in the head, else their union.

    A value is tried in every graph at once, and holds in those whose domain has it.
    """
    rest = tuple(v for v in head if v != variable)
    union_of_rest = empty(len(rest), size)
    found: list[Tuples] = []
    domain = domain_of(factors, variable, size)
    values = k.xp().flatnonzero(k.unpack(k.or_reduce(domain, axis=0), size.nodes))
    for node in values.tolist():
        allowed = Factor((), Value(0, size, k.contains(domain, node)))
        value = solve([allowed, *(select(f, variable, node) for f in factors)], rest, size)
        if variable in head:
            if not is_empty(value):
                tuples = to_tuples(value)
                at = 1 + head.index(variable)
                placed = k.xp().full((len(tuples), 1), node, np.intp)
                found.append(k.xp().concatenate([tuples[:, :at], placed, tuples[:, at:]], axis=1))
            continue
        union_of_rest = union(union_of_rest, value)
        if not head and union_of_rest.truth.all():
            break  # holds everywhere: no need for more values
    if variable not in head:
        return union_of_rest
    tuples = k.xp().concatenate(found) if found else k.xp().empty((0, 1 + len(head)), np.intp)
    return from_tuples(len(head), size, tuples)


def select(factor: Factor, variable: VarId, node: NodeIndex) -> Factor:
    """The factor with ``variable`` set to ``node``."""
    if variable not in factor.variables:
        return factor
    place = factor.variables.index(variable)
    variables = factor.variables[:place] + factor.variables[place + 1 :]
    value, size = factor.value, factor.value.size
    match value.arity:
        case 1:
            return Factor((), Value(0, size, k.contains(value.vector, node)))
        case 2:
            matrix = value.matrix
            vector = k.row(matrix, node) if place == 0 else k.column(matrix, node, size.nodes)
            return Factor(variables, Value(1, size, vector))
        case _:
            tuples = value.array[value.array[:, 1 + place] == node]
            rest = k.xp().delete(tuples, 1 + place, axis=1)
            return make_factor(variables, from_tuples(len(variables), size, rest), factor.negated)


def complete(
    rows: Rows, pending: list[Factor], unbound: list[VarId], size: Size, lookups: Lookups
) -> Iterator[Rows]:
    """Blocks of complete rows from ``rows``: bind one more variable or wide factor
    (:func:`grow`), and complete each block of the result before the next."""
    if not unbound:
        yield rows
        return
    blocks, used = grow(rows, pending, unbound, size, lookups)
    left = [f for i, f in enumerate(pending) if i not in used]
    for block in blocks:
        block, rest = keep(block, left, lookups)
        if len(block.tuples):
            still_unbound = [v for v in unbound if v not in block.variables]
            yield from complete(block, rest, still_unbound, size, lookups)


def blocks_of(counts: Indices, limit: int) -> Iterator[tuple[int, int]]:
    """Consecutive ranges of rows whose ``counts`` add up to at most ``limit``, but for a
    single row over it; ranges that add up to nothing are skipped."""
    total = k.xp().cumsum(counts)
    first = 0
    while first < len(counts):
        before = int(total[first - 1]) if first else 0
        last = max(int(k.xp().searchsorted(total, before + limit, "right")), first + 1)
        if int(total[last - 1]) > before:
            yield first, last
        first = last


def eliminate_existentials(
    factors: list[Factor], head: tuple[VarId, ...], size: Size
) -> tuple[list[Factor], Truths]:
    """The factors, with the variables outside the head and wide factors that have at most
    two neighbours eliminated, but for those of no variables; and the graphs where those
    hold (where nothing satisfies the factors, they don't)."""
    while True:
        wide = {v for f in factors if f.value.arity > 2 for v in f.variables}
        others = {v for f in factors for v in f.variables} - set(head) - wide
        candidates = [v for v in others if len(neighbours_of(factors, v)) <= 2]
        if not candidates:
            break
        variable = min(candidates, key=lambda v: (len(neighbours_of(factors, v)), v))
        factors = eliminate_one(factors, variable, neighbours_of(factors, variable), size)
    graphs = k.xp().ones(size.graphs, bool)
    for factor in factors:
        if not factor.variables:
            graphs &= factor.value.truth
    return [f for f in factors if f.variables], graphs


def grow(
    rows: Rows, pending: list[Factor], unbound: list[VarId], size: Size, lookups: Lookups
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
        factors = [pending[i] for i in sorted(used)]
        return extend(rows, variable, factors, k.full_vector(*size), size), used
    if wide:
        i = min(wide, key=lambda i: len(pending[i].value.array))
        return join_tuples(rows, pending[i], lookups), {i}
    variable = min(unbound, key=lambda v: k.count(domain_of(pending, v, size)))
    domain = domain_of(pending, variable, size)  # the unary factors on it, and more
    return extend(rows, variable, [], domain, size), set(unary[variable])


def extend(
    rows: Rows, variable: VarId, factors: list[Factor], domain: Vector, size: Size
) -> Iterator[Rows]:
    """Each row, with each value of ``variable`` in its graph's ``domain`` that ``factors``
    (unary on it, or binary from a bound variable to it) allow with that row: in blocks
    of at most :data:`BLOCK_ROWS` rows (but for a single row with more values)."""
    links: list[tuple[Indices, Matrix]] = []  # a bound variable's values, rows of what it allows
    for factor in factors:
        if factor.value.arity == 1:
            domain = domain & factor.value.vector
            continue
        matrix = factor.value.matrix
        if factor.variables[1] != variable:
            matrix = k.transpose(matrix, size.nodes)
        (other,) = set(factor.variables) - {variable}
        links.append((rows.column(other), matrix))
    variables = (*rows.variables, variable)
    graphs = rows.graphs
    words = max(k.words(size.nodes), 1)
    chunk = max(1, min(BLOCK_ROWS, CHUNK_WORDS // words))
    for start in range(0, len(rows.tuples), chunk):
        stop = min(start + chunk, len(rows.tuples))
        of = graphs[start:stop]
        candidates = domain[of]  # a copy: the domain of each row's graph
        for values, matrix in links:
            candidates &= matrix[of, values[start:stop]]
        counts = k.xp().bitwise_count(candidates).sum(axis=1)  # values per row
        for first, last in blocks_of(counts, BLOCK_ROWS):
            at, value = k.nonzero_bits(candidates[first:last])
            yield Rows(variables, k.xp().column_stack([rows.tuples[start + first + at], value]))


def join_tuples(rows: Rows, factor: Factor, lookups: Lookups) -> Iterator[Rows]:
    """Each row, with each tuple of a wide ``factor`` of its graph that agrees with it: in
    blocks of at most :data:`BLOCK_ROWS` rows (but for a single row with more tuples)."""
    tuples = factor.value.array
    shared = [v for v in factor.variables if v in rows.variables]
    added = [1 + factor.variables.index(v) for v in factor.variables if v not in rows.variables]
    # The tuples agreeing with row i (on its graph, and the variables both have):
    # order[low[i] : low[i] + matches[i]].
    found = lookup(lookups, factor, [0, *(1 + factor.variables.index(v) for v in shared)])
    order = found.order
    low, matches = found.matching(rows.columns(shared))
    variables = (*rows.variables, *(factor.variables[c - 1] for c in added))
    for first, last in blocks_of(matches, BLOCK_ROWS):
        some = matches[first:last]
        at = k.xp().repeat(k.xp().arange(first, last), some)
        within = k.xp().arange(int(some.sum())) - k.xp().repeat(k.xp().cumsum(some) - some, some)
        theirs = order[k.xp().repeat(low[first:last], some) + within]
        yield Rows(variables, k.xp().column_stack([rows.tuples[at], tuples[theirs][:, added]]))


def keep(rows: Rows, pending: list[Factor], lookups: Lookups) -> tuple[Rows, list[Factor]]:
    """The rows every factor over bound variables allows, and the factors left."""
    bound = set(rows.variables)
    keeping = k.xp().ones(len(rows.tuples), bool)
    left = []
    for factor in pending:
        if set(factor.variables) <= bound:
            keeping &= allows(rows, factor, lookups)
        else:
            left.append(factor)
    return Rows(rows.variables, rows.tuples[keeping]), left


def allows(rows: Rows, factor: Factor, lookups: Lookups) -> Truths:
    """Which rows a factor over bound variables allows."""
    of = rows.graphs
    columns = [rows.column(v) for v in factor.variables]
    match factor.value.arity:
        case 1:
            (c,) = columns
            found = (factor.value.vector[of, c >> 6] >> (c & 63).astype(np.uint64)) & k.ONE
        case 2:
            r, c = columns
            found = (factor.value.matrix[of, r, c >> 6] >> (c & 63).astype(np.uint64)) & k.ONE
        case _:
            every = lookup(lookups, factor, list(range(1 + len(factor.variables))))
            found = every.matching(rows.columns(factor.variables))[1] > 0
    allowed: Truths = found.astype(bool) != factor.negated
    return allowed


@dataclass(frozen=True)
class Lookup:
    """A wide factor's tuples sorted once by some of their columns, to find in bulk the
    tuples that agree with each row on those columns."""

    order: Indices
    """The tuples, by index, sorted by the columns."""
    keys: Any
    """Those columns of the tuples, in that order: one key per tuple (see ``row_keys``)."""
    base: int
    """More than any value in the columns."""

    @classmethod
    def of(cls, tuples: Tuples, columns: list[int], base: int) -> Lookup:
        keys = row_keys(tuples[:, columns], base)
        order = keys.argsort(kind="stable")
        return cls(k.asarray(order), keys[order], base)

    def matching(self, values: Tuples) -> tuple[Indices, Indices]:
        """For each row of ``values`` (of the columns): where its tuples start in
        :attr:`order`, and how many there are."""
        keys = row_keys(values, self.base)  # on the CPU if too wide for numbers
        low = self.keys.searchsorted(keys, "left")
        high = self.keys.searchsorted(keys, "right")
        return k.asarray(low), k.asarray(high - low)


type Lookups = dict[tuple[int, tuple[int, ...]], Lookup]
"""A join's lookups, by the ``id`` of the factor and the columns (the graph's is ``0``):
each is made once, and every block of rows uses it."""


def lookup(lookups: Lookups, factor: Factor, columns: list[int]) -> Lookup:
    key = (id(factor), tuple(columns))
    if key not in lookups:
        lookups[key] = Lookup.of(factor.value.array, columns, key_base(factor.value.size))
    return lookups[key]
