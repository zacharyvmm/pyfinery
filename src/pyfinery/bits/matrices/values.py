# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Sets of tuples of nodes, of each graph of a batch: packed bits up to two nodes,
arrays of tuples beyond."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from . import kernels as k
from .kernels import Indices, Matrix, NodeIndex, Truths, Vector

type NodeTuple = tuple[NodeIndex, ...]
"""A tuple of nodes, e.g. the arguments of a predicate."""

type Tuples = NDArray[np.intp]
"""Tuples of nodes: an array of shape ``(m, arity)``, a :data:`NodeTuple` per row. Of a
batch, each row starts with its graph: shape ``(m, 1 + arity)``."""

type Bits = bool | Vector | Matrix | Tuples
"""A set of tuples of nodes, as numpy holds it, by arity: whether the empty tuple is
in it (0); a :data:`~.kernels.Vector` (1); a :data:`~.kernels.Matrix` (2); or
:data:`Tuples` without repeats (3 or more). Of one graph, without the batch's axis
(or its column); of a batch, with it: :data:`~.kernels.Truths` for 0."""


type BatchBits = Truths | Vector | Matrix | Tuples
"""A set of tuples of nodes of each graph of a batch, by arity: :data:`~.kernels.Truths`
(0); a :data:`~.kernels.Vector` (1); a :data:`~.kernels.Matrix` (2); or :data:`Tuples`
without repeats, each after its graph (3 or more)."""


class Size(NamedTuple):
    """How many graphs a batch has, and how many node slots each."""

    graphs: int
    nodes: int


@dataclass(frozen=True)
class Value:
    """A set of tuples of ``arity`` nodes, of each graph of a batch of ``size``.

    ``data`` is batched :data:`Bits` of that arity; read it as what it is with
    :attr:`truth`, :attr:`vector`, :attr:`matrix` or :attr:`array`. The set
    operations below work on any arity, on every graph at once.
    """

    arity: int
    size: Size
    data: BatchBits

    @property
    def n(self) -> int:
        return self.size.nodes

    @property
    def truth(self) -> Truths:
        assert self.arity == 0
        return cast(Truths, self.data)

    @property
    def vector(self) -> Vector:
        assert self.arity == 1
        return cast(Vector, self.data)

    @property
    def matrix(self) -> Matrix:
        assert self.arity == 2
        return cast(Matrix, self.data)

    @property
    def array(self) -> Tuples:
        """The tuples, each after its graph: shape ``(m, 1 + arity)``."""
        assert self.arity >= 3
        return cast(Tuples, self.data)


def empty(arity: int, size: Size) -> Value:
    graphs, n = size
    match arity:
        case 0:
            return Value(0, size, np.zeros(graphs, bool))
        case 1:
            return Value(1, size, k.empty_vector(graphs, n))
        case 2:
            return Value(2, size, k.empty_matrix(graphs, n))
        case _:
            return Value(arity, size, np.empty((0, 1 + arity), np.intp))


def full(arity: int, size: Size) -> Value:
    graphs, n = size
    match arity:
        case 0:
            return Value(0, size, np.ones(graphs, bool))
        case 1:
            return Value(1, size, k.full_vector(graphs, n))
        case 2:
            return Value(2, size, k.full_matrix(graphs, n))
        case _:
            raise ValueError("a full relation of three or more nodes is too large to hold")


def from_tuples(arity: int, size: Size, tuples: Tuples) -> Value:
    """The tuples of ``tuples``, each row its graph and then its nodes."""
    graphs, n = size
    tuples = np.asarray(tuples, np.intp).reshape(-1, 1 + arity)
    match arity:
        case 0:
            truth = np.zeros(graphs, bool)
            truth[tuples[:, 0]] = True
            return Value(0, size, truth)
        case 1:
            return Value(1, size, k.vector_of(graphs, n, tuples[:, 0], tuples[:, 1]))
        case 2:
            return Value(2, size, k.matrix_of(graphs, n, tuples[:, 0], tuples[:, 1], tuples[:, 2]))
        case _:
            return Value(arity, size, unique_rows(tuples, size))


def unique_rows(tuples: Tuples, size: Size) -> Tuples:
    """The distinct rows (a graph, then nodes), in order: sorted as numbers, one per row,
    when the rows fit in one, else as rows."""
    if len(tuples) < 2:
        return tuples
    width = tuples.shape[1]
    base = max(size.graphs, size.nodes, 1)
    if base**width >= 2**63:
        return np.unique(tuples, axis=0)
    bases = base ** np.arange(width - 1, -1, -1, dtype=np.int64)
    keys = np.unique(tuples.astype(np.int64) @ bases)
    rows: Tuples = ((keys[:, None] // bases) % base).astype(np.intp)
    return rows


def to_tuples(value: Value) -> Tuples:
    """The tuples, each row its graph and then its nodes, ordered by graph."""
    match value.arity:
        case 0:
            return np.flatnonzero(value.truth).astype(np.intp)[:, None]
        case 1:
            return np.stack(k.indices(value.vector), axis=1)
        case 2:
            return np.stack(k.pairs(value.matrix), axis=1)
        case _:
            return value.array


def is_empty(value: Value) -> bool:
    """Whether no graph has a tuple."""
    match value.arity:
        case 0:
            return not value.truth.any()
        case 1 | 2:
            return not np.any(value.data)
        case _:
            return len(value.array) == 0


def only_in(value: Value, graphs: Truths) -> Value:
    """The tuples of the graphs ``graphs`` holds for."""
    match value.arity:
        case 0:
            return Value(0, value.size, value.truth & graphs)
        case 1:
            return Value(1, value.size, np.where(graphs[:, None], value.vector, np.uint64(0)))
        case 2:
            return Value(2, value.size, np.where(graphs[:, None, None], value.matrix, np.uint64(0)))
        case _:
            return Value(value.arity, value.size, value.array[graphs[value.array[:, 0]]])


def union(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.size, left.truth | right.truth)
        case 1 | 2:
            return Value(left.arity, left.size, left.data | right.data)
        case _:
            return from_tuples(left.arity, left.size, np.concatenate([left.array, right.array]))


def intersection(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.size, left.truth & right.truth)
        case 1 | 2:
            return Value(left.arity, left.size, left.data & right.data)
        case _:
            kept = left.array[members(left.array, right.array)]
            return Value(left.arity, left.size, kept)


def complement(value: Value) -> Value:
    match value.arity:
        case 0:
            return Value(0, value.size, ~value.truth)
        case 1 | 2:
            return Value(value.arity, value.size, k.complement(value.data, value.n))  # type: ignore[arg-type]
        case _:
            raise ValueError("the complement of three or more nodes is too large to hold")


def difference(left: Value, right: Value) -> Value:
    """The tuples of ``left`` that aren't in ``right``."""
    match left.arity:
        case 0:
            return Value(0, left.size, left.truth & ~right.truth)
        case 1 | 2:
            return Value(left.arity, left.size, left.data & ~right.data)
        case _:
            kept = left.array[~members(left.array, right.array)]
            return Value(left.arity, left.size, kept)


def transpose(value: Value) -> Value:
    return Value(2, value.size, k.transpose(value.matrix, value.n))


def relabel(value: Value, size: Size, placement: Indices) -> Value:
    """The same tuples among ``size``'s slots, where node ``i`` of graph ``g`` is now
    ``placement[g, i]``."""
    tuples = to_tuples(value).copy()
    tuples[:, 1:] = placement[tuples[:, :1], tuples[:, 1:]]
    return from_tuples(value.arity, size, tuples)


# -- tuples as keys -------------------------------------------------------------------


def members(tuples: Tuples, of: Tuples) -> NDArray[np.bool_]:
    """Which rows of ``tuples`` are rows of ``of``."""
    if not len(tuples) or not len(of):
        return np.zeros(len(tuples), bool)
    keys = np.sort(row_keys(of))
    found = row_keys(tuples)
    at = np.minimum(np.searchsorted(keys, found), len(keys) - 1)
    found_at: NDArray[np.bool_] = keys[at] == found
    return found_at


def row_keys(columns: Tuples) -> NDArray[Any]:
    """A key per row, equal where the rows are: the value for one column, else the row's
    bytes (whose order isn't numeric, but is the same for every array of rows)."""
    if columns.shape[1] == 1:
        return np.ascontiguousarray(columns[:, 0], np.intp)
    rows = np.ascontiguousarray(columns, np.intp)
    return rows.view(np.dtype((np.void, rows.itemsize * rows.shape[1]))).ravel()


# -- one graph ------------------------------------------------------------------------


def of_graph(arity: int, n: int, bits: Bits) -> Value:
    """One graph's :data:`Bits`, as a batch of that graph."""
    size = Size(1, n)
    match arity:
        case 0:
            return Value(0, size, np.array([bool(bits)]))
        case 1 | 2:
            return Value(arity, size, np.asarray(bits)[None])
        case _:
            tuples = np.asarray(bits, np.intp).reshape(-1, arity)
            return Value(arity, size, np.column_stack([np.zeros(len(tuples), np.intp), tuples]))


def graph_bits(value: Value, graph: int = 0) -> Bits:
    """The :data:`Bits` of one graph of a batch."""
    match value.arity:
        case 0:
            return bool(value.truth[graph])
        case 1 | 2:
            found: Vector | Matrix = value.data[graph]
            return found
        case _:
            tuples = value.array
            own: Tuples = tuples[tuples[:, 0] == graph][:, 1:]
            return own


def with_graph(tuples: Tuples, graph: int = 0) -> Tuples:
    """Tuples of one graph, each after the graph's index."""
    tuples = np.asarray(tuples, np.intp)
    if tuples.ndim == 1:  # tuples of no nodes
        tuples = tuples.reshape(len(tuples), 0)
    return np.column_stack([np.full(len(tuples), graph, np.intp), tuples])
