# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Sets of tuples of nodes: packed bits up to two nodes, arrays of tuples beyond."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from . import kernels as k
from .kernels import Indices, Matrix, NodeIndex, Vector
from .syntax import Modality

type NodeTuple = tuple[NodeIndex, ...]
"""A tuple of nodes, e.g. the arguments of a predicate."""

type Tuples = NDArray[np.intp]
"""Tuples of nodes: an array of shape ``(m, arity)``, a :data:`NodeTuple` per row."""


@dataclass(frozen=True)
class Value:
    """A set of tuples of ``arity`` nodes out of ``n``.

    ``data`` depends on the arity: whether the empty tuple is in the set for 0,
    a :data:`~.kernels.Vector` for 1, a :data:`~.kernels.Matrix` for 2, and for
    3 or more, :data:`Tuples` without repeats. Read it as one of those with
    :attr:`truth`, :attr:`vector`, :attr:`matrix` or :attr:`array`.
    """

    arity: int
    n: int
    data: bool | Vector | Matrix | Tuples

    @property
    def truth(self) -> bool:
        assert self.arity == 0
        return cast(bool, self.data)

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
        assert self.arity >= 3
        return cast(Tuples, self.data)


class Modal(NamedTuple):
    """A relation's values: the tuples that must hold, and those that may."""

    must: Value
    may: Value

    def read(self, modality: Modality) -> Value:
        return self.must if modality == "must" else self.may


def empty(arity: int, n: int) -> Value:
    match arity:
        case 0:
            return Value(0, n, False)
        case 1:
            return Value(1, n, k.empty_vector(n))
        case 2:
            return Value(2, n, k.empty_matrix(n))
        case _:
            return Value(arity, n, np.empty((0, arity), np.intp))


def full(arity: int, n: int) -> Value:
    match arity:
        case 0:
            return Value(0, n, True)
        case 1:
            return Value(1, n, k.full_vector(n))
        case 2:
            return Value(2, n, np.tile(k.full_vector(n), (n, 1)))
        case _:
            raise ValueError("a full relation of three or more nodes is too large to hold")


def from_tuples(arity: int, n: int, tuples: Tuples) -> Value:
    if arity == 0:  # an array of empty tuples: is there one?
        return Value(0, n, len(tuples) > 0)
    tuples = np.asarray(tuples, np.intp).reshape(-1, arity)
    match arity:
        case 1:
            return Value(1, n, k.vector_of(n, tuples[:, 0]))
        case 2:
            return Value(2, n, k.matrix_of(n, tuples[:, 0], tuples[:, 1]))
        case _:
            return Value(arity, n, np.unique(tuples, axis=0) if len(tuples) else tuples)


def to_tuples(value: Value) -> Tuples:
    match value.arity:
        case 0:
            return np.empty((1 if value.truth else 0, 0), np.intp)
        case 1:
            return k.indices(value.vector, value.n)[:, None]
        case 2:
            rows, cols = k.pairs(value.matrix, value.n)
            return np.stack([rows, cols], axis=1)
        case _:
            return value.array


def is_empty(value: Value) -> bool:
    match value.arity:
        case 0:
            return not value.truth
        case 1:
            return not np.any(value.vector)
        case 2:
            return not np.any(value.matrix)
        case _:
            return len(value.array) == 0


def union(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.n, left.truth or right.truth)
        case 1:
            return Value(1, left.n, left.vector | right.vector)
        case 2:
            return Value(2, left.n, left.matrix | right.matrix)
        case _:
            return from_tuples(left.arity, left.n, np.concatenate([left.array, right.array]))


def intersection(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.n, left.truth and right.truth)
        case 1:
            return Value(1, left.n, left.vector & right.vector)
        case 2:
            return Value(2, left.n, left.matrix & right.matrix)
        case _:
            mine = {tuple(t) for t in left.array.tolist()}
            kept = [t for t in right.array.tolist() if tuple(t) in mine]
            return from_tuples(left.arity, left.n, np.array(kept, np.intp))


def complement(value: Value) -> Value:
    match value.arity:
        case 0:
            return Value(0, value.n, not value.truth)
        case 1:
            return Value(1, value.n, k.complement_vector(value.vector, value.n))
        case 2:
            return Value(2, value.n, k.complement_matrix(value.matrix, value.n))
        case _:
            raise ValueError("the complement of three or more nodes is too large to hold")


def difference(left: Value, right: Value) -> Value:
    """The tuples of ``left`` that aren't in ``right``."""
    match left.arity:
        case 0:
            return Value(0, left.n, left.truth and not right.truth)
        case 1:
            return Value(1, left.n, left.vector & ~right.vector)
        case 2:
            return Value(2, left.n, left.matrix & ~right.matrix)
        case _:
            theirs = {tuple(t) for t in right.array.tolist()}
            kept = [t for t in left.array.tolist() if tuple(t) not in theirs]
            return from_tuples(left.arity, left.n, np.array(kept, np.intp))


def transpose(value: Value) -> Value:
    return Value(2, value.n, k.transpose(value.matrix, value.n))


def relabel(value: Value, n: int, placement: Indices) -> Value:
    """The same tuples among ``n`` nodes, where node ``i`` is now ``placement[i]``."""
    if value.arity == 0:
        return Value(0, n, value.truth)
    return from_tuples(value.arity, n, placement[to_tuples(value)])
