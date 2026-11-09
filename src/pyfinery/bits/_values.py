# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Sets of tuples of nodes: packed bits up to two nodes, arrays of tuples beyond."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _kernels as k

Tuples = NDArray[np.intp]


@dataclass(frozen=True)
class Value:
    """A set of tuples of ``arity`` nodes out of ``n``.

    ``data`` is a ``bool`` for arity 0, a packed vector for 1, a packed matrix
    for 2, and for 3 or more, an array of unique tuples of shape ``(m, arity)``.
    """

    arity: int
    n: int
    data: Any


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
            return np.empty((1 if value.data else 0, 0), np.intp)
        case 1:
            return k.indices(value.data, value.n)[:, None]
        case 2:
            rows, cols = k.pairs(value.data, value.n)
            return np.stack([rows, cols], axis=1)
        case _:
            return value.data


def is_empty(value: Value) -> bool:
    if value.arity == 0:
        return not value.data
    return not np.any(value.data) if value.arity <= 2 else len(value.data) == 0


def union(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.n, left.data or right.data)
        case 1 | 2:
            return Value(left.arity, left.n, left.data | right.data)
        case _:
            return from_tuples(left.arity, left.n, np.concatenate([left.data, right.data]))


def intersection(left: Value, right: Value) -> Value:
    match left.arity:
        case 0:
            return Value(0, left.n, left.data and right.data)
        case 1 | 2:
            return Value(left.arity, left.n, left.data & right.data)
        case _:
            mine = {tuple(t) for t in left.data.tolist()}
            kept = [t for t in right.data.tolist() if tuple(t) in mine]
            return from_tuples(left.arity, left.n, np.array(kept, np.intp))


def complement(value: Value) -> Value:
    match value.arity:
        case 0:
            return Value(0, value.n, not value.data)
        case 1:
            return Value(1, value.n, k.complement_vector(value.data, value.n))
        case 2:
            return Value(2, value.n, k.complement_matrix(value.data, value.n))
        case _:
            raise ValueError("the complement of three or more nodes is too large to hold")


def difference(left: Value, right: Value) -> Value:
    """The tuples of ``left`` that aren't in ``right``."""
    match left.arity:
        case 0:
            return Value(0, left.n, left.data and not right.data)
        case 1 | 2:
            return Value(left.arity, left.n, left.data & ~right.data)
        case _:
            theirs = {tuple(t) for t in right.data.tolist()}
            kept = [t for t in left.data.tolist() if tuple(t) not in theirs]
            return from_tuples(left.arity, left.n, np.array(kept, np.intp))


def transpose(value: Value) -> Value:
    assert value.arity == 2
    return Value(2, value.n, k.transpose(value.data, value.n))
