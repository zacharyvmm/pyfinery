# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Sets and binary relations over ``n`` nodes, as NumPy arrays of packed bits.

A *vector* is a set of nodes: ``uint64`` words of shape ``(words(n),)``, bit
``i % 64`` of word ``i // 64`` for node ``i``. A *matrix* is a binary relation:
shape ``(n, words(n))``, row ``i`` the set of ``j`` with ``(i, j)``. Bits past
``n`` are always zero, so that comparing, counting and complementing work on
whole words.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NodeIndex = int
"""A node, by its place in the model: ``0`` to ``n - 1``."""

type Vector = NDArray[np.uint64]
"""A set of nodes: ``words(n)`` words, bit ``i % 64`` of word ``i // 64`` for node ``i``."""

type Matrix = NDArray[np.uint64]
"""A binary relation: ``n`` rows of ``words(n)`` words, row ``i`` the set of ``j``
with ``(i, j)``."""

type Indices = NDArray[np.intp]
"""Nodes, by index: an array of :data:`NodeIndex`."""

ONE = np.uint64(1)


def words(n: int) -> int:
    return (n + 63) // 64


def empty_vector(n: int) -> Vector:
    return np.zeros(words(n), np.uint64)


def full_vector(n: int) -> Vector:
    vector = np.full(words(n), np.uint64(0xFFFF_FFFF_FFFF_FFFF), np.uint64)
    if n % 64:
        vector[-1] = (ONE << np.uint64(n % 64)) - ONE
    return vector


def empty_matrix(n: int) -> Matrix:
    return np.zeros((n, words(n)), np.uint64)


def vector_of(n: int, nodes: NDArray[np.integer] | list[NodeIndex]) -> Vector:
    """The set of ``nodes``."""
    found = np.asarray(nodes, dtype=np.int64)
    vector = empty_vector(n)
    np.bitwise_or.at(vector, found >> 6, ONE << (found & 63).astype(np.uint64))
    return vector


def matrix_of(n: int, rows: NDArray[np.integer], cols: NDArray[np.integer]) -> Matrix:
    """The relation of the pairs ``(rows[k], cols[k])``: work in the number of pairs."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    matrix = empty_matrix(n)
    np.bitwise_or.at(matrix, (rows, cols >> 6), ONE << (cols & 63).astype(np.uint64))
    return matrix


def pack(dense: NDArray[np.bool_]) -> NDArray[np.uint64]:
    """Pack the last axis of a boolean array, whose length is a multiple of 64."""
    packed = np.packbits(dense, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def unpack(bits: NDArray[np.uint64], n: int) -> NDArray[np.bool_]:
    """The last axis as booleans, ``n`` long."""
    raw = np.ascontiguousarray(bits).view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little", count=n).astype(bool)


def indices(vector: Vector, n: int) -> Indices:
    """The members of a vector, in order: work in the number of nonzero words."""
    (word,) = np.nonzero(vector)
    bits = bits_of(vector[word])
    at, bit = np.nonzero(bits)
    return (word[at] * 64 + bit).astype(np.intp)


def pairs(matrix: Matrix, n: int) -> tuple[Indices, Indices]:
    """The pairs of a matrix, by row then column: work in the number of nonzero words."""
    row, word = np.nonzero(matrix)
    bits = bits_of(matrix[row, word])
    at, bit = np.nonzero(bits)
    return row[at].astype(np.intp), (word[at] * 64 + bit).astype(np.intp)


def bits_of(values: NDArray[np.uint64]) -> NDArray[np.uint8]:
    """The 64 bits of each word, least significant first: shape ``(len(values), 64)``."""
    raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(raw, axis=1, bitorder="little")


def count(bits: Vector | Matrix) -> int:
    return int(np.bitwise_count(bits).sum())


def complement_vector(vector: Vector, n: int) -> Vector:
    return ~vector & full_vector(n)


def complement_matrix(matrix: Matrix, n: int) -> Matrix:
    return ~matrix & full_vector(n)


def transpose(matrix: Matrix, n: int) -> Matrix:
    rows, cols = pairs(matrix, n)
    return matrix_of(n, cols, rows)


def row_any(matrix: Matrix, n: int) -> Vector:
    """The rows with a pair: ``{i | ∃j (i, j)}``."""
    rows = np.zeros(words(n) * 64, bool)
    rows[:n] = matrix.any(axis=1)
    return pack(rows)


def column_any(matrix: Matrix) -> Vector:
    """The columns with a pair: ``{j | ∃i (i, j)}``."""
    if not len(matrix):  # no nodes
        return np.zeros(matrix.shape[1], np.uint64)
    columns: Vector = np.bitwise_or.reduce(matrix, axis=0)
    return columns


def mask_rows(matrix: Matrix, vector: Vector, n: int) -> Matrix:
    """The pairs whose row is in ``vector``."""
    keep = unpack(vector, n)
    return np.where(keep[:, None], matrix, np.uint64(0))


def mask_columns(matrix: Matrix, vector: Vector) -> Matrix:
    """The pairs whose column is in ``vector``."""
    return matrix & vector


def outer(rows: Vector, columns: Vector, n: int) -> Matrix:
    """The pairs of a row in ``rows`` and a column in ``columns``."""
    keep = unpack(rows, n)
    return np.where(keep[:, None], columns[None, :], np.uint64(0))


def diagonal(matrix: Matrix, n: int) -> Vector:
    """``{i | (i, i)}``."""
    nodes = np.arange(n)
    on = (matrix[nodes, nodes >> 6] >> (nodes & 63).astype(np.uint64)) & ONE
    return vector_of(n, nodes[on.astype(bool)])


def identity(n: int) -> Matrix:
    nodes = np.arange(n)
    return matrix_of(n, nodes, nodes)


def row(matrix: Matrix, i: NodeIndex) -> Vector:
    found: Vector = matrix[i].copy()
    return found


def column(matrix: Matrix, j: NodeIndex, n: int) -> Vector:
    word, bit = divmod(j, 64)
    selected = (matrix[:, word] >> np.uint64(bit)) & ONE
    dense = np.zeros(words(n) * 64, bool)
    dense[:n] = selected.astype(bool)
    return pack(dense)


def contains(vector: Vector, i: NodeIndex) -> bool:
    word, bit = divmod(i, 64)
    return bool((vector[word] >> np.uint64(bit)) & ONE)


def product(left: Matrix, right: Matrix, n: int) -> Matrix:
    """Boolean matrix product: ``{(i, j) | ∃k (i, k) ∈ left, (k, j) ∈ right}``.

    Gathers the rows of ``right`` that each row of ``left`` names and ORs them
    together, so the work grows with the pairs of ``left``. When ``left`` is
    dense, a float product of the unpacked matrices is cheaper.
    """
    pairs_left = count(left)
    if pairs_left == 0:
        return empty_matrix(n)
    if pairs_left > n * n // 16 and n > 64:
        dense = unpack(left, n).astype(np.float32) @ unpack(right, n).astype(np.float32)
        padded = np.zeros((n, words(n) * 64), bool)
        padded[:, :n] = dense > 0
        return pack(padded)
    rows, cols = pairs(left, n)
    result = empty_matrix(n)
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    result[rows[starts]] = np.bitwise_or.reduceat(right[cols], starts, axis=0)
    return result


def closure(matrix: Matrix, n: int) -> Matrix:
    """Transitive closure: the pairs joined by a path of one or more steps.

    Strongly connected components are condensed (Tarjan's algorithm), then
    visited from the sinks back: every node of a component reaches the same
    nodes, its own successors and what the other components it steps to
    reach. Each component is visited once, so the cost grows with the edges
    and the size of the result, whatever order the nodes are in.
    """
    rows, cols = pairs(matrix, n)
    starts = np.searchsorted(rows, np.arange(n + 1))
    successors = [cols[starts[i] : starts[i + 1]].tolist() for i in range(n)]
    result = empty_matrix(n)
    for component in strongly_connected_components(successors):
        members = np.asarray(component)
        # In a cycle, the successors of the members include all of them.
        reached = np.bitwise_or.reduce(matrix[members], axis=0)
        inside = set(component)
        outside = sorted({w for v in component for w in successors[v]} - inside)
        if outside:
            reached = reached | np.bitwise_or.reduce(result[outside], axis=0)
        result[members] = reached
    return result


def strongly_connected_components(
    successors: list[list[NodeIndex]],
) -> list[list[NodeIndex]]:
    """Strongly connected components, sinks first (Tarjan's algorithm, iteratively)."""
    n = len(successors)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            node, child = work.pop()
            if child == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            if child < len(successors[node]):
                work.append((node, child + 1))
                successor = successors[node][child]
                if index[successor] == -1:
                    work.append((successor, 0))
                elif on_stack[successor]:
                    low[node] = min(low[node], index[successor])
                continue
            for successor in successors[node]:
                if on_stack[successor] and index[successor] > index[node]:
                    low[node] = min(low[node], low[successor])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components
