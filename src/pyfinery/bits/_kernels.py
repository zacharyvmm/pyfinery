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

Bits = NDArray[np.uint64]

_ONE = np.uint64(1)


def words(n: int) -> int:
    return (n + 63) // 64


def empty_vector(n: int) -> Bits:
    return np.zeros(words(n), np.uint64)


def full_vector(n: int) -> Bits:
    vector = np.full(words(n), np.uint64(0xFFFF_FFFF_FFFF_FFFF), np.uint64)
    if n % 64:
        vector[-1] = (_ONE << np.uint64(n % 64)) - _ONE
    return vector


def empty_matrix(n: int) -> Bits:
    return np.zeros((n, words(n)), np.uint64)


def vector_of(n: int, indices: NDArray[np.integer] | list[int]) -> Bits:
    """The set of ``indices``."""
    dense = np.zeros(words(n) * 64, bool)
    dense[np.asarray(indices, dtype=np.int64)] = True
    return pack(dense)


def matrix_of(n: int, rows: NDArray[np.integer], cols: NDArray[np.integer]) -> Bits:
    """The relation of the pairs ``(rows[k], cols[k])``."""
    dense = np.zeros((n, words(n) * 64), bool)
    dense[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = True
    return pack(dense)


def pack(dense: NDArray[np.bool_]) -> Bits:
    """Pack the last axis of a boolean array, whose length is a multiple of 64."""
    packed = np.packbits(dense, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def unpack(bits: Bits, n: int) -> NDArray[np.bool_]:
    """The last axis as booleans, ``n`` long."""
    raw = np.ascontiguousarray(bits).view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little", count=n).astype(bool)


def indices(vector: Bits, n: int) -> NDArray[np.intp]:
    return np.flatnonzero(unpack(vector, n))


def pairs(matrix: Bits, n: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    rows, cols = np.nonzero(unpack(matrix, n))
    return rows, cols


def count(bits: Bits) -> int:
    return int(np.bitwise_count(bits).sum())


def complement_vector(vector: Bits, n: int) -> Bits:
    return ~vector & full_vector(n)


def complement_matrix(matrix: Bits, n: int) -> Bits:
    return ~matrix & full_vector(n)


def transpose(matrix: Bits, n: int) -> Bits:
    dense = unpack(matrix, n)
    padded = np.zeros((n, words(n) * 64), bool)
    padded[:, :n] = dense.T
    return pack(padded)


def row_any(matrix: Bits, n: int) -> Bits:
    """The rows with a pair: ``{i | ∃j (i, j)}``."""
    rows = np.zeros(words(n) * 64, bool)
    rows[:n] = matrix.any(axis=1)
    return pack(rows)


def column_any(matrix: Bits) -> Bits:
    """The columns with a pair: ``{j | ∃i (i, j)}``."""
    return np.bitwise_or.reduce(matrix, axis=0) if len(matrix) else matrix.sum(axis=0)


def mask_rows(matrix: Bits, vector: Bits, n: int) -> Bits:
    """The pairs whose row is in ``vector``."""
    keep = unpack(vector, n)
    return np.where(keep[:, None], matrix, np.uint64(0))


def mask_columns(matrix: Bits, vector: Bits) -> Bits:
    """The pairs whose column is in ``vector``."""
    return matrix & vector


def outer(rows: Bits, columns: Bits, n: int) -> Bits:
    """The pairs of a row in ``rows`` and a column in ``columns``."""
    keep = unpack(rows, n)
    return np.where(keep[:, None], columns[None, :], np.uint64(0))


def diagonal(matrix: Bits, n: int) -> Bits:
    """``{i | (i, i)}``."""
    dense = unpack(matrix, n)
    return vector_of(n, np.flatnonzero(np.diagonal(dense)))


def identity(n: int) -> Bits:
    nodes = np.arange(n)
    return matrix_of(n, nodes, nodes)


def row(matrix: Bits, i: int) -> Bits:
    return matrix[i].copy()


def column(matrix: Bits, j: int, n: int) -> Bits:
    word, bit = divmod(j, 64)
    selected = (matrix[:, word] >> np.uint64(bit)) & _ONE
    dense = np.zeros(words(n) * 64, bool)
    dense[:n] = selected.astype(bool)
    return pack(dense)


def contains(vector: Bits, i: int) -> bool:
    word, bit = divmod(i, 64)
    return bool((vector[word] >> np.uint64(bit)) & _ONE)


def product(left: Bits, right: Bits, n: int) -> Bits:
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


def closure(matrix: Bits, n: int) -> Bits:
    """Transitive closure: the pairs joined by a path of one or more steps.

    Strongly connected components are condensed (Tarjan's algorithm), then
    visited from the sinks back: every node of a component reaches the same
    nodes, its own successors and what the other components it steps to
    reach. Each component is visited once, so the cost grows with the edges
    and the size of the result, whatever order the nodes are in.
    """
    successors = [np.flatnonzero(r).tolist() for r in unpack(matrix, n)]
    result = empty_matrix(n)
    for component in _components(successors):
        members = np.asarray(component)
        # In a cycle, the successors of the members include all of them.
        reached = np.bitwise_or.reduce(matrix[members], axis=0)
        inside = set(component)
        outside = sorted({w for v in component for w in successors[v]} - inside)
        if outside:
            reached = reached | np.bitwise_or.reduce(result[outside], axis=0)
        result[members] = reached
    return result


def _components(successors: list[list[int]]) -> list[list[int]]:
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
