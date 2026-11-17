# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Sets and binary relations over the nodes of a batch of graphs, as NumPy arrays of
packed bits.

Every graph of a batch has the same ``n`` node slots (a smaller graph leaves some
empty). A *vector* is a set of nodes of each graph: ``uint64`` words of shape
``(graphs, words(n))``, bit ``i % 64`` of word ``i // 64`` for node ``i``. A
*matrix* is a binary relation of each graph: shape ``(graphs, n, words(n))``, row
``i`` of graph ``g`` the set of ``j`` with ``(i, j)`` in ``g``. Bits past ``n``
are always zero, so that comparing, counting and complementing work on whole
words. Graphs never share a tuple: each kernel works on each graph at once, as
the same array operation over the leading axis.

These are the only functions that do arithmetic on bits, and each is one or a
few whole-array operations whatever the number of graphs; another array library
(e.g. on a GPU) would reimplement them, and nothing else. The exception is
:func:`closure` of graphs of more than :data:`SQUARING_NODES` nodes, which visits
strongly connected components in Python; that of smaller ones
(:func:`closure_by_squaring`) is array operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NodeIndex = int
"""A node, by its slot in each graph: ``0`` to ``n - 1``."""

type Vector = NDArray[np.uint64]
"""A set of nodes of each graph: shape ``(graphs, words(n))``, bit ``i % 64`` of word
``i // 64`` for node ``i``."""

type Matrix = NDArray[np.uint64]
"""A binary relation of each graph: shape ``(graphs, n, words(n))``, row ``i`` of graph
``g`` the set of ``j`` with ``(i, j)``."""

type Truths = NDArray[np.bool_]
"""Whether something holds in each graph: shape ``(graphs,)``."""

type Indices = NDArray[np.intp]
"""Nodes or graphs, by index."""

ONE = np.uint64(1)

CHUNK_WORDS = 1 << 22
"""How many words :func:`product` gathers, or unpacks as floats, at a time: 32 MiB."""


def words(n: int) -> int:
    return (n + 63) // 64


def full_words(n: int) -> NDArray[np.uint64]:
    """The words of the set of every slot: shape ``(words(n),)``, broadcast over graphs."""
    vector = np.full(words(n), np.uint64(0xFFFF_FFFF_FFFF_FFFF), np.uint64)
    if n % 64:
        vector[-1] = (ONE << np.uint64(n % 64)) - ONE
    return vector


def empty_vector(graphs: int, n: int) -> Vector:
    return np.zeros((graphs, words(n)), np.uint64)


def full_vector(graphs: int, n: int) -> Vector:
    return np.tile(full_words(n), (graphs, 1))


def empty_matrix(graphs: int, n: int) -> Matrix:
    return np.zeros((graphs, n, words(n)), np.uint64)


def full_matrix(graphs: int, n: int) -> Matrix:
    return np.tile(full_words(n), (graphs, n, 1))


def vector_of(graphs: int, n: int, of: NDArray[np.integer], nodes: NDArray[np.integer]) -> Vector:
    """The set of node ``nodes[k]`` of graph ``of[k]``, for each ``k``."""
    found = np.asarray(nodes, dtype=np.int64)
    vector = empty_vector(graphs, n)
    np.bitwise_or.at(vector, (np.asarray(of), found >> 6), ONE << (found & 63).astype(np.uint64))
    return vector


def matrix_of(
    graphs: int,
    n: int,
    of: NDArray[np.integer],
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
) -> Matrix:
    """The relation of the pairs ``(rows[k], cols[k])`` of graph ``of[k]``: work in the
    number of pairs."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    matrix = empty_matrix(graphs, n)
    at = (np.asarray(of), rows, cols >> 6)
    np.bitwise_or.at(matrix, at, ONE << (cols & 63).astype(np.uint64))
    return matrix


def pack(dense: NDArray[np.bool_]) -> NDArray[np.uint64]:
    """Pack the last axis of a boolean array, whose length is a multiple of 64."""
    packed = np.packbits(dense, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def pack_padded(dense: NDArray[np.bool_], n: int) -> NDArray[np.uint64]:
    """Pack the last axis of a boolean array, ``n`` long."""
    padded = np.zeros((*dense.shape[:-1], words(n) * 64), bool)
    padded[..., :n] = dense
    return pack(padded)


def unpack(bits: NDArray[np.uint64], n: int) -> NDArray[np.bool_]:
    """The last axis as booleans, ``n`` long."""
    raw = np.ascontiguousarray(bits).view(np.uint8)
    return np.unpackbits(raw, axis=-1, bitorder="little", count=n).astype(bool)


def nonzero_bits(bits: NDArray[np.uint64]) -> tuple[Indices, ...]:
    """Where the bits are set, as :func:`numpy.nonzero` would say of the unpacked array:
    an index array per leading axis, then the bit's place, in order: work in the number
    of nonzero words."""
    *leading, word = np.nonzero(bits)
    found = bits_of(bits[(*leading, word)])
    at, bit = np.nonzero(found)
    return (*(axis[at].astype(np.intp) for axis in leading), (word[at] * 64 + bit).astype(np.intp))


def indices(vector: Vector) -> tuple[Indices, Indices]:
    """The members of a vector: their graphs and nodes, in order."""
    graphs, nodes = nonzero_bits(vector)
    return graphs, nodes


def pairs(matrix: Matrix) -> tuple[Indices, Indices, Indices]:
    """The pairs of a matrix: their graphs, rows and columns, in order."""
    graphs, rows, cols = nonzero_bits(matrix)
    return graphs, rows, cols


def bits_of(values: NDArray[np.uint64]) -> NDArray[np.uint8]:
    """The 64 bits of each word, least significant first: shape ``(len(values), 64)``."""
    raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(raw, axis=1, bitorder="little")


def count(bits: Vector | Matrix) -> int:
    """How many members, in every graph."""
    return int(np.bitwise_count(bits).sum())


def complement(bits: Vector | Matrix, n: int) -> Vector | Matrix:
    return ~bits & full_words(n)


def transpose(matrix: Matrix, n: int) -> Matrix:
    graphs, rows, cols = pairs(matrix)
    return matrix_of(len(matrix), n, graphs, cols, rows)


def row_any(matrix: Matrix, n: int) -> Vector:
    """The rows with a pair: ``{i | ∃j (i, j)}``."""
    return pack_padded(matrix.any(axis=-1), n)


def column_any(matrix: Matrix) -> Vector:
    """The columns with a pair: ``{j | ∃i (i, j)}``."""
    if not matrix.shape[1]:  # no nodes
        return np.zeros((len(matrix), matrix.shape[2]), np.uint64)
    columns: Vector = np.bitwise_or.reduce(matrix, axis=1)
    return columns


def mask_rows(matrix: Matrix, vector: Vector, n: int) -> Matrix:
    """The pairs whose row is in ``vector``."""
    keep = unpack(vector, n)
    return np.where(keep[..., None], matrix, np.uint64(0))


def mask_columns(matrix: Matrix, vector: Vector) -> Matrix:
    """The pairs whose column is in ``vector``."""
    return matrix & vector[:, None, :]


def outer(rows: Vector, columns: Vector, n: int) -> Matrix:
    """The pairs of a row in ``rows`` and a column in ``columns``."""
    keep = unpack(rows, n)
    return np.where(keep[..., None], columns[:, None, :], np.uint64(0))


def diagonal(matrix: Matrix, n: int) -> Vector:
    """``{i | (i, i)}``."""
    nodes = np.arange(n)
    on = (matrix[:, nodes, nodes >> 6] >> (nodes & 63).astype(np.uint64)) & ONE
    return pack_padded(on.astype(bool), n)


def identity(graphs: int, n: int) -> Matrix:
    nodes = np.arange(n)
    one = matrix_of(1, n, np.zeros(n, np.intp), nodes, nodes)
    return np.repeat(one, graphs, axis=0)


def row(matrix: Matrix, i: NodeIndex) -> Vector:
    found: Vector = matrix[:, i].copy()
    return found


def column(matrix: Matrix, j: NodeIndex, n: int) -> Vector:
    word, bit = divmod(j, 64)
    selected = (matrix[:, :, word] >> np.uint64(bit)) & ONE
    return pack_padded(selected.astype(bool), n)


def contains(vector: Vector, i: NodeIndex) -> Truths:
    """Whether node ``i`` is in the vector, in each graph."""
    word, bit = divmod(i, 64)
    found: Truths = ((vector[:, word] >> np.uint64(bit)) & ONE).astype(bool)
    return found


def product(left: Matrix, right: Matrix, n: int) -> Matrix:
    """Boolean matrix product, of each graph: ``{(i, j) | ∃k (i, k) ∈ left, (k, j) ∈ right}``.

    Gathers the rows of ``right`` that each row of ``left`` names and ORs them
    together, so the work grows with the pairs of ``left``. When ``left`` is
    dense, a float product of the unpacked matrices is cheaper.
    """
    graphs = len(left)
    pairs_left = count(left)
    if pairs_left == 0:
        return empty_matrix(graphs, n)
    if pairs_left > graphs * n * n // 16 and n > 64:
        return dense_product(left, right, n)
    of, rows, cols = pairs(left)
    flat_right = right.reshape(graphs * n, -1)
    targets = of * n + rows  # the row of the result each pair adds to
    starts = np.flatnonzero(np.r_[True, targets[1:] != targets[:-1]])
    result = np.zeros_like(flat_right)
    # A bounded number of gathered rows at a time, whole rows of the result each.
    step = max(1, CHUNK_WORDS // max(flat_right.shape[1], 1))
    first = 0
    while first < len(starts):
        last = int(np.searchsorted(starts, starts[first] + step, "left"))
        last = max(last, first + 1)
        begin = starts[first]
        end = starts[last] if last < len(starts) else len(targets)
        gathered = flat_right[(of * n + cols)[begin:end]]
        result[targets[starts[first:last]]] = np.bitwise_or.reduceat(
            gathered, starts[first:last] - begin, axis=0
        )
        first = last
    return result.reshape(right.shape)


def dense_product(left: Matrix, right: Matrix, n: int) -> Matrix:
    """:func:`product` as a float product of the unpacked matrices, a few graphs at a time."""
    graphs = len(left)
    result = np.empty_like(left)
    step = max(1, CHUNK_WORDS * 2 // max(n * n, 1))  # float32: half a word an entry
    for start in range(0, graphs, step):
        part = slice(start, start + step)
        dense = unpack(left[part], n).astype(np.float32) @ unpack(right[part], n).astype(np.float32)
        result[part] = pack_padded(dense > 0, n)
    return result


SQUARING_NODES = 256
"""The most nodes graphs may have for :func:`closure` to square matrices rather than
visit components: up to this, squaring was faster for batches of 1 to 1,000 graphs,
sparse or dense; from twice it, slower."""


def closure(matrix: Matrix, n: int) -> Matrix:
    """Transitive closure, of each graph: the pairs joined by a path of one or more steps.

    Small graphs square the relation (:func:`closure_by_squaring`), in array
    operations; larger ones visit their strongly connected components
    (:func:`closure_by_components`), whose cost grows with the edges and the result.
    """
    if n <= SQUARING_NODES:
        return closure_by_squaring(matrix, n)
    return closure_by_components(matrix, n)


def closure_by_squaring(matrix: Matrix, n: int) -> Matrix:
    """``R+`` as ``R ∪ R·R ∪ ...``: each step doubles the length of the paths, until a
    step adds nothing (at most ``log2(n)`` steps)."""
    reached = matrix
    while True:
        longer = reached | product(reached, reached, n)
        if np.array_equal(longer, reached):
            return reached
        reached = longer


def closure_by_components(matrix: Matrix, n: int) -> Matrix:
    """``R+`` by strongly connected components (Tarjan's algorithm).

    Components are condensed, then visited from the sinks back: every node of a
    component reaches the same nodes, its own successors and what the other
    components it steps to reach. Each component is visited once, so the cost
    grows with the edges and the size of the result, whatever order the nodes
    are in. The graphs of the batch are visited as one graph, their nodes apart.
    """
    graphs = len(matrix)
    flat = matrix.reshape(graphs * n, -1)
    of, rows, cols = pairs(matrix)
    sources, targets = of * n + rows, of * n + cols
    starts = np.searchsorted(sources, np.arange(graphs * n + 1))
    successors = [targets[starts[i] : starts[i + 1]].tolist() for i in range(graphs * n)]
    result = np.zeros_like(flat)
    for component in strongly_connected_components(successors):
        members = np.asarray(component)
        # In a cycle, the successors of the members include all of them.
        reached = np.bitwise_or.reduce(flat[members], axis=0)
        inside = set(component)
        outside = sorted({w for v in component for w in successors[v]} - inside)
        if outside:
            reached = reached | np.bitwise_or.reduce(result[outside], axis=0)
        result[members] = reached
    return result.reshape(matrix.shape)


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
