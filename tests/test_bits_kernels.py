# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The packed-bit kernels of the bits engine, on batches of graphs, against dense NumPy
references computed graph by graph: on the CPU, and (PYFINERY_GPU_TESTS=1) a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from pyfinery.bits.matrices import kernels as k

SIZES = [1, 2, 63, 64, 65, 130]
GRAPHS = 3


def _random(n: int, density: float, seed: int, graphs: int = GRAPHS) -> np.ndarray:
    """A dense relation per graph; each graph's density differs, the last is empty."""
    rng = np.random.default_rng(seed)
    dense = rng.random((graphs, n, n)) < density * rng.random((graphs, 1, 1)) * 2
    dense[-1] = False
    return dense


@pytest.fixture(autouse=True)
def on_device(device):
    """Every kernel of a test runs on ``device``, and makes its arrays."""
    with k.on(device):
        yield


def _packed(dense: np.ndarray) -> k.Matrix:
    graphs, n, _ = dense.shape
    return k.matrix_of(graphs, n, *np.nonzero(dense))


def _dense(bits, n: int) -> np.ndarray:
    """Packed bits, on whatever device, as NumPy booleans."""
    return k.to_host(k.unpack(bits, n))


def _closure(dense: np.ndarray) -> np.ndarray:
    result = dense.copy()
    while True:
        step = result | ((result.astype(np.int64) @ dense.astype(np.int64)) > 0)
        if (step == result).all():
            return result
        result = step


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("density", [0.0, 0.05, 0.3, 0.9])
def test_product(n, density):
    left, right = _random(n, density, 1), _random(n, density, 2)
    expected = (left.astype(np.int64) @ right.astype(np.int64)) > 0
    assert (_dense(k.product(_packed(left), _packed(right), n), n) == expected).all()


@pytest.mark.parametrize("dense_nodes", [k.GPU_DENSE_NODES, 0])  # on a GPU: unpacked, scattered
def test_product_in_chunks(monkeypatch, dense_nodes):
    monkeypatch.setattr(k, "CHUNK_WORDS", 5)  # a few rows, or a graph, at a time
    monkeypatch.setattr(k, "GPU_DENSE_NODES", dense_nodes)
    for n, density in [(70, 0.02), (130, 0.5)]:
        left, right = _random(n, density, 3), _random(n, density, 4)
        expected = (left.astype(np.int64) @ right.astype(np.int64)) > 0
        assert (_dense(k.product(_packed(left), _packed(right), n), n) == expected).all()


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("density", [0.0, 0.02, 0.1, 0.5])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("closure", [k.closure_by_components, k.closure_by_squaring])
def test_closure(n, density, seed, closure):
    dense = _random(n, density, seed)
    expected = np.stack([_closure(graph) for graph in dense])
    assert (_dense(closure(_packed(dense), n), n) == expected).all()


@pytest.mark.parametrize("closure", [k.closure_by_components, k.closure_by_squaring])
def test_closure_of_long_paths_and_cycles(closure):
    n = 200
    dense = np.zeros((2, n, n), bool)
    for graph, seed in enumerate((0, 1)):
        order = np.random.default_rng(seed).permutation(n)
        dense[graph, order[:-1], order[1:]] = True  # one long path, in a random order
        dense[graph, order[150], order[100 + graph]] = True  # and a cycle in it
    expected = np.stack([_closure(graph) for graph in dense])
    assert (_dense(closure(_packed(dense), n), n) == expected).all()


@pytest.mark.parametrize("n", SIZES)
def test_transpose_and_projections(n):
    dense = _random(n, 0.2, 3)
    matrix = _packed(dense)
    assert (_dense(k.transpose(matrix, n), n) == dense.transpose(0, 2, 1)).all()
    assert (_dense(k.row_any(matrix, n), n) == dense.any(axis=2)).all()
    assert (_dense(k.column_any(matrix), n) == dense.any(axis=1)).all()
    assert (_dense(k.diagonal(matrix, n), n) == np.diagonal(dense, axis1=1, axis2=2)).all()
    for j in {0, n - 1, n // 2}:
        assert (_dense(k.column(matrix, j, n), n) == dense[:, :, j]).all()
        assert (_dense(k.row(matrix, j), n) == dense[:, j]).all()
        assert (k.to_host(k.contains(k.row(matrix, j), 0)) == dense[:, j, 0]).all()


@pytest.mark.parametrize("density", [0.0, 0.001, 0.01, 0.5])  # listed as pairs, or unpacked
def test_transpose_of_sparse_and_dense_matrices(density):
    for n in (70, 300):
        dense = _random(n, density, 7)
        assert (_dense(k.transpose(_packed(dense), n), n) == dense.transpose(0, 2, 1)).all()


@pytest.mark.parametrize("n", SIZES)
def test_complements_keep_padding_clear(n):
    vector = k.vector_of(GRAPHS, n, [0, 1], [0, 0])
    complement = k.complement(vector, n)
    assert [k.count(graph) for graph in complement] == [n - 1, n - 1, n]
    assert k.count(k.complement(k.empty_matrix(GRAPHS, n), n)) == GRAPHS * n * n
    assert k.count(k.full_vector(GRAPHS, n)) == GRAPHS * n
    assert k.count(k.full_matrix(GRAPHS, n)) == GRAPHS * n * n
    assert (_dense(k.identity(GRAPHS, n), n) == np.eye(n, dtype=bool)).all()


def test_masks_and_outer():
    n = 70
    rows = k.vector_of(2, n, [0, 0, 1], [1, 69, 3])
    cols = k.vector_of(2, n, [0, 0, 1], [0, 68, 5])
    outer = _dense(k.outer(rows, cols, n), n)
    expected = {(0, 1, 0), (0, 1, 68), (0, 69, 0), (0, 69, 68), (1, 3, 5)}
    assert set(zip(*np.nonzero(outer), strict=True)) == expected
    full = k.full_matrix(2, n)
    assert k.count(k.mask_rows(full, rows, n)) == 3 * n
    assert k.count(k.mask_columns(full, cols)) == 3 * n


def test_bits_are_found_in_order_whatever_the_leading_axes():
    dense = _random(70, 0.1, 5)
    expected = np.nonzero(dense)
    for found in (k.nonzero_bits(k.pack_padded(k.asarray(dense), 70)), k.pairs(_packed(dense))):
        assert all((k.to_host(a) == b).all() for a, b in zip(found, expected, strict=True))
