# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The packed-bit kernels of the bits engine, against dense NumPy references."""

from __future__ import annotations

import numpy as np
import pytest

from pyfinery.bits import _kernels as k

SIZES = [1, 2, 63, 64, 65, 130]


def _random(n: int, density: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, n)) < density


def _packed(dense: np.ndarray) -> k.Bits:
    n = len(dense)
    return k.matrix_of(n, *np.nonzero(dense))


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
    assert (k.unpack(k.product(_packed(left), _packed(right), n), n) == expected).all()


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("density", [0.0, 0.02, 0.1, 0.5])
@pytest.mark.parametrize("seed", range(3))
def test_closure(n, density, seed):
    dense = _random(n, density, seed)
    assert (k.unpack(k.closure(_packed(dense), n), n) == _closure(dense)).all()


def test_closure_of_long_paths_and_cycles():
    n = 200
    dense = np.zeros((n, n), bool)
    order = np.random.default_rng(0).permutation(n)
    dense[order[:-1], order[1:]] = True  # one long path, in a random order
    dense[order[150], order[100]] = True  # and a cycle in it
    assert (k.unpack(k.closure(_packed(dense), n), n) == _closure(dense)).all()


@pytest.mark.parametrize("n", SIZES)
def test_transpose_and_projections(n):
    dense = _random(n, 0.2, 3)
    matrix = _packed(dense)
    assert (k.unpack(k.transpose(matrix, n), n) == dense.T).all()
    assert (k.unpack(k.row_any(matrix, n), n) == dense.any(axis=1)).all()
    assert (k.unpack(k.column_any(matrix), n) == dense.any(axis=0)).all()
    assert (k.unpack(k.diagonal(matrix, n), n) == np.diagonal(dense)).all()
    for j in {0, n - 1, n // 2}:
        assert (k.unpack(k.column(matrix, j, n), n) == dense[:, j]).all()


@pytest.mark.parametrize("n", SIZES)
def test_complements_keep_padding_clear(n):
    vector = k.vector_of(n, [0])
    complement = k.complement_vector(vector, n)
    assert k.count(complement) == n - 1
    assert k.count(k.complement_matrix(k.empty_matrix(n), n)) == n * n
    assert k.count(k.full_vector(n)) == n


def test_masks_and_outer():
    n = 70
    rows, cols = k.vector_of(n, [1, 69]), k.vector_of(n, [0, 68])
    outer = k.unpack(k.outer(rows, cols, n), n)
    assert set(zip(*np.nonzero(outer), strict=True)) == {(1, 0), (1, 68), (69, 0), (69, 68)}
    full = np.tile(k.full_vector(n), (n, 1))
    assert k.count(k.mask_rows(full, rows, n)) == 2 * n
    assert k.count(k.mask_columns(full, cols)) == 2 * n
