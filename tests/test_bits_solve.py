# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The clause solver of the bits engine, against enumerating every assignment."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from pyfinery.bits.evaluation import solve as solve_module
from pyfinery.bits.evaluation.solve import Factor, Node, Var, bind, make_factor, solve
from pyfinery.bits.matrices.values import from_tuples, to_tuples


def _brute_force(factors, head, variables, n):
    found = set()
    for values in itertools.product(range(n), repeat=len(variables)):
        assignment = dict(zip(variables, values, strict=True))
        if all(_holds(f, assignment) for f in factors):
            found.add(tuple(assignment[v] for v in head))
    return found


def _holds(factor, assignment):
    key = tuple(assignment[v] for v in factor.variables)
    present = key in factor.tuples
    return present != factor.negated


class _Spec:
    def __init__(self, variables, tuples, negated):
        self.variables, self.tuples, self.negated = variables, tuples, negated


def _draw(rng: random.Random, n: int):
    count = rng.randint(2, 5)
    variables = list(range(count))
    specs = []
    for _ in range(rng.randint(1, 6)):
        arity = rng.choice([0, 1, 1, 2, 2, 2, 3])
        chosen = tuple(rng.sample(variables, min(arity, count)))
        all_tuples = list(itertools.product(range(n), repeat=len(chosen)))
        tuples = {t for t in all_tuples if rng.random() < rng.choice([0.2, 0.5, 0.8])}
        negated = len(chosen) >= 3 and rng.random() < 0.4
        specs.append(_Spec(chosen, tuples, negated))
    # Every variable in a positive factor, as in a clause.
    for v in variables:
        if not any(v in s.variables and not s.negated for s in specs):
            specs.append(_Spec((v,), {(i,) for i in range(n) if rng.random() < 0.7}, False))
    head = tuple(rng.sample(variables, rng.randint(0, min(4, count))))
    return variables, specs, head


@pytest.mark.parametrize("seed", range(300))
def test_solve_matches_enumeration(seed):
    _check_against_enumeration(seed)


@pytest.mark.parametrize("seed", range(50))
def test_bulk_joins_in_chunks_match_enumeration(seed, monkeypatch):
    monkeypatch.setattr(solve_module, "CHUNK_WORDS", 3)  # a few rows at a time
    _check_against_enumeration(seed)


def _check_against_enumeration(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 5)
    variables, specs, head = _draw(rng, n)
    factors = [
        make_factor(
            s.variables,
            from_tuples(len(s.variables), n, np.array(sorted(s.tuples), np.intp)),
            s.negated,
        )
        for s in specs
    ]
    result = solve(factors, head, n)
    found = {tuple(int(i) for i in t) for t in to_tuples(result)}
    assert found == _brute_force(specs, head, variables, n)


def test_bind_selects_nodes_and_repeated_variables():
    n = 4
    tuples = np.array([[0, 1, 1], [1, 1, 2], [2, 3, 3], [0, 2, 1]], np.intp)
    value = from_tuples(3, n, tuples)
    factor = bind(value, [Node(0), Var(5), Var(5)])
    assert factor.variables == (5,)
    assert {tuple(t) for t in to_tuples(factor.value).tolist()} == {(1,)}
    factor = bind(value, [Var(1), Var(2), Var(2)])
    assert {tuple(t) for t in to_tuples(factor.value).tolist()} == {(0, 1), (2, 3)}
    negated = bind(from_tuples(1, n, np.array([[2]])), [Node(2)], negated=True)
    assert negated == Factor((), negated.value) and negated.value.data is False
