"""Compiled problems, kept for reuse: compiling costs tenths of a second, a model milliseconds."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence

from .._options import Predicate
from .._schema import Schema
from ._compile import CompiledProblem

SIZE = 16
"""How many compiled problems are kept."""

_lock = threading.Lock()
_problems: OrderedDict[tuple[object, ...], CompiledProblem] = OrderedDict()


def compiled(
    schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate] = ()
) -> CompiledProblem:
    """``schema``, ``rules`` and the declarations of ``predicates``, compiled once and reused
    while they are among the last :data:`SIZE` used."""
    key = (id(schema), tuple(rules), tuple((p.name, tuple(p.parameters)) for p in predicates))
    with _lock:
        found = _problems.get(key)
        if found is not None and found.schema is schema:
            _problems.move_to_end(key)
            return found
    problem = CompiledProblem(schema, rules, predicates)
    with _lock:
        _problems[key] = problem
        _problems.move_to_end(key)
        while len(_problems) > SIZE:
            _problems.popitem(last=False)
    return problem


def clear() -> None:
    """Forget every compiled problem."""
    with _lock:
        _problems.clear()
