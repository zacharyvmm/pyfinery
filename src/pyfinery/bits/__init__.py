# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Evaluate Refinery predicates in Python, over packed bit matrices, without a server.

Given a partial model (nodes, and which tuples of each base relation must
and may hold), :func:`evaluate` computes the predicates of Refinery rules as
Refinery's ``semantics`` endpoint does: each tuple is ``true``, ``unknown``,
``false`` or ``error``. It doesn't propagate: what the rules imply for the
base relations (e.g. through ``error`` predicates or propagation rules) must
already be in the model.

Rules may use atoms of classes, references, predicates and attributes,
transitive closure (``+``), negation, ``must`` and ``may``, ``==`` and
``!=`` of nodes, and comparisons of attribute values. Aggregation,
arithmetic and the ``candidate`` modality raise :class:`UnsupportedRuleError`.

:class:`BitsEngine`, passed instead of a client, makes
:meth:`~pyfinery.Metamodel.query` evaluate its rules this way.

Needs NumPy: ``pip install 'pyfinery[bits]'``.
"""

from __future__ import annotations

try:
    import numpy  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("pyfinery.bits needs NumPy: pip install 'pyfinery[bits]'") from e

from ._errors import RuleError, UnsupportedRuleError
from ._evaluate import PredicateValues, evaluate
from ._metamodel import BitsEngine
from ._model import Attribute, LogicValue, Node, PartialModel, Relation

__all__ = [
    "Attribute",
    "BitsEngine",
    "LogicValue",
    "Node",
    "PartialModel",
    "PredicateValues",
    "Relation",
    "RuleError",
    "UnsupportedRuleError",
    "evaluate",
]
