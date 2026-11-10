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

# The packages, in the order data flows through them (each imports only those above it):
#
#   matrices/    the data: a PartialModel (tuples) built into a PackedModel (numpy bits)
#     kernels      packed-bit vectors and matrices: product, transitive closure, ...
#     values       sets of tuples of any arity, and their set operations
#     model        PartialModel: nodes, and the must/may tuples of base relations
#     packed       PackedModel: each relation's must/may arrays; combine, connect
#   rules/       the text: Rules.compile reads and checks rules, without a model
#     syntax       Refinery source -> Source (the pred declarations, as written)
#     compiled     Rules: the predicates with and without a body, checked
#   evaluation/  rules over a model: evaluate() plans the relations to compute, then
#     pipeline     computes each: the model's matrices in, the predicates' matrices out
#     plan         the steps (closures, predicates), each after the relations it reads
#     clauses      a clause checked and rewritten for a model (ClausePostProcessor)
#     solve        a clause's literals joined: matrix products, bulk joins
#     compare      comparisons of attribute values (IntInterval, RealInterval, StringValue)
#   engine/      BitsEngine: Metamodel.query's Pydantic graph -> PackedModel -> QueryResult
#
# Only the names below are the API; the modules' other names may change.

try:
    import numpy  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("pyfinery.bits needs NumPy: pip install 'pyfinery[bits]'") from e

from .engine import BitsEngine
from .errors import RuleError, UnsupportedRuleError
from .evaluation.pipeline import PredicateValues, evaluate
from .matrices.model import Attribute, LogicValue, Node, PartialModel, Relation
from .matrices.packed import PackedModel, PackedRelation
from .rules.compiled import Rules

__all__ = [
    "Attribute",
    "BitsEngine",
    "LogicValue",
    "Node",
    "PackedModel",
    "PackedRelation",
    "PartialModel",
    "PredicateValues",
    "Relation",
    "Rules",
    "RuleError",
    "UnsupportedRuleError",
    "evaluate",
]
