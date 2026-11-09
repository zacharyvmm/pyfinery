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

# The modules, in the order data flows through them:
#
#   metamodel   BitsEngine.query_graph: Graph -> PartialModel -> PackedModel -> QueryResult
#   syntax      Refinery source -> Source (the pred declarations, as written)
#   packed      PackedModel: a PartialModel's relations as bits; evaluate(), combine(), connect()
#   rules       Rules: rules compiled once (read, checked), to evaluate on PackedModels
#   evaluation  plan, then run each step, building its must and may matrices
#   plan        the steps: closures and predicates, each after the relations it reads
#   clauses     a clause, checked and rewritten as Refinery's ClausePostProcessor does
#   compare     comparisons of attribute values (IntInterval, RealInterval, StringValue)
#   solve       a clause's factors -> its parameters' tuples: matrix products, bulk joins
#   values      sets of tuples of nodes: packed bits up to two nodes, arrays beyond
#   kernels     packed-bit vectors and matrices: product, transitive closure, ...
#   model       PartialModel: nodes, and the must/may tuples of base relations
#
# Only the names below are the API; the modules' other names may change.

try:
    import numpy  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("pyfinery.bits needs NumPy: pip install 'pyfinery[bits]'") from e

from .errors import RuleError, UnsupportedRuleError
from .evaluation import PredicateValues
from .metamodel import BitsEngine
from .model import Attribute, LogicValue, Node, PartialModel, Relation
from .packed import PackedModel, PackedRelation, evaluate
from .rules import Rules

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
