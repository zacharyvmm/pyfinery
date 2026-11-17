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

:func:`evaluate_batch` evaluates rules on many graphs at once, each as if alone:
their relations are stacked into arrays with a leading axis of graphs
(:class:`PackedBatch`), and every step of the evaluation is the same array
operation over all of them.

Rules may use atoms of classes, references, predicates and attributes,
transitive closure (``+``), negation, ``must`` and ``may``, ``==`` and
``!=`` of nodes, and comparisons of attribute values. Aggregation,
arithmetic and the ``candidate`` modality raise :class:`UnsupportedRuleError`.

:class:`BitsEngine`, passed instead of a client, makes
:meth:`~pyfinery.Metamodel.query` evaluate its rules this way, and
:meth:`~pyfinery.Metamodel.query_batch` evaluate them on many graphs at once.

Needs NumPy: ``pip install 'pyfinery[bits]'``.
"""

from __future__ import annotations

# The packages, in the order data flows through them (each imports only those above it):
#
#   matrices/    the data: PartialModels (tuples) built into a PackedBatch (numpy bits)
#     kernels      packed bits of a batch of graphs: product, transitive closure, ...
#                  (the only arithmetic on bits: what a GPU port would reimplement)
#     values       sets of tuples of any arity, of each graph, and their set operations
#     model        PartialModel: nodes, and the must/may tuples of base relations
#     packed       PackedModel: one graph's must/may arrays; combine, connect
#     batch        PackedBatch: graphs stacked on a leading axis, each its own
#   rules/       the text: Rules.compile reads and checks rules, without a model
#     syntax       Refinery source -> Source (the pred declarations, as written)
#     compiled     Rules: the predicates with and without a body, checked
#   evaluation/  rules over a batch: evaluate_batch() plans the relations to compute, then
#     pipeline     computes each: the batch's matrices in, the predicates' matrices out
#                  (evaluate() is a batch of one graph)
#     plan         the steps (closures, predicates), each after the relations it reads
#     clauses      a clause checked and rewritten for a model (ClausePostProcessor)
#     solve        a clause's literals joined: matrix products, bulk joins
#     compare      comparisons of attribute values (IntInterval, RealInterval, StringValue)
#   engine/      BitsEngine: Metamodel.query's Pydantic graphs -> PackedBatch -> QueryResults
#
# Only the names below are the API; the modules' other names may change.

try:
    import numpy  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("pyfinery.bits needs NumPy: pip install 'pyfinery[bits]'") from e

from .engine import BitsEngine
from .errors import RuleError, UnsupportedRuleError
from .evaluation.pipeline import BatchValues, PredicateValues, evaluate, evaluate_batch
from .matrices.batch import PackedBatch
from .matrices.model import Attribute, LogicValue, Node, PartialModel, Relation
from .matrices.packed import PackedModel, PackedRelation
from .rules.compiled import Rules

__all__ = [
    "Attribute",
    "BatchValues",
    "BitsEngine",
    "LogicValue",
    "Node",
    "PackedBatch",
    "PackedModel",
    "PackedRelation",
    "PartialModel",
    "PredicateValues",
    "Relation",
    "Rules",
    "RuleError",
    "UnsupportedRuleError",
    "evaluate",
    "evaluate_batch",
]
