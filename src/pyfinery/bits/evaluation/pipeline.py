# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Evaluating predicates over a built model, as Refinery does.

Refinery evaluates a predicate twice: for the tuples that *must* satisfy it,
and those that *may*. Each literal of a clause is read in the clause's
modality, except that negation swaps them: ``!R`` must hold where ``R`` may
not, and may hold where ``R`` must not (``DnfLifter`` in Refinery's
``store-reasoning``). Besides:

- a variable that isn't a parameter, in a positive literal, ranges over the
  nodes that (must or may) exist;
- the variables only a negative literal has are quantified inside it, over
  the nodes that exist in the negation's modality: ``!friend(p, _)`` is "no
  friend of ``p`` that exists";
- a ``must`` or ``may`` literal is read in that modality, whatever the clause's;
- ``R+`` steps through nodes that exist (in the modality) to reach the last;
- an attribute value is an interval (or an unknown string), so a comparison
  must hold if it holds for every pair of values, and may if for some;
- an ``error`` predicate is ``error`` where its body must hold and false
  elsewhere: it never may hold.

The result of a predicate is its ``must`` and ``may`` tuples: ``true`` where
both hold, ``unknown`` where only ``may`` does, ``error`` where only ``must``.

:func:`evaluate_batch` plans the relations to compute (:mod:`.plan`), then
computes each in order, from the batch's matrices and those computed before
it, for every graph of the batch at once; :func:`evaluate` evaluates a batch of
one graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..errors import RuleError
from ..matrices import kernels as k
from ..matrices.batch import PackedBatch
from ..matrices.kernels import Matrix, Vector
from ..matrices.model import LogicValue, PartialModel
from ..matrices.packed import PackedModel
from ..matrices.values import (
    BatchBits,
    Bits,
    NodeTuple,
    Size,
    Tuples,
    Value,
    difference,
    empty,
    from_tuples,
    graph_bits,
    intersection,
    of_graph,
    to_tuples,
    union,
    with_graph,
)
from ..rules.compiled import Rules
from ..rules.syntax import Modality, Position, Predicate, PredicateKind
from .clauses import CallLiteral, Clause, CompareLiteral, Operand, Target, named_attribute, resolve
from .compare import Side, compare
from .plan import EXISTS_TARGET, Closure, Derivation, Undetermined, plan
from .solve import Factor, Var, VarId, bind, make_factor, solve


def evaluate(
    rules: str | Sequence[str] | Rules,
    model: PartialModel | PackedModel,
    select: Sequence[str] | None = None,
) -> dict[str, PredicateValues]:
    """Evaluate the predicates ``rules`` define over ``model``.

    ``rules`` is Refinery source (statements other than ``pred`` declarations
    are skipped), or :class:`~.rules.compiled.Rules` compiled from it; ``model``
    is a :class:`PartialModel`, or a :class:`PackedModel` built from one. Compile
    and build them yourself to evaluate them again. The predicates named in
    ``select`` are evaluated, or else every one with a body.
    """
    batch = PackedBatch.build([model])
    return {name: values[0] for name, values in evaluate_batch(rules, batch, select).items()}


def evaluate_batch(
    rules: str | Sequence[str] | Rules,
    models: PackedBatch | Sequence[PartialModel | PackedModel],
    select: Sequence[str] | None = None,
) -> dict[str, BatchValues]:
    """Evaluate the predicates ``rules`` define over every graph of a batch at once.

    ``models`` is a :class:`~.matrices.batch.PackedBatch`, or the models to build
    one of. Each graph gets what :func:`evaluate` would give it alone: ``values[g]``
    is graph ``g``'s :class:`PredicateValues`.
    """
    compiled = rules if isinstance(rules, Rules) else Rules.compile(rules)
    batch = models if isinstance(models, PackedBatch) else PackedBatch.build(models)
    predicates = compiled.select(select)
    targets = selected_targets(predicates, batch, compiled)
    steps = plan(targets, batch, compiled.defined, compiled.declared)
    relations = base_relations(batch)
    for step in steps:
        match step:
            case Undetermined():
                relations[step.target] = undetermined(step, relations, batch)
            case Closure():
                relations[step.target] = closure(step, relations, batch.size)
            case Derivation():
                relations[step.target] = derive(step, relations, batch)
    results = {}
    for predicate, target in zip(predicates, targets, strict=True):
        must, may = relations[target]
        arity = len(predicate.parameters)
        values = BatchValues(predicate.name, predicate.kind, arity, must.data, may.data, batch)
        results[predicate.name] = values
    return results


def selected_targets(predicates: list[Predicate], model: PackedBatch, rules: Rules) -> list[Target]:
    """What each selected predicate is: its own, or (without a body) the relation of the
    model it names, which must relate as many nodes as the predicate has parameters."""
    targets = []
    for predicate in predicates:
        target = resolve(predicate.name, predicate.position, model, rules.defined, rules.declared)
        arity = len(predicate.parameters)
        if target.kind == "base" and model.relations[target.name].arity != arity:
            raise RuleError(
                f"{predicate.name} has {arity} parameters, but {target.name} in the model "
                f"relates {model.relations[target.name].arity} nodes",
                predicate.position,
            )
        targets.append(target)
    return targets


@dataclass(frozen=True)
class PredicateValues:
    """What a predicate evaluates to over the model's ``n`` nodes: the tuples that must
    hold and those that may, as packed bits (:data:`~.matrices.values.Bits`), e.g.
    ``values.must`` is the numpy matrix of a predicate of two parameters."""

    name: str
    kind: PredicateKind
    arity: int
    must: Bits
    may: Bits
    n: int

    def tuples(self) -> dict[LogicValue, Tuples]:
        """The tuples that are ``true``, ``unknown`` and ``error``, as arrays of shape
        ``(m, arity)``, in no particular order."""
        must, may = of_graph(self.arity, self.n, self.must), of_graph(self.arity, self.n, self.may)
        return {value: to_tuples(found)[:, 1:] for value, found in logic_values(must, may).items()}

    def items(self) -> Iterator[tuple[NodeTuple, LogicValue]]:
        """The tuples that aren't false, and their values, ordered by tuple."""
        found: list[tuple[NodeTuple, LogicValue]] = []
        for logic_value, tuples in self.tuples().items():
            found += [(key, logic_value) for key in map(tuple, tuples.tolist())]
        return iter(sorted(found))

    def value(self, key: NodeTuple) -> LogicValue:
        must = is_member(of_graph(self.arity, self.n, self.must), key)
        may = is_member(of_graph(self.arity, self.n, self.may), key)
        if must:
            return "true" if may else "error"
        return "unknown" if may else "false"


@dataclass(frozen=True)
class BatchValues:
    """What a predicate evaluates to over each graph of a batch: the tuples that must hold
    and those that may, as batched packed bits over the batch's node slots.

    ``values[g]`` is graph ``g``'s :class:`PredicateValues`, in its own node indices.
    """

    name: str
    kind: PredicateKind
    arity: int
    must: BatchBits
    may: BatchBits
    batch: PackedBatch

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, graph: int) -> PredicateValues:
        if not -len(self) <= graph < len(self):
            raise IndexError(graph)
        graph %= len(self)
        n = self.batch.sizes[graph]

        def own(bits: BatchBits) -> Bits:
            value = self.value(bits)
            if self.batch.keeps_indices:
                return graph_bits(value, graph)
            slots = self.batch.size.nodes
            tuples = to_tuples(of_graph(self.arity, slots, graph_bits(value, graph)))[:, 1:]
            nodes = self.batch.owners[graph][tuples]  # the slots, as the graph's own nodes
            return graph_bits(from_tuples(self.arity, Size(1, n), with_graph(nodes)))

        must = own(self.must)
        may = must if self.may is self.must else own(self.may)
        return PredicateValues(self.name, self.kind, self.arity, must, may, n)

    def tuples(self) -> dict[LogicValue, Tuples]:
        """The tuples that are ``true``, ``unknown`` and ``error``, as arrays of shape
        ``(m, 1 + arity)``: each row its graph, then its nodes by their indices in the
        graph; ordered by graph, otherwise in no particular order."""
        found = logic_values(self.value(self.must), self.value(self.may))
        return {value: self.own_tuples(tuples) for value, tuples in found.items()}

    def value(self, bits: BatchBits) -> Value:
        return Value(self.arity, self.batch.size, bits)

    def own_tuples(self, value: Value) -> Tuples:
        """The tuples of ``value``, with each node's slot replaced by its index in its graph."""
        tuples = to_tuples(value)
        if self.batch.keeps_indices or not len(tuples):
            return tuples
        tuples = tuples.copy()
        tuples[:, 1:] = self.batch.owners[tuples[:, :1], tuples[:, 1:]]
        return tuples


def logic_values(must: Value, may: Value) -> dict[LogicValue, Value]:
    """The tuples that are ``true``, ``unknown`` and ``error``."""
    return {
        "true": intersection(must, may),
        "unknown": difference(may, must),
        "error": difference(must, may),
    }


def is_member(value: Value, key: NodeTuple) -> bool:
    match value.arity:
        case 0:
            return bool(value.truth[0])
        case 1:
            return bool(k.contains(value.vector, key[0])[0])
        case 2:
            return bool(k.contains(value.matrix[:, key[0]], key[1])[0])
        case _:
            return bool(np.any(np.all(value.array[:, 1:] == np.asarray(key), axis=1)))


class Modal(NamedTuple):
    """A relation's values as evaluation reads them: the tuples that must hold, and
    those that may."""

    must: Value
    may: Value

    def read(self, modality: Modality) -> Value:
        return self.must if modality == "must" else self.may


type Relations = dict[Target, Modal]
"""The values of the relations computed so far, by what reads them."""


def base_relations(batch: PackedBatch) -> Relations:
    """The batch's matrices, as evaluation reads them: where it starts."""
    return {
        Target("base", r.name): Modal(batch.value(r, "must"), batch.value(r, "may"))
        for r in batch.relations.values()
    }


# -- steps --------------------------------------------------------------------


def undetermined(step: Undetermined, relations: Relations, batch: PackedBatch) -> Modal:
    """Unknown for the nodes of its parameter types: an untyped parameter, any node of its
    graph."""
    predicate = step.predicate
    size = batch.size
    arity = len(predicate.parameters)
    domains: list[Vector] = []  # the nodes each parameter may be, in each graph
    for parameter, domain in zip(predicate.parameters, step.domains, strict=True):
        if domain is None:
            domains.append(batch.nodes)
            continue
        kind = relations[domain].may
        if kind.arity != 1:
            raise RuleError(f"{parameter.type} isn't a class", predicate.position)
        domains.append(kind.vector)
    match arity:
        case 0:
            may = Value(0, size, np.ones(size.graphs, bool))
        case 1:
            may = Value(1, size, domains[0])
        case 2:
            may = Value(2, size, k.outer(domains[0], domains[1], size.nodes))
        case _:
            grids = []
            for graph in range(size.graphs):
                axes = [k.indices(d[graph : graph + 1])[1] for d in domains]
                grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(arity, -1).T
                grids.append(with_graph(grid, graph))
            may = from_tuples(arity, size, np.concatenate(grids))
    return Modal(empty(arity, size), may)


def closure(step: Closure, relations: Relations, size: Size) -> Modal:
    """``R+`` in each modality, from ``R`` and the nodes that exist."""
    steps, exists = relations[step.target.steps], relations[EXISTS_TARGET]
    if steps.must.arity != 2:
        raise RuleError(f"{step.written}+ needs a relation of two nodes", step.position)
    must = transitive_closure(steps.must.matrix, exists.must.vector, size.nodes)
    may = transitive_closure(steps.may.matrix, exists.may.vector, size.nodes)
    return Modal(Value(2, size, must), Value(2, size, may))


def transitive_closure(steps: Matrix, exists: Vector, n: int) -> Matrix:
    """``R+``: steps of ``R``, each but the last to a node that exists."""
    through = k.mask_columns(steps, exists)  # the steps to a node that exists
    return steps | k.product(k.closure(through, n), steps, n)


def derive(step: Derivation, relations: Relations, batch: PackedBatch) -> Modal:
    """The union of the predicate's clauses in each modality; an error predicate
    never may hold."""
    arity = len(step.predicate.parameters)

    def union_of_clauses(modality: Modality) -> Value:
        result = empty(arity, batch.size)
        for clause in step.clauses:
            result = union(result, evaluate_clause(clause, modality, relations, batch))
        return result

    must = union_of_clauses("must")
    may = empty(arity, batch.size) if step.predicate.kind == "error" else union_of_clauses("may")
    return Modal(must, may)


# -- clauses ------------------------------------------------------------------


def evaluate_clause(
    clause: Clause, modality: Modality, relations: Relations, batch: PackedBatch
) -> Value:
    """The tuples of the parameters for which every literal of ``clause`` holds."""
    size = batch.size
    exists = relations[EXISTS_TARGET].read(modality)
    factors: list[Factor] = []
    for call in clause.calls:
        if not call.atom.negated:
            value = read(call, call.atom.modality or modality, relations)
            factors.append(bind(value, call.arguments))
    parameters = set(range(clause.parameters))
    existential = {
        v for c in clause.calls if not c.atom.negated and not c.attribute for v in c.variables
    } - parameters
    for variable in sorted(existential):
        factors.append(Factor((variable,), exists))
    for call in clause.calls:
        if call.atom.negated:
            negation = call.atom.modality or flip(modality)
            factors.append(negative(call, clause.bound, negation, relations, size))
    for compared in clause.comparisons:
        factors.append(comparison(compared, modality, batch))
    return solve(factors, tuple(range(clause.parameters)), size)


def flip(modality: Modality) -> Modality:
    return "may" if modality == "must" else "must"


def read(call: CallLiteral, modality: Modality, relations: Relations) -> Value:
    """The values of the relation ``call`` reads, in ``modality``."""
    value = relations[call.target].read(modality)
    if value.arity != len(call.arguments):
        raise RuleError(
            f"{call.atom.relation} has {value.arity} parameters, given {len(call.arguments)}",
            call.atom.position,
        )
    return value


def negative(
    call: CallLiteral, bound: frozenset[VarId], modality: Modality, relations: Relations, size: Size
) -> Factor:
    """``!call``, read in ``modality`` (the clause's flipped, or the atom's own)."""
    value = read(call, modality, relations)
    args = call.arguments
    variables = [a.id for a in args if isinstance(a, Var)]
    local = sorted(set(variables) - bound)
    if not local:
        return bind(value, args, negated=True)
    exists = relations[EXISTS_TARGET].read(modality)
    outer = tuple(dict.fromkeys(v for v in variables if v in bound))
    inner = [bind(value, args), *(Factor((v,), exists) for v in local)]
    return make_factor(outer, solve(inner, outer, size), negated=True)


def comparison(compared: CompareLiteral, modality: Modality, batch: PackedBatch) -> Factor:
    position = compared.comparison.position
    left, right = side(compared.left, position, batch), side(compared.right, position, batch)
    return compare(left, right, compared.comparison, modality, batch.size, batch.strings)


def side(operand: Operand, position: Position, batch: PackedBatch) -> Side:
    if operand.attribute is None:
        return Side(None, None, operand.constant, None)
    attribute = named_attribute(batch, operand.attribute, position)
    return Side(operand.variable, operand.node, None, attribute)
