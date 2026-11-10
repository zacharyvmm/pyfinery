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

:func:`evaluate` plans the relations to compute (:mod:`.plan`), then computes
each in order, from the model's matrices and those computed before it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..errors import RuleError
from ..matrices import kernels as k
from ..matrices.kernels import Matrix, Vector
from ..matrices.model import LogicValue, PartialModel
from ..matrices.packed import PackedModel
from ..matrices.values import (
    Bits,
    NodeTuple,
    Tuples,
    Value,
    difference,
    empty,
    from_tuples,
    intersection,
    to_tuples,
    union,
)
from ..rules.compiled import Rules
from ..rules.syntax import Modality, Position, PredicateKind
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
    compiled = rules if isinstance(rules, Rules) else Rules.compile(rules)
    packed = model if isinstance(model, PackedModel) else PackedModel.build(model)
    predicates = compiled.select(select)
    targets = [
        resolve(p.name, p.position, packed, compiled.defined, compiled.declared) for p in predicates
    ]
    steps = plan(targets, packed, compiled.defined, compiled.declared)
    relations = base_relations(packed)
    for step in steps:
        match step:
            case Undetermined():
                relations[step.target] = undetermined(step, relations, packed.size)
            case Closure():
                relations[step.target] = closure(step, relations, packed.size)
            case Derivation():
                relations[step.target] = derive(step, relations, packed)
    results = {}
    for predicate, target in zip(predicates, targets, strict=True):
        must, may = relations[target]
        arity = len(predicate.parameters)
        values = PredicateValues(
            predicate.name, predicate.kind, arity, must.data, may.data, packed.size
        )
        results[predicate.name] = values
    return results


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
        must, may = Value(self.arity, self.n, self.must), Value(self.arity, self.n, self.may)
        return {
            "true": to_tuples(intersection(must, may)),
            "unknown": to_tuples(difference(may, must)),
            "error": to_tuples(difference(must, may)),
        }

    def items(self) -> Iterator[tuple[NodeTuple, LogicValue]]:
        """The tuples that aren't false, and their values, ordered by tuple."""
        found: list[tuple[NodeTuple, LogicValue]] = []
        for logic_value, tuples in self.tuples().items():
            found += [(key, logic_value) for key in map(tuple, tuples.tolist())]
        return iter(sorted(found))

    def value(self, key: NodeTuple) -> LogicValue:
        must = is_member(Value(self.arity, self.n, self.must), key)
        may = is_member(Value(self.arity, self.n, self.may), key)
        if must:
            return "true" if may else "error"
        return "unknown" if may else "false"


def is_member(value: Value, key: NodeTuple) -> bool:
    match value.arity:
        case 0:
            return value.truth
        case 1:
            return k.contains(value.vector, key[0])
        case 2:
            return k.contains(value.matrix[key[0]], key[1])
        case _:
            return bool(np.any(np.all(value.array == np.asarray(key), axis=1)))


class Modal(NamedTuple):
    """A relation's values as evaluation reads them: the tuples that must hold, and
    those that may."""

    must: Value
    may: Value

    def read(self, modality: Modality) -> Value:
        return self.must if modality == "must" else self.may


type Relations = dict[Target, Modal]
"""The values of the relations computed so far, by what reads them."""


def base_relations(model: PackedModel) -> Relations:
    """The model's matrices, as evaluation reads them: where it starts."""
    n = model.size
    return {
        Target("base", r.name): Modal(Value(r.arity, n, r.must), Value(r.arity, n, r.may))
        for r in model.relations.values()
    }


# -- steps --------------------------------------------------------------------


def undetermined(step: Undetermined, relations: Relations, n: int) -> Modal:
    predicate = step.predicate
    arity = len(predicate.parameters)
    domains: list[Vector] = []  # the nodes each parameter may be
    for parameter, domain in zip(predicate.parameters, step.domains, strict=True):
        if domain is None:
            domains.append(k.full_vector(n))
            continue
        kind = relations[domain].may
        if kind.arity != 1:
            raise RuleError(f"{parameter.type} isn't a class", predicate.position)
        domains.append(kind.vector)
    match arity:
        case 0:
            may = Value(0, n, True)
        case 1:
            may = Value(1, n, domains[0])
        case 2:
            may = Value(2, n, k.outer(domains[0], domains[1], n))
        case _:
            axes = [k.indices(d, n) for d in domains]
            grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(arity, -1).T
            may = from_tuples(arity, n, grid)
    return Modal(empty(arity, n), may)


def closure(step: Closure, relations: Relations, n: int) -> Modal:
    """``R+`` in each modality, from ``R`` and the nodes that exist."""
    steps, exists = relations[step.target.steps], relations[EXISTS_TARGET]
    if steps.must.arity != 2:
        raise RuleError(f"{step.written}+ needs a relation of two nodes", step.position)
    must = transitive_closure(steps.must.matrix, exists.must.vector, n)
    may = transitive_closure(steps.may.matrix, exists.may.vector, n)
    return Modal(Value(2, n, must), Value(2, n, may))


def transitive_closure(steps: Matrix, exists: Vector, n: int) -> Matrix:
    """``R+``: steps of ``R``, each but the last to a node that exists."""
    through = k.mask_columns(steps, exists)  # the steps to a node that exists
    return steps | k.product(k.closure(through, n), steps, n)


def derive(step: Derivation, relations: Relations, model: PackedModel) -> Modal:
    """The union of the predicate's clauses in each modality; an error predicate
    never may hold."""
    arity = len(step.predicate.parameters)

    def union_of_clauses(modality: Modality) -> Value:
        result = empty(arity, model.size)
        for clause in step.clauses:
            result = union(result, evaluate_clause(clause, modality, relations, model))
        return result

    must = union_of_clauses("must")
    may = empty(arity, model.size) if step.predicate.kind == "error" else union_of_clauses("may")
    return Modal(must, may)


# -- clauses ------------------------------------------------------------------


def evaluate_clause(
    clause: Clause, modality: Modality, relations: Relations, model: PackedModel
) -> Value:
    """The tuples of the parameters for which every literal of ``clause`` holds."""
    n = model.size
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
            factors.append(negative(call, clause.bound, negation, relations, n))
    for compared in clause.comparisons:
        factors.append(comparison(compared, modality, model))
    return solve(factors, tuple(range(clause.parameters)), n)


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
    call: CallLiteral, bound: frozenset[VarId], modality: Modality, relations: Relations, n: int
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
    return make_factor(outer, solve(inner, outer, n), negated=True)


def comparison(compared: CompareLiteral, modality: Modality, model: PackedModel) -> Factor:
    position = compared.comparison.position
    left, right = side(compared.left, position, model), side(compared.right, position, model)
    return compare(left, right, compared.comparison, modality, model.size)


def side(operand: Operand, position: Position, model: PackedModel) -> Side:
    if operand.attribute is None:
        return Side(None, None, operand.constant, None)
    attribute = named_attribute(model, operand.attribute, position)
    return Side(operand.variable, operand.node, None, attribute)
