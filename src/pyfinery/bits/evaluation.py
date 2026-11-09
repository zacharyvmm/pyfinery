# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Evaluating predicates over a partial model, as Refinery does.

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
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import kernels as k
from .clauses import CallLiteral, Clause, CompareLiteral, Operand, Target, resolve
from .compare import Side, compare
from .errors import RuleError
from .kernels import Vector
from .model import LogicValue
from .plan import (
    EXISTS_TARGET,
    Closure,
    Derivation,
    Undetermined,
    find_predicate,
    plan,
)
from .solve import Factor, Var, VarId, bind, make_factor, solve
from .syntax import Modality, Position, PredicateKind
from .values import (
    Modal,
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

if TYPE_CHECKING:
    from .packed import PackedModel
    from .rules import Rules


@dataclass(frozen=True)
class PredicateValues:
    """What a predicate evaluates to, over the nodes of the model (by index)."""

    name: str
    kind: PredicateKind
    arity: int
    must: Value
    may: Value

    def tuples(self) -> dict[LogicValue, Tuples]:
        """The tuples that are ``true``, ``unknown`` and ``error``, as arrays of shape
        ``(m, arity)``, in no particular order."""
        return {
            "true": to_tuples(intersection(self.must, self.may)),
            "unknown": to_tuples(difference(self.may, self.must)),
            "error": to_tuples(difference(self.must, self.may)),
        }

    def items(self) -> Iterator[tuple[NodeTuple, LogicValue]]:
        """The tuples that aren't false, and their values, ordered by tuple."""
        found: list[tuple[NodeTuple, LogicValue]] = []
        for logic_value, tuples in self.tuples().items():
            found += [(key, logic_value) for key in map(tuple, tuples.tolist())]
        return iter(sorted(found))

    def value(self, key: NodeTuple) -> LogicValue:
        must, may = is_member(self.must, key), is_member(self.may, key)
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


def evaluate_rules(
    rules: Rules, model: PackedModel, select: Sequence[str] | None = None
) -> dict[str, PredicateValues]:
    """Evaluate the predicates of ``rules`` over ``model``: plan the relations to compute,
    then compute each, in order, from the model's and those before it."""
    defined, declared = rules.defined, rules.declared
    names = list(select) if select is not None else list(defined)
    selected = [find_predicate(name, defined, declared) for name in names]
    targets = [resolve(p.name, p.position, model, defined, declared) for p in selected]
    steps = plan(targets, model, defined, declared)
    relations = base_relations(model)
    for step in steps:
        match step:
            case Undetermined():
                relations[step.target] = undetermined(step, relations, model.size)
            case Closure():
                relations[step.target] = transitive_closure(step, relations, model.size)
            case Derivation():
                relations[step.target] = derive(step, relations, model)
    return {
        p.name: PredicateValues(p.name, p.kind, len(p.parameters), *relations[target])
        for p, target in zip(selected, targets, strict=True)
    }


type Relations = dict[Target, Modal]
"""The values of the relations computed so far, by what reads them."""


def base_relations(model: PackedModel) -> Relations:
    """The relations of the model, packed already: where evaluation starts."""
    return {Target("base", r.name): r.values for r in model.relations.values()}


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


def transitive_closure(step: Closure, relations: Relations, n: int) -> Modal:
    steps, exists = relations[step.target.steps], relations[EXISTS_TARGET]
    if steps.must.arity != 2:
        raise RuleError(f"{step.written}+ needs a relation of two nodes", step.position)

    def close(steps: Value, exists: Value) -> Value:
        through = k.mask_columns(steps.matrix, exists.vector)
        paths = k.product(k.closure(through, n), steps.matrix, n)
        return Value(2, n, steps.matrix | paths)

    return Modal(close(steps.must, exists.must), close(steps.may, exists.may))


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
    attribute = model.attribute(operand.attribute, position)
    return Side(operand.variable, operand.node, None, attribute)
