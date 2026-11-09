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
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from . import _kernels as k
from ._clauses import CallLiteral, Clause, CompareLiteral, Context, Operand, normalize
from ._errors import RuleError, UnsupportedRuleError
from ._model import Attribute, LogicValue, PartialModel, Relation, real_interval
from ._solve import Factor, Var, bind, make_factor, solve
from ._syntax import (
    Atom,
    Comparison,
    Modality,
    Position,
    Predicate,
    PredicateKind,
    Source,
    parse,
    refinery_name,
)
from ._values import Value, difference, empty, from_tuples, intersection, to_tuples, union


@dataclass(frozen=True)
class PredicateValues:
    """What a predicate evaluates to, over the nodes of the model (by index)."""

    name: str
    kind: PredicateKind
    arity: int
    must: Value
    may: Value

    def tuples(self) -> dict[LogicValue, NDArray[np.intp]]:
        """The tuples that are ``true``, ``unknown`` and ``error``, as arrays of shape
        ``(m, arity)``, in no particular order."""
        return {
            "true": to_tuples(intersection(self.must, self.may)),
            "unknown": to_tuples(difference(self.may, self.must)),
            "error": to_tuples(difference(self.must, self.may)),
        }

    def items(self) -> Iterator[tuple[tuple[int, ...], LogicValue]]:
        """The tuples that aren't false, and their values, ordered by tuple."""
        found: list[tuple[tuple[int, ...], LogicValue]] = []
        for logic_value, tuples in self.tuples().items():
            found += [(key, logic_value) for key in map(tuple, tuples.tolist())]
        return iter(sorted(found))

    def value(self, key: tuple[int, ...]) -> LogicValue:
        must, may = _member(self.must, key), _member(self.may, key)
        if must:
            return "true" if may else "error"
        return "unknown" if may else "false"


def _member(value: Value, key: tuple[int, ...]) -> bool:
    match value.arity:
        case 0:
            return bool(value.data)
        case 1:
            return k.contains(value.data, key[0])
        case 2:
            return k.contains(value.data[key[0]], key[1])
        case _:
            return bool(np.any(np.all(value.data == np.asarray(key), axis=1)))


def evaluate(
    rules: str | Sequence[str], model: PartialModel, select: Sequence[str] | None = None
) -> dict[str, PredicateValues]:
    """Evaluate the predicates ``rules`` define over ``model``.

    ``rules`` is Refinery source (statements other than ``pred`` declarations
    are skipped). The predicates named in ``select`` are evaluated, or else
    every one with a body.
    """
    source = rules if isinstance(rules, str) else "\n".join(rules)
    parsed = parse(source)
    _check_not_refined(parsed)
    evaluator = _Evaluator(model, list(parsed.predicates))
    names = list(select) if select is not None else list(evaluator.defined)
    results = {}
    for name in names:
        predicate = evaluator.predicate(name)
        if predicate is None:
            raise RuleError(f"no predicate named {name!r} in the rules")
        must = evaluator.relation(predicate.name, "must", predicate.position)
        may = evaluator.relation(predicate.name, "may", predicate.position)
        arity = len(predicate.parameters)
        results[predicate.name] = PredicateValues(predicate.name, predicate.kind, arity, must, may)
    return results


def _check_not_refined(source: Source) -> None:
    """Refuse sources where Refinery would refine a predicate beyond what its body says."""
    defined = {p.name: p for p in source.predicates if p.body is not None}
    for predicate in source.predicates:
        if predicate.subsets is None:
            continue
        if predicate.body is not None:
            raise UnsupportedRuleError(
                f"subsets on a predicate with a body ({predicate.name})", predicate.position
            )
        if predicate.subsets in defined:
            raise UnsupportedRuleError(
                f"subsets of a predicate with a body ({predicate.subsets})", predicate.position
            )
    for name in defined:
        if name in source.asserted:
            raise UnsupportedRuleError(
                f"assertions about a predicate with a body ({name})", source.asserted[name]
            )
        if name in source.in_classes:
            raise UnsupportedRuleError(
                f"a predicate with a body as the type of a reference ({name})",
                source.in_classes[name],
            )


def _flip(modality: Modality) -> Modality:
    return "may" if modality == "must" else "must"


class _Evaluator:
    def __init__(self, model: PartialModel, predicates: list[Predicate]) -> None:
        self.model = model
        self.n = model.size
        self.defined: dict[str, Predicate] = {}
        self.declared: dict[str, Predicate] = {}
        """Predicates without a body: their values are the model's, or else unknown."""
        for predicate in predicates:
            if predicate.body is None:
                self.declared.setdefault(predicate.name, predicate)
                continue
            if predicate.name in self.defined:
                raise RuleError(f"{predicate.name} is defined twice", predicate.position)
            self.defined[predicate.name] = predicate
        self._derived: dict[tuple[str, Modality], Value] = {}
        self._base: dict[tuple[str, Modality], Value] = {}
        self._closures: dict[tuple[str, Modality], Value] = {}
        self._active: list[str] = []
        self._clauses: dict[str, list[Clause]] = {}
        self.context = Context(model, self.is_attribute, self.identity)
        exists = self._lookup_relation("exists", None)
        self.exists = {m: self.base(exists, m) for m in ("must", "may")}

    # -- relations --------------------------------------------------------------

    def _lookup_relation(self, name: str, position: Position | None) -> Relation:
        try:
            relation = self.model.relation_named(name)
        except ValueError as e:
            raise RuleError(str(e), position) from None
        if relation is None:
            raise RuleError(f"no relation named {name!r}", position)
        return relation

    def predicate(self, name: str) -> Predicate | None:
        """A predicate of the rules by name, as Refinery writes it or (if it needs
        quotes) unquoted, e.g. ``'odd name'`` or ``odd name``."""
        for candidate in (name, refinery_name(name)):
            found = self.defined.get(candidate) or self.declared.get(candidate)
            if found is not None:
                return found
        return None

    def base(self, relation: Relation, modality: Modality) -> Value:
        key = (relation.name, modality)
        if key not in self._base:
            tuples = relation.must if modality == "must" else relation.may
            self._base[key] = from_tuples(relation.arity, self.n, tuples)
        return self._base[key]

    def relation(self, name: str, modality: Modality, position: Position) -> Value:
        predicate = self.defined.get(name)
        if predicate is not None:
            return self.derived(predicate, modality)
        declared = self.declared.get(name)
        if declared is not None and self.model.relation_named(name) is None:
            return self.undetermined(declared, modality)
        return self.base(self._lookup_relation(name, position), modality)

    def undetermined(self, predicate: Predicate, modality: Modality) -> Value:
        """A predicate without a body or values: unknown for the nodes of its parameter
        types, false for others, as Refinery leaves one nothing is asserted about."""
        arity = len(predicate.parameters)
        if modality == "must":
            return empty(arity, self.n)
        domains = []
        for parameter in predicate.parameters:
            if parameter.type is None:
                domains.append(k.full_vector(self.n))
                continue
            kind = self.relation(parameter.type, "may", predicate.position)
            if kind.arity != 1:
                raise RuleError(f"{parameter.type} isn't a class", predicate.position)
            domains.append(kind.data)
        match arity:
            case 0:
                return Value(0, self.n, True)
            case 1:
                return Value(1, self.n, domains[0])
            case 2:
                return Value(2, self.n, k.outer(domains[0], domains[1], self.n))
            case _:
                axes = [k.indices(d, self.n) for d in domains]
                grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(arity, -1).T
                return from_tuples(arity, self.n, grid)

    def derived(self, predicate: Predicate, modality: Modality) -> Value:
        key = (predicate.name, modality)
        if key in self._derived:
            return self._derived[key]
        if predicate.name in self._active:
            cycle = " -> ".join(
                [*self._active[self._active.index(predicate.name) :], predicate.name]
            )
            raise RuleError(
                f"recursive predicates ({cycle}): Refinery allows none; use + for transitive "
                "closure",
                predicate.position,
            )
        self._active.append(predicate.name)
        try:
            arity = len(predicate.parameters)
            result = empty(arity, self.n)
            if predicate.kind != "error" or modality == "must":
                for clause in self.normalized(predicate):
                    result = union(result, self.clause(clause, modality))
        finally:
            self._active.pop()
        self._derived[key] = result
        return result

    def closure(self, name: str, modality: Modality, position: Position) -> Value:
        """``name+``: steps of ``name``, each but the last to a node that exists."""
        key = (name, modality)
        if key not in self._closures:
            steps = self.relation(name, modality, position)
            if steps.arity != 2:
                raise RuleError(f"{name}+ needs a relation of two nodes", position)
            through = k.mask_columns(steps.data, self.exists[modality].data)
            paths = k.product(k.closure(through, self.n), steps.data, self.n)
            self._closures[key] = Value(2, self.n, steps.data | paths)
        return self._closures[key]

    def call(self, call: CallLiteral, modality: Modality) -> Value:
        atom = call.atom
        if atom.transitive:
            value = self.closure(atom.relation, modality, atom.position)
        else:
            value = self.relation(atom.relation, modality, atom.position)
        if value.arity != len(call.arguments):
            raise RuleError(
                f"{atom.relation} has {value.arity} parameters, given {len(call.arguments)}",
                atom.position,
            )
        return value

    # -- clauses --------------------------------------------------------------

    def normalized(self, predicate: Predicate) -> list[Clause]:
        """The clauses of ``predicate`` as Refinery rewrites them, without those always false."""
        if predicate.name not in self._clauses:
            assert predicate.body is not None
            clauses = [normalize(predicate, literals, self.context) for literals in predicate.body]
            self._clauses[predicate.name] = [c for c in clauses if c is not None]
        return self._clauses[predicate.name]

    def is_attribute(self, name: str) -> bool:
        if name in self.defined:
            return False
        try:
            attribute = self.model.attribute_named(name)
        except ValueError:
            return False
        return attribute is not None and attribute.data_type == "boolean"

    def identity(self, atom: Atom) -> str:
        if atom.relation in self.defined or atom.relation in self.declared:
            return atom.relation
        return self._lookup_relation(atom.relation, atom.position).name

    def clause(self, clause: Clause, modality: Modality) -> Value:
        parameters = set(range(clause.parameters))
        factors: list[Factor] = []
        for call in clause.calls:
            if not call.atom.negated:
                value = self.call(call, call.atom.modality or modality)
                factors.append(bind(value, call.arguments))
        existential = {
            v for c in clause.calls if not c.atom.negated and not c.attribute for v in c.variables
        } - parameters
        for variable in sorted(existential):
            factors.append(Factor((variable,), self.exists[modality]))
        for call in clause.calls:
            if call.atom.negated:
                negation = call.atom.modality or _flip(modality)
                factors.append(self.negative(call, clause.bound, negation))
        for compared in clause.comparisons:
            factors.append(self.comparison(compared, modality))
        return solve(factors, tuple(range(clause.parameters)), self.n)

    def negative(self, call: CallLiteral, bound: frozenset[int], modality: Modality) -> Factor:
        """``!call``, read in ``modality`` (the clause's flipped, or the atom's own)."""
        value = self.call(call, modality)
        args = call.arguments
        variables = [a.id for a in args if isinstance(a, Var)]
        local = sorted(set(variables) - bound)
        if not local:
            return bind(value, args, negated=True)
        outer = tuple(dict.fromkeys(v for v in variables if v in bound))
        inner = [bind(value, args), *(Factor((v,), self.exists[modality]) for v in local)]
        return make_factor(outer, solve(inner, outer, self.n), negated=True)

    def comparison(self, compared: CompareLiteral, modality: Modality) -> Factor:
        comparison = compared.comparison
        left = self.operand(compared.left, comparison.position)
        right = self.operand(compared.right, comparison.position)
        variables = tuple(
            dict.fromkeys(v for v in (left.variable, right.variable) if v is not None)
        )
        strings = {left.is_string, right.is_string} - {None}
        if len(strings) > 1:
            raise RuleError("a comparison of a string and a number", comparison.position)
        booleans = any(
            o.attribute is not None and o.attribute.data_type == "boolean" for o in (left, right)
        )
        if booleans and comparison.operator not in ("==", "!="):
            raise UnsupportedRuleError("ordering booleans", comparison.position)
        if strings == {True}:
            must, may = _compare_strings(left, right, variables, comparison)
        else:
            must, may = _compare_numbers(left, right, variables, comparison)
        holds = must if modality == "must" else may
        if comparison.negated:
            holds = ~(may if modality == "must" else must)
        return Factor(variables, _pack(holds, len(variables), self.n))

    def operand(self, operand: Operand, position: Position) -> _Operand:
        if operand.attribute is None:
            return _Operand(None, None, operand.constant, None)
        try:
            attribute = self.model.attribute_named(operand.attribute)
        except ValueError as e:
            raise RuleError(str(e), position) from None
        if attribute is None:
            raise RuleError(f"no attribute named {operand.attribute!r}", position)
        return _Operand(operand.variable, operand.node, None, attribute)


# -- comparisons --------------------------------------------------------------


@dataclass(frozen=True)
class _Operand:
    variable: int | None
    node: int | None
    constant: object
    attribute: Attribute | None

    @property
    def is_string(self) -> bool | None:
        if self.attribute is not None:
            return self.attribute.data_type == "string"
        return isinstance(self.constant, str)


def _shape(operand: _Operand, variables: tuple[int, ...]) -> tuple[int, ...]:
    """Where an operand's values go, to broadcast over the compared variables."""
    shape = [1] * len(variables)
    if operand.variable is not None:
        shape[variables.index(operand.variable)] = -1
    return tuple(shape)


def _bounds(operand: _Operand, variables: tuple[int, ...]) -> tuple[NDArray, NDArray]:
    if operand.attribute is None:
        value = operand.constant
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, Decimal):
            low, high = real_interval(value)  # a real constant, as RealInterval.of rounds it
        elif isinstance(value, int):
            low = high = value
        else:
            raise RuleError(f"not a number: {value!r}")
        # Floats, unless they can't hold the numbers exactly.
        if float(low) == low and float(high) == high:
            return np.asarray(float(low)), np.asarray(float(high))
        return np.asarray(low, dtype=object), np.asarray(high, dtype=object)
    lower, upper = operand.attribute.lower, operand.attribute.upper
    if operand.node is not None:
        return lower[operand.node : operand.node + 1].reshape(()), upper[
            operand.node : operand.node + 1
        ].reshape(())
    shape = _shape(operand, variables)
    return lower.reshape(shape), upper.reshape(shape)


def _exact_together(*arrays: NDArray) -> tuple[NDArray, ...]:
    """The arrays, as exact Python numbers if any of them needs to be."""
    if any(array.dtype == object for array in arrays):
        return tuple(array.astype(object) for array in arrays)
    return arrays


def _compare_numbers(
    left: _Operand, right: _Operand, variables: tuple[int, ...], comparison: Comparison
) -> tuple[NDArray, NDArray]:
    """Whether the comparison must hold, and may: ``IntInterval`` and ``RealInterval``
    in Refinery's ``logic``, whose error is the empty interval ``[inf, -inf]``.

    ``a <= b`` may hold if ``lo(a) <= hi(b)``, and must if ``hi(a) <= lo(b)``;
    ``a == b`` is ``a <= b`` and ``b <= a``; ``a < b`` is ``not (b <= a)``, and
    ``not`` turns ``(must, may)`` into ``(not may, not must)``.
    """
    a_low, a_high, b_low, b_high = _exact_together(
        *_bounds(left, variables), *_bounds(right, variables)
    )
    a, b = (a_low, a_high), (b_low, b_high)

    def less_eq(x: tuple[NDArray, NDArray], y: tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
        return np.asarray(x[1] <= y[0], bool), np.asarray(x[0] <= y[1], bool)

    def negation(value: tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
        return ~value[1], ~value[0]

    def equal() -> tuple[NDArray, NDArray]:
        (must_ab, may_ab), (must_ba, may_ba) = less_eq(a, b), less_eq(b, a)
        return must_ab & must_ba, may_ab & may_ba

    match comparison.operator:
        case "<=":
            return less_eq(a, b)
        case ">=":
            return less_eq(b, a)
        case "<":
            return negation(less_eq(b, a))
        case ">":
            return negation(less_eq(a, b))
        case "==":
            return equal()
        case "!=":
            return negation(equal())
    raise RuleError(f"unknown operator {comparison.operator}", comparison.position)


def _with_errors(must: NDArray, may: NDArray, error: NDArray) -> tuple[NDArray, NDArray]:
    """A comparison of strings is an error if either is (``StringValue.checkEquals``)."""
    return must | error, may & ~error


_UNKNOWN, _ERROR = -1, -2


def _compare_strings(
    left: _Operand, right: _Operand, variables: tuple[int, ...], comparison: Comparison
) -> tuple[NDArray, NDArray]:
    if comparison.operator not in ("==", "!="):
        raise RuleError("strings can only be compared with == and !=", comparison.position)
    codes: dict[str, int] = {}

    def code(attribute: Attribute, node: int) -> int:
        if attribute.lower[node] > attribute.upper[node]:
            return _ERROR
        text = attribute.strings[node]
        return _UNKNOWN if text is None else codes.setdefault(text, len(codes))

    def encode(operand: _Operand) -> NDArray:
        attribute = operand.attribute
        if attribute is None:
            return np.asarray(codes.setdefault(str(operand.constant), len(codes)))
        if operand.node is not None:
            return np.asarray(code(attribute, operand.node))
        array = np.array([code(attribute, node) for node in range(len(attribute.lower))])
        return array.reshape(_shape(operand, variables))

    a, b = encode(left), encode(right)
    known = (a >= 0) & (b >= 0)
    equal = known & (a == b)
    if comparison.operator == "==":
        must, may = equal, ~known | (a == b)
    else:
        must, may = known & (a != b), ~equal
    return _with_errors(must, may, (a == _ERROR) | (b == _ERROR))


def _pack(holds: NDArray, arity: int, n: int) -> Value:
    holds = np.broadcast_to(holds, (n,) * arity)
    match arity:
        case 0:
            return Value(0, n, bool(holds))
        case 1:
            dense = np.zeros(k.words(n) * 64, bool)
            dense[:n] = holds
            return Value(1, n, k.pack(dense))
        case _:
            dense = np.zeros((n, k.words(n) * 64), bool)
            dense[:, :n] = holds
            return Value(2, n, k.pack(dense))
