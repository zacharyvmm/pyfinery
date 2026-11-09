# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Clauses as Refinery reads them before evaluating them: a port of ``ClausePostProcessor``.

Refinery checks and rewrites each clause before reading it in a modality
(``subprojects/logic/src/main/java/tools/refinery/logic/dnf/ClausePostProcessor.java``,
commit ``7470710``):

- a parameter must be bound by a literal that isn't negated, and the inputs of
  comparisons and attribute atoms must be bound;
- a variable no literal binds may appear in only one literal (besides
  negated atoms, whose own variables are their own);
- a clause that calls a relation and its negation with the same bound
  arguments, e.g. ``R(x), !R(x, _)``, is false, whatever the modality.

``x == y`` and ``x != y`` are calls of the builtin ``equals``, as Refinery's
language translates them: they bind variables like other atoms.

This module checks and rewrites; :mod:`._evaluate` reads the result in a modality.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ._errors import RuleError
from ._model import PartialModel
from ._solve import Argument, Node, Var
from ._syntax import Atom, Call, Comparison, Constant, Literal_, Name, Predicate, Term


@dataclass(frozen=True)
class CallLiteral:
    """An atom with its arguments resolved."""

    atom: Atom
    arguments: tuple[Argument, ...]
    attribute: bool = False
    """A boolean attribute read as an atom: its arguments are inputs, which it doesn't bind."""

    @property
    def variables(self) -> set[int]:
        return {a.id for a in self.arguments if isinstance(a, Var)}


@dataclass(frozen=True)
class Operand:
    variable: int | None = None
    node: int | None = None
    constant: object = None
    attribute: str | None = None


@dataclass(frozen=True)
class CompareLiteral:
    comparison: Comparison
    left: Operand
    right: Operand

    @property
    def variables(self) -> set[int]:
        return {o.variable for o in (self.left, self.right) if o.variable is not None}


@dataclass(frozen=True)
class Clause:
    """A clause, rewritten: ``None`` from :func:`normalize` if it's always false."""

    calls: tuple[CallLiteral, ...]
    comparisons: tuple[CompareLiteral, ...]
    bound: frozenset[int]
    """The variables literals bind: parameters, and those in positive atoms."""
    parameters: int


@dataclass(frozen=True)
class Context:
    model: PartialModel
    is_attribute: Callable[[str], bool]
    """Whether a name is that of a boolean attribute (and no predicate's)."""
    identity: Callable[[Atom], str]
    """What an atom calls: the same for two atoms of one relation."""


def normalize(
    predicate: Predicate, literals: Sequence[Literal_], context: Context
) -> Clause | None:
    scope = _Scope(predicate, context.model)
    parameters = len(predicate.parameters)
    calls: list[CallLiteral] = []
    comparisons: list[CompareLiteral] = []
    clause_level: set[int] = set()  # variables of literals that aren't negated atoms

    for index, parameter in enumerate(predicate.parameters):
        if parameter.type is not None:
            type_atom = Atom(parameter.type, (), False, None, False, predicate.position)
            calls.append(CallLiteral(type_atom, (Var(index),)))
    for literal in literals:
        if isinstance(literal, Comparison) and _of_nodes(literal):
            if literal.negated:
                raise RuleError(
                    "a negated comparison of nodes: write != or == instead", literal.position
                )
            left = scope.argument(literal.left)  # type: ignore[arg-type]
            right = scope.argument(literal.right)  # type: ignore[arg-type]
            negated = literal.operator == "!="
            atom = Atom("equals", (), negated, None, False, literal.position)
            call = CallLiteral(atom, (left, right))
            clause_level |= call.variables  # ``!=`` isn't a negation scope
            calls.append(call)
        elif isinstance(literal, Comparison):
            compared = CompareLiteral(
                literal,
                _operand(literal.left, scope, literal),
                _operand(literal.right, scope, literal),
            )
            clause_level |= compared.variables
            comparisons.append(compared)
        else:
            attribute = context.is_attribute(literal.relation)
            if attribute and literal.transitive:
                raise RuleError(f"{literal.relation} is an attribute", literal.position)
            call = CallLiteral(
                literal, tuple(scope.argument(a) for a in literal.arguments), attribute
            )
            if attribute or not literal.negated:
                clause_level |= call.variables
            calls.append(call)

    bound = {v for c in calls if not c.atom.negated and not c.attribute for v in c.variables}
    for index, parameter in enumerate(predicate.parameters):
        if index not in bound:
            raise RuleError(
                f"parameter {parameter.name} of {predicate.name} is bound by no literal",
                predicate.position,
            )
    inputs = {v for c in calls if c.attribute for v in c.variables}
    inputs |= {v for c in comparisons for v in c.variables}
    unbound_inputs = sorted(inputs - bound)
    if unbound_inputs:
        names = ", ".join(scope.name(v) for v in unbound_inputs)
        raise RuleError(f"unbound input variables: {names}", predicate.position)
    for variable in clause_level - bound:
        users = [c for c in calls if variable in c.variables]
        if len(users) > 1:
            raise RuleError(
                f"variable {scope.name(variable)} is bound by no literal, and appears in "
                "several literals",
                predicate.position,
            )
    if _contradicts(calls, bound, context):
        return None
    return Clause(tuple(calls), tuple(comparisons), frozenset(bound), parameters)


def _of_nodes(comparison: Comparison) -> bool:
    sides = (comparison.left, comparison.right)
    return comparison.operator in ("==", "!=") and all(isinstance(s, Name) for s in sides)


def _operand(term: Term, scope: _Scope, comparison: Comparison) -> Operand:
    if isinstance(term, Constant):
        return Operand(constant=term.value)
    if isinstance(term, Call):
        if len(term.arguments) != 1:
            raise RuleError(f"{term.relation} takes one argument", term.position)
        argument = scope.argument(term.arguments[0])
        if isinstance(argument, Var):
            return Operand(variable=argument.id, attribute=term.relation)
        return Operand(node=argument.index, attribute=term.relation)
    raise RuleError("only attribute values and constants can be compared", comparison.position)


def _contradicts(calls: list[CallLiteral], bound: set[int], context: Context) -> bool:
    """Whether the clause calls a relation and its negation at the same bound arguments."""

    def target(call: CallLiteral) -> tuple[str, str | None] | None:
        if call.atom.transitive or call.attribute:
            return None
        return context.identity(call.atom), call.atom.modality

    positive: dict[tuple[str, str | None], list[CallLiteral]] = {}
    for call in calls:
        key = target(call)
        if key is not None and not call.atom.negated:
            positive.setdefault(key, []).append(call)
    for call in calls:
        key = target(call)
        if key is None or not call.atom.negated:
            continue
        for other in positive.get(key, []):
            if all(
                (isinstance(n, Var) and n.id not in bound) or n == p
                for n, p in zip(call.arguments, other.arguments, strict=True)
            ):
                return True
    return False


# -- names --------------------------------------------------------------------


class _Scope:
    """The variables of a clause: parameters first, then the rest as they come."""

    def __init__(self, predicate: Predicate, model: PartialModel) -> None:
        self.model = model
        self.ids: dict[str, int] = {}
        self.names: dict[int, str] = {}
        for parameter in predicate.parameters:
            if parameter.name in self.ids:
                raise RuleError(f"two parameters named {parameter.name}", predicate.position)
            self.ids[parameter.name] = len(self.ids)
            self.names[len(self.names)] = parameter.name
        self.next = len(self.ids)

    def name(self, variable: int) -> str:
        return self.names.get(variable, "_")

    def argument(self, name: Name) -> Argument:
        if name.text == "_":
            self.next += 1
            return Var(self.next - 1)
        if name.text in self.ids:
            return Var(self.ids[name.text])
        try:
            atom = self.model.atom_named(name.text)
        except ValueError as e:
            raise RuleError(str(e), name.position) from None
        if atom is not None:
            return Node(atom)
        if "::" in name.text:
            raise RuleError(f"no atom named {name.text!r}", name.position)
        self.ids[name.text] = self.next
        self.names[self.next] = name.text
        self.next += 1
        return Var(self.ids[name.text])
