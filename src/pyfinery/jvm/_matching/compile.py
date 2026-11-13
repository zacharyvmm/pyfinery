"""Rules compiled with Refinery's own compiler and rewritten to match a plain model.

:class:`CompiledProblem` parses a metamodel, its rules, the declarations of
:class:`~pyfinery.Predicate` objects and the metamodel's structural checks
(:mod:`.structure`) as one Refinery problem, compiles every predicate with
Refinery's ``QueryCompiler``, and rewrites the result from the partial-model
symbols it targets onto plain two-valued ones: a store with Refinery's
incremental (Rete) query interpreter and no reasoning layer, so a model is
matched as given, not concretized.

On a complete model, Refinery's built-in predicates have their plain meaning:

- ``node(x)`` and ``exists(x)`` hold for every object of the model;
- ``equals(x, y)`` is identity;
- ``container(x)`` and ``contained(x)`` hold for the instances of classes with an
  outgoing, or incoming, containment reference;
- ``contains(x, y)`` is any containment reference, and ``invalidContainer(y)`` a
  contained object with no container, several, or on a containment cycle;
- ``Owner::feature::invalidMultiplicity(x)`` is the metamodel's multiplicity check;
- the modal operators ``must``, ``may`` and ``candidate`` change nothing.

What a plain store can't match yet, attribute values and ``count``, makes the
predicates that use them, and those that call them, unsupported: they are
listed in :attr:`CompiledProblem.unsupported` and not matched.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..._mapping import Catalogue
from ..._options import Predicate
from ..._render import _declare_predicate, render_metamodel
from ..._results import ViolationKind
from ..._schema import Schema
from ...errors import InvalidProblemError
from .classes import java
from .structure import CONTAINS, INVALID_CONTAINER, Check, structure

NODE = "builtin::node"
"""The plain symbol every object of a model is in: ``node`` and ``exists``."""
CONTAINER = "container"
CONTAINED = "contained"


@dataclass(frozen=True, slots=True)
class ErrorQuery:
    """A predicate whose tuples are violations, and how they are reported."""

    predicate: str
    kind: ViolationKind
    constraint: str
    order: int
    """Where a server would list them: the position of ``constraint`` among the relations."""


class CompiledProblem:
    """A metamodel, its rules and its predicates, compiled into a plain store.

    ``rules`` are Refinery source; ``predicates`` are declared as base predicates,
    whose facts come from Python.
    """

    def __init__(
        self, schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate] = ()
    ) -> None:
        j = java()
        self.schema = schema
        self.catalogue = Catalogue(schema)
        self.timings: dict[str, float] = {}
        structural = structure(schema, self.catalogue)
        self.unchecked_multiplicities = list(structural.unchecked)
        declarations = [_declare_predicate(schema, p) for p in predicates]
        self.source = render_metamodel(schema, [*rules, *declarations])
        text = self.source + "\n" + structural.text + "\n"

        clock = time.perf_counter()
        injector = j.StandaloneRefinery.getInjector()
        try:
            problem = injector.getInstance(j.ProblemLoader).loadString(text)
        except j.InvalidProblemException as error:
            raise InvalidProblemError(str(error.getMessage())) from None
        initializer = injector.getInstance(j.ModelInitializer)
        initializer.readProblem(problem)
        trace = initializer.getProblemTrace()
        compiler = injector.getInstance(j.QueryCompiler)
        compiler.setProblemTrace(trace)
        self.timings["parse"] = time.perf_counter() - clock

        clock = time.perf_counter()
        self.reserved_ids = int(trace.getNodeTrace().size())
        """Ids the problem's own nodes take (enum literals, ``Class::new``): objects start here."""
        self.constants = {
            name: int(trace.getNodeId(name)) for name in self.catalogue.literal_names()
        }
        """Enum literals by Refinery name, and their node ids."""

        self._trace = trace
        self._lookups: dict[str, Any] = {}
        self.parameters: dict[str, tuple[str, ...]] = {}
        """The names of each compiled predicate's parameters."""
        self.symbols: dict[str, Any] = {}
        """Plain symbols by relation name: classes, enums, references, base predicates."""
        self._views: dict[Any, Any] = {}  # PartialRelation -> KeyOnlyView of its symbol
        self._compiled: dict[Any, Any] = {}  # PartialRelation -> RelationalQuery (partial)
        self._names: dict[Any, str] = {}  # PartialRelation -> name
        order: dict[str, int] = {}
        kinds: dict[str, str] = {}  # error predicates of the rules

        builtins = {
            str(j.ReasoningAdapter.EXISTS_SYMBOL.name()),
            str(j.ReasoningAdapter.EQUALS_SYMBOL.name()),
            str(j.ContainmentHierarchyTranslator.CONTAINS_SYMBOL.name()),
            str(j.ContainmentHierarchyTranslator.INVALID_CONTAINER.name()),
        }
        for position, entry in enumerate(trace.getRelationTrace().entrySet()):
            relation, partial = entry.getKey(), entry.getValue()
            name = str(partial.name())
            order.setdefault(name, position)
            if not isinstance(partial, j.PartialRelation) or name.endswith("::computed"):
                continue  # attributes (functions), and Refinery's copies of rules
            self._names[partial] = name
            if name in builtins or name.endswith("::invalidMultiplicity"):
                continue  # given their plain meaning below
            if isinstance(relation, j.PredicateDefinition) and not j.ProblemUtil.isBasePredicate(
                relation
            ):
                self._compiled[partial] = compiler.toQuery(partial.name(), relation)
                self.parameters[name] = tuple(str(p.getName()) for p in relation.getParameters())
                if relation.getKind() == j.PredicateKind.ERROR:
                    kinds[name] = "rule"
            else:  # a class, enum, reference, or base predicate (node, container, contained too)
                symbol = j.Symbol.of(name, int(partial.arity()))
                self.symbols[name] = symbol
                self._views[partial] = j.KeyOnlyView(symbol)

        # Built-ins with their plain meaning.
        node_view = self._views[trace.getPartialRelation(NODE)]
        self._views[j.ReasoningAdapter.EXISTS_SYMBOL] = node_view
        self._equals = j.ReasoningAdapter.EQUALS_SYMBOL
        self._plain_queries: dict[Any, str] = {}  # PartialRelation -> structural predicate
        self._extra_symbols: list[Any] = []
        if structural.containment:
            self._plain_queries[j.ContainmentHierarchyTranslator.CONTAINS_SYMBOL] = CONTAINS
            self._plain_queries[j.ContainmentHierarchyTranslator.INVALID_CONTAINER] = (
                INVALID_CONTAINER
            )
        else:
            # Without containment references, nothing contains anything: relations that stay empty
            # (rather than false, which would leave a rule's variables unbound).
            for partial in (
                j.ContainmentHierarchyTranslator.CONTAINS_SYMBOL,
                j.ContainmentHierarchyTranslator.INVALID_CONTAINER,
            ):
                empty = j.Symbol.of(f"pyfinery_empty::{partial.name()}", int(partial.arity()))
                self._extra_symbols.append(empty)
                self._views[partial] = j.KeyOnlyView(empty)
        checks = {check.predicate: check for check in structural.checks}
        for check in structural.checks:
            if check.kind == "multiplicity":
                self._plain_queries[trace.getPartialRelation(check.constraint)] = check.predicate
        self._by_name = {name: partial for partial, name in self._names.items()}

        # Rewrite every compiled predicate onto the plain symbols.
        self._rewritten: dict[Any, Any] = {}
        self._quantified = 0
        self._unsupported_dnfs: dict[Any, str] = {}
        self.queries: dict[str, Any] = {}
        """Matchable predicates by name, as plain queries."""
        self.unsupported: dict[str, str] = {}
        """Predicates the direct path can't match yet, and why."""
        for partial, query in self._compiled.items():
            name = self._names[partial]
            rewritten = self._rewrite(query.getDnf())
            reason = self._unsupported_dnfs.get(rewritten)
            if reason is not None:
                self.unsupported[name] = reason
            else:
                self.queries[name] = rewritten.asRelation()

        self.errors: list[ErrorQuery] = []
        """The predicates whose tuples are violations, in the order a server lists them."""
        for name, kind in kinds.items():
            self.errors.append(ErrorQuery(name, kind, name, order[name]))  # type: ignore[arg-type]
        for check in checks.values():
            self.errors.append(_error(check, order))
        self.errors.sort(key=lambda error: error.order)
        self.timings["compile"] = time.perf_counter() - clock

        clock = time.perf_counter()
        queries = j.ArrayList()
        for query in self.queries.values():
            queries.add(query)
        self.store = (
            j.ModelStore.builder()
            .symbols(j.ArrayList([*self.symbols.values(), *self._extra_symbols]))
            .with_(j.QueryInterpreterAdapter.builder().queries(queries))
            .build()
        )
        self.timings["store"] = time.perf_counter() - clock

    def symbol(self, relation: str) -> Any:
        """The plain symbol of ``relation``, named as Refinery prints it or as pyfinery renders it
        (quoted where it must be)."""
        found = self.symbols.get(relation) or self._lookups.get(relation)
        if found is None:
            partial = self._trace.getPartialRelation(relation)
            found = self._lookups[relation] = self.symbols[str(partial.name())]
        return found

    @property
    def rule_errors(self) -> list[str]:
        """The rules' error predicates."""
        return [error.predicate for error in self.errors if error.kind == "rule"]

    # -- rewriting -------------------------------------------------------------------------------

    def _rewrite(self, dnf: Any) -> Any:
        j = java()
        known = self._rewritten.get(dnf)
        if known is not None:
            return known
        builder = j.Dnf.builderFrom(dnf)
        reason: str | None = None
        parameters = {p.getVariable() for p in dnf.getSymbolicParameters()}
        for clause in dnf.getClauses():
            bound = set(parameters)
            for literal in clause.literals():
                bound.update(literal.getOutputVariables())
            literals = j.ArrayList()
            for literal in clause.literals():
                rewritten, why = self._rewrite_literal(literal)
                reason = reason or why
                literals.add(self._quantify_negation(rewritten, bound))
            builder.clause(literals)
        result = builder.build()
        self._rewritten[dnf] = result
        if reason is not None:
            self._unsupported_dnfs[result] = reason
        return result

    def _quantify_negation(self, literal: Any, bound: set[Any]) -> Any:
        """A negated call with variables of its own (``!Link(_)``: no link exists) as the negation
        of a predicate over its other variables only, which is how Refinery's matcher takes it."""
        j = java()
        negative = isinstance(literal, j.CallLiteral) and (
            literal.getPolarity() == j.CallPolarity.NEGATIVE
        )
        if not negative:
            return literal
        arguments = list(literal.getArguments())
        if all(argument in bound for argument in arguments):
            return literal
        kept = list(dict.fromkeys(a for a in arguments if a in bound))
        target = literal.getTarget()
        positive = j.CallLiteral(j.CallPolarity.POSITIVE, target, j.ArrayList(arguments))
        self._quantified += 1
        inner = (
            j.Dnf.builder(f"pyfinery_exists{self._quantified}")
            .parameters(j.ArrayList(kept))
            .clause(j.ArrayList([positive]))
            .build()
        )
        return j.CallLiteral(j.CallPolarity.NEGATIVE, inner, j.ArrayList(kept))

    def _rewrite_literal(self, literal: Any) -> tuple[Any, str | None]:
        """The literal on plain symbols, and why it can't be matched, if it can't."""
        j = java()
        if not isinstance(literal, j.CallLiteral):
            if isinstance(literal, (j.ConstantLiteral, j.BooleanLiteral)):
                return literal, None
            return literal, _why(literal)
        target = literal.getTarget()
        if isinstance(target, j.ModalConstraint):
            target = target.constraint()  # on a complete model, every modality is the fact
            literal = literal.withTarget(target)
        if isinstance(target, j.Dnf):
            inner = self._rewrite(target)
            return literal.withTarget(inner), self._unsupported_dnfs.get(inner)
        if not isinstance(target, j.PartialRelation):
            return literal, f"a call of {target}"
        positive = literal.getPolarity() != j.CallPolarity.NEGATIVE
        if target.equals(self._equals):
            arguments = literal.getArguments()
            return j.EquivalenceLiteral(positive, arguments.get(0), arguments.get(1)), None
        structural = self._plain_queries.get(target)
        if structural is not None:
            target = self._by_name[structural]
        query = self._compiled.get(target)
        if query is not None:
            inner = self._rewrite(query.getDnf())
            return literal.withTarget(inner), self._unsupported_dnfs.get(inner)
        view = self._views.get(target)
        if view is None:
            return literal, f"{target.name()}, which has no plain counterpart"
        return literal.withTarget(view), None


def _why(literal: Any) -> str:
    kind = str(literal.getClass().getSimpleName())
    if kind == "PartialCheckLiteral":
        return f"attribute values or count ({literal})"
    return f"{kind} ({literal})"


def _error(check: Check, order: dict[str, int]) -> ErrorQuery:
    # A multiplicity is listed after its reference, a contradiction with its relation.
    name = check.constraint.removeprefix("builtin::")  # Refinery's trace has no prefix there
    position = order.get(name, len(order))
    return ErrorQuery(check.predicate, check.kind, check.constraint, position)
