"""Rules compiled with Refinery's own compiler and rewritten to match a plain model.

:func:`compile_problem` parses a metamodel, its rules, the declarations of
:class:`~pyfinery.Predicate` objects and the metamodel's structural checks
(:mod:`.structure`) as one Refinery problem, compiles every predicate with
Refinery's ``QueryCompiler``, and rewrites the result from the partial-model
symbols it targets onto plain two-valued ones: a store with Refinery's
incremental (Rete) query interpreter and no reasoning layer, so a model is
matched as given, not concretized. :func:`compiled` keeps the last problems
compiled for reuse.

On a complete model, Refinery's built-in predicates have their plain meaning:

- ``node(x)`` and ``exists(x)`` hold for every object of the model;
- ``equals(x, y)`` is identity;
- ``container(x)`` and ``contained(x)`` hold for the instances of classes with an
  outgoing, or incoming, containment reference;
- ``contains(x, y)`` is any containment reference, and ``invalidContainer(y)`` a
  contained object with no container, several, or on a containment cycle;
- ``Owner::feature::invalidMultiplicity(x)`` is the metamodel's multiplicity check;
- the modal operators ``must``, ``may`` and ``candidate`` change nothing.

Attribute values are held as exact values of Refinery's domains
(:func:`.model.to_java`), and a check over them, ``File::size(f) > 3``, holds when it
is certainly true. An attribute with no value has an unknown one: neither a
check over it nor the negation of the check holds. An attribute read on an
object of another class (``File::size(d)`` for a directory ``d``) is a
contradiction, ``error``, as on a server: every check over it holds, and so is
reported. ``count``, ``sum``, ``min``
and ``max`` aggregate over the matches of their condition; over none, a count
and a sum are 0, a ``min`` is positive infinity and a ``max`` negative
infinity, as in Refinery.

What the direct path can't match (e.g. the ``meet``, ``join`` and ``reify``
aggregators, which are about partial models) makes the predicates that use it,
and those that call them, unsupported: they are listed in
:attr:`CompiledProblem.unsupported` and not matched.

The model holds the graph's facts and those of :class:`~pyfinery.Predicate`
objects only. So the rules may define predicates (``pred``, ``error``,
``shadow``) and nothing else: anything that would add facts or nodes, such as
an assertion (``flag().``), a node, class or enum declaration, or a
propagation rule, is listed in :attr:`CompiledProblem.refused`, rather than
left out without a word.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from .._options import Predicate
from .._render import _declare_predicate, render_metamodel
from .._results import ViolationKind
from .._schema import AttributeFeature, AttributeType, Schema, class_name
from .._syntax import quote_id
from ..errors import InvalidProblemError
from .classes import java
from .mapping import Catalogue
from .structure import CONTAINS, INVALID_CONTAINER, Check, Structure, structure

NODE = "builtin::node"
"""The plain symbol every object of a model is in: ``node`` and ``exists``."""
CONTAINER = "container"
CONTAINED = "contained"

CACHE_SIZE = 16
"""How many compiled problems :func:`compiled` keeps."""


@dataclass(frozen=True, slots=True)
class ErrorQuery:
    """A predicate whose tuples are violations, and how they are reported."""

    predicate: str
    kind: ViolationKind
    constraint: str
    order: int
    """Where a server would list them: the position of ``constraint`` among the relations."""


@dataclass(frozen=True, slots=True, eq=False)
class CompiledProblem:
    """A metamodel, its rules and its predicates, compiled into a plain store by
    :func:`compile_problem`."""

    schema: Schema
    catalogue: Catalogue
    source: str
    """The rendered metamodel, rules and predicate declarations, without the structural checks."""
    timings: dict[str, float]
    """Seconds spent parsing, compiling and building the store."""
    reserved_ids: int
    """Ids the problem's own nodes take (enum literals, ``Class::new``): objects start here."""
    constants: dict[str, int]
    """Enum literals by Refinery name, and their node ids."""
    refused: list[str]
    """Why the rules can't be matched on a plain model: what they state besides predicates."""
    parameters: dict[str, tuple[str, ...]]
    """The names of each compiled predicate's parameters."""
    attributes: dict[str, AttributeType]
    """The attributes the rules read, by relation name, and the type of their values."""
    queries: dict[str, Any]
    """Matchable predicates by name, as plain queries."""
    unsupported: dict[str, str]
    """Predicates the direct path can't match, and why."""
    errors: list[ErrorQuery]
    """The predicates whose tuples are violations, in the order a server lists them."""
    store: Any
    """The plain ``ModelStore`` the queries are matched in."""
    _symbols: dict[str, Any]
    """Plain symbols by relation name, as Refinery prints it and as pyfinery renders it: classes,
    enums, references, base predicates, and the attributes the rules read, whose symbols hold
    values."""

    def symbol(self, relation: str) -> Any:
        """The plain symbol of ``relation``, named as Refinery prints it or as pyfinery renders it
        (quoted where it must be)."""
        try:
            return self._symbols[relation]
        except KeyError:
            raise KeyError(f"{relation!r} has no plain symbol in this problem") from None

    @property
    def rule_errors(self) -> list[str]:
        """The rules' error predicates."""
        return [error.predicate for error in self.errors if error.kind == "rule"]


def compile_problem(
    schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate] = ()
) -> CompiledProblem:
    """Compile ``schema``, ``rules`` (Refinery source) and ``predicates`` (declared as base
    predicates, whose facts come from Python) into a plain store."""
    timings: dict[str, float] = {}

    catalogue = Catalogue(schema)
    structural = structure(schema, catalogue)
    declarations = [_declare_predicate(schema, p) for p in predicates]
    source = render_metamodel(schema, [*rules, *declarations])

    clock = time.perf_counter()
    injector = java().StandaloneRefinery.getInjector()
    problem = _load(injector, f"{source}\n{structural.text}\n")
    refused = _refused_statements(schema, problem)
    trace = _read(injector, problem)
    timings["parse"] = time.perf_counter() - clock

    clock = time.perf_counter()
    reserved_ids = int(trace.getNodeTrace().size())
    constants = {name: int(trace.getNodeId(name)) for name in catalogue.literal_names()}
    relations = _collect_relations(trace, _query_compiler(injector, trace))
    builtins = _plain_builtins(trace, relations.views, structural)
    compiler = _Compiler.start(catalogue, trace, relations, builtins)
    queries, unsupported = compiler.rewrite_all()
    errors = _error_queries(relations.rule_errors, structural.checks, relations.order)
    timings["compile"] = time.perf_counter() - clock

    clock = time.perf_counter()
    store = _store([*compiler.symbols.values(), *builtins.extra_symbols], queries.values())
    timings["store"] = time.perf_counter() - clock

    written = _written_relations(catalogue, predicates)
    return CompiledProblem(
        schema=schema,
        catalogue=catalogue,
        source=source,
        timings=timings,
        reserved_ids=reserved_ids,
        constants=constants,
        refused=refused,
        parameters=relations.parameters,
        attributes=compiler.attributes,
        queries=queries,
        unsupported=unsupported,
        errors=errors,
        store=store,
        _symbols=_with_rendered_names(compiler.symbols, trace, written),
    )


def _load(injector: Any, text: str) -> Any:
    """The Refinery problem of ``text``; raises :class:`InvalidProblemError` if it doesn't parse."""
    j = java()
    try:
        return injector.getInstance(j.ProblemLoader).loadString(text)
    except j.InvalidProblemException as error:
        raise InvalidProblemError(str(error.getMessage())) from None


def _read(injector: Any, problem: Any) -> Any:
    """The trace of ``problem``: its relations and nodes, by name, as partial-model symbols."""
    initializer = injector.getInstance(java().ModelInitializer)
    initializer.readProblem(problem)
    return initializer.getProblemTrace()


def _query_compiler(injector: Any, trace: Any) -> Any:
    compiler = injector.getInstance(java().QueryCompiler)
    compiler.setProblemTrace(trace)
    return compiler


def _refused_statements(schema: Schema, problem: Any) -> list[str]:
    """Why the statements of ``problem`` that a plain model can't hold are refused: all but
    predicate definitions and the declarations pyfinery renders for ``schema``."""
    j = java()
    rendered = {class_name(t) for t in (*schema.classes, *schema.unions, *schema.enums)}
    reasons = []
    for statement in problem.getStatements():
        if isinstance(statement, j.PredicateDefinition):
            continue
        if isinstance(statement, (j.ClassDeclaration, j.EnumDeclaration)):
            name = quote_id(str(statement.getName()))
            if name in rendered:
                continue
            what = f"{'a class' if isinstance(statement, j.ClassDeclaration) else 'an enum'} {name}"
        elif isinstance(statement, j.AbstractAssertion):
            relation = statement.getRelation()
            what = f"an assertion about {relation.getName() if relation is not None else '?'}"
        elif isinstance(statement, j.NodeDeclaration):
            what = f"nodes ({', '.join(str(node.getName()) for node in statement.getNodes())})"
        else:
            what = f"a {statement.eClass().getName()}"  # e.g. a RuleDefinition: a propagation rule
            if isinstance(statement, j.NamedElement) and statement.getName() is not None:
                what += f" {statement.getName()}"
        reasons.append(f"{what} in the rules: only predicates; give facts as Predicate objects")
    return reasons


@dataclass(frozen=True, slots=True)
class _Relations:
    """The relations of a problem's trace: compiled predicates, and the plain symbols of the
    others."""

    names: dict[Any, str] = field(default_factory=dict)
    """Every partial relation, by its name."""
    compiled: dict[Any, Any] = field(default_factory=dict)
    """Defined predicates: partial relation -> its query on partial-model symbols."""
    parameters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    symbols: dict[str, Any] = field(default_factory=dict)
    """Plain symbols by name: classes, enums, references, base predicates."""
    views: dict[Any, Any] = field(default_factory=dict)
    """Partial relation -> the view of its plain symbol."""
    rule_errors: list[str] = field(default_factory=list)
    order: dict[str, int] = field(default_factory=dict)
    """The position of each relation in the trace: where a server lists its violations."""


def _collect_relations(trace: Any, query_compiler: Any) -> _Relations:
    """Compile each predicate the rules define, and give every other relation a plain symbol."""
    j = java()
    builtins = {
        str(j.ReasoningAdapter.EXISTS_SYMBOL.name()),
        str(j.ReasoningAdapter.EQUALS_SYMBOL.name()),
        str(j.ContainmentHierarchyTranslator.CONTAINS_SYMBOL.name()),
        str(j.ContainmentHierarchyTranslator.INVALID_CONTAINER.name()),
    }
    found = _Relations()
    for position, entry in enumerate(trace.getRelationTrace().entrySet()):
        relation, partial = entry.getKey(), entry.getValue()
        name = str(partial.name())
        found.order.setdefault(name, position)
        if not isinstance(partial, j.PartialRelation) or name.endswith("::computed"):
            continue  # attributes (functions), and Refinery's copies of rules
        found.names[partial] = name
        if name in builtins or name.endswith("::invalidMultiplicity"):
            continue  # given their plain meaning by _plain_builtins
        if isinstance(relation, j.PredicateDefinition) and not j.ProblemUtil.isBasePredicate(
            relation
        ):
            found.compiled[partial] = query_compiler.toQuery(partial.name(), relation)
            found.parameters[name] = tuple(str(p.getName()) for p in relation.getParameters())
            if relation.getKind() == j.PredicateKind.ERROR:
                found.rule_errors.append(name)
        else:  # a class, enum, reference, or base predicate (node, container, contained too)
            symbol = j.Symbol.of(name, int(partial.arity()))
            found.symbols[name] = symbol
            found.views[partial] = j.KeyOnlyView(symbol)
    return found


@dataclass(frozen=True, slots=True)
class _Builtins:
    """Refinery's built-in relations with their plain meaning."""

    views: dict[Any, Any] = field(default_factory=dict)
    """Built-in partial relation -> the view of the plain symbol that stands for it."""
    plain_queries: dict[Any, str] = field(default_factory=dict)
    """Built-in partial relation -> the structural predicate that stands for it."""
    extra_symbols: list[Any] = field(default_factory=list)
    """Plain symbols only these views read."""


def _plain_builtins(trace: Any, views: dict[Any, Any], structural: Structure) -> _Builtins:
    j = java()
    hierarchy = j.ContainmentHierarchyTranslator
    found = _Builtins(
        views={j.ReasoningAdapter.EXISTS_SYMBOL: views[trace.getPartialRelation(NODE)]}
    )
    if structural.containment:
        found.plain_queries[hierarchy.CONTAINS_SYMBOL] = CONTAINS
        found.plain_queries[hierarchy.INVALID_CONTAINER] = INVALID_CONTAINER
    else:
        # Without containment references, nothing contains anything: relations that stay empty
        # (rather than false, which would leave a rule's variables unbound).
        for partial in (hierarchy.CONTAINS_SYMBOL, hierarchy.INVALID_CONTAINER):
            empty = j.Symbol.of(f"pyfinery_empty::{partial.name()}", int(partial.arity()))
            found.extra_symbols.append(empty)
            found.views[partial] = j.KeyOnlyView(empty)
    for check in structural.checks:
        if check.kind == "multiplicity":
            found.plain_queries[trace.getPartialRelation(check.constraint)] = check.predicate
    return found


@cache
def _term_rewriter_class() -> Any:
    """A Java ``TermRewriter`` that calls a Python function, defined once the JVM runs."""
    from .runtime import _jpype

    jpype = _jpype()

    @jpype.JImplements(java().TermRewriter)
    class TermRewriter:
        def __init__(self, rewrite: Callable[[Any], Any]) -> None:
            self._rewrite = rewrite

        @jpype.JOverride
        def rewriteTerm(self, term: Any) -> Any:
            return self._rewrite(term)

    return TermRewriter


@dataclass(slots=True)
class _Compiler:
    """What rewriting compiled predicates onto plain symbols reads and remembers; it lives only
    while :func:`compile_problem` runs."""

    catalogue: Catalogue
    names: dict[Any, str]
    by_name: dict[str, Any]
    compiled: dict[Any, Any]
    views: dict[Any, Any]
    plain_queries: dict[Any, str]
    symbols: dict[str, Any]
    """Plain symbols by name; rewriting adds those of the attributes the rules read."""
    attribute_names: dict[str, str]
    """Attributes as Refinery names them -> as pyfinery renders them."""
    attributes: dict[str, AttributeType] = field(default_factory=dict)
    rewritten: dict[Any, Any] = field(default_factory=dict)
    unsupported_dnfs: dict[Any, str] = field(default_factory=dict)
    quantified: int = 0
    """How many helper predicates rewriting has made: their names are numbered."""

    @classmethod
    def start(
        cls, catalogue: Catalogue, trace: Any, relations: _Relations, builtins: _Builtins
    ) -> _Compiler:
        return cls(
            catalogue=catalogue,
            names=relations.names,
            by_name={name: partial for partial, name in relations.names.items()},
            compiled=relations.compiled,
            views={**relations.views, **builtins.views},
            plain_queries=builtins.plain_queries,
            symbols=dict(relations.symbols),
            attribute_names={
                str(trace.getPartialFunction(name).name()): name
                for name, feature in catalogue.features.items()
                if isinstance(feature, AttributeFeature)
            },
        )

    def rewrite_all(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Every compiled predicate on the plain symbols: the matchable ones as plain queries by
        name, and why each of the others can't be matched."""
        queries: dict[str, Any] = {}
        unsupported: dict[str, str] = {}
        for partial, query in self.compiled.items():
            name = self.names[partial]
            rewritten = self.rewrite(query.getDnf())
            reason = self.unsupported_dnfs.get(rewritten)
            if reason is not None:
                unsupported[name] = reason
            else:
                queries[name] = rewritten.asRelation()
        return queries, unsupported

    def rewrite(self, dnf: Any) -> Any:
        j = java()
        known = self.rewritten.get(dnf)
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
                rewritten, why = self.rewrite_literal(literal)
                reason = reason or why
                literals.add(self.quantify_negation(rewritten, bound))
            builder.clause(literals)
        result = builder.build()
        self.rewritten[dnf] = result
        if reason is not None:
            self.unsupported_dnfs[result] = reason
        return result

    def quantify_negation(self, literal: Any, bound: set[Any]) -> Any:
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
        self.quantified += 1
        inner = (
            j.Dnf.builder(f"pyfinery_exists{self.quantified}")
            .parameters(j.ArrayList(kept))
            .clause(j.ArrayList([positive]))
            .build()
        )
        return j.CallLiteral(j.CallPolarity.NEGATIVE, inner, j.ArrayList(kept))

    def rewrite_literal(self, literal: Any) -> tuple[Any, str | None]:
        """The literal on plain symbols, and why it can't be matched, if it can't."""
        j = java()
        if isinstance(literal, j.PartialCheckLiteral):
            reasons: list[str] = []
            term = self.rewrite_term(literal.getTerm(), reasons)
            check = j.CheckLiteral(j.TruthValueTerms.must(term))  # holds when certainly true
            return check, reasons[0] if reasons else None
        if not isinstance(literal, j.CallLiteral):
            if isinstance(literal, (j.ConstantLiteral, j.BooleanLiteral)):
                return literal, None
            return literal, f"{literal.getClass().getSimpleName()} ({literal})"
        target = literal.getTarget()
        if isinstance(target, j.ModalConstraint):
            target = target.constraint()  # on a complete model, every modality is the fact
            literal = literal.withTarget(target)
        if isinstance(target, j.Dnf):
            inner = self.rewrite(target)
            return literal.withTarget(inner), self.unsupported_dnfs.get(inner)
        if not isinstance(target, j.PartialRelation):
            return literal, f"a call of {target}"
        positive = literal.getPolarity() != j.CallPolarity.NEGATIVE
        if target.equals(j.ReasoningAdapter.EQUALS_SYMBOL):
            arguments = literal.getArguments()
            return j.EquivalenceLiteral(positive, arguments.get(0), arguments.get(1)), None
        structural = self.plain_queries.get(target)
        if structural is not None:
            target = self.by_name[structural]
        query = self.compiled.get(target)
        if query is not None:
            inner = self.rewrite(query.getDnf())
            return literal.withTarget(inner), self.unsupported_dnfs.get(inner)
        view = self.views.get(target)
        if view is None:
            return literal, f"{target.name()}, which has no plain counterpart"
        return literal.withTarget(view), None

    def rewrite_term(self, term: Any, reasons: list[str]) -> Any:
        """``term``, of a check, reading plain symbols; what it can't read is added to
        ``reasons``. Only attribute reads and aggregations change: the comparisons and
        arithmetic around them are Refinery's."""
        j = java()
        one = j.IntIntervalTerms.constant(j.IntInterval.ONE)
        if isinstance(term, j.PartialFunctionCallTerm):
            function = term.getPartialFunction()
            view = j.FunctionView(self.attribute_symbol(function))
            # The model holds a value, or unknown, for each object of the attribute's class; on
            # an object of another class the attribute is a contradiction, error, as on a server.
            return view.leftJoin(function.abstractDomain().error(), term.getArguments())
        if isinstance(term, j.PartialCountTerm):  # the sum of a 1 for each match
            return self.aggregate(term, j.IntIntervalTerms.INT_INTERVAL_SUM, one, reasons)
        if isinstance(term, j.PartialAggregationTerm):
            aggregator = term.getAggregator()
            value = self.rewrite_term(term.getBody(), reasons)
            if isinstance(aggregator, j.WeightedAggregator):  # sum: each match counts once
                value = aggregator.withWeight(one, value)
            elif not isinstance(aggregator, j.UnweightedAggregator):  # not min or max
                reasons.append(f"an aggregator other than count, sum, min and max ({term})")
                return term
            return self.aggregate(term, aggregator.getInnerAggregator(), value, reasons)
        if isinstance(term, j.PartialTerm):
            reasons.append(f"{term.getClass().getSimpleName()} ({term})")
            return term
        rewriter = _term_rewriter_class()(lambda sub: self.rewrite_term(sub, reasons))
        return term.rewriteSubTerms(rewriter)

    def attribute_symbol(self, function: Any) -> Any:
        """The plain symbol of an attribute, which holds its values: made when a rule reads it."""
        name = self.attribute_names[str(function.name())]
        if name not in self.symbols:
            feature = self.catalogue.features[name]
            assert isinstance(feature, AttributeFeature)
            self.attributes[name] = feature.data_type
            domain = function.abstractDomain()
            self.symbols[name] = java().Symbol.of(
                name, int(function.arity()), domain.abstractType()
            )
        return self.symbols[name]

    def aggregate(self, call: Any, aggregator: Any, value: Any, reasons: list[str]) -> Any:
        """``value`` aggregated over the matches of the condition of ``call`` (a count or an
        aggregation): of a predicate that gives ``value`` for each match."""
        j = java()
        arguments = list(call.getArguments())
        condition = j.CallLiteral(j.CallPolarity.POSITIVE, call.getTarget(), j.ArrayList(arguments))
        matched, why = self.rewrite_literal(condition)
        if why is not None:
            reasons.append(why)
        self.quantified += 1
        result = j.Variable.of("value", value.getType())
        inner = (
            j.Dnf.builder(f"pyfinery_values{self.quantified}")
            .parameters(j.ArrayList([*dict.fromkeys(arguments), result]))
            .clause(j.ArrayList([matched, j.AssignLiteral(result, value)]))
            .build()
        )
        return inner.aggregateBy(result, aggregator, j.ArrayList([*arguments, result]))


def _error_queries(
    rule_errors: Sequence[str], checks: Sequence[Check], order: dict[str, int]
) -> list[ErrorQuery]:
    """The rules' error predicates and the structural checks, in the order a server lists them."""
    errors = [ErrorQuery(name, "rule", name, order[name]) for name in rule_errors]
    unique = {check.predicate: check for check in checks}
    errors += [_structural_error(check, order) for check in unique.values()]
    return sorted(errors, key=lambda error: error.order)


def _structural_error(check: Check, order: dict[str, int]) -> ErrorQuery:
    # A multiplicity is listed after its reference, a contradiction with its relation.
    name = check.constraint.removeprefix("builtin::")  # Refinery's trace has no prefix there
    position = order.get(name, len(order))
    return ErrorQuery(check.predicate, check.kind, check.constraint, position)


def _store(symbols: Iterable[Any], queries: Iterable[Any]) -> Any:
    """A plain store of ``symbols`` that matches ``queries`` incrementally."""
    j = java()
    return (
        j.ModelStore.builder()
        .symbols(j.ArrayList(list(symbols)))
        .with_(j.QueryInterpreterAdapter.builder().queries(j.ArrayList(list(queries))))
        .build()
    )


def _written_relations(catalogue: Catalogue, predicates: Sequence[Predicate]) -> list[str]:
    """The names a plain model writes facts under, as pyfinery renders them: classes, unions,
    enums, references, built-ins, and :class:`~pyfinery.Predicate` objects."""
    features = [
        name
        for name, feature in catalogue.features.items()
        if not isinstance(feature, AttributeFeature)  # those the rules read are named alike
    ]
    return [
        NODE,
        CONTAINER,
        CONTAINED,
        *catalogue.classes,
        *catalogue.enums,
        *features,
        *(quote_id(p.name) for p in predicates),
    ]


def _with_rendered_names(
    symbols: dict[str, Any], trace: Any, written: Iterable[str]
) -> dict[str, Any]:
    """``symbols``, also under each name in ``written`` that Refinery prints differently (e.g.
    unquoted)."""
    named = dict(symbols)
    for name in written:
        if name in named:
            continue
        try:
            partial = trace.getPartialRelation(name)
        except Exception:  # a Java exception: no such relation, and so no symbol to write to
            continue
        found = symbols.get(str(partial.name()))
        if found is not None:
            named[name] = found
    return named


_cache_lock = threading.Lock()
_problems: OrderedDict[tuple[object, ...], CompiledProblem] = OrderedDict()


def compiled(
    schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate] = ()
) -> CompiledProblem:
    """``schema``, ``rules`` and the declarations of ``predicates``, compiled once and reused
    while they are among the last :data:`CACHE_SIZE` used."""
    key = _cache_key(schema, rules, predicates)
    with _cache_lock:
        found = _problems.get(key)
        if found is not None and found.schema is schema:
            _problems.move_to_end(key)
            return found
    problem = compile_problem(schema, rules, predicates)
    with _cache_lock:
        _problems[key] = problem
        _problems.move_to_end(key)
        while len(_problems) > CACHE_SIZE:
            _problems.popitem(last=False)
    return problem


def clear_compiled() -> None:
    """Forget every compiled problem."""
    with _cache_lock:
        _problems.clear()


def _cache_key(
    schema: Schema, rules: Sequence[str], predicates: Sequence[Predicate]
) -> tuple[object, ...]:
    # A predicate's facts don't change what is compiled: its name and parameters do.
    return (id(schema), tuple(rules), tuple((p.name, tuple(p.parameters)) for p in predicates))
