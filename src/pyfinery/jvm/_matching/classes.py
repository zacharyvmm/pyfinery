"""The Java classes pyfinery uses, resolved once the JVM runs."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

_NAMES = {
    # Problems: parsing, tracing, compiling rules.
    "StandaloneRefinery": "tools.refinery.generator.standalone.StandaloneRefinery",
    "ProblemLoader": "tools.refinery.generator.ProblemLoader",
    "InvalidProblemException": "tools.refinery.generator.InvalidProblemException",
    "ModelInitializer": "tools.refinery.language.semantics.ModelInitializer",
    "QueryCompiler": "tools.refinery.language.semantics.internal.query.QueryCompiler",
    "ProblemUtil": "tools.refinery.language.utils.ProblemUtil",
    "ClassDeclaration": "tools.refinery.language.model.problem.ClassDeclaration",
    "EnumDeclaration": "tools.refinery.language.model.problem.EnumDeclaration",
    "ReferenceDeclaration": "tools.refinery.language.model.problem.ReferenceDeclaration",
    "PredicateDefinition": "tools.refinery.language.model.problem.PredicateDefinition",
    "PredicateKind": "tools.refinery.language.model.problem.PredicateKind",
    # Partial-model symbols the compiler targets, and the built-in ones.
    "PartialRelation": "tools.refinery.store.reasoning.representation.PartialRelation",
    "ModalConstraint": "tools.refinery.store.reasoning.literal.ModalConstraint",
    "ReasoningAdapter": "tools.refinery.store.reasoning.ReasoningAdapter",
    "ContainmentHierarchyTranslator": (
        "tools.refinery.store.reasoning.translator.containment.ContainmentHierarchyTranslator"
    ),
    "PartialCheckLiteral": "tools.refinery.store.reasoning.literal.PartialCheckLiteral",
    "PartialTerm": "tools.refinery.store.reasoning.literal.PartialTerm",
    "PartialFunctionCallTerm": "tools.refinery.store.reasoning.literal.PartialFunctionCallTerm",
    "PartialCountTerm": "tools.refinery.store.reasoning.literal.PartialCountTerm",
    "PartialAggregationTerm": "tools.refinery.store.reasoning.literal.PartialAggregationTerm",
    # Logic: rewriting the compiled rules.
    "Dnf": "tools.refinery.logic.dnf.Dnf",
    "CallLiteral": "tools.refinery.logic.literal.CallLiteral",
    "CallPolarity": "tools.refinery.logic.literal.CallPolarity",
    "EquivalenceLiteral": "tools.refinery.logic.literal.EquivalenceLiteral",
    "BooleanLiteral": "tools.refinery.logic.literal.BooleanLiteral",
    "ConstantLiteral": "tools.refinery.logic.literal.ConstantLiteral",
    "CheckLiteral": "tools.refinery.logic.literal.CheckLiteral",
    "AssignLiteral": "tools.refinery.logic.literal.AssignLiteral",
    "Variable": "tools.refinery.logic.term.Variable",
    "TermRewriter": "tools.refinery.logic.rewriter.TermRewriter",
    "WeightedAggregator": "tools.refinery.logic.term.PartialAggregator$MultiplicitySensitive",
    "UnweightedAggregator": "tools.refinery.logic.term.PartialAggregator$MultiplicityInsensitive",
    # Attribute values: exact values of Refinery's abstract domains.
    "TruthValue": "tools.refinery.logic.term.truthvalue.TruthValue",
    "TruthValueTerms": "tools.refinery.logic.term.truthvalue.TruthValueTerms",
    "IntInterval": "tools.refinery.logic.term.intinterval.IntInterval",
    "IntIntervalTerms": "tools.refinery.logic.term.intinterval.IntIntervalTerms",
    "RealInterval": "tools.refinery.logic.term.realinterval.RealInterval",
    "StringValue": "tools.refinery.logic.term.string.StringValue",
    "BigInteger": "java.math.BigInteger",
    # The plain store and its queries.
    "Symbol": "tools.refinery.store.representation.Symbol",
    "KeyOnlyView": "tools.refinery.store.query.view.KeyOnlyView",
    "FunctionView": "tools.refinery.store.query.view.FunctionView",
    "ModelStore": "tools.refinery.store.model.ModelStore",
    "QueryInterpreterAdapter": "tools.refinery.store.query.interpreter.QueryInterpreterAdapter",
    "ModelQueryAdapter": "tools.refinery.store.query.ModelQueryAdapter",
    "Tuple": "tools.refinery.store.tuple.Tuple",
    "ArrayList": "java.util.ArrayList",
}


@cache
def java() -> SimpleNamespace:
    """The classes, by short name; starts the JVM if it hasn't started."""
    from .._runtime import _jpype, start

    start()
    jpype = _jpype()
    return SimpleNamespace(**{name: jpype.JClass(path) for name, path in _NAMES.items()})
