"""The Java classes pyfinery uses, resolved once the JVM runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cache
from typing import Any


@dataclass(frozen=True, slots=True)
class JavaClasses:
    """The Java classes pyfinery uses, by short name; each field's ``java`` metadata is the class's
    full name."""

    # Problems: parsing, tracing, compiling rules.
    StandaloneRefinery: Any = field(
        metadata={"java": "tools.refinery.generator.standalone.StandaloneRefinery"}
    )
    ProblemLoader: Any = field(metadata={"java": "tools.refinery.generator.ProblemLoader"})
    InvalidProblemException: Any = field(
        metadata={"java": "tools.refinery.generator.InvalidProblemException"}
    )
    ModelInitializer: Any = field(
        metadata={"java": "tools.refinery.language.semantics.ModelInitializer"}
    )
    QueryCompiler: Any = field(
        metadata={"java": "tools.refinery.language.semantics.internal.query.QueryCompiler"}
    )
    ProblemUtil: Any = field(metadata={"java": "tools.refinery.language.utils.ProblemUtil"})
    ClassDeclaration: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.ClassDeclaration"}
    )
    EnumDeclaration: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.EnumDeclaration"}
    )
    ReferenceDeclaration: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.ReferenceDeclaration"}
    )
    PredicateDefinition: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.PredicateDefinition"}
    )
    PredicateKind: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.PredicateKind"}
    )
    AbstractAssertion: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.AbstractAssertion"}
    )
    NodeDeclaration: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.NodeDeclaration"}
    )
    NamedElement: Any = field(
        metadata={"java": "tools.refinery.language.model.problem.NamedElement"}
    )
    # Partial-model symbols the compiler targets, and the built-in ones.
    PartialRelation: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.representation.PartialRelation"}
    )
    ModalConstraint: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.ModalConstraint"}
    )
    ReasoningAdapter: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.ReasoningAdapter"}
    )
    ContainmentHierarchyTranslator: Any = field(
        metadata={
            "java": (
                "tools.refinery.store.reasoning.translator.containment."
                "ContainmentHierarchyTranslator"
            )
        }
    )
    PartialCheckLiteral: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.PartialCheckLiteral"}
    )
    PartialTerm: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.PartialTerm"}
    )
    PartialFunctionCallTerm: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.PartialFunctionCallTerm"}
    )
    PartialCountTerm: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.PartialCountTerm"}
    )
    PartialAggregationTerm: Any = field(
        metadata={"java": "tools.refinery.store.reasoning.literal.PartialAggregationTerm"}
    )
    # Logic: rewriting the compiled rules.
    Dnf: Any = field(metadata={"java": "tools.refinery.logic.dnf.Dnf"})
    CallLiteral: Any = field(metadata={"java": "tools.refinery.logic.literal.CallLiteral"})
    CallPolarity: Any = field(metadata={"java": "tools.refinery.logic.literal.CallPolarity"})
    EquivalenceLiteral: Any = field(
        metadata={"java": "tools.refinery.logic.literal.EquivalenceLiteral"}
    )
    BooleanLiteral: Any = field(metadata={"java": "tools.refinery.logic.literal.BooleanLiteral"})
    ConstantLiteral: Any = field(metadata={"java": "tools.refinery.logic.literal.ConstantLiteral"})
    CheckLiteral: Any = field(metadata={"java": "tools.refinery.logic.literal.CheckLiteral"})
    AssignLiteral: Any = field(metadata={"java": "tools.refinery.logic.literal.AssignLiteral"})
    Variable: Any = field(metadata={"java": "tools.refinery.logic.term.Variable"})
    TermRewriter: Any = field(metadata={"java": "tools.refinery.logic.rewriter.TermRewriter"})
    WeightedAggregator: Any = field(
        metadata={"java": "tools.refinery.logic.term.PartialAggregator$MultiplicitySensitive"}
    )
    UnweightedAggregator: Any = field(
        metadata={"java": "tools.refinery.logic.term.PartialAggregator$MultiplicityInsensitive"}
    )
    # Attribute values: exact values of Refinery's abstract domains.
    TruthValue: Any = field(metadata={"java": "tools.refinery.logic.term.truthvalue.TruthValue"})
    TruthValueTerms: Any = field(
        metadata={"java": "tools.refinery.logic.term.truthvalue.TruthValueTerms"}
    )
    IntInterval: Any = field(metadata={"java": "tools.refinery.logic.term.intinterval.IntInterval"})
    IntIntervalTerms: Any = field(
        metadata={"java": "tools.refinery.logic.term.intinterval.IntIntervalTerms"}
    )
    RealInterval: Any = field(
        metadata={"java": "tools.refinery.logic.term.realinterval.RealInterval"}
    )
    StringValue: Any = field(metadata={"java": "tools.refinery.logic.term.string.StringValue"})
    BigInteger: Any = field(metadata={"java": "java.math.BigInteger"})
    # The plain store and its queries.
    Symbol: Any = field(metadata={"java": "tools.refinery.store.representation.Symbol"})
    KeyOnlyView: Any = field(metadata={"java": "tools.refinery.store.query.view.KeyOnlyView"})
    FunctionView: Any = field(metadata={"java": "tools.refinery.store.query.view.FunctionView"})
    ModelStore: Any = field(metadata={"java": "tools.refinery.store.model.ModelStore"})
    QueryInterpreterAdapter: Any = field(
        metadata={"java": "tools.refinery.store.query.interpreter.QueryInterpreterAdapter"}
    )
    ModelQueryAdapter: Any = field(
        metadata={"java": "tools.refinery.store.query.ModelQueryAdapter"}
    )
    Tuple: Any = field(metadata={"java": "tools.refinery.store.tuple.Tuple"})
    ArrayList: Any = field(metadata={"java": "java.util.ArrayList"})


@cache
def java() -> JavaClasses:
    """The classes, by short name; starts the JVM if it hasn't started."""
    from .runtime import _jpype, start

    start()
    jpype = _jpype()
    return JavaClasses(
        **{entry.name: jpype.JClass(entry.metadata["java"]) for entry in fields(JavaClasses)}
    )
