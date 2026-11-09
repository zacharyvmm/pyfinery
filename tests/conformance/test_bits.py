# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The bits engine computes every predicate as Refinery did, for every conformance case.

The engine reads the base relations of Refinery's recorded partial model (after
Refinery's propagation), evaluates the predicates of the case's source, and
must give each tuple of each predicate the value Refinery gave it. The
case's expectations must then hold of the model with the engine's values.
"""

from __future__ import annotations

import pytest

from pyfinery.bits import PartialModel, UnsupportedRuleError, evaluate
from pyfinery.bits.syntax import Atom, Call, Name, parse
from pyfinery.models import JsonOutput

from .problems import Case
from .test_conformance import CASES, _id, _outcome, failures

# Files whose predicates Refinery refines beyond their bodies, which the engine
# refuses to evaluate rather than evaluate differently.
UNSUPPORTED = {
    "upstream/concretization/predicateSubsetting.problem": "subsets on a predicate with a body",
    "upstream/semantics/predicateSubsetting.problem": "subsets on a predicate with a body",
    "upstream/semantics/crossreference/basePredicateSuperset.problem": "subsets of a predicate",
    "upstream/semantics/crossreference/typeConstraint.problem": "type of a reference|assertions",
}


# Refinery (pinned image) evaluates a call that repeats a variable used nowhere
# else in its clause as true: ``pred cyc() <-> r+(z, z).`` holds with no cycles,
# and ``r(z, z)`` with only unknown loops. Predicates that depend on such a call
# aren't compared; the engine evaluates them as the semantics says.
LONE_DIAGONAL = "Refinery evaluates a call repeating a variable used nowhere else as true"


def _lone_diagonal(predicate, atoms: set[str], attributes: set[str]) -> bool:
    """Whether a clause repeats, within one atom, a variable no other literal has. A
    negated atom (not an attribute) has its own variables, as in Refinery's language."""
    bound = {p.name for p in predicate.parameters} | atoms | {"_"}
    for clause in predicate.body or ():

        def scoped(literal) -> bool:
            return (
                isinstance(literal, Atom) and literal.negated and literal.relation not in attributes
            )

        clause_level = {n for lit in clause if not scoped(lit) for n in _names(lit)}
        for literal in clause:
            if not isinstance(literal, Atom):
                continue
            names = _names(literal)
            for name in set(names) - bound:
                if names.count(name) < 2:
                    continue
                if scoped(literal) and name not in clause_level:
                    return True
                if sum(name in _names(other) for other in clause) == 1:
                    return True
    return False


def _names(literal) -> list[str]:
    if isinstance(literal, Atom):
        return [a.text for a in literal.arguments]
    names = []
    for side in (literal.left, literal.right):
        if isinstance(side, Name):
            names.append(side.text)
        elif isinstance(side, Call):
            names += [a.text for a in side.arguments]
    return names


def _affected(predicates, atoms: set[str], attributes: set[str]) -> set[str]:
    """The predicates with a lone diagonal, and those that call them."""
    affected = {
        p.name for p in predicates if p.body is not None and _lone_diagonal(p, atoms, attributes)
    }
    changed = True
    while changed:
        changed = False
        for predicate in predicates:
            if predicate.name in affected or predicate.body is None:
                continue
            called = {
                lit.relation for clause in predicate.body for lit in clause if isinstance(lit, Atom)
            }
            if called & affected:
                affected.add(predicate.name)
                changed = True
    return affected


def _defined(case: Case) -> list[str]:
    return [p.name for p in parse(case.source).predicates if p.body is not None]


def _with_engine_values(model: JsonOutput, values) -> JsonOutput:
    """Refinery's model, with the engine's values for the predicates it evaluated."""
    interpretation = dict(model.partial_interpretation)
    for name, result in values.items():
        interpretation[name] = [[*key, value] for key, value in result.items()]
    return model.model_copy(update={"partial_interpretation": interpretation})


def _refinery_values(model: JsonOutput, name: str) -> dict[tuple[int, ...], str]:
    relation = model.relation(name)
    return {
        tuple(raw[: relation.arity]): raw[relation.arity]  # type: ignore[misc]
        for raw in model.partial_interpretation.get(name, [])
        if raw[relation.arity] != "false"
    }


@pytest.mark.parametrize(("path", "case"), CASES, ids=[f"{_id(p)}::{c.label}" for p, c in CASES])
def test_engine_computes_what_refinery_did(path, case):
    outcome = _outcome(path, case)
    assert outcome.partial is not None
    defined = _defined(case)
    if not defined:
        pytest.skip("no predicates to evaluate")
    refinery = outcome.partial
    model = PartialModel.from_json(refinery, exclude=defined)
    reason = UNSUPPORTED.get(_id(path))
    if reason is not None:
        with pytest.raises(UnsupportedRuleError, match=reason):
            evaluate(case.source, model)
        return
    values = evaluate(case.source, model)
    names = [n.name for n in refinery.nodes]
    atoms = {n.simple_name for n in refinery.nodes if n.kind == "atom"}
    atoms |= {n.name for n in refinery.nodes if n.kind == "atom"}
    attributes = {
        name
        for r in refinery.relations
        if r.detail.type == "attribute"
        for name in (r.name, r.simple_name)
    }
    affected = _affected(parse(case.source).predicates, atoms, attributes)
    differing = set()
    for name, result in values.items():
        expected = _refinery_values(refinery, name)
        actual = dict(result.items())
        differences = {
            tuple(names[i] for i in key): (actual.get(key, "false"), expected.get(key, "false"))
            for key in expected.keys() | actual.keys()
            if actual.get(key, "false") != expected.get(key, "false")
        }
        if name in affected and differences:
            differing.add(name)
            continue
        assert differences == {}, f"{name}: (engine, Refinery) differ"
    if differing:
        return  # the expectations may be about these predicates
    partial_expectations = [e for e in case.expectations if e.concreteness == "partial"]
    engine_case = Case(
        case.path, case.index, case.name, case.kind, case.source, tuple(partial_expectations)
    )
    engine_outcome = outcome.__class__(
        _with_engine_values(refinery, values), None, outcome.issues, None
    )
    assert failures(engine_case, engine_outcome) == []
