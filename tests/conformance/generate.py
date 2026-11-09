# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Random problems for the conformance suite: small partial models and random predicates.

Each problem is drawn from a seed: objects of two classes, some of which only
may exist, references and attributes that hold, may hold or don't, and a
handful of predicates that call each other. The predicates use what Refinery's
language has for patterns: negation (of bound and of quantified variables),
transitive closure, ``==`` and ``!=``, ``exists``, constants, comparisons of
attribute intervals, ``must`` and ``may`` in shadow predicates, and up to
four parameters. A draw that Refinery rejects is drawn again, and so is a
clause that repeats, in one literal, a variable no other literal has (e.g.
``r+(z, z)``): the pinned image evaluates that as true, even without cycles.

The problems have no expectations: what Refinery answers, recorded, is what
another implementation must compute. Generate them again (needs Docker, or a
server at ``REFINERY_API_BASE``), then record them::

    uv run python -m tests.conformance.generate [COUNT]
    uv run python -m tests.conformance.recording
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass, field

from pyfinery import Refinery, RefineryError, RefineryServer
from pyfinery.models import JsonOutput

from .interpretation import Interpretation, errors
from .recording import FORMAT, IMAGE, ROOT

DIRECTORY = ROOT / "generated"

METAMODEL = """\
class A {
    A[] r
    B[] s
    int n
}

class B {
    A[] t
    boolean f
}

atom c.
A(c).
"""

# Binary relations, by the classes of their places.
REFERENCES = {"r": ("A", "A"), "s": ("A", "B"), "t": ("B", "A")}


@dataclass
class _Predicate:
    name: str
    parameters: list[str | None]
    shadow: bool
    clauses: list[list[str]] = field(default_factory=list)

    @property
    def arity(self) -> int:
        return len(self.parameters)


def problem(seed: int) -> str:
    """The problem drawn from ``seed``."""
    rng = random.Random(seed)
    lines = [METAMODEL]
    lines += _objects(rng)
    lines.append("")
    predicates: list[_Predicate] = []
    for index in range(rng.randint(3, 6)):
        predicates.append(_predicate(rng, f"p{index}", predicates))
    lines += [_render(p) for p in predicates]
    return "\n".join(lines) + "\n"


def _objects(rng: random.Random) -> list[str]:
    a_objects = ["c", *(f"a{i}" for i in range(rng.randint(1, 4)))]
    b_objects = [f"b{i}" for i in range(rng.randint(1, 3))]
    lines = [f"A({a})." for a in a_objects[1:]] + [f"B({b})." for b in b_objects]
    for name in a_objects[1:] + b_objects:
        if rng.random() < 0.25:
            lines.append(f"?exists({name}).")
    if rng.random() < 0.75:  # mostly a closed world, as pyfinery's queries have
        lines += ["!exists(A::new).", "!exists(B::new)."]
        lines += [f"default !{name}(*, *)." for name in REFERENCES]
    of_class = {"A": a_objects, "B": b_objects}
    for name, (source, target) in REFERENCES.items():
        for x in of_class[source]:
            for y in of_class[target]:
                roll = rng.random()
                if roll < 0.25:
                    lines.append(f"{name}({x}, {y}).")
                elif roll < 0.4:
                    lines.append(f"?{name}({x}, {y}).")
    for a in a_objects:
        roll = rng.random()
        if roll < 0.5:
            lines.append(f"n({a}): {rng.randint(0, 5)}.")
        elif roll < 0.75:
            low = rng.randint(0, 4)
            lines.append(f"n({a}): {low}..{low + rng.randint(1, 3)}.")
    for b in b_objects:
        roll = rng.random()
        if roll < 0.7:
            lines.append(f"f({b}): {'true' if roll < 0.35 else 'false'}.")
    return lines


def _predicate(rng: random.Random, name: str, earlier: list[_Predicate]) -> _Predicate:
    shadow = rng.random() < 0.25
    arity = rng.choices([0, 1, 2, 3, 4], weights=[1, 4, 5, 2, 1])[0]
    parameters: list[str | None] = [rng.choice(["A", "A", "B", None]) for _ in range(arity)]
    predicate = _Predicate(name, parameters, shadow)
    callable_ = [p for p in earlier if shadow or not p.shadow]
    for _ in range(rng.choice([1, 1, 2])):
        clause = _clause(rng, predicate, callable_)
        while _lone_diagonal(clause):
            clause = _clause(rng, predicate, callable_)
        predicate.clauses.append(clause)
    return predicate


_VARIABLE = re.compile(r"\b(x\d+|y|z)\b")


def _lone_diagonal(clause: list[str]) -> bool:
    """Whether a literal repeats a variable (not a parameter) no other literal has, e.g.
    ``r+(z, z)``: Refinery evaluates that as true, even without cycles."""
    for variable in ("y", "z"):
        users = [lit for lit in clause if variable in _VARIABLE.findall(lit)]
        if len(users) == 1 and _VARIABLE.findall(users[0]).count(variable) > 1:
            return True
    return False


def _clause(rng: random.Random, predicate: _Predicate, callable_: list[_Predicate]) -> list[str]:
    parameters = [f"x{i}" for i in range(predicate.arity)]
    variables = parameters + ["y", "z"]
    bound: set[str] = set()
    literals: list[str] = []

    def argument() -> str:
        roll = rng.random()
        if roll < 0.08:
            return "c"
        if roll < 0.18:
            return "_"
        return rng.choice(variables)

    def atom() -> tuple[str, list[str]]:
        choices: list[tuple[str, int]] = [
            *((r, 2) for r in REFERENCES),
            ("r+", 2),
            ("A", 1),
            ("B", 1),
            ("exists", 1),
            *((p.name, p.arity) for p in callable_),
        ]
        binary = [p.name for p in callable_ if p.arity == 2]
        choices += [(f"{name}+", 2) for name in binary]
        relation, arity = rng.choice(choices)
        return relation, [argument() for _ in range(arity)]

    for _ in range(rng.randint(1, 4)):
        roll = rng.random()
        if roll < 0.55:
            relation, args = atom()
            modal = rng.choice(["must ", "may "]) if predicate.shadow and rng.random() < 0.4 else ""
            literals.append(f"{modal}{relation}({', '.join(args)})")
            bound |= {a for a in args if a not in ("_", "c")}
        elif roll < 0.8:
            relation, args = atom()
            modal = rng.choice(["must ", "may "]) if predicate.shadow and rng.random() < 0.4 else ""
            literals.append(f"!{modal}{relation}({', '.join(args)})")
        elif roll < 0.9:
            left, right = rng.choice(variables), rng.choice([*variables, "c"])
            literals.append(f"{left} {rng.choice(['==', '!='])} {right}")
            if literals[-1].split()[1] == "==":
                bound |= {v for v in (left, right) if v != "c"}
        else:
            literals.append("<comparison>")
    # Parameters without a type must be bound; comparisons need bound variables.
    for index, parameter in enumerate(parameters):
        if predicate.parameters[index] is None and parameter not in bound:
            literals.append(f"{rng.choice(['A', 'exists'])}({parameter})")
            bound.add(parameter)
    typed = {p for i, p in enumerate(parameters) if predicate.parameters[i] is not None}
    candidates = sorted(bound | typed)
    for index, literal in enumerate(literals):
        if literal == "<comparison>":
            if not candidates:
                literals[index] = "A(c)"
                continue
            if rng.random() < 0.3:  # a boolean attribute: its argument is an input
                literals[index] = f"{rng.choice(['', '!'])}f({rng.choice(candidates)})"
                continue
            left = f"n({rng.choice(candidates)})"
            right = rng.choice([str(rng.randint(0, 5)), f"n({rng.choice(candidates)})"])
            comparison = f"{left} {rng.choice(['==', '!=', '<', '<=', '>', '>='])} {right}"
            literals[index] = f"!({comparison})" if rng.random() < 0.3 else comparison
    return literals


def _render(predicate: _Predicate) -> str:
    parameters = ", ".join(
        f"{kind} x{i}" if kind else f"x{i}" for i, kind in enumerate(predicate.parameters)
    )
    body = " ;\n    ".join(", ".join(clause) for clause in predicate.clauses)
    prefix = "shadow pred" if predicate.shadow else "pred"
    return f"{prefix} {predicate.name}({parameters}) <->\n    {body}.\n"


HEADER = """\
% SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
%
% SPDX-License-Identifier: EPL-2.0

% Generated by tests/conformance/generate.py from seed {seed}: what Refinery answers,
% recorded, is the expectation.

% TEST: seed {seed}

"""


# An inconsistent model (e.g. comparing an attribute of an object outside its
# class) is a case WITH ERRORS, which needs an expectation: this one always holds.
ERRORS_EXPECTATION = "\n% EXPECT:\nA(c).\n"


def answer(refinery: Refinery, source: str) -> JsonOutput | None:
    """Refinery's partial model for the problem, unless it has errors."""
    try:
        result = refinery.semantics(source, format=FORMAT)
    except RefineryError:
        return None
    return None if result.issues else result.json_


def generate(refinery: Refinery, count: int) -> None:
    DIRECTORY.mkdir(exist_ok=True)
    for old in DIRECTORY.glob("*.problem"):
        old.unlink()
    seed = 0
    for index in range(count):
        while (model := answer(refinery, source := problem(seed))) is None:
            seed += 1
        text = HEADER.format(seed=seed) + source
        if errors(Interpretation(model)):
            text = text.replace("% TEST: seed", "% TEST WITH ERRORS: seed") + ERRORS_EXPECTATION
        path = DIRECTORY / f"random{index:03}.problem"
        path.write_text(text, encoding="utf-8")
        seed += 1


def main(argv: list[str]) -> None:
    count = int(argv[0]) if argv else 100
    base_url = os.environ.get("REFINERY_API_BASE")
    if base_url:
        with Refinery(base_url, timeout=None) as refinery:
            generate(refinery, count)
        return
    with (
        RefineryServer(image=IMAGE, semantics_timeout=60_000) as server,
        server.client(timeout=None) as refinery,
    ):
        generate(refinery, count)


if __name__ == "__main__":
    main(sys.argv[1:])
