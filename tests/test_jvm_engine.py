"""The JVM engine (:class:`pyfinery.jvm.JvmEngine`): validation and queries on Refinery's own
matcher, each test in a subprocess (a process has one JVM, started once).

Opt in with ``PYFINERY_JVM_TESTS=1``, with a JDK 25 in ``PYFINERY_JAVA_HOME`` or ``JAVA_HOME``.
Each script makes several checks, to start the JVM fewer times.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.jvm,
    pytest.mark.skipif(
        os.environ.get("PYFINERY_JVM_TESTS") != "1", reason="set PYFINERY_JVM_TESTS=1"
    ),
]

ROOT = Path(__file__).resolve().parent.parent

PRELUDE = """
import json
from tests.graph_models import Dir, File, Filesystem, Link, metamodel
from pyfinery.jvm import JvmEngine, UnsupportedRuleError

mm = metamodel()
engine = JvmEngine()


def violations(objects, rules=(), **kwargs):
    return sorted(str(v) for v in mm.validate(objects, engine, rules, **kwargs).violations)


def raises(error, call):
    try:
        call()
    except error as found:
        return str(found)
    return None
"""


def run(script: str) -> dict:
    """Run ``script`` after :data:`PRELUDE` in a fresh Python process, from the repository root
    (to import ``tests``); it prints one JSON object as its last line."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PRELUDE) + textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_validates_as_a_server_does():
    # Each case's violations were checked against a Refinery server.
    out = run(
        """
        out = {}
        out["one"] = violations([Filesystem(root=Dir(name="home"))])
        out["two"] = violations([Filesystem(root=Dir(name="h2", contents=[
            Dir(name="a"), Dir(name="b")]))])
        out["orphan"] = violations([Filesystem(root=Dir(name="r")), Dir(name="orphan")])

        x = Dir(name="x")
        y = Dir(name="y", contents=[x])
        x.contents.append(y)
        out["cycle"] = violations([Filesystem(root=Dir(name="r2")), x])

        s = Dir(name="s")
        out["shared"] = violations([Filesystem(root=Dir(name="r3", contents=[
            Dir(name="t1", contents=[s]), Dir(name="t2", contents=[s])]))])

        m = Dir(name="m", contents=[Dir(name=f"c{i}") for i in range(4)])
        out["many"] = violations([Filesystem(root=m)])

        out["link"] = violations([Filesystem(root=Dir(name="r5", contents=[
            Link(target=Dir(name="tt"))]))])

        # Validating twice gives the same: nothing is left over from the first model.
        out["again"] = violations([Filesystem(root=Dir(name="home"))])
        print(json.dumps(out))
        """
    )
    assert out["one"] == ["rule: emptyDir(home)"]
    assert out["two"] == ["rule: emptyDir(a)", "rule: emptyDir(b)"]
    assert out["orphan"] == [
        "containment: builtin::invalidContainer(orphan)",
        "rule: emptyDir(orphan)",
        "rule: emptyDir(r)",
    ]
    # A server also reports emptyDir(x) and emptyDir(y), as contradictions make every fact
    # about x and y an error; the JVM engine matches the rules on the facts as given.
    assert out["cycle"] == [
        "containment: builtin::invalidContainer(x)",
        "containment: builtin::invalidContainer(y)",
        "contradiction: Dir::contents(x, y)",
        "contradiction: Dir::contents(y, x)",
        "contradiction: FSObject::parent(x, y)",
        "contradiction: FSObject::parent(y, x)",
        "rule: emptyDir(r2)",
    ]
    assert out["shared"] == ["containment: builtin::invalidContainer(s)", "rule: emptyDir(s)"]
    assert out["many"] == [
        "multiplicity: Dir::contents::invalidMultiplicity(m)",
        *(f"rule: emptyDir(c{i})" for i in range(4)),
    ]
    assert out["link"] == ["containment: builtin::invalidContainer(tt)", "rule: emptyDir(tt)"]
    assert out["again"] == out["one"]


def test_builtins_predicates_and_names():
    out = run(
        """
        from pyfinery import InvalidProblemError, Predicate
        from pyfinery.jvm._cache import compiled

        leaf = Dir(name="leaf")
        sub = Dir(name="sub", contents=[leaf])
        w = Dir(name="w", contents=[File(size=1), sub])
        objects = [Filesystem(root=w)]
        out = {}
        out["builtins"] = violations(objects, [
            "error pred everything(x) <-> node(x), exists(x).",
            "error pred inside(x) <-> contained(x), !container(x).",
            "error pred holds(x, y) <-> contains(x, y), Dir(x), Dir(y).",
            "error pred two(Dir d) <-> contents(d, a), contents(d, b), a != b, !equals(a, b).",
            "error pred same(x) <-> File(x), equals(x, x).",
            "error pred bad(x) <-> invalidContainer(x).",
            "error pred over(Dir d) <-> Dir::contents::invalidMultiplicity(d).",
        ])

        marked = Predicate("marked", (Dir,), tuples=[(w,)])
        is_leaf = Predicate("isLeaf", (Dir,), evaluate=lambda d: d.name == "leaf")
        maybe = Predicate("maybe", (Dir,), tuples=[((sub,), "unknown")])
        out["predicates"] = violations(objects, [
            "error pred m(Dir d) <-> marked(d).",
            "error pred l(Dir d) <-> isLeaf(d).",
            "error pred u(Dir d) <-> maybe(d).",
        ], predicates=[marked, is_leaf, maybe])
        out["quoted"] = violations(objects, ["error pred 'odd name'(Dir d) <-> contents(d, _)."])
        out["invalid"] = raises(InvalidProblemError, lambda: violations(
            objects, ["error pred broken(Dir d) <-> nothing(d)."]))

        rules = [*mm.rules, "pred extra(Dir d) <-> Dir(d)."]
        first = compiled(mm.schema, rules)
        out["cached"] = compiled(mm.schema, list(rules)) is first
        out["recompiled"] = compiled(mm.schema, [*rules, "pred more(Dir d) <-> Dir(d)."]) is first
        print(json.dumps(out))
        """
    )
    # Enum literals are nodes too, as on a server.
    assert out["builtins"] == [
        "rule: emptyDir(leaf)",
        "rule: everything(Permission::read)",
        "rule: everything(Permission::write)",
        "rule: everything(file_1)",
        "rule: everything(filesystem_1)",
        "rule: everything(leaf)",
        "rule: everything(sub)",
        "rule: everything(w)",
        "rule: holds(sub, leaf)",
        "rule: holds(w, sub)",
        "rule: inside(file_1)",
        "rule: same(file_1)",
        "rule: two(w)",
    ]
    # "unknown" Predicate facts don't hold when validating.
    assert out["predicates"] == ["rule: emptyDir(leaf)", "rule: l(leaf)", "rule: m(w)"]
    assert out["quoted"] == [
        "rule: 'odd name'(sub)",
        "rule: 'odd name'(w)",
        "rule: emptyDir(leaf)",
    ]
    assert out["invalid"] is not None and "nothing" in out["invalid"]
    assert out["cached"] and not out["recompiled"]


def test_unsupported_rules():
    out = run(
        """
        from typing import Annotated
        from pydantic import Field
        from pyfinery import Metamodel, Ref, RefineryNode

        objects = [Filesystem(root=Dir(name="home"))]
        out = {}
        out["attribute"] = raises(UnsupportedRuleError, lambda: violations(
            objects, ["error pred big(File f) <-> File::size(f) > 3."]))
        out["count"] = raises(UnsupportedRuleError, lambda: violations(
            objects, ["error pred full(Dir d) <-> count { Dir::contents(d, _) } > 1."]))
        # Through a helper: the error predicate that calls it is unsupported.
        out["helper"] = raises(UnsupportedRuleError, lambda: violations(objects, [
            "pred large(File f) <-> File::size(f) > 3.",
            "error pred hasLarge(Dir d) <-> contents(d, f), large(f).",
        ]))
        # An unsupported helper no error predicate uses is left alone.
        out["unused"] = violations(objects, ["pred large(File f) <-> File::size(f) > 3."])
        out["is_value_error"] = issubclass(UnsupportedRuleError, ValueError)

        class Wide(RefineryNode):
            __node_id__ = "name"
            name: str
            items: Annotated[list["Wide"], Ref(multiplicity="0..10")] = Field(
                default_factory=list)

        class Two(RefineryNode):
            __node_id__ = "name"
            name: str
            items: Annotated[list["Two"], Ref(multiplicity="2..4")] = Field(
                default_factory=list)

        out["wide"] = raises(UnsupportedRuleError, lambda: Metamodel(Wide).validate(
            [Wide(name="a")], engine))
        a = Two(name="a")
        b = Two(name="b", items=[a])
        c = Two(name="c", items=[a, b])
        out["two"] = sorted(str(v) for v in Metamodel(Two).validate([a, b, c], engine).violations)
        print(json.dumps(out))
        """
    )
    assert "big: uses attribute values or count" in out["attribute"]
    assert "full: uses" in out["count"]
    assert "hasLarge: uses" in out["helper"]
    assert out["unused"] == ["rule: emptyDir(home)"]
    assert out["is_value_error"]
    assert "the multiplicity of Wide::items [0..10], which needs count" in out["wide"]
    # At least two: a has none and b one; c has two.
    assert out["two"] == [
        "multiplicity: Two::items::invalidMultiplicity(a)",
        "multiplicity: Two::items::invalidMultiplicity(b)",
    ]


def test_queries():
    out = run(
        """
        from pyfinery import ClassOptions, Metamodel, Predicate
        from tests.graph_models import RULES

        leaf = Dir(name="leaf")
        w = Dir(name="w", contents=[File(size=1), Dir(name="sub", contents=[leaf])])
        objects = [Filesystem(root=w)]


        def matches(result):
            return sorted(
                [m.predicate, list(m.node_names), m.logic_value, list(m.parameters)]
                for m in result.matches
            )


        out = {}
        result = mm.query(objects, engine, [
            "pred holds(Dir d, FSObject o) <-> contents(d, o).",
            "pred hidden(Dir d) <-> Dir(d).",
        ], select=["holds"])
        out["selected"] = matches(result)
        out["errors"] = sorted(str(v) for v in result.errors)
        out["all"] = sorted({m.predicate for m in mm.query(objects, engine, [
            "pred one(Dir d) <-> Dir(d).", "pred other(File f) <-> File(f)."]).matches})
        out["quoted"] = matches(mm.query(
            objects, engine, ["pred 'odd name'(Dir d) <-> contents(d, _)."], select=["odd name"]))
        out["shadow"] = matches(mm.query(objects, engine, [
            "shadow pred full(Dir d) <-> must contents(d, f), may File(f), candidate Dir(d)."]))
        out["predicate"] = matches(mm.query(
            objects, engine, ["pred tagged(Dir d) <-> marked(d), contents(d, _)."],
            predicates=[Predicate("marked", (Dir,), evaluate=lambda d: d.name != "leaf")]))
        out["missing"] = raises(ValueError, lambda: mm.query(
            objects, engine, ["pred one(Dir d) <-> Dir(d)."], select=["nothere"]))

        out["unknown_fact"] = raises(UnsupportedRuleError, lambda: mm.query(
            objects, engine, ["pred tagged(Dir d) <-> marked(d)."],
            predicates=[Predicate("marked", (Dir,), tuples=[((w,), "unknown")])]))
        maybe = Metamodel(Filesystem, rules=RULES,
                          options={File: ClassOptions(exists=lambda f: "unknown")})
        out["may_exist"] = raises(UnsupportedRuleError, lambda: maybe.query(
            objects, engine, ["pred one(Dir d) <-> Dir(d)."]))
        out["unsupported"] = raises(UnsupportedRuleError, lambda: mm.query(
            objects, engine, ["pred big(File f) <-> File::size(f) > 3."], select=["big"]))
        print(json.dumps(out))
        """
    )
    assert out["selected"] == [
        ["holds", ["sub", "leaf"], "true", ["d", "o"]],
        ["holds", ["w", "file_1"], "true", ["d", "o"]],
        ["holds", ["w", "sub"], "true", ["d", "o"]],
    ]
    assert out["errors"] == ["rule: emptyDir(leaf)"]
    assert out["all"] == ["one", "other"]
    assert out["quoted"] == [
        ["odd name", ["sub"], "true", ["d"]],
        ["odd name", ["w"], "true", ["d"]],
    ]
    assert out["shadow"] == [["full", ["w"], "true", ["d"]]]
    assert out["predicate"] == [
        ["tagged", ["sub"], "true", ["d"]],
        ["tagged", ["w"], "true", ["d"]],
    ]
    assert "nothere" in out["missing"]
    assert "marked(w) only may hold" in out["unknown_fact"]
    assert "file_1 only may exist" in out["may_exist"]
    assert "big: uses" in out["unsupported"]


def test_negation_of_a_pattern_with_only_quantified_variables():
    out = run(
        """
        objects = [Filesystem(root=Dir(name="w", contents=[File()]))]
        out = {}
        out["one"] = violations(objects, ["error pred noLink(Dir d) <-> Dir(d), !Link(_)."])
        out["nullary"] = violations(objects, ["error pred noLinks() <-> !Link(_)."])
        out["two"] = raises(Exception, lambda: violations(
            objects, ["error pred neither() <-> !Link(_), !Link::target(_, _)."]))
        print(json.dumps(out))
        """
    )
    assert out["one"] == ["rule: noLink(w)"]
    assert out["nullary"] == ["rule: noLinks()"]
    assert out["two"] is None


def test_containment_builtins_without_containment():
    out = run(
        """
        from typing import Annotated
        from pydantic import Field
        from pyfinery import Metamodel, Ref, RefineryNode

        class N(RefineryNode):
            __node_id__ = "name"
            name: str
            next: Annotated[list["N"], Ref()] = Field(default_factory=list)

        b = N(name="b", next=[N(name="a")])
        found = Metamodel(N).validate([b], engine, [
            "error pred holds(x, y) <-> contains(x, y).",
            "error pred bad(x) <-> invalidContainer(x).",
        ])
        print(json.dumps(sorted(str(v) for v in found.violations)))
        """
    )
    assert out == []
