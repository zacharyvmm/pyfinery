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
        from pyfinery.jvm._matching.cache import compiled

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
        meet = "meet { Dir::contents(d, f) -> File::size(f) } > 3"
        out["aggregator"] = raises(UnsupportedRuleError, lambda: violations(
            objects, [f"error pred odd(Dir d) <-> {meet}."]))
        out["reify"] = raises(UnsupportedRuleError, lambda: violations(
            objects, ["error pred odd(File f) <-> reify { File(f) } == true."]))
        # Through a helper: the error predicate that calls it is unsupported.
        out["helper"] = raises(UnsupportedRuleError, lambda: violations(objects, [
            f"pred odd(Dir d) <-> {meet}.",
            "error pred hasOdd(Dir d) <-> contents(d, e), odd(e).",
        ]))
        # An unsupported helper no error predicate uses is left alone.
        out["unused"] = violations(objects, [f"pred odd(Dir d) <-> {meet}."])
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

        wide = Wide(name="w", items=[Wide(name=f"w{i}") for i in range(11)])
        out["wide"] = sorted(str(v) for v in Metamodel(Wide).validate([wide], engine).violations)
        a = Two(name="a")
        b = Two(name="b", items=[a])
        c = Two(name="c", items=[a, b])
        out["two"] = sorted(str(v) for v in Metamodel(Two).validate([a, b, c], engine).violations)
        print(json.dumps(out))
        """
    )
    assert "odd: uses an aggregator other than count, sum, min and max" in out["aggregator"]
    assert "odd: uses ReifyTerm" in out["reify"]
    assert "hasOdd: uses" in out["helper"]
    assert out["unused"] == ["rule: emptyDir(home)"]
    assert out["is_value_error"]
    # A bound of any size is counted: eleven items, at most ten.
    assert out["wide"] == ["multiplicity: Wide::items::invalidMultiplicity(w)"]
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
            objects, engine, ["pred odd(File f) <-> reify { File(f) } == true."],
            select=["odd"]))
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
    assert "odd: uses" in out["unsupported"]


ATTRIBUTE_PRELUDE = """
def matched(objects, rules):
    names = [rule.split()[1].split("(")[0] for rule in rules]
    result = mm.query(objects, engine, rules, select=names)
    found = {name: [] for name in names}
    for match in result.matches:
        found[match.predicate].append(" ".join(match.node_names))
    return {name: sorted(rows) for name, rows in found.items()}
"""


def test_attribute_values():
    out = run(
        ATTRIBUTE_PRELUDE
        + """
a = File(size=7, ratio=0.5, hidden=True, label="x")
b = File(size=8, ratio=7.5, label="a b")
c = File(size=2**40, ratio=-1e-3)  # no label
objects = [Filesystem(root=Dir(name="r", contents=[a, b, c]))]
out = matched(objects, [
    "pred big(File f) <-> File::size(f) > 7.",
    "pred atLeast(File f) <-> File::size(f) >= 8.",
    "pred small(File f) <-> File::size(f) < 8.",
    "pred atMost(File f) <-> File::size(f) <= 7.",
    "pred seven(File f) <-> File::size(f) == 7.",
    "pred notSeven(File f) <-> File::size(f) != 7.",
    "pred huge(File f) <-> File::size(f) == 1099511627776.",
    "pred before(File a, File b) <-> File::size(a) < File::size(b).",
    "pred next(File a, File b) <-> File::size(a) + 1 == File::size(b).",
    "pred half(File f) <-> File::ratio(f) == 0.5.",
    "pred low(File f) <-> File::ratio(f) < 0.0.",
    "pred double(File f) <-> File::ratio(f) * 2.0 >= 15.0.",
    "pred isHidden(File f) <-> File::hidden(f).",
    "pred hiddenTrue(File f) <-> File::hidden(f) == true.",
    "pred shown(File f) <-> !File::hidden(f).",
    'pred x(File f) <-> File::label(f) == "x".',
    'pred spaced(File f) <-> File::label(f) == "a b".',
    'pred notX(File f) <-> !(File::label(f) == "x").',
    'pred otherThanX(File f) <-> File::label(f) != "x".',
    "pred sameLabel(File a, File b) <-> File::label(a) == File::label(b).",
    "pred large(File f) <-> File::size(f) > 7.",
    "pred hasLarge(Dir d) <-> contents(d, f), large(f).",
])
out["violations"] = violations(objects, ["error pred tooBig(File f) <-> File::size(f) > 7."])
out["not_finite"] = raises(ValueError, lambda: violations(
    [Filesystem(root=Dir(name="r", contents=[File(ratio=float("inf"))]))],
    ["error pred low(File f) <-> File::ratio(f) < 0.0."]))
print(json.dumps(out))
"""
    )
    a, b, c = "file_1", "file_2", "file_3"
    assert out["big"] == [b, c] and out["atLeast"] == [b, c]
    assert out["small"] == [a] and out["atMost"] == [a]
    assert out["seven"] == [a] and out["notSeven"] == [b, c]
    assert out["huge"] == [c]  # beyond 32 bits
    assert out["before"] == [f"{a} {b}", f"{a} {c}", f"{b} {c}"]
    assert out["next"] == [f"{a} {b}"]
    assert out["half"] == [a] and out["low"] == [c] and out["double"] == [b]
    assert out["isHidden"] == [a] and out["hiddenTrue"] == [a] and out["shown"] == [b, c]
    assert out["x"] == [a] and out["spaced"] == [b]
    # No value: neither a check nor its negation holds of the file without a label.
    assert out["notX"] == [b] and out["otherThanX"] == [b]
    assert out["sameLabel"] == [f"{a} {a}", f"{b} {b}"]
    assert out["hasLarge"] == ["r"]
    assert out["violations"] == ["rule: tooBig(file_2)", "rule: tooBig(file_3)"]
    assert "cannot represent the real value inf" in out["not_finite"]


def test_aggregations():
    out = run(
        ATTRIBUTE_PRELUDE
        + """
full = Dir(name="full", contents=[File(size=7, ratio=0.5), File(size=8, ratio=1.5)])
empty = Dir(name="empty")
objects = [Filesystem(root=Dir(name="r", contents=[full, empty]))]
sizes = "{ Dir::contents(d, f) -> File::size(f) }"
ratios = "{ Dir::contents(d, f) -> File::ratio(f) }"
out = matched(objects, [
    "pred two(Dir d) <-> count { Dir::contents(d, _) } == 2.",
    "pred none(Dir d) <-> count { Dir::contents(d, _) } == 0.",
    "pred notTwo(Dir d) <-> !(count { Dir::contents(d, _) } == 2).",
    "pred big(Dir d) <-> count { bigIn(d, _) } == 1.",
    "pred bigIn(Dir d, File f) <-> contents(d, f), File::size(f) > 7.",
    f"pred sum15(Dir d) <-> sum {sizes} == 15.",
    f"pred sum0(Dir d) <-> sum {sizes} == 0.",
    f"pred sumReal(Dir d) <-> sum {ratios} == 2.0.",
    f"pred min7(Dir d) <-> min {sizes} == 7.",
    f"pred max8(Dir d) <-> max {sizes} == 8.",
    f"pred maxReal(Dir d) <-> max {ratios} > 1.0.",
    f"pred spread(Dir d) <-> max {sizes} - min {sizes} == 1.",
    # Of no file, a min is positive infinity and a max negative infinity.
    f"pred minAbove(Dir d) <-> min {sizes} > 100.",
    f"pred minBelow(Dir d) <-> min {sizes} < 100.",
    f"pred minNotBelow(Dir d) <-> !(min {sizes} < 100).",
    f"pred maxBelow(Dir d) <-> max {sizes} < 0.",
    f"pred maxAbove(Dir d) <-> max {sizes} > 0.",
    "pred grand(Dir d) <-> sum { Dir::contents(d, e) -> count { Dir::contents(e, _) } } == 2.",
    # Guarded by the class, the sum reads no directory's size.
    "pred fileIn(Dir d, File f) <-> Dir::contents(d, f).",
    "pred guarded(Dir d) <-> sum { fileIn(d, f) -> File::size(f) } == 0.",
])
print(json.dumps(out))
"""
    )
    assert out["two"] == ["full", "r"] and out["none"] == ["empty"]
    assert out["notTwo"] == ["empty"]
    assert out["big"] == ["full"]
    # The contents of r are directories: File::size of a directory is a contradiction, error,
    # so every check over r's sizes holds, as on a server.
    assert out["sum15"] == ["full", "r"] and out["sum0"] == ["empty", "r"]
    assert out["sumReal"] == ["full", "r"]
    assert out["min7"] == ["full", "r"] and out["max8"] == ["full", "r"]
    assert out["maxReal"] == ["full", "r"] and out["spread"] == ["full", "r"]
    assert out["minAbove"] == ["empty", "r"] and out["minBelow"] == ["full", "r"]
    assert out["minNotBelow"] == ["empty", "r"]
    assert out["maxBelow"] == ["empty", "r"] and out["maxAbove"] == ["full", "r"]
    assert out["grand"] == ["r"]
    assert out["guarded"] == ["empty", "r"]


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
