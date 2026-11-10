# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The bits engine's API: models built by hand, rules, results and errors.

What it computes is checked against Refinery in ``tests/conformance/test_bits.py``.
"""

from __future__ import annotations

import pytest

from pyfinery.bits import (
    Attribute,
    Node,
    PartialModel,
    Relation,
    RuleError,
    UnsupportedRuleError,
    evaluate,
)
from pyfinery.bits.rules.syntax import parse


def model() -> PartialModel:
    """alice -> bob -> carl, bob only may exist, carl may follow alice back."""
    nodes = [
        Node("alice", "alice"),
        Node("bob", "bob"),
        Node("carl", "carl"),
        Node("E::x", "x", "atom"),
    ]
    people = [((i,), "true") for i in range(3)]
    return PartialModel(
        nodes,
        [
            Relation.of("Person", 1, people),
            Relation.of(
                "Person::friend", 2, [((0, 1), "true"), ((1, 2), "true"), ((2, 0), "unknown")]
            ),
            Relation.of(
                "builtin::exists",
                1,
                [((0,), "true"), ((1,), "unknown"), ((2,), "true"), ((3,), "true")],
            ),
        ],
        [Attribute.of("Person::age", "int", 4, [(0, 30), (1, "10..20"), (2, None)])],
    )


def values(result) -> dict:
    return {key: value for key, value in result.items()}


def test_evaluates_every_predicate_with_a_body():
    results = evaluate(
        [
            "pred reaches(Person p, Person q) <-> friend+(p, q).",
            "pred adult(Person p) <-> age(p) >= 18.",
            "pred trusted(Person p).",  # no body: not evaluated
        ],
        model(),
    )
    assert set(results) == {"reaches", "adult"}
    reaches = results["reaches"]
    assert reaches.value((0, 1)) == "true"
    assert reaches.value((0, 2)) == "unknown"  # through bob, who only may exist
    assert reaches.value((1, 1)) == "unknown"
    assert reaches.value((3, 0)) == "false"
    assert values(results["adult"]) == {(0,): "true", (1,): "unknown", (2,): "unknown"}


def test_select_and_sources_with_other_statements():
    source = (
        "class Person { Person[] friend }\nPerson(alice).\n"
        "pred self(Person p) <-> p == p.\npred other(Person p) <-> !self(p)."
    )
    results = evaluate(source, model(), select=["other"])
    assert set(results) == {"other"}
    with pytest.raises(RuleError, match="no predicate named 'missing'"):
        evaluate(source, model(), select=["missing"])


def test_error_predicates_are_errors_where_they_must_hold():
    results = evaluate("error pred befriends(Person p) <-> friend(p, _).", model())
    # alice's only friend, bob, only may exist: the body must hold for bob alone.
    assert values(results["befriends"]) == {(1,): "error"}
    assert results["befriends"].kind == "error"


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ("pred p(Person x) <-> nothing(x).", "no relation named 'nothing'"),
        ("pred p(Person x) <-> friend(x).", "friend has 2 parameters, given 1"),
        ("pred p(x) <-> !friend(x, _).", "parameter x of p is bound by no literal"),
        ("pred p(Person x) <-> age(y) > 3.", "unbound input variables: y"),
        ("pred p(Person x) <-> !friend(x, y), x != y.", "appears in several literals"),
        ("pred p(Person x) <-> q(x).\npred q(Person x) <-> p(x).", "recursive predicates"),
        ("pred p(Person x) <-> friend(x, nope::thing).", "no atom named 'nope::thing'"),
        ("pred p(Person x) <-> !(x == x).", "negated comparison of nodes"),
        ("pred p(Person x) <-> name(x) == 3.", "no attribute named 'name'"),
        ("pred p(Person x, Person x) <-> friend(x, x).", "two parameters named x"),
        ("pred p(Person x) <-> friend(x, y).\npred p(Person x) <-> friend(y, x).", "defined twice"),
    ],
)
def test_rule_errors(rules, message):
    with pytest.raises(RuleError, match=message):
        evaluate(rules, model())


@pytest.mark.parametrize(
    ("rules", "feature"),
    [
        ("pred p(Person x) <-> count { friend(x, _) } > 1.", "aggregation"),
        ("pred p(Person x) <-> age(x) + 1 > 3.", "arithmetic"),
        ("shadow pred p(Person x) <-> candidate must friend(x, x).", "candidate"),
        (
            "pred p(Person x) subsets q <-> friend(x, x).\npred q(Person x).",
            "subsets on a predicate",
        ),
        (
            "pred p(Person x) <-> friend(x, x).\npred q(Person x) subsets p.",
            "subsets of a predicate",
        ),
        ("pred p(Person x) <-> friend(x, x).\n!p(alice).", "assertions about a predicate"),
        ("class A { p[] r }\npred p(Person x) <-> friend(x, x).", "type of a reference"),
        ("pred p(Person x) <-> vip(x) < true.", "ordering booleans"),
    ],
)
def test_unsupported(rules, feature):
    m = model()
    if "vip" in rules:
        m = PartialModel(
            m.nodes, m.relations, [Attribute.of("Person::vip", "boolean", 4, [(0, True)])]
        )
        m = PartialModel(
            m.nodes, [*m.relations, Relation.of("Person::vip", 1, [((0,), "true")])], m.attributes
        )
    with pytest.raises(UnsupportedRuleError, match=feature):
        evaluate(rules, m)


def test_parser_skips_other_statements_and_reads_kinds():
    source = """
    class A { int n  A[] r }
    enum E { x, y }
    A(a). n(a): 1..2. scope A = 1..5.
    propagation rule r1(A a) <-> r(a, a) ==> !r(a, a).
    error pred e(A a) <-> r(a, a).
    shadow pred s(a) <-> must r(a, _).
    error short(A a) <-> r(a, _).
    pred 'odd name'(A a, b) <-> r(a, b) ; r(b, a).
    pred base(A a).
    """
    predicates = parse(source).predicates
    assert [(p.name, p.kind) for p in predicates] == [
        ("e", "error"),
        ("s", "shadow"),
        ("short", "error"),
        ("'odd name'", "pred"),
        ("base", "pred"),
    ]
    assert len(predicates[3].body) == 2 and predicates[4].body is None
    assert [(p.type, p.name) for p in predicates[3].parameters] == [("A", "a"), (None, "b")]


def test_rule_errors_have_positions():
    with pytest.raises(RuleError) as raised:
        evaluate("pred p(Person x) <->\n    friend(x, ).", model())
    assert raised.value.position is not None and raised.value.position.line == 2


def test_partial_models_check_their_input():
    with pytest.raises(ValueError, match="arity"):
        Relation.of("r", 2, [((0,), "true")])
    with pytest.raises(ValueError, match="two relations"):
        PartialModel([Node("a", "a")], [Relation.of("r", 1, []), Relation.of("r", 1, [])])
    # Without builtin::exists, every node exists; without equals, each equals itself.
    bare = PartialModel([Node("a", "a"), Node("b", "b")], [Relation.of("R", 1, [((0,), "true")])])
    results = evaluate("pred p(R x) <-> R(y), x == y.", bare)
    assert dict(results["p"].items()) == {(0,): "true"}


def test_predicates_without_a_body_are_unknown_on_their_types():
    rules = [
        "pred trusted(Person p).",
        "pred pair(Person p, q).",
        "pred vouched(Person p, Person q) <-> friend(p, q), trusted(q).",
    ]
    results = evaluate(rules, model(), select=["trusted", "pair", "vouched"])
    assert values(results["trusted"]) == {(0,): "unknown", (1,): "unknown", (2,): "unknown"}
    assert len(values(results["pair"])) == 3 * 4 and set(values(results["pair"]).values()) == {
        "unknown"
    }
    assert values(results["vouched"]) == {(0, 1): "unknown", (1, 2): "unknown", (2, 0): "unknown"}
    # Values the model has for one are its own.
    given = PartialModel(
        model().nodes, [*model().relations, Relation.of("trusted", 1, [((2,), "true")])]
    )
    assert values(evaluate(rules, given, select=["trusted"])["trusted"]) == {(2,): "true"}


def test_numbers_are_exact_and_reals_rounded_as_refinery_does():
    nodes = [Node(f"n{i}", f"n{i}") for i in range(3)]
    big = 2**53
    numbers = PartialModel(
        nodes,
        [Relation.of("A", 1, [((i,), "true") for i in range(3)])],
        [
            Attribute.of("A::n", "int", 3, [(0, big), (1, big + 1), (2, "3..9")]),
            # 0.30000000000000004 has 17 significant digits: an interval of two.
            Attribute.of("A::x", "real", 3, [(0, 0.1), (1, 0.30000000000000004), (2, "2.5")]),
        ],
    )
    results = evaluate(
        [
            "pred same(A p, A q) <-> n(p) == n(q).",
            f"pred above(A p) <-> n(p) > {big}.",
            "pred third(A p) <-> x(p) == 0.3.",
            "pred tenth(A p) <-> x(p) == 0.1.",
        ],
        numbers,
    )
    assert results["same"].value((0, 1)) == "false"
    assert values(results["above"]) == {(1,): "true"}
    assert results["third"].value((1,)) == "unknown"
    assert values(results["tenth"]) == {(0,): "true"}


def test_strings_are_values_even_when_spelled_like_logic_values():
    nodes = [Node(f"n{i}", f"n{i}") for i in range(4)]
    strings = [(0, "unknown"), (1, "error"), (2, None), (3, Attribute.ERROR)]
    model = PartialModel(
        nodes,
        [Relation.of("A", 1, [((i,), "true") for i in range(4)])],
        [Attribute.of("A::s", "string", 4, strings)],
    )
    results = evaluate(
        ['pred word(A p) <-> s(p) == "unknown".', 'pred err(A p) <-> s(p) == "error".'], model
    )
    assert values(results["word"]) == {(0,): "true", (2,): "unknown", (3,): "error"}
    assert values(results["err"]) == {(1,): "true", (2,): "unknown", (3,): "error"}


@pytest.mark.parametrize(
    ("literal", "value"),
    [
        (r'"a\nb"', "a\nb"),
        (r'"\t\r\b\f"', "\t\r\b\f"),
        (r'"\u0061\u00e9"', "aé"),
        (r'"\uD83D\uDE00"', "\U0001f600"),  # UTF-16: a surrogate pair is one character
        (r'"\uD83D"', "\ud83d"),  # a lone surrogate stays
        (r'"a\\b\"c\'d"', "a\\b\"c'd"),
    ],
)
def test_string_literals_decode_java_escapes(literal, value):
    from pyfinery.bits.names import string_literal

    assert string_literal(literal) == value


@pytest.mark.parametrize(
    ("value", "name"),
    [
        ("plain", "plain"),
        ("e", "e"),
        ("atom", "atom"),  # keywords Refinery also takes as names aren't quoted
        ("contains", "contains"),
        ("pred", "'pred'"),
        ("true", "'true'"),
        ("my atom", "'my atom'"),
        ("it's", "'it\\'s'"),
        ("x\\y", "'x\\\\y'"),
    ],
)
def test_names_are_written_as_refinery_writes_them(value, name):
    """As checked against the pinned image: these are the names of its JSON output."""
    from pyfinery.bits.names import refinery_name

    assert refinery_name(value) == name


def test_quoted_names_resolve_to_the_models_names():
    nodes = [Node("'Jane Doe'", "'Jane Doe'"), Node("'my atom'", "'my atom'", "atom")]
    model = PartialModel(
        nodes,
        [
            Relation.of("'My Class'", 1, [((0,), "true"), ((1,), "true")]),
            Relation.of("'My Class'::'odd ref'", 2, [((0, 1), "true")], "'odd ref'"),
        ],
    )
    rules = "pred 'odd pred'('My Class' x) <-> 'My Class'::'odd ref'(x, 'my atom')."
    results = evaluate(rules, model, select=["odd pred"])
    assert values(results["'odd pred'"]) == {(0,): "true"}


def test_quoted_parameters_are_the_variables_their_bodies_name():
    """Review of #28, round 3: the declaration and the body spelled 'odd var' apart."""
    rules = "pred q(Person 'odd var') <-> Person::friend('odd var', other)."
    # alice's friend only may exist; bob's, carl, does; carl's friendship is unknown.
    assert values(evaluate(rules, model())["q"]) == {(0,): "unknown", (1,): "true", (2,): "unknown"}


def test_only_reference_types_count_as_predicates_typing_references():
    from pyfinery.bits.rules.syntax import parse

    source = "class A { contains bar[0..1] foo opposite baz  refers B b; int n }"
    assert set(parse(source).in_classes) == {"bar", "B", "int"}
    # A predicate named after a keyword isn't a type because 'class' is in the statement.
    results = evaluate("class B { B[] r }\npred 'class'(Person x) <-> Person(x).", model())
    assert set(results) == {"'class'"}


@pytest.mark.parametrize(
    ("source", "refused"),
    [
        # Review of #28, round 4: s's facts refine q through its parameter type.
        ("pred q(Person x) <-> Person(x).\npred s(q x).\ns(alice).", "parameter type of s"),
        ("pred q(Person x) <-> Person(x).\nscope q = 1.", "named outside"),
        (
            "pred q(Person x) <-> Person(x).\npropagation rule p(Person x) <-> Person(x) ==> q(x).",
            "named outside",
        ),
        # Where a predicate with a body is only read, it is evaluated.
        ("pred q(Person x) <-> Person(x).\npred t(q x) <-> Person(x).", None),
        ("pred q(Person x) <-> Person(x).\npropagation rule p(Person x) <-> q(x) ==> r(x).", None),
        ("pred q(Person x) <-> Person(x).\nenum E { q }\nPerson(q).", None),
    ],
)
def test_predicates_with_bodies_are_only_read(source, refused):
    """A predicate with a body may be named in bodies, as a parameter type of a predicate
    with a body, and in rules' premises; elsewhere Refinery may refine it."""
    if refused is None:
        evaluate(source, model())
    else:
        with pytest.raises(UnsupportedRuleError, match=refused):
            evaluate(source, model())


def test_a_selected_predicate_without_a_body_has_the_models_arity():
    # Person relates one node in the model; a declaration of none can't select it.
    with pytest.raises(
        RuleError, match="Person has 0 parameters, but Person in the model relates 1"
    ):
        evaluate("pred Person().", model(), select=["Person"])
    assert len(list(evaluate("pred Person(x).", model(), select=["Person"])["Person"].items())) == 3
