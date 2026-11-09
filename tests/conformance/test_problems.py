# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Splitting ``.problem`` test files, and checking values against expectations."""

from __future__ import annotations

import pytest

from pyfinery.models import JsonOutput

from .interpretation import Interpretation, check, errors, refines
from .problems import Assertion, Expectation, ProblemFileError, split

FILE = """\
class A.
A(a).
% TEST: first
A(b).
% EXPECT:
A(b).
% TEST WITH ERRORS: second
!A(a).
% EXPECT EXACTLY: exact
A(a): error.
% EXPECT CANDIDATE:
?A(*).
"""


def test_cases_get_the_common_part_and_their_own():
    first, second = split(FILE)
    assert (first.index, first.name, first.kind) == (0, "first", "no errors")
    assert (second.index, second.name, second.kind) == (1, "second", "allow errors")
    assert "A(b)." in first.source and "!A(a)." not in first.source
    assert "!A(a)." in second.source and "A(b)." not in second.source
    assert first.source.startswith("class A.\nA(a).\n")
    assert second.source.startswith("class A.\nA(a).\n")


def test_line_numbers_are_kept():
    for case in split(FILE):
        for line_number, line in enumerate(case.source.splitlines(), start=1):
            if line:
                assert FILE.splitlines()[line_number - 1] == line


def test_expectations_are_left_out_of_the_problem():
    first, second = split(FILE)
    assert first.source.count("A(b).") == 1
    assert "error" not in second.source
    assert [e.exact for e in second.expectations] == [True, False]
    assert [e.concreteness for e in second.expectations] == ["partial", "candidate"]
    assert second.expectations[0].description == "exact"
    assert second.needs_candidate and not first.needs_candidate


def test_assertions():
    (case,) = split(
        "class P { int age }\n% EXPECT:\nP(a).\n!P(b).\n?'odd name'(x, *).\n"
        "P::age(a): 1..5.\nage(b): *..3.\nage(c): -2.\nage(d): 1.5.\n"
        "% a comment\nP(c): error.\n"
    )
    assertions = case.expectations[0].assertions
    assert [(a.relation, a.arguments, a.value) for a in assertions] == [
        ("P", ("a",), "true"),
        ("P", ("b",), "false"),
        ("odd name", ("x", None), "unknown"),
        ("P::age", ("a",), "1..5"),
        ("age", ("b",), "*..3"),
        ("age", ("c",), "-2"),
        ("age", ("d",), "1.5"),
        ("P", ("c",), "error"),
    ]
    assert [a.line for a in assertions] == [3, 4, 5, 6, 7, 8, 9, 11]
    assert assertions[2].source == "?'odd name'(x, *)."


def test_a_file_with_expectations_and_no_test_is_one_case():
    (case,) = split("class A.\nA(a).\n% EXPECT:\nA(a).\n% EXPECT EXACTLY:\nA(a).\n")
    assert case.name is None and len(case.expectations) == 2


def test_header_comments_in_strings_are_not_headers():
    (case,) = split('class A { string s }\nA(a).\ns(a): "% TEST: no".\n% EXPECT:\nA(a).\n')
    assert case.name is None


@pytest.mark.parametrize(
    "text",
    [
        "class A.\n",  # no cases
        "% EXPECT:\nA(a).\n% TEST:\nA(b).\n",  # a test after a single test's expectation
        "% TEST WITH ERRORS:\nA(a).\n",  # errors allowed, but nothing expected
        "% EXPECT:\nclass B.\n",  # not an assertion
        "% EXPECT:\ndefault A(*).\n",  # default assertions
        "% EXPECT:\nA(a)\n",  # unterminated
    ],
)
def test_malformed_files(text):
    with pytest.raises(ProblemFileError):
        split(text)


@pytest.mark.parametrize(
    ("actual", "expected", "exact", "result"),
    [
        ("true", "true", False, True),
        ("error", "true", False, True),
        ("error", "true", True, False),
        ("true", "unknown", False, True),
        ("true", "unknown", True, False),
        ("unknown", "true", False, False),
        ("false", "true", False, False),
        ("5", "1..10", False, True),
        ("5", "1..10", True, False),
        ("5", "*..4", False, False),
        ("3..4", "1..*", False, True),
        ("error", "1..2", False, True),
        ("unknown", "unknown", True, True),
        ("1.5", "1.5", True, True),
        ('"x"', '"x"', True, True),
        ('"x"', "unknown", False, True),
        # Integers and reals are exact, as Refinery's BigInteger and BigDecimal.
        ("9007199254740992", "9007199254740993", True, False),
        ("9007199254740992", "9007199254740993..9007199254740994", False, False),
        ("0.1", "0.1000000000000000055511151231257827", True, False),
        ("0.30", "0.3", True, True),
        # Strings are compared by value, as Refinery escapes them.
        ('"it\\\'s"', '"it\'s"', True, True),
        ('"a\\u0041"', '"aA"', True, True),
        ('"x"', '"y"', False, False),
        # An error refines every value, of every type.
        ("error", '"x"', False, True),
        ("error", '"x"', True, False),
        ("error", "5", False, True),
    ],
)
def test_refines(actual, expected, exact, result):
    assert refines(actual, expected, exact=exact) is result


def test_reals_are_rounded_to_16_significant_digits():
    # As Refinery's RealInterval.of: 0.10000000000000001 is 0.1000000000000000..0.1000000000000001.
    rounded = "0.1000000000000000..0.1000000000000001"
    assert refines(rounded, "0.10000000000000001", exact=True, real=True)
    assert not refines(rounded, "0.10000000000000001", exact=True)
    assert refines("0.1", "0.1", exact=True, real=True)


MODEL = JsonOutput.model_validate(
    {
        "concreteness": "partial",
        "nodes": [
            {"name": "a", "simpleName": "a", "kind": "default"},
            {"name": "b", "simpleName": "b", "kind": "default"},
            {"name": "E::lit", "simpleName": "lit", "kind": "atom"},
        ],
        "relations": [
            *(
                {
                    "name": name,
                    "simpleName": name.split("::")[-1],
                    "arity": arity,
                    "detail": {"type": "pred", "kind": "defined"},
                }
                for name, arity in [("builtin::exists", 1), ("p", 1), ("r", 2)]
            ),
        ],
        "partialInterpretation": {
            "builtin::exists": [[0, "true"], [2, "true"]],
            "p": [[0, "true"], [1, "true"], [2, "error"]],
            "r": [[0, 2, "unknown"]],
        },
    }
)


def _expect(*assertions: tuple[str, tuple[str | None, ...], str], exact=True) -> Expectation:
    return Expectation(
        "partial", exact, None, 1, tuple(Assertion(r, a, v, 1, f"{r}{a}") for r, a, v in assertions)
    )


def test_values_of_objects_that_do_not_exist_are_the_default():
    interpretation = Interpretation(MODEL)
    assert check(_expect(("p", ("a",), "true")), interpretation) == []
    # b doesn't exist: p(b) reads false, as in Refinery's FilteredInterpretation.
    assert check(_expect(("p", ("b",), "false")), interpretation) == []


def test_nodes_by_simple_name_and_wildcards():
    interpretation = Interpretation(MODEL)
    assert check(_expect(("r", ("a", "lit"), "unknown")), interpretation) == []
    assert check(_expect(("r", ("a", None), "unknown")), interpretation) == []
    assert check(_expect(("r", (None, None), "true")), interpretation) != []
    assert check(_expect(("exists", ("a",), "true")), interpretation) == []


def test_errors_are_inconsistencies():
    assert errors(Interpretation(MODEL)) == ["p(E::lit)"]


def test_an_attribute_without_a_value_is_an_error():
    model = JsonOutput.model_validate(
        {
            "concreteness": "partial",
            "nodes": [{"name": "a", "simpleName": "a", "kind": "default"}],
            "relations": [
                {
                    "name": "A::n",
                    "simpleName": "n",
                    "arity": 1,
                    "dataType": "builtin::int",
                    "detail": {"type": "attribute"},
                },
            ],
            "partialInterpretation": {"A::n": []},
        }
    )
    interpretation = Interpretation(model)
    assert check(_expect(("n", ("a",), "error")), interpretation) == []
