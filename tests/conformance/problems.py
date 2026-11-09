# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Test cases from Refinery's ``.problem`` test files.

Ported from Refinery's file-based semantics tests, ``ProblemSplitter``,
``SemanticsTestBuilder`` and ``SemanticsTestCaseBuilder`` in
``subprojects/generator-core/src/testFixtures`` (commit ``7470710``):
https://github.com/graphs4value/refinery/tree/747071004a969288113bb09d85299bbd28c9f9df/subprojects/generator-core/src/testFixtures/java/tools/refinery/generator/tests

A file is a problem split into chunks by header comments::

    class Person { Person[] friend }   % common to every case
    Person(a).
    % TEST: a case, adding to the common part
    friend(a, a).
    % EXPECT:
    friend(a, a).
    % EXPECT EXACTLY: what must hold, no more and no less precisely
    !friend(a, a).

- ``% TEST[ WITH ERRORS | WITH PROPAGATION FAILURE | WITH CONCRETIZATION FAILURE][: name]``
  starts a case: the common part, then its own statements.
- ``% EXPECT[ CANDIDATE][ EXACTLY][: description]`` starts assertions that must
  hold in the case's partial model (or, with ``CANDIDATE``, its concretization).
  An actual value passes if it refines the expected one, or with ``EXACTLY``,
  if it equals it. An ``EXPECT`` before any ``TEST`` makes the file one case.

Each case's problem keeps the line numbers of the file: the chunks of other
cases, and its own expectations, are replaced by empty lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TestKind = Literal["no errors", "allow errors", "propagation failure", "concretization failure"]
Concreteness = Literal["partial", "candidate"]

_COMMENT_PREFIX = r"(//|%)\s*"
_TEST_CASE = re.compile(
    _COMMENT_PREFIX + r"TEST(\s+WITH\s+((?P<allow_errors>ERRORS)|"
    r"((?P<propagation>PROPAGATION)|(?P<concretization>CONCRETIZATION)) FAILURE))?"
    r"(\s*:\s*(?P<name>\S.*)?)?"
)
_EXPECTATION = re.compile(
    _COMMENT_PREFIX + r"EXPECT(?P<candidate>\s+CANDIDATE)?(?P<exact>\s+EXACTLY)?"
    r"(\s*:\s*(?P<description>\S.*)?)?"
)


class ProblemFileError(ValueError):
    """A ``.problem`` test file that can't be split into cases."""


@dataclass(frozen=True)
class Assertion:
    """An assertion of an ``EXPECT`` chunk, e.g. ``?friend(a, b).`` or ``age(a): 1..5.``"""

    relation: str
    arguments: tuple[str | None, ...]
    """Node names; ``None`` for the wildcard ``*``."""
    value: str
    """``"true"``, ``"false"``, ``"unknown"`` or ``"error"``, or an attribute value
    as written, e.g. ``"20"``, ``"10..30"``, ``"*..5"`` or ``'"text"'``."""
    line: int
    source: str


@dataclass(frozen=True)
class Expectation:
    concreteness: Concreteness
    exact: bool
    description: str | None
    line: int
    assertions: tuple[Assertion, ...]


@dataclass(frozen=True)
class Case:
    path: Path
    index: int
    name: str | None
    kind: TestKind
    source: str
    """The problem, without the expectations."""
    expectations: tuple[Expectation, ...]

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}" if self.name else str(self.index)

    @property
    def needs_candidate(self) -> bool:
        return any(e.concreteness == "candidate" for e in self.expectations)


def load(path: Path) -> list[Case]:
    """The test cases of the ``.problem`` file at ``path``."""
    return split(path.read_text(encoding="utf-8"), path)


def split(text: str, path: Path = Path("<string>")) -> list[Case]:
    """The test cases of the ``.problem`` file ``text``."""
    builder = _Builder(path)
    header: _Header = _Common()
    start = 0
    for offset, comment in _line_comments(text):
        next_header = _parse_header(comment, text.count("\n", 0, offset) + 1)
        if next_header is not None:
            builder.accept(header, text[start:offset])
            header, start = next_header, offset
    builder.accept(header, text[start:])
    return builder.build()


# -- chunk headers ------------------------------------------------------------


@dataclass(frozen=True)
class _Common:
    pass


@dataclass(frozen=True)
class _TestCase:
    kind: TestKind
    name: str | None


@dataclass(frozen=True)
class _Expect:
    concreteness: Concreteness
    exact: bool
    description: str | None
    line: int


_Header = _Common | _TestCase | _Expect


def _parse_header(comment: str, line: int) -> _Header | None:
    text = comment.strip()
    if match := _TEST_CASE.fullmatch(text):
        kind: TestKind
        if match.group("allow_errors"):
            kind = "allow errors"
        elif match.group("propagation"):
            kind = "propagation failure"
        elif match.group("concretization"):
            kind = "concretization failure"
        else:
            kind = "no errors"
        return _TestCase(kind, _strip(match.group("name")))
    if match := _EXPECTATION.fullmatch(text):
        concreteness: Concreteness = "candidate" if match.group("candidate") else "partial"
        exact = match.group("exact") is not None
        return _Expect(concreteness, exact, _strip(match.group("description")), line)
    return None


def _strip(text: str | None) -> str | None:
    return text.strip() if text is not None else None


class _Builder:
    """``SemanticsTestBuilder``: assembles cases from chunks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.common: list[str] = []
        self.cases: list[Case] = []
        self.current: _CaseBuilder | None = None
        self.single = False

    def accept(self, header: _Header, body: str) -> None:
        match header:
            case _Common():
                if self.current is not None:
                    raise ProblemFileError("common code after the first test case")
                self.common.append(body)
            case _TestCase():
                if self.single:
                    raise ProblemFileError("TEST chunk after an EXPECT chunk of a single test")
                self._start(header, body)
                self._blank(body)
            case _Expect():
                if self.current is None:
                    self._start(_TestCase("no errors", None), "")
                    self.single = True
                assert self.current is not None
                self.current.expect(header, body)
                self._blank(body)

    def _start(self, header: _TestCase, body: str) -> None:
        if self.current is not None:
            self.cases.append(self.current.build())
        source = "".join(self.common) + body
        self.current = _CaseBuilder(self.path, len(self.cases), header, source)

    def _blank(self, body: str) -> None:
        """Keep the common part's line numbers in later cases."""
        if not self.single:
            self.common.append(_blank_lines(body))

    def build(self) -> list[Case]:
        if self.current is None:
            raise ProblemFileError("the file has no TEST or EXPECT chunks")
        self.cases.append(self.current.build())
        return self.cases


class _CaseBuilder:
    """``SemanticsTestCaseBuilder``: a case and its expectations."""

    def __init__(self, path: Path, index: int, header: _TestCase, source: str) -> None:
        self.path = path
        self.index = index
        self.header = header
        self.source = [source]
        self.expectations: list[Expectation] = []

    def expect(self, header: _Expect, body: str) -> None:
        # Like Refinery, which removes the assertions of expectations from the
        # problem, but keeps the line numbers of what follows.
        self.source.append(_blank_lines(body))
        assertions = tuple(_parse_assertions(body, header.line))
        expectation = Expectation(
            header.concreteness, header.exact, header.description, header.line, assertions
        )
        self.expectations.append(expectation)

    def build(self) -> Case:
        if not self.expectations and self.header.kind == "allow errors":
            raise ProblemFileError("a TEST WITH ERRORS has no EXPECT chunks")
        return Case(
            self.path,
            self.index,
            self.header.name,
            self.header.kind,
            "".join(self.source),
            tuple(self.expectations),
        )


def _blank_lines(text: str) -> str:
    return re.sub(r"(?m)^.+$", "", text)


# -- tokens -------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<line_comment>(?:%|//)[^\n]*)
    | (?P<block_comment>/\*.*?\*/)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<quoted>'(?:[^'\\]|\\.)*')
    | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<punctuation>::|\.\.|[()\[\]{},.:;!?*<>=+\-])
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    offset: int


def _tokens(text: str) -> list[_Token]:
    tokens = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        assert match is not None  # ``other`` matches any character
        kind = match.lastgroup
        assert kind is not None
        tokens.append(_Token(kind, match.group(), offset))
        offset = match.end()
    return tokens


def _line_comments(text: str) -> list[tuple[int, str]]:
    """The offset and text of each line comment, outside strings and quoted names."""
    return [(t.offset, t.text) for t in _tokens(text) if t.kind == "line_comment"]


# -- assertions ---------------------------------------------------------------

_LOGIC_VALUES = {"true", "false", "unknown", "error"}


def _parse_assertions(body: str, first_line: int) -> list[Assertion]:
    tokens = [t for t in _tokens(body) if t.kind not in ("space", "line_comment", "block_comment")]
    assertions = []
    statement: list[_Token] = []
    for token in tokens:
        if token.text == ".":  # a range is one token, ``..``
            line = first_line + body.count("\n", 0, statement[0].offset) if statement else 0
            end = token.offset + 1
            source = body[statement[0].offset : end] if statement else "."
            assertions.append(_parse_assertion(statement, line, source))
            statement = []
        else:
            statement.append(token)
    if statement:
        raise ProblemFileError(f"unterminated assertion in EXPECT chunk: {body!r}")
    return assertions


def _parse_assertion(tokens: list[_Token], line: int, source: str) -> Assertion:
    def fail(reason: str) -> ProblemFileError:
        return ProblemFileError(f"line {line}: {reason}: {source!r}")

    if not tokens:
        raise fail("empty assertion")
    if tokens[0].text == "default":
        raise fail("default assertions are not supported in EXPECT chunks")
    position = 0
    value = "true"
    if tokens[0].text in ("!", "?"):
        value = "false" if tokens[0].text == "!" else "unknown"
        position = 1
    relation, position = _qualified_name(tokens, position)
    if relation is None:
        raise fail("only assertions are supported in EXPECT chunks")
    if position >= len(tokens) or tokens[position].text != "(":
        raise fail("expected '('")
    position += 1
    arguments: list[str | None] = []
    while tokens[position].text != ")":
        if tokens[position].text == "*":
            arguments.append(None)
            position += 1
        else:
            name, position = _qualified_name(tokens, position)
            if name is None:
                raise fail("expected a node name or '*'")
            arguments.append(name)
        if tokens[position].text == ",":
            position += 1
        elif tokens[position].text != ")":
            raise fail("expected ',' or ')'")
    position += 1
    if position < len(tokens):
        if tokens[position].text != ":" or value != "true":
            raise fail("expected ':' and a value after the arguments")
        value = "".join(t.text for t in tokens[position + 1 :])
        if not value:
            raise fail("expected a value after ':'")
    return Assertion(relation, tuple(arguments), value, line, source)


def _qualified_name(tokens: list[_Token], position: int) -> tuple[str | None, int]:
    """A name such as ``Person::friend``, with quoted parts unquoted."""
    parts = []
    while position < len(tokens) and tokens[position].kind in ("name", "quoted"):
        token = tokens[position]
        parts.append(_unquote(token.text) if token.kind == "quoted" else token.text)
        position += 1
        if position < len(tokens) and tokens[position].text == "::":
            position += 1
        else:
            break
    return ("::".join(parts) if parts else None), position


def _unquote(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text[1:-1])


def is_logic_value(value: str) -> bool:
    return value in _LOGIC_VALUES
