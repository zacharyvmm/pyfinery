# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Reading predicate declarations from Refinery source.

Only ``pred`` declarations (with ``error`` and ``shadow``) are read; other
statements, such as classes, assertions and rules, are skipped. A body is a
disjunction (``;``) of conjunctions (``,``) of literals:

- an atom, ``Class::feature(x, y)``, ``pred+(x, y)`` for transitive closure,
  with arguments that are variables, ``_`` or atoms (resolved later);
- ``!`` before an atom or a parenthesized comparison;
- ``must`` and ``may`` before an atom;
- a comparison of variables, attribute values and constants, with ``==``,
  ``!=``, ``<``, ``<=``, ``>`` or ``>=``.

What Refinery has beyond that (aggregation, arithmetic, ``candidate``, ...)
raises :class:`UnsupportedRuleError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ._errors import RuleError, UnsupportedRuleError

PredicateKind = Literal["pred", "error", "shadow"]
Modality = Literal["must", "may"]
Operator = Literal["==", "!=", "<", "<=", ">", ">="]


@dataclass(frozen=True)
class Position:
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


@dataclass(frozen=True)
class Name:
    """A variable, an atom, or ``_``: which one is decided when names are resolved."""

    text: str
    position: Position


@dataclass(frozen=True)
class Atom:
    relation: str
    arguments: tuple[Name, ...]
    negated: bool
    modality: Modality | None
    transitive: bool
    position: Position


@dataclass(frozen=True)
class Constant:
    value: int | Decimal | str | bool


@dataclass(frozen=True)
class Call:
    """``feature(x)`` in a comparison: an attribute value."""

    relation: str
    arguments: tuple[Name, ...]
    position: Position


Term = Name | Constant | Call


@dataclass(frozen=True)
class Comparison:
    left: Term
    operator: Operator
    right: Term
    negated: bool
    position: Position


Literal_ = Atom | Comparison


@dataclass(frozen=True)
class Parameter:
    type: str | None
    name: str


@dataclass(frozen=True)
class Predicate:
    name: str
    kind: PredicateKind
    parameters: tuple[Parameter, ...]
    body: tuple[tuple[Literal_, ...], ...] | None
    """Clauses of literals; ``None`` for a predicate without a body."""
    position: Position
    subsets: str | None = None


@dataclass(frozen=True)
class Source:
    """The predicates of a source, and where the rest of it refines a predicate's
    values beyond what its body says, which the bits engine doesn't do."""

    predicates: tuple[Predicate, ...]
    asserted: dict[str, Position]
    """Relations that assertions are about, e.g. ``bar`` in ``!bar(b1).``"""
    in_classes: dict[str, Position]
    """Names in class declarations, e.g. ``bar`` in ``class A { bar[] foo }``."""


# -- tokens -------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<comment>(?:%|//)[^\n]*|/\*.*?\*/)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<quoted>'(?:[^'\\]|\\.)*')
    | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<symbol><->|==>|->|::|\.\.|==|!=|<=|>=|[()\[\]{},.:;!?*<>=+\-/@#~^&|])
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    position: Position

    @property
    def value(self) -> str:
        """The text, unquoted (and its escapes decoded) for quoted names."""
        if self.kind == "quoted":
            return string_literal(self.text)
        return self.text

    @property
    def name(self) -> str:
        """The name as Refinery writes it in its output: quoted only if it must be,
        e.g. ``'My Class'``, or ``aA`` for ``'a\\u0041'``."""
        return refinery_name(self.value)


_EOF = "<end of input>"


def _tokens(source: str) -> list[_Token]:
    tokens = []
    line, line_start = 1, 0
    for match in _TOKEN.finditer(source):
        kind = match.lastgroup
        assert kind is not None
        text = match.group()
        if kind not in ("space", "comment"):
            tokens.append(_Token(kind, text, Position(line, match.start() - line_start + 1)))
        newlines = text.count("\n")
        if newlines:
            line += newlines
            line_start = match.start() + text.rindex("\n") + 1
    tokens.append(_Token("eof", _EOF, Position(line, len(source) - line_start + 1)))
    return tokens


# -- parser -------------------------------------------------------------------

_MODIFIERS = {"error", "shadow"}
_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}
_UNSUPPORTED_IN_BODY = {
    "count": "aggregation",
    "sum": "aggregation",
    "prod": "aggregation",
    "min": "aggregation",
    "max": "aggregation",
    "candidate": "the candidate modality",
}


def parse(source: str) -> Source:
    """The predicate declarations of ``source``, which may be a whole problem."""
    return _Parser(_tokens(source)).statements()


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.at = 0

    # -- helpers --

    def peek(self, ahead: int = 0) -> _Token:
        return self.tokens[min(self.at + ahead, len(self.tokens) - 1)]

    def next(self) -> _Token:
        token = self.peek()
        self.at = min(self.at + 1, len(self.tokens) - 1)
        return token

    def accept(self, text: str) -> bool:
        if self.peek().text == text and self.peek().kind not in ("string", "quoted"):
            self.next()
            return True
        return False

    def expect(self, text: str) -> _Token:
        token = self.peek()
        if token.text != text or token.kind in ("string", "quoted"):
            raise RuleError(f"expected {text!r}, found {token.text!r}", token.position)
        return self.next()

    def identifier(self) -> _Token:
        token = self.peek()
        if token.kind not in ("name", "quoted"):
            raise RuleError(f"expected a name, found {token.text!r}", token.position)
        return self.next()

    def qualified_name(self) -> str:
        """A name as Refinery writes it in its output, e.g. ``'My Class'::name``."""
        parts = [self.identifier().name]
        while self.peek().text == "::" and self.peek(1).kind in ("name", "quoted"):
            self.next()
            parts.append(self.identifier().name)
        return "::".join(parts)

    # -- statements --

    def statements(self) -> Source:
        predicates = []
        asserted: dict[str, Position] = {}
        in_classes: dict[str, Position] = {}
        while self.peek().kind != "eof":
            start = self.at
            modifiers = []
            while self.peek().text in _MODIFIERS and self.peek().kind == "name":
                modifiers.append(self.next().text)
            if self.peek().text == "pred" and self.peek().kind == "name":
                self.next()
                predicates.append(self.predicate(modifiers))
            elif modifiers == ["error"] and self.peek().kind in ("name", "quoted"):
                predicates.append(self.predicate(modifiers))  # ``error name(...) <-> ...``
            else:
                self.at = start
                self.skip_statement()
                statement = self.tokens[start : self.at]
                if (asserted_name := _asserted(statement)) is not None:
                    asserted.setdefault(asserted_name, statement[0].position)
                if _is_class(statement):
                    for token in statement:
                        if token.kind in ("name", "quoted"):
                            in_classes.setdefault(token.name, token.position)
        return Source(tuple(predicates), asserted, in_classes)

    def skip_statement(self) -> None:
        """Skip to the end of a statement: a ``.``, or the ``}`` closing a class or enum."""
        depth = 0
        while True:
            token = self.next()
            if token.kind == "eof":
                return
            if token.kind in ("string", "quoted"):
                continue
            if token.text in ("(", "[", "{"):
                depth += 1
            elif token.text in (")", "]", "}"):
                depth -= 1
                if depth == 0 and token.text == "}":
                    self.accept(".")
                    return
            elif token.text == "." and depth <= 0:
                return

    def predicate(self, modifiers: list[str]) -> Predicate:
        position = self.peek().position
        name = self.qualified_name()
        kind: PredicateKind = "pred"
        if "error" in modifiers:
            kind = "error"
        elif "shadow" in modifiers:
            kind = "shadow"
        self.expect("(")
        parameters = []
        if not self.accept(")"):
            while True:
                parameters.append(self.parameter())
                if self.accept(")"):
                    break
                self.expect(",")
        subsets = self.qualified_name() if self.accept("subsets") else None
        body = None
        if self.accept("<->"):
            body = self.body()
        self.expect(".")
        return Predicate(name, kind, tuple(parameters), body, position, subsets)

    def parameter(self) -> Parameter:
        if self.peek().text in ("in", "out") and self.peek(1).kind in ("name", "quoted"):
            raise UnsupportedRuleError("directed parameters", self.peek().position)
        first = self.qualified_name()
        if self.peek().kind in ("name", "quoted"):
            return Parameter(first, self.identifier().value)
        if "::" in first:
            raise RuleError(f"expected a parameter name after {first}", self.peek().position)
        return Parameter(None, first)

    # -- bodies --

    def body(self) -> tuple[tuple[Literal_, ...], ...]:
        clauses = [self.clause()]
        while self.accept(";"):
            clauses.append(self.clause())
        return tuple(clauses)

    def clause(self) -> tuple[Literal_, ...]:
        literals = [self.literal()]
        while self.accept(","):
            literals.append(self.literal())
        return tuple(literals)

    def literal(self) -> Literal_:
        token = self.peek()
        if token.kind == "name" and token.text in _UNSUPPORTED_IN_BODY:
            raise UnsupportedRuleError(_UNSUPPORTED_IN_BODY[token.text], token.position)
        if (
            token.text in ("true", "false")
            and token.kind == "name"
            and self.peek(1).text
            in (
                ",",
                ";",
                ".",
            )
        ):
            raise UnsupportedRuleError("constant literals", token.position)
        if self.accept("!"):
            literal = self.parenthesized() if self.peek().text == "(" else self.modal_atom()
            return _negate(literal)
        if self.peek().text == "(":
            return self.parenthesized()
        if self.peek().text in ("must", "may") and self.peek(1).kind in ("name", "quoted"):
            return self.modal_atom()
        left = self.term()
        if self.peek().text in _OPERATORS:
            return self.comparison(left, token.position)
        if isinstance(left, Call):
            return Atom(left.relation, left.arguments, False, None, False, left.position)
        if isinstance(left, _Transitive):
            return Atom(left.relation, left.arguments, False, None, True, left.position)
        raise RuleError(f"expected a literal, found {token.text!r}", token.position)

    def parenthesized(self) -> Literal_:
        start = self.expect("(")
        literal = self.literal()
        if self.peek().text in (",", ";"):
            raise UnsupportedRuleError("nested conjunctions or disjunctions", start.position)
        self.expect(")")
        return literal

    def modal_atom(self) -> Atom:
        modality: Modality | None = None
        if self.peek().text in ("must", "may") and self.peek(1).kind in ("name", "quoted"):
            modality = "must" if self.next().text == "must" else "may"
        if self.peek().text == "candidate":
            raise UnsupportedRuleError("the candidate modality", self.peek().position)
        term = self.term()
        if isinstance(term, Call):
            atom = Atom(term.relation, term.arguments, False, modality, False, term.position)
        elif isinstance(term, _Transitive):
            atom = Atom(term.relation, term.arguments, False, modality, True, term.position)
        else:
            raise RuleError("expected an atom", self.peek().position)
        if self.peek().text in _OPERATORS:
            raise RuleError("comparisons must be parenthesized after '!'", self.peek().position)
        return atom

    def comparison(self, left: Term, position: Position) -> Comparison:
        operator = self.next().text
        right = self.term()
        if self.peek().text in _OPERATORS:
            raise RuleError("comparisons don't chain", self.peek().position)
        for side in (left, right):
            if isinstance(side, _Transitive):
                raise RuleError("transitive closure in a comparison", position)
        return Comparison(left, operator, right, False, position)  # type: ignore[arg-type]

    def term(self) -> Term | _Transitive:
        token = self.peek()
        if token.text == "-" and self.peek(1).kind == "number":
            self.next()
            term: Term | _Transitive = Constant(_number("-" + self.next().text))
        elif token.kind == "number":
            term = Constant(_number(self.next().text))
        elif token.kind == "string":
            term = Constant(string_literal(self.next().text))
        elif token.kind == "name" and token.text in ("true", "false"):
            self.next()
            term = Constant(token.text == "true")
        elif token.kind in ("name", "quoted"):
            name = self.qualified_name()
            transitive = False
            if self.peek().text == "+" and self.peek(1).text == "(":
                self.next()
                transitive = True
            if self.peek().text == "(":
                arguments = self.arguments()
                term = (_Transitive if transitive else Call)(name, arguments, token.position)
            elif transitive:
                raise RuleError("expected '(' after '+'", self.peek().position)
            else:
                term = Name(name, token.position)
        elif token.text == "(":
            raise UnsupportedRuleError("parenthesized expressions", token.position)
        else:
            raise RuleError(f"expected a term, found {token.text!r}", token.position)
        if self.peek().text in ("+", "-", "*", "/"):
            raise UnsupportedRuleError("arithmetic", self.peek().position)
        if self.peek().text in ("{", "["):
            raise UnsupportedRuleError(f"{self.peek().text!r} in a body", self.peek().position)
        return term

    def arguments(self) -> tuple[Name, ...]:
        self.expect("(")
        arguments: list[Name] = []
        if self.accept(")"):
            return ()
        while True:
            token = self.peek()
            if token.kind not in ("name", "quoted"):
                if token.kind in ("number", "string"):
                    raise UnsupportedRuleError("constant arguments", token.position)
                if token.text == "*":
                    raise RuleError("'*' is only for assertions", token.position)
                raise RuleError(f"expected an argument, found {token.text!r}", token.position)
            arguments.append(Name(self.qualified_name(), token.position))
            if self.accept(")"):
                return tuple(arguments)
            self.expect(",")


_STATEMENT_KEYWORDS = {
    "abstract",
    "aggregator",
    "atom",
    "class",
    "concretization",
    "datatype",
    "declare",
    "decision",
    "enum",
    "import",
    "multi",
    "partial",
    "pred",
    "problem",
    "propagation",
    "rule",
    "scope",
    "shadow",
    "error",
}


def _asserted(statement: list[_Token]) -> str | None:
    """The relation an assertion is about, e.g. ``bar`` in ``default !bar(*).``"""
    at = 0
    if at < len(statement) and statement[at].text == "default":
        at += 1
    if at < len(statement) and statement[at].text in ("!", "?"):
        at += 1
    parts = []
    while at < len(statement) and statement[at].kind in ("name", "quoted"):
        parts.append(statement[at].name)
        at += 1
        if at < len(statement) and statement[at].text == "::":
            at += 1
        else:
            break
    if not parts or (statement[0].kind == "name" and parts[0] in _STATEMENT_KEYWORDS):
        return None
    if at < len(statement) and statement[at].text == "(":
        return "::".join(parts)
    return None


def _is_class(statement: list[_Token]) -> bool:
    leading = [t.text for t in statement[:2]]
    return "class" in leading


@dataclass(frozen=True)
class _Transitive:
    relation: str
    arguments: tuple[Name, ...]
    position: Position


def _negate(literal: Literal_) -> Literal_:
    match literal:
        case Atom():
            return Atom(
                literal.relation,
                literal.arguments,
                not literal.negated,
                literal.modality,
                literal.transitive,
                literal.position,
            )
        case Comparison():
            return Comparison(
                literal.left, literal.operator, literal.right, not literal.negated, literal.position
            )


# The keywords Refinery quotes when it writes a name: its grammar's keywords, but those
# its Identifier rule also takes as names (IdentifierValueConverter, Problem.xtext).
_QUOTED_KEYWORDS = frozenset(
    {
        *("abstract", "candidate", "class", "declare", "default", "enum", "error"),
        *("extends", "false", "import", "is", "may", "must", "partial", "pred"),
        *("refers", "rule", "scope", "true", "unknown", "__proto__"),
    }
)
_SIMPLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_JAVA_ESCAPES = {"\\": "\\\\", "'": "\\'", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}
_JAVA_ESCAPES |= {"\b": "\\b", "\f": "\\f"}


def refinery_name(value: str) -> str:
    """A name as Refinery writes it in its output (``IdentifierValueConverter``):
    as it is if it is an identifier and not a keyword, else quoted, Java-escaped."""
    if _SIMPLE_NAME.match(value) and value not in _QUOTED_KEYWORDS:
        return value
    return "'" + "".join(_JAVA_ESCAPES.get(c, c) for c in value) + "'"


_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}


def string_literal(text: str) -> str:
    """The value of a string literal as Refinery writes it, e.g. ``"a\\n\\u00e9"``.

    Escapes are Java's: ``\\n``, ``\\t``, ``\\uXXXX`` (UTF-16 code units, so a
    surrogate pair is one character), and a backslash before anything else.
    """

    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped.startswith("u"):
            return chr(int(escaped[1:], 16))
        return _ESCAPES.get(escaped, escaped)

    value = re.sub(r"\\(u[0-9a-fA-F]{4}|.)", replace, text[1:-1], flags=re.DOTALL)
    try:  # join surrogate pairs; a lone surrogate stays as it is
        return value.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        return value


def _number(text: str) -> int | Decimal:
    """A number as written: an int, or a real as a decimal (BigInteger and BigDecimal)."""
    try:
        return int(text)
    except ValueError:
        return Decimal(text)
