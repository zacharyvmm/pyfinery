# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Names and strings as Refinery writes them: identifiers quoted when they must be,
string literals with Java's escapes."""

from __future__ import annotations

import re

# The keywords Refinery quotes when it writes a name: its grammar's keywords, but those
# its Identifier rule also takes as names (IdentifierValueConverter, Problem.xtext).
QUOTED_KEYWORDS = frozenset(
    {
        *("abstract", "candidate", "class", "declare", "default", "enum", "error"),
        *("extends", "false", "import", "is", "may", "must", "partial", "pred"),
        *("refers", "rule", "scope", "true", "unknown", "__proto__"),
    }
)
SIMPLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
JAVA_ESCAPES = {"\\": "\\\\", "'": "\\'", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}
JAVA_ESCAPES |= {"\b": "\\b", "\f": "\\f"}


def refinery_name(value: str) -> str:
    """A name as Refinery writes it in its output (``IdentifierValueConverter``):
    as it is if it is an identifier and not a keyword, else quoted, Java-escaped."""
    if SIMPLE_NAME.match(value) and value not in QUOTED_KEYWORDS:
        return value
    return "'" + "".join(JAVA_ESCAPES.get(c, c) for c in value) + "'"


ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}


def string_literal(text: str) -> str:
    """The value of a string literal as Refinery writes it, e.g. ``"a\\n\\u00e9"``.

    Escapes are Java's: ``\\n``, ``\\t``, ``\\uXXXX`` (UTF-16 code units, so a
    surrogate pair is one character), and a backslash before anything else.
    """

    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped.startswith("u"):
            return chr(int(escaped[1:], 16))
        return ESCAPES.get(escaped, escaped)

    value = re.sub(r"\\(u[0-9a-fA-F]{4}|.)", replace, text[1:-1], flags=re.DOTALL)
    try:  # join surrogate pairs; a lone surrogate stays as it is
        return value.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        return value
