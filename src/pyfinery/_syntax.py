# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Writing Refinery source: identifiers, literals, and multiplicities."""

from __future__ import annotations

import math
import re
from typing import Any

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEYWORDS = frozenset(
    [
        "abstract",
        "aggregator",
        "as",
        "assert",
        "atom",
        "candidate",
        "class",
        "concretization",
        "container",
        "contains",
        "datatype",
        "decision",
        "declare",
        "default",
        "e",
        "E",
        "enum",
        "error",
        "extends",
        "false",
        "import",
        "is",
        "may",
        "module",
        "multi",
        "must",
        "opposite",
        "partial",
        "pred",
        "primitive",
        "problem",
        "propagation",
        "refers",
        "rule",
        "scope",
        "shadow",
        "subsets",
        "theory",
        "true",
        "unknown",
        "using",
    ]
)


def quote_id(name: str) -> str:
    """Render ``name`` as a Refinery identifier, quoting it if necessary."""
    if _ID.match(name) and name not in _KEYWORDS:
        return name
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def encodable(text: str) -> bool:
    """Whether ``text`` can be sent as JSON: not if it holds a lone surrogate."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _string_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    if not encodable(escaped):
        # Refinery reads ``\uXXXX`` as a UTF-16 code unit, as Java does.
        escaped = "".join(c if encodable(c) else f"\\u{ord(c):04x}" for c in escaped)
    return f'"{escaped}"'


def attribute_literal(value: Any, data_type: str) -> str:
    match data_type:
        case "boolean":
            return "true" if value else "false"
        case "int":
            return str(int(value))
        case "real":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Refinery cannot represent the real value {number!r}")
            text = repr(number)
            return text if ("." in text or "e" in text) else f"{text}.0"
        case _:
            return _string_literal(str(value))


def multiplicity(lower: int, upper: int | None) -> str:
    if upper == lower:
        return f"[{lower}]"
    return f"[{lower}..{'*' if upper is None else upper}]"


def parse_multiplicity(value: str | tuple[int, int | None]) -> tuple[int, int | None]:
    if isinstance(value, tuple):
        return value
    text = value.strip().strip("[]")
    if not text:
        return 0, None
    lower_text, _, upper_text = text.partition("..")
    lower = int(lower_text)
    if not upper_text:
        return lower, lower
    return lower, None if upper_text.strip() == "*" else int(upper_text)


# Builtin symbols a node name shouldn't shadow.
RESERVED_NODE_NAMES = frozenset(
    {
        *("node", "exists", "equals", "count"),
        *("contains", "contained", "container"),
        *("int", "real", "string", "boolean"),
    }
)
