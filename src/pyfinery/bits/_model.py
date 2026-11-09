# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The partial model predicates are evaluated over: nodes, and the values of base relations."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..models import ErrorValue, JsonOutput, UnknownValue

LogicValue = Literal["true", "false", "unknown", "error"]
NodeKind = Literal["default", "atom", "multi"]
DataType = Literal["int", "real", "string", "boolean"]

EXISTS = "builtin::exists"
EQUALS = "builtin::equals"


@dataclass(frozen=True)
class Node:
    name: str
    simple_name: str
    kind: NodeKind = "default"
    """``atom`` nodes (atoms and enum literals) are the ones rules may name."""


def _tuples(arity: int, keys: list[tuple[int, ...]]) -> NDArray[np.intp]:
    return np.array(keys, dtype=np.intp).reshape(len(keys), arity)


@dataclass(frozen=True)
class Relation:
    """A base relation: the tuples that must hold and those that may.

    A tuple that is ``true`` is in both; ``unknown``, only in ``may``;
    ``error``, only in ``must``; ``false``, in neither.
    """

    name: str
    arity: int
    must: NDArray[np.intp]
    may: NDArray[np.intp]
    simple_name: str = ""

    @classmethod
    def of(
        cls,
        name: str,
        arity: int,
        values: Iterable[tuple[tuple[int, ...], LogicValue]],
        simple_name: str | None = None,
    ) -> Relation:
        """A relation from ``(tuple, value)`` pairs; tuples that aren't given are false."""
        must, may = [], []
        for key, value in values:
            if len(key) != arity:
                raise ValueError(f"{name}: {key} has arity {len(key)}, not {arity}")
            if value in ("true", "error"):
                must.append(tuple(key))
            if value in ("true", "unknown"):
                may.append(tuple(key))
        return cls(
            name, arity, _tuples(arity, must), _tuples(arity, may), simple_name or _simple(name)
        )


class _Error:
    def __repr__(self) -> str:
        return "Attribute.ERROR"


Number = int | Decimal | float
"""Exact, as Refinery's BigInteger and BigDecimal bounds; ``float`` only for infinities,
or values a float holds exactly."""


class Attribute:
    """The values of an attribute, as an interval per node: for booleans, ``false``
    is 0 and ``true`` 1; for strings, a known value is ``[0, 0]`` and in ``strings``.

    An unknown value is ``[-inf, inf]``. A node outside the attribute's class
    has no value, which Refinery treats as an error: the empty ``[inf, -inf]``.
    Bounds are float64 arrays when every value is a float exactly, and object
    arrays of exact ``int`` and ``Fraction`` otherwise.
    """

    ERROR = _Error()
    """The value of an attribute that is an error, for :meth:`of`."""

    def __init__(
        self,
        name: str,
        data_type: DataType,
        lower: NDArray,
        upper: NDArray,
        strings: tuple[str | None, ...] = (),
        simple_name: str = "",
    ) -> None:
        self.name = name
        self.data_type = data_type
        self.lower = lower
        self.upper = upper
        self.strings = strings
        self.simple_name = simple_name or _simple(name)

    def __repr__(self) -> str:
        return f"Attribute({self.name!r}, {self.data_type!r})"

    @classmethod
    def of(
        cls,
        name: str,
        data_type: DataType,
        n: int,
        values: Iterable[tuple[int, object]],
        simple_name: str | None = None,
    ) -> Attribute:
        """An attribute from ``(node, value)`` pairs, given for the nodes of its class.

        A value is ``None`` for unknown, :attr:`ERROR`, or else: a string for a
        string attribute; a bool for a boolean one; and for numbers an ``int``,
        a ``float`` or ``Decimal`` (reals), or text as Refinery writes it:
        ``"5"``, ``"0.25"``, an interval ``"3..9"`` or ``"*..5"``,
        ``"unknown"`` or ``"error"``. As in Refinery, a real is a decimal (a
        ``float``, the one its ``repr`` writes, which is what Refinery is sent)
        rounded outward to 16 significant digits. Nodes not given have no
        value (error).
        """
        lower: list[Number] = [math.inf] * n
        upper: list[Number] = [-math.inf] * n
        strings: list[str | None] = [None] * n
        for node, value in values:
            if data_type == "string" and isinstance(value, str):
                strings[node] = value
                lower[node] = upper[node] = 0
                continue
            lower[node], upper[node] = _interval(value, data_type)
        bounds = [b for b in (*lower, *upper) if not isinstance(b, float)]
        dtype = np.float64 if all(float(b) == b for b in bounds) else object
        return cls(
            name,
            data_type,
            np.array(lower, dtype=dtype),
            np.array(upper, dtype=dtype),
            tuple(strings) if data_type == "string" else (),
            simple_name or _simple(name),
        )


_FLOOR = Context(prec=16, rounding=ROUND_FLOOR)
_CEILING = Context(prec=16, rounding=ROUND_CEILING)


def real_interval(value: Decimal) -> tuple[Decimal, Decimal]:
    """A real as Refinery's ``RealInterval.of`` holds it: rounded outward to 16 digits."""
    return _FLOOR.plus(value), _CEILING.plus(value)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"not a number: {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Refinery cannot represent the number {value!r}")
        return Decimal(repr(value))
    if isinstance(value, int | Decimal):
        return Decimal(value)
    return Decimal(str(value).strip())


def _integer(value: object) -> int:
    if isinstance(value, bool | float | Decimal):
        raise ValueError(f"not an integer: {value!r}")
    return int(value) if isinstance(value, int) else int(str(value).strip())


def _interval(value: object, data_type: DataType) -> tuple[Number, Number]:
    if value is None or value == "unknown":
        return -math.inf, math.inf
    if value is Attribute.ERROR or value == "error":
        return math.inf, -math.inf
    if data_type == "boolean":
        if value in (True, "true"):
            return 1, 1
        if value in (False, "false"):
            return 0, 0
        raise ValueError(f"not a boolean: {value!r}")
    if data_type == "string":
        raise ValueError(f"not a string: {value!r}")
    lower: object = value
    upper: object = value
    if isinstance(value, str) and ".." in value:
        lower, _, upper = value.partition("..")
    if data_type == "int":
        low = -math.inf if str(lower).strip() == "*" else _integer(lower)
        high = math.inf if str(upper).strip() == "*" else _integer(upper)
        return low, high
    low = -math.inf if str(lower).strip() == "*" else real_interval(_decimal(lower))[0]
    high = math.inf if str(upper).strip() == "*" else real_interval(_decimal(upper))[1]
    return low, high


def _simple(name: str) -> str:
    return name.rsplit("::", 1)[-1]


@dataclass
class PartialModel:
    """Nodes, base relations and attributes, by name.

    Without ``builtin::exists``, every node exists; without ``builtin::equals``,
    each node equals itself and nothing else.
    """

    nodes: Sequence[Node]
    relations: Sequence[Relation] = ()
    attributes: Sequence[Attribute] = ()
    _relations: dict[str, Relation] = field(init=False, repr=False)
    _attributes: dict[str, Attribute] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._relations = {}
        for relation in self.relations:
            if relation.name in self._relations:
                raise ValueError(f"two relations named {relation.name!r}")
            self._relations[relation.name] = relation
        self._attributes = {a.name: a for a in self.attributes}
        n = len(self.nodes)
        if EXISTS not in self._relations:
            self._relations[EXISTS] = Relation.of(EXISTS, 1, (((i,), "true") for i in range(n)))
        if EQUALS not in self._relations:
            self._relations[EQUALS] = Relation.of(EQUALS, 2, (((i, i), "true") for i in range(n)))

    @property
    def size(self) -> int:
        return len(self.nodes)

    def relation_named(self, name: str) -> Relation | None:
        """A relation by name, or by simple name if only one has it.

        ``exists`` and ``equals`` are the builtin ones.
        """
        if name in ("exists", "equals"):
            name = f"builtin::{name}"
        return _lookup(self._relations, name)

    def attribute_named(self, name: str) -> Attribute | None:
        """An attribute by name, or by simple name if only one has it."""
        return _lookup(self._attributes, name)

    def atom_named(self, name: str) -> int | None:
        """The index of the atom (or enum literal) ``name``, by name or unique simple name."""
        found = [
            i for i, node in enumerate(self.nodes) if node.kind == "atom" and node.name == name
        ]
        if not found:
            found = [
                i
                for i, node in enumerate(self.nodes)
                if node.kind == "atom" and node.simple_name == name
            ]
        if len(found) > 1:
            raise ValueError(f"atom {name!r} is ambiguous")
        return found[0] if found else None

    @classmethod
    def from_json(cls, output: JsonOutput, *, exclude: Iterable[str] = ()) -> PartialModel:
        """The model in Refinery's JSON output, keeping objects that don't exist.

        Relations named in ``exclude`` (e.g. the predicates to evaluate) and
        ``computed`` ones are left out.
        """
        excluded = set(exclude)
        n = len(output.nodes)
        nodes = [Node(node.name, node.simple_name, node.kind) for node in output.nodes]
        relations, attributes = [], []
        for metadata in output.relations:
            if metadata.name in excluded or metadata.detail.type in ("computed", "function"):
                continue
            raw = output.partial_interpretation.get(metadata.name, [])
            arity = metadata.arity
            data_type = (metadata.data_type or "").removeprefix("builtin::")
            if metadata.detail.type == "attribute" and data_type != "boolean":
                if data_type not in ("int", "real", "string"):
                    continue
                values = [
                    (int(t[0]), _attribute_value(t[arity], data_type))  # type: ignore[arg-type]
                    for t in raw
                ]
                attributes.append(
                    Attribute.of(metadata.name, data_type, n, values, metadata.simple_name)  # type: ignore[arg-type]
                )
                continue
            keyed = [(tuple(int(i) for i in t[:arity]), _logic_value(t[arity])) for t in raw]  # type: ignore[arg-type]
            if data_type == "boolean":
                # A boolean attribute is an error on the nodes it isn't listed for.
                listed = {key for key, _ in keyed}
                keyed += [((i,), "error") for i in range(n) if (i,) not in listed]
            relations.append(Relation.of(metadata.name, arity, keyed, metadata.simple_name))
            if data_type == "boolean":
                values = [(key[0], value) for key, value in keyed]
                attributes.append(
                    Attribute.of(metadata.name, "boolean", n, values, metadata.simple_name)
                )
        return cls(nodes, relations, attributes)


def _lookup[T: (Relation, Attribute)](table: dict[str, T], name: str) -> T | None:
    found = table.get(name)
    if found is not None:
        return found
    matches = [item for item in table.values() if item.simple_name == name]
    if len(matches) > 1:
        names = ", ".join(sorted(item.name for item in matches))
        raise ValueError(f"{name!r} is ambiguous: {names}")
    return matches[0] if matches else None


def _attribute_value(value: object, data_type: str) -> object:
    match value:
        case UnknownValue():
            return value.unknown
        case ErrorValue():
            return "error"
        case str() if data_type == "string" and value.startswith('"'):
            return _unquote(value)  # strings are quoted; ``unknown`` and ``error`` aren't
        case "unknown":
            return None
        case "error":
            return Attribute.ERROR
        case _:
            return value


def _unquote(text: str) -> str:
    """A string literal as Refinery writes it, e.g. ``"a \\"b\\""``."""
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}

    def replace(match: re.Match[str]) -> str:
        escaped = match.group(1)
        if escaped.startswith("u"):
            return chr(int(escaped[1:], 16))
        return escapes.get(escaped, escaped)

    return re.sub(r"\\(u[0-9a-fA-F]{4}|.)", replace, text[1:-1])


def _logic_value(value: object) -> LogicValue:
    if value not in ("true", "false", "unknown", "error"):
        raise ValueError(f"not a logic value: {value!r}")
    return value  # type: ignore[return-value]
