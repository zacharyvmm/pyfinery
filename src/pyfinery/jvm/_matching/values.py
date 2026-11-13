"""Attribute values, between Python and Refinery.

Refinery keeps the value of an attribute in an abstract domain: an interval
of integers or of reals, a four-valued truth value, a string that may be
unknown. A plain model holds each value as the exact element of its domain,
``7`` as the interval ``[7, 7]``, so the comparisons and arithmetic Refinery's
compiler emits work on it unchanged, and a check holds when it is certainly
true. An attribute with no value reads as the domain's ``unknown``, of which
no check is certain.
"""

from __future__ import annotations

import math
from typing import Any

from ..._mapping import AttributeType
from .classes import java

NO_VALUE = object()
"""The value of an attribute an object of its class has no value for: unknown."""

_DOMAINS = {"boolean": "TruthValue", "int": "IntInterval", "real": "RealInterval"}


def to_java(data_type: AttributeType, value: object) -> Any:
    """The exact Refinery value of ``value``, an attribute of type ``data_type``."""
    j = java()
    if value is NO_VALUE:
        return getattr(j, _DOMAINS.get(data_type, "StringValue")).UNKNOWN
    match data_type:
        case "boolean":
            return j.TruthValue.TRUE if value else j.TruthValue.FALSE
        case "int":
            return j.IntInterval.of(j.BigInteger(str(int(value))))
        case "real":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Refinery cannot represent the real value {number!r}")
            return j.RealInterval.of(repr(number))
        case _:
            return j.StringValue.of(str(value))
