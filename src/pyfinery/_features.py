# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""How the fields of Pydantic model classes are described before they are mapped
to Refinery features.
"""

from __future__ import annotations

import enum
import types
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from ._options import ReferenceKind, RefineryNode

_DATA_TYPES: dict[type, str] = {bool: "boolean", int: "int", float: "real", str: "string"}


@dataclass(frozen=True, slots=True)
class UnionClass:
    """An abstract Refinery superclass standing for a union of model classes."""

    name: str
    members: tuple[type[BaseModel], ...]


@dataclass(frozen=True, slots=True)
class LiteralEnum:
    """A Refinery enum standing for a ``Literal`` of strings."""

    name: str
    values: tuple[str, ...]


ClassLike = type[BaseModel] | UnionClass
EnumLike = type[enum.Enum] | LiteralEnum


@dataclass(frozen=True, slots=True)
class Feature:
    """A field of a Pydantic model class mapped to Refinery."""

    owner: type[BaseModel]
    name: str
    target: ClassLike | EnumLike | str
    """Model class, union, enum, ``Literal`` enum, or data type name for attributes."""
    many: bool
    lower: int
    upper: int | None
    kind: ReferenceKind
    opposite: str | None
    by_key: bool = False
    """Whether the field holds keys of the target objects rather than the objects."""

    @property
    def is_attribute(self) -> bool:
        return isinstance(self.target, str)

    @property
    def is_enum(self) -> bool:
        return isinstance(self.target, LiteralEnum) or _is_enum_class(self.target)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == len(get_args(annotation)) - 1:
            return (args[0] if len(args) == 1 else Union[tuple(args)]), True  # noqa: UP007
    return annotation, False


def _strip_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        inner, more = _strip_annotated(base)
        return inner, [*metadata, *more]
    return annotation, []


_COLLECTIONS = (list, set, frozenset, tuple, Sequence, Iterable)


def _is_enum_class(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, enum.Enum)


def _is_model_class(value: Any) -> bool:
    """Pydantic model classes that can be mapped to Refinery classes."""
    return (
        isinstance(value, type)
        and issubclass(value, BaseModel)
        and value not in (BaseModel, RefineryNode)
    )


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


class _Unsupported(TypeError):
    """A field type with no Refinery counterpart (skipped with ``unsupported="skip"``)."""
