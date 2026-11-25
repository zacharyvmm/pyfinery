# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Reading a field's annotation: ``Annotated``, ``Optional`` and collections taken off.

Both ``_options`` (defaults of marked fields) and ``_schema`` (Refinery features)
read annotations, so this module imports neither.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin

import annotated_types
from pydantic.fields import FieldInfo

COLLECTIONS = (list, set, frozenset, tuple, Sequence, Iterable)
_CONCRETE_COLLECTIONS = (list, set, frozenset, tuple)

# ``None`` is a type (``list[None]``), so a collection without exactly one item type
# needs a marker of its own.
NO_ITEM_TYPE: Any = object()


@dataclass(frozen=True, slots=True)
class Shape:
    """A field's annotation with ``Annotated``, ``Optional`` and a collection taken off."""

    type: Any
    """The type of the value, or of each item; :data:`NO_ITEM_TYPE` for a collection
    without exactly one item type."""
    optional: bool
    many: bool
    metadata: list[Any]
    """``Annotated`` metadata of the field, outside ``Optional``."""
    inner_metadata: list[Any]
    """``Annotated`` metadata inside ``Optional``, and of the items of a collection."""

    @classmethod
    def of(cls, info: FieldInfo) -> Shape:
        annotation, outer = strip_annotated(info.annotation)
        metadata = [*info.metadata, *outer]
        annotation, optional = strip_optional(annotation)
        annotation, inner_metadata = strip_annotated(annotation)
        many = get_origin(annotation) in COLLECTIONS or annotation in _CONCRETE_COLLECTIONS
        if many:
            items = [a for a in get_args(annotation) if a is not Ellipsis]
            annotation = NO_ITEM_TYPE
            if len(items) == 1:
                annotation, item_metadata = strip_annotated(items[0])
                inner_metadata += item_metadata
        return cls(annotation, optional, many, metadata, inner_metadata)

    def bounds(self, metadata: list[Any]) -> tuple[int, int | None]:
        """The multiplicity: from ``MinLen``/``MaxLen`` in ``metadata`` for a collection."""
        if not self.many:
            return (0 if self.optional else 1), 1
        min_lengths = (m.min_length for m in metadata if isinstance(m, annotated_types.MinLen))
        max_lengths = (m.max_length for m in metadata if isinstance(m, annotated_types.MaxLen))
        return next(min_lengths, 0), next(max_lengths, None)


def empty_factory(info: FieldInfo) -> Callable[[], Any] | None:
    """What a reference field holds when left out: an empty collection of its type, ``None``
    if it is optional; ``None`` (no factory) for a required single reference."""
    shape = Shape.of(info)
    if shape.many:
        collection, _ = strip_optional(strip_annotated(info.annotation)[0])
        origin = get_origin(collection) or collection
        return origin if origin in _CONCRETE_COLLECTIONS else list
    if shape.optional:
        return _none
    return None


def _none() -> None:
    return None


def strip_optional(annotation: Any) -> tuple[Any, bool]:
    if is_union(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == len(get_args(annotation)) - 1:
            return (args[0] if len(args) == 1 else Union[tuple(args)]), True  # noqa: UP007
    return annotation, False


def strip_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        inner, more = strip_annotated(base)
        return inner, [*metadata, *more]
    return annotation, []


def is_union(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)
