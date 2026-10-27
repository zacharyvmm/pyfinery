# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The Refinery types derived from Pydantic model classes: which classes, unions
and enums there are, and the :class:`Feature` each field maps to.
"""

from __future__ import annotations

import enum
import inspect
import types
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ._options import ClassOptions, Ref, ReferenceKind, RefineryNode
from ._syntax import parse_multiplicity, quote_id

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


@dataclass(slots=True)
class Schema:
    """The Refinery types of some model classes, and what decides how they map."""

    options: dict[type, ClassOptions]
    """:class:`ClassOptions` by class, as given to :class:`Metamodel`."""
    named_unions: dict[str, tuple[type[BaseModel], ...]]
    """Unions by the name given to :class:`Metamodel`."""
    classes: list[type[BaseModel]] = field(default_factory=list)
    """Superclasses before subclasses."""
    unions: list[UnionClass] = field(default_factory=list)
    enums: list[EnumLike] = field(default_factory=list)
    features: dict[type[BaseModel], list[Feature]] = field(default_factory=dict)
    """The features declared in each class (inherited ones are its superclass's)."""
    skipped: list[tuple[type[BaseModel], str, str]] = field(default_factory=list)
    """Fields left out: ``(class, field, reason)``."""
    type_names: set[str] = field(default_factory=set)
    """The Refinery names of the classes, unions and enums, which nodes can't take."""

    def options_for(self, cls: type) -> ClassOptions:
        """The options of the nearest class in the MRO of ``cls`` that has some."""
        for base in cls.__mro__:
            if base in self.options:
                return self.options[base]
        return _DEFAULT_OPTIONS

    def is_entity(self, cls: type) -> bool:
        """Whether instances are shared nodes (RefineryNode or keyed), not owned values."""
        return issubclass(cls, RefineryNode) or self.options_for(cls).key is not None

    def is_abstract(self, cls: type) -> bool:
        abstract = self.options[cls].abstract if cls in self.options else None
        return bool(cls.__dict__.get("__abstract__", False)) if abstract is None else abstract

    def identity(self, node: BaseModel) -> Hashable:
        """What makes ``node`` one Refinery node: its key if its class has one, else itself."""
        key = self.options_for(type(node)).key
        return (type(node), key(node)) if key is not None else id(node)

    def find_feature(self, cls: type, name: str) -> Feature | None:
        """The feature named ``name`` of ``cls``, declared in it or inherited."""
        for base in cls.__mro__:
            for feature in self.features.get(base, []):  # type: ignore[call-overload]
                if feature.name == name:
                    return feature
        return None

    def subclasses(self, cls: type) -> list[type[BaseModel]]:
        """Classes of the metamodel strictly below ``cls``."""
        return [c for c in self.classes if c is not cls and issubclass(c, cls)]

    def union_named(self, name: str) -> UnionClass | None:
        return next((u for u in self.unions if u.name == name), None)


_DEFAULT_OPTIONS = ClassOptions()


def class_name(node_class: type | UnionClass | LiteralEnum) -> str:
    """The Refinery name of a class, union, or enum."""
    if isinstance(node_class, UnionClass | LiteralEnum):
        return quote_id(node_class.name)
    return quote_id(node_class.__name__)


def discover(
    classes: Iterable[type[BaseModel]],
    options: Mapping[type, ClassOptions] | None,
    unions: Mapping[str, Iterable[type[BaseModel]]] | None,
    unsupported: Literal["error", "skip"],
) -> Schema:
    """The Refinery types of ``classes`` and of the classes reachable from them.

    Those are their superclasses, the classes their fields refer to, and the
    already-imported subclasses of those (but not every subclass of a shared
    base class).
    """
    named_unions = {name: tuple(members) for name, members in (unions or {}).items()}
    schema = Schema(dict(options or {}), named_unions)
    # (class, whether to include its subclasses): roots and field types are
    # expanded downwards, superclasses only upwards, so that a base class
    # shared by unrelated models doesn't pull all of them in.
    pending: list[tuple[type[BaseModel], bool]] = [(cls, True) for cls in classes]
    expanded: set[type[BaseModel]] = set()
    while pending:
        cls, downwards = pending.pop(0)
        if cls is RefineryNode:
            continue
        if not is_model_class(cls):
            raise TypeError(f"{cls!r} is not a RefineryNode subclass or Pydantic model")
        if cls not in schema.features:
            _rebuild(cls)
            schema.classes.append(cls)
            schema.features[cls] = _features(schema, cls, unsupported)
            pending.extend((b, False) for b in cls.__bases__ if is_model_class(b))
            for feature in schema.features[cls]:
                target = feature.target
                if feature.is_enum:
                    if target not in schema.enums:
                        schema.enums.append(target)  # type: ignore[arg-type]
                elif isinstance(target, UnionClass):
                    if target not in schema.unions:
                        schema.unions.append(target)
                    pending.extend((member, True) for member in target.members)
                elif not feature.is_attribute:
                    pending.append((target, True))  # type: ignore[arg-type]
        if downwards and cls not in expanded:
            expanded.add(cls)
            pending.extend((sub, True) for sub in cls.__subclasses__())
    # Superclasses must be declared before subclasses for readability.
    schema.classes.sort(key=lambda c: (len(c.__mro__), c.__qualname__))
    schema.type_names = {class_name(t) for t in [*schema.classes, *schema.unions, *schema.enums]}
    return schema


def check(schema: Schema) -> None:
    """Raise ``TypeError`` for what Refinery can't represent: two types with one
    name, mismatched opposites, and references by key to classes without a key."""
    _check_names(schema)
    _check_opposites(schema)
    _check_key_references(schema)


# -- fields -----------------------------------------------------------------


def _features(schema: Schema, cls: type[BaseModel], unsupported: str) -> list[Feature]:
    exclude = set(schema.options_for(cls).exclude)
    features = []
    for name in _own_fields(cls):
        if name in exclude:
            continue
        try:
            feature = _feature(schema, cls, name)
        except _Unsupported as e:
            if unsupported == "error":
                raise TypeError(
                    f"{e} (exclude the field with ClassOptions(exclude=...) or pass "
                    "unsupported='skip')"
                ) from None
            schema.skipped.append((cls, name, str(e)))
            continue
        if feature is not None:
            features.append(feature)
    return features


def _feature(schema: Schema, cls: type[BaseModel], name: str) -> Feature | None:
    by_key = schema.options_for(cls).references or {}
    if name in by_key:
        return _key_reference(schema, cls, name, by_key[name])
    info = cls.model_fields[name]
    where = f"{cls.__name__}.{name}"
    shape = _Shape.of(info)
    annotation = shape.type
    metadata = [*shape.metadata, *shape.inner_metadata]
    if shape.many:
        if annotation is None:
            raise TypeError(f"{where}: collections must have exactly one item type")
    elif get_origin(annotation) not in (None, Literal) and not _is_union(annotation):
        raise _Unsupported(f"{where}: unsupported field type {info.annotation!r}")

    refs = [m for m in metadata if isinstance(m, Ref)]
    if len(refs) > 1:
        raise TypeError(f"{where}: more than one Ref/Contains/Container marker")
    ref = refs[0] if refs else Ref()

    if annotation in _DATA_TYPES:
        if shape.many:
            raise _Unsupported(f"{where}: Refinery attributes can't hold collections")
        if refs:
            raise TypeError(f"{where}: attributes can't have Ref markers")
        return Feature(cls, name, _DATA_TYPES[annotation], False, 0, 1, "refers", None)

    target: ClassLike | EnumLike
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not all(isinstance(v, str) for v in values):
            raise _Unsupported(f"{where}: only Literals of strings map to Refinery enums")
        if len(values) == 1:
            # A constant, e.g. a discriminator tag: the class says it all.
            return None
        target = _literal_enum(schema, f"{cls.__name__}{_camel(name)}", values)
    elif _is_union(annotation):
        target = _union(schema, where, annotation)
    elif is_model_class(annotation) or _is_enum_class(annotation):
        target = annotation
    else:
        raise _Unsupported(
            f"{where}: unsupported field type {annotation!r}; use a RefineryNode subclass, "
            "a Pydantic model, a union of them, an Enum, a Literal of strings, int, float, "
            "str, or bool"
        )
    if isinstance(target, LiteralEnum) or _is_enum_class(target):
        if ref.kind != "refers":
            raise TypeError(f"{where}: enums cannot be containment references")
        kind: ReferenceKind = "refers"
    else:
        members = target.members if isinstance(target, UnionClass) else (target,)
        entities = [schema.is_entity(m) for m in members]  # type: ignore[arg-type]
        if any(entities) and not all(entities):
            raise TypeError(
                f"{where}: mixes owned value classes with entity classes (RefineryNode or "
                "keyed); Refinery needs them all to be one or the other"
            )
        kind = ref.kind
        if not any(entities):
            # Plain models without a key are value objects owned by this object.
            if ref.kind == "container" or ref.opposite is not None:
                shown = getattr(target, "name", None) or target.__name__  # type: ignore[union-attr]
                raise TypeError(
                    f"{where}: {shown} is a plain Pydantic model, which is always "
                    "contained in its owner and cannot have opposites; make it a RefineryNode "
                    "(or give it a key) to use opposites or Container()"
                )
            kind = "contains"

    lower, upper = shape.bounds(metadata)
    if ref.multiplicity is not None:
        lower, upper = parse_multiplicity(ref.multiplicity)
    if ref.kind == "container":
        if shape.many:
            raise TypeError(f"{where}: a container reference must be a single node")
        if ref.opposite is None:
            raise TypeError(f"{where}: a container reference needs an opposite")
    return Feature(cls, name, target, shape.many, lower, upper, kind, ref.opposite)


def _key_reference(
    schema: Schema,
    cls: type[BaseModel],
    name: str,
    spec: type[BaseModel] | Sequence[type[BaseModel]] | str,
) -> Feature:
    where = f"{cls.__name__}.{name}"
    shape = _Shape.of(cls.model_fields[name])
    if shape.type not in (str, int):
        raise TypeError(
            f"{where}: a reference by key must hold str or int keys, or a collection of them"
        )
    target: ClassLike
    if isinstance(spec, str):
        if spec not in schema.named_unions:
            raise TypeError(f"{where}: no union named {spec!r} was given to Metamodel(unions=)")
        target = _union_of(schema, schema.named_unions[spec])
    elif isinstance(spec, type):
        if not is_model_class(spec):
            raise TypeError(f"{where}: {spec!r} is not a Pydantic model class")
        target = spec
    else:
        target = _union_of(schema, tuple(spec))
    lower, upper = shape.bounds(shape.metadata)
    return Feature(cls, name, target, shape.many, lower, upper, "refers", None, by_key=True)


@dataclass(frozen=True, slots=True)
class _Shape:
    """A field's annotation with ``Annotated``, ``Optional`` and a collection taken off."""

    type: Any
    """The type of the value, or of each item; ``None`` for a collection without
    exactly one item type."""
    optional: bool
    many: bool
    metadata: list[Any]
    """``Annotated`` metadata of the field, outside ``Optional``."""
    inner_metadata: list[Any]
    """``Annotated`` metadata inside ``Optional``, and of the items of a collection."""

    @classmethod
    def of(cls, info: FieldInfo) -> _Shape:
        annotation, metadata = _strip_annotated(info.annotation)
        metadata = [*info.metadata, *metadata]
        annotation, optional = _strip_optional(annotation)
        annotation, inner_metadata = _strip_annotated(annotation)
        many = get_origin(annotation) in _COLLECTIONS or annotation in (list, set, frozenset, tuple)
        if many:
            args = [a for a in get_args(annotation) if a is not Ellipsis]
            annotation = None
            if len(args) == 1:
                annotation, more = _strip_annotated(args[0])
                inner_metadata += more
        return cls(annotation, optional, many, metadata, inner_metadata)

    def bounds(self, metadata: list[Any]) -> tuple[int, int | None]:
        """The multiplicity: from ``MinLen``/``MaxLen`` in ``metadata`` for a collection."""
        if not self.many:
            return (0 if self.optional else 1), 1
        lower = next((m.min_length for m in metadata if isinstance(m, annotated_types.MinLen)), 0)
        upper = next(
            (m.max_length for m in metadata if isinstance(m, annotated_types.MaxLen)), None
        )
        return lower, upper


def _own_fields(node_class: type[BaseModel]) -> list[str]:
    """Fields first declared in ``node_class`` (inherited ones belong to a superclass)."""
    own = inspect.get_annotations(node_class)
    inherited = {
        name
        for base in node_class.__mro__[1:]
        if is_model_class(base)
        for name in base.model_fields  # type: ignore[union-attr]
    }
    return [name for name in node_class.model_fields if name in own and name not in inherited]


def _rebuild(node_class: type[BaseModel]) -> None:
    if not node_class.__pydantic_complete__:
        try:
            node_class.model_rebuild()
        except Exception as e:
            raise TypeError(
                f"Cannot resolve the field types of {node_class.__name__}; define all "
                f"referenced classes before building the Metamodel ({e})"
            ) from e


def _literal_enum(schema: Schema, name: str, values: tuple[str, ...]) -> LiteralEnum:
    for existing in schema.enums:
        if isinstance(existing, LiteralEnum) and existing.values == values:
            return existing
    return LiteralEnum(name, values)


def _union(schema: Schema, where: str, annotation: Any) -> UnionClass:
    members = []
    for arg in get_args(annotation):
        arg, _ = _strip_annotated(arg)
        if not is_model_class(arg):
            raise _Unsupported(f"{where}: unions must only contain Pydantic model classes")
        members.append(arg)
    return _union_of(schema, tuple(members))


def _union_of(schema: Schema, members: tuple[type[BaseModel], ...]) -> UnionClass:
    key = frozenset(members)
    for existing in schema.unions:
        if frozenset(existing.members) == key:
            return existing
    names = {frozenset(m): name for name, m in schema.named_unions.items()}
    return UnionClass(names.get(key) or "Or".join(m.__name__ for m in members), members)


# -- checks -----------------------------------------------------------------


def _check_names(schema: Schema) -> None:
    seen: dict[str, object] = {}
    named: list[tuple[str, object]] = [(c.__name__, c) for c in schema.classes]
    named += [(u.name, u) for u in schema.unions]
    named += [(e.name if isinstance(e, LiteralEnum) else e.__name__, e) for e in schema.enums]
    for name, owner in named:
        if seen.setdefault(name, owner) is not owner:
            raise TypeError(
                f"Two different types are both named {name!r}; Refinery needs unique names"
            )


def _check_key_references(schema: Schema) -> None:
    for features in schema.features.values():
        for feature in features:
            if not feature.by_key:
                continue
            target = feature.target
            members = target.members if isinstance(target, UnionClass) else (target,)
            for member in members:
                if schema.options_for(member).key is None:  # type: ignore[arg-type]
                    raise TypeError(
                        f"{feature.owner.__name__}.{feature.name}: references by key need "
                        f"ClassOptions(key=...) on {member.__name__}"  # type: ignore[union-attr]
                    )


def _check_opposites(schema: Schema) -> None:
    for features in schema.features.values():
        for feature in features:
            if feature.opposite is None:
                continue
            where = f"{feature.owner.__name__}.{feature.name}"
            if isinstance(feature.target, UnionClass):
                raise TypeError(f"{where}: references to unions can't have opposites")
            other = schema.find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
            if other is None:
                raise TypeError(
                    f"{where}: opposite {feature.opposite!r} is not a field of "
                    f"{feature.target.__name__}"  # type: ignore[union-attr]
                )
            if other.opposite != feature.name:
                raise TypeError(
                    f"{where}: {other.owner.__name__}.{other.name} must declare "
                    f"opposite={feature.name!r} too"
                )
            if feature.kind == "contains" and other.kind != "container":
                raise TypeError(
                    f"{where}: the opposite of a containment reference must use Container()"
                )
            if feature.kind == "container" and other.kind != "contains":
                raise TypeError(
                    f"{where}: the opposite of a container reference must use Contains(); "
                    f"{other.owner.__name__}.{other.name} is not a containment reference"
                )


# -- annotations --------------------------------------------------------------


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


def is_model_class(value: Any) -> bool:
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
