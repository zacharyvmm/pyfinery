# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The Refinery types derived from Pydantic model classes: which classes, unions
and enums there are, and the feature each field maps to.
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

from ._annotations import NO_ITEM_TYPE, Shape, is_union, strip_annotated
from ._options import ClassOptions, ReferenceKind, RefineryNode, RefOptions
from ._syntax import parse_multiplicity, quote_id

AttributeType = Literal["boolean", "int", "real", "string"]

_DATA_TYPES: dict[type, AttributeType] = {bool: "boolean", int: "int", float: "real", str: "string"}


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


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributeFeature:
    """A field holding an ``int``, ``float``, ``str`` or ``bool``."""

    owner: type[BaseModel]
    name: str
    data_type: AttributeType
    many: bool
    lower: int
    upper: int | None

    @property
    def target(self) -> AttributeType:
        return self.data_type

    @property
    def kind(self) -> Literal["ref"]:
        return "ref"

    @property
    def opposite(self) -> None:
        return None

    @property
    def by_key(self) -> Literal[False]:
        return False

    @property
    def is_attribute(self) -> Literal[True]:
        return True

    @property
    def is_enum(self) -> Literal[False]:
        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class EnumFeature:
    """A field holding members of an ``Enum``, or strings of a ``Literal``."""

    owner: type[BaseModel]
    name: str
    target: EnumLike
    many: bool
    lower: int
    upper: int | None

    @property
    def kind(self) -> Literal["ref"]:
        return "ref"

    @property
    def opposite(self) -> None:
        return None

    @property
    def by_key(self) -> Literal[False]:
        return False

    @property
    def is_attribute(self) -> Literal[False]:
        return False

    @property
    def is_enum(self) -> Literal[True]:
        return True


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceFeature:
    """A field holding model objects, or (``by_key``) the keys of model objects."""

    owner: type[BaseModel]
    name: str
    target: ClassLike
    many: bool
    lower: int
    upper: int | None
    kind: ReferenceKind
    opposite: str | None = None
    by_key: bool = False

    @property
    def is_attribute(self) -> Literal[False]:
        return False

    @property
    def is_enum(self) -> Literal[False]:
        return False


# Runtime union so callers can check ``isinstance(feature, Feature)``.
Feature = AttributeFeature | EnumFeature | ReferenceFeature


@dataclass(slots=True)
class Schema:
    """The Refinery types of some model classes, and what decides how they map."""

    options: dict[type, ClassOptions]
    named_unions: dict[str, tuple[type[BaseModel], ...]]
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
    options_of: dict[type, ClassOptions] = field(default_factory=dict, repr=False, compare=False)
    """:meth:`options_for` of each class asked about (``options`` don't change)."""
    features_of: dict[type, tuple[Feature, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    """:meth:`all_features` of each class asked about, once the classes are discovered."""

    def options_for(self, cls: type) -> ClassOptions:
        """The options of the nearest class in the MRO of ``cls`` that has some."""
        found = self.options_of.get(cls)
        if found is None:
            options = (self.options[base] for base in cls.__mro__ if base in self.options)
            found = self.options_of[cls] = next(options, _DEFAULT_OPTIONS)
        return found

    def all_features(self, cls: type) -> tuple[Feature, ...]:
        """The features of ``cls``, its own and inherited, by its MRO."""
        found = self.features_of.get(cls)
        if found is None:
            declared = (self.features.get(base, ()) for base in cls.__mro__)  # type: ignore[call-overload]
            found = self.features_of[cls] = tuple(f for features in declared for f in features)
        return found

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
                if isinstance(feature, EnumFeature):
                    if feature.target not in schema.enums:
                        schema.enums.append(feature.target)
                elif isinstance(feature, ReferenceFeature):
                    target = feature.target
                    if isinstance(target, UnionClass):
                        if target not in schema.unions:
                            schema.unions.append(target)
                        pending.extend((member, True) for member in target.members)
                    else:
                        pending.append((target, True))
        if downwards and cls not in expanded:
            expanded.add(cls)
            pending.extend((sub, True) for sub in cls.__subclasses__())
    _infer_opposites(schema)
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


def _infer_opposites(schema: Schema) -> None:
    """Pair each ``Container()`` without an opposite with the one containment reference it can
    be the opposite of: declared in (or inherited by) its target class, holding objects that
    have the container reference. That reference gets the container as its opposite too."""
    claimed: dict[tuple[type, str], ReferenceFeature] = {}
    for features in list(schema.features.values()):
        for i, feature in enumerate(features):
            if not isinstance(feature, ReferenceFeature):
                continue
            if feature.kind != "container" or feature.opposite is not None:
                continue
            where = f"{feature.owner.__name__}.{feature.name}"
            target = feature.target
            if not isinstance(target, type):
                raise TypeError(f"{where}: a container reference must refer to one class")
            candidates = _containment_candidates(schema, feature, target, claimed)
            named = [c for c in candidates if c.opposite == feature.name]
            if len(named) == 1:
                candidates = named
            if len(candidates) != 1:
                found = ", ".join(f"{c.owner.__name__}.{c.name}" for c in candidates)
                raise TypeError(
                    f"{where}: give Container(opposite=...): "
                    + (
                        f"it could be the opposite of any of {found}"
                        if candidates
                        else f"no containment reference of {target.__name__} holds "
                        f"{feature.owner.__name__} objects"
                    )
                )
            (other,) = candidates
            key = (other.owner, other.name)
            if key in claimed:
                first = claimed[key]
                raise TypeError(
                    f"{where}: {first.owner.__name__}.{first.name} could be the opposite of "
                    f"{other.owner.__name__}.{other.name} too; give Container(opposite=...)"
                )
            claimed[key] = feature
            features[i] = replace(feature, opposite=other.name)
            if other.opposite is None:
                own = schema.features[other.owner]
                own[own.index(other)] = replace(other, opposite=feature.name)


def _containment_candidates(
    schema: Schema,
    container: ReferenceFeature,
    target: type[BaseModel],
    claimed: Mapping[tuple[type, str], ReferenceFeature],
) -> list[ReferenceFeature]:
    """The containment references of ``target`` that ``container`` can be the opposite of."""
    candidates = []
    for base in target.__mro__:
        for other in schema.features.get(base, []):  # type: ignore[call-overload]
            if not isinstance(other, ReferenceFeature) or other.kind != "contains":
                continue
            holds_owner = isinstance(other.target, type) and issubclass(
                other.target, container.owner
            )
            if not holds_owner:
                continue
            # One already paired with another Container() stays a candidate, so that the
            # clash is reported rather than the reference silently skipped.
            free = other.opposite in (None, container.name)
            if free or (other.owner, other.name) in claimed:
                candidates.append(other)
    return candidates


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
    shape = Shape.of(info)
    annotation = shape.type
    metadata = [*shape.metadata, *shape.inner_metadata]
    if shape.many:
        if annotation is NO_ITEM_TYPE:
            raise TypeError(f"{where}: collections must have exactly one item type")
    elif get_origin(annotation) not in (None, Literal) and not is_union(annotation):
        raise _Unsupported(f"{where}: unsupported field type {info.annotation!r}")

    refs = [m for m in metadata if isinstance(m, RefOptions)]
    if len(refs) > 1:
        raise TypeError(f"{where}: more than one Ref/Contains/Container marker")
    ref = refs[0] if refs else RefOptions()

    if annotation in _DATA_TYPES:
        if shape.many:
            raise _Unsupported(f"{where}: Refinery attributes can't hold collections")
        if refs:
            raise TypeError(f"{where}: attributes can't have Ref markers")
        data_type = _DATA_TYPES[annotation]
        return AttributeFeature(
            owner=cls, name=name, data_type=data_type, many=False, lower=0, upper=1
        )

    enum_target: EnumLike | None = None
    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not all(isinstance(v, str) for v in values):
            raise _Unsupported(f"{where}: only Literals of strings map to Refinery enums")
        if len(values) == 1:
            # A constant, e.g. a discriminator tag: the class says it all.
            return None
        enum_target = _literal_enum(schema, f"{cls.__name__}{_camel(name)}", values)
    elif _is_enum_class(annotation):
        enum_target = annotation
    if enum_target is not None:
        if ref.kind != "refers":
            raise TypeError(f"{where}: enums cannot be containment references")
        if ref.opposite is not None:
            raise TypeError(f"{where}: enums can't have opposites")
        lower, upper = _bounds(shape, metadata, ref)
        return EnumFeature(
            owner=cls, name=name, target=enum_target, many=shape.many, lower=lower, upper=upper
        )

    target: ClassLike
    if is_union(annotation):
        target = _union(schema, where, annotation)
    elif is_model_class(annotation):
        target = annotation
    else:
        raise _Unsupported(
            f"{where}: unsupported field type {annotation!r}; use a RefineryNode subclass, "
            "a Pydantic model, a union of them, an Enum, a Literal of strings, int, float, "
            "str, or bool"
        )
    kind = _reference_kind(schema, where, target, ref)
    lower, upper = _bounds(shape, metadata, ref)
    if ref.kind == "container" and shape.many:
        raise TypeError(f"{where}: a container reference must be a single node")
    return ReferenceFeature(
        owner=cls,
        name=name,
        target=target,
        many=shape.many,
        lower=lower,
        upper=upper,
        kind=kind,
        opposite=ref.opposite,
    )


def _bounds(shape: Shape, metadata: list[Any], ref: RefOptions) -> tuple[int, int | None]:
    if ref.multiplicity is not None:
        return parse_multiplicity(ref.multiplicity)
    return shape.bounds(metadata)


def _reference_kind(
    schema: Schema, where: str, target: ClassLike, ref: RefOptions
) -> ReferenceKind:
    """The kind ``ref`` asks for, or ``contains`` for owned values: plain models without a key
    are value objects owned by the object holding them."""
    members = target.members if isinstance(target, UnionClass) else (target,)
    entities = [schema.is_entity(m) for m in members]
    if all(entities):
        return ref.kind
    if any(entities):
        raise TypeError(
            f"{where}: mixes owned value classes with entity classes (RefineryNode or "
            "keyed); Refinery needs them all to be one or the other"
        )
    if ref.kind == "container" or ref.opposite is not None:
        shown = target.name if isinstance(target, UnionClass) else target.__name__
        raise TypeError(
            f"{where}: {shown} is a plain Pydantic model, which is always "
            "contained in its owner and cannot have opposites; make it a RefineryNode "
            "(or give it a key) to use opposites or Container()"
        )
    return "contains"


def _key_reference(
    schema: Schema,
    cls: type[BaseModel],
    name: str,
    spec: type[BaseModel] | Sequence[type[BaseModel]] | str,
) -> ReferenceFeature:
    where = f"{cls.__name__}.{name}"
    shape = Shape.of(cls.model_fields[name])
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
    return ReferenceFeature(
        owner=cls,
        name=name,
        target=target,
        many=shape.many,
        lower=lower,
        upper=upper,
        kind="refers",
        by_key=True,
    )


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
        arg, _ = strip_annotated(arg)
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
            if not isinstance(feature, ReferenceFeature) or not feature.by_key:
                continue
            target = feature.target
            members = target.members if isinstance(target, UnionClass) else (target,)
            for member in members:
                if schema.options_for(member).key is None:
                    raise TypeError(
                        f"{feature.owner.__name__}.{feature.name}: references by key need "
                        f"ClassOptions(key=...) on {member.__name__}"
                    )


def _check_opposites(schema: Schema) -> None:
    for features in schema.features.values():
        for feature in features:
            if not isinstance(feature, ReferenceFeature) or feature.opposite is None:
                continue
            where = f"{feature.owner.__name__}.{feature.name}"
            target = feature.target
            if isinstance(target, UnionClass):
                raise TypeError(f"{where}: references to unions can't have opposites")
            other = schema.find_feature(target, feature.opposite)
            if other is None:
                raise TypeError(
                    f"{where}: opposite {feature.opposite!r} is not a field of {target.__name__}"
                )
            if not isinstance(other, ReferenceFeature) or other.opposite != feature.name:
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


def _is_enum_class(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, enum.Enum)


def is_model_class(value: Any) -> bool:
    """Pydantic model classes that can be mapped to Refinery classes."""
    return (
        isinstance(value, type)
        and issubclass(value, BaseModel)
        and value not in (BaseModel, RefineryNode)
    )


def _camel(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


class _Unsupported(TypeError):
    """A field type with no Refinery counterpart (skipped with ``unsupported="skip"``)."""
