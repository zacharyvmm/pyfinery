# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Validate and query Pydantic object graphs with Refinery.

A :class:`Metamodel` derives a Refinery metamodel from Pydantic model classes,
and turns instances of them into Refinery facts:

- :meth:`Metamodel.validate` checks a graph against the metamodel and
  ``error pred`` rules, closing the world (anything not asserted is false)
  with Refinery's ``concretize`` operation, and reports each constraint that
  evaluates to ``error`` as a :class:`Violation`.
- :meth:`Metamodel.query` evaluates graph patterns written as Refinery
  predicates over a closed world that may hold *uncertain* facts, and
  reports each :class:`Match` as ``must`` (true in Refinery's four-valued
  logic) or ``may`` (unknown).

How fields map to Refinery:

========================================  =========================================
Python                                    Refinery
========================================  =========================================
``int`` / ``float`` / ``str`` / ``bool``  ``int`` / ``real`` / ``string`` / ``boolean``
``Literal["a", "b"]``                     an enum (single-value tags are skipped)
an :class:`enum.Enum` subclass            an enum
``Model`` / ``Model | None`` / ``list``   ``Model[1]`` / ``Model[0..1]`` / ``Model[0..*]``
``A | B``                                 an abstract superclass of ``A`` and ``B``
subclassing                               ``extends``
========================================  =========================================

Whether a field typed as a model class is containment or a plain reference
depends on the identity of that class:

- plain Pydantic models are *owned values*: each instance is a node
  contained in the object holding it;
- classes with ``ClassOptions(key=...)`` are *entities*: instances with equal
  keys are one node, and fields typed as them are references. Fields that
  hold keys rather than objects are mapped with
  ``ClassOptions(references=...)``;
- :class:`RefineryNode` subclasses are entities by object identity, for
  hand-built graphs, with :func:`Contains`, :func:`Container` and
  :class:`Ref` markers for containment, opposites, and multiplicities.

:class:`ClassOptions` configure classes from outside, so generated or
third-party models can be used unchanged; :class:`Predicate` provides
relations computed in Python (e.g. regular expressions) to rules.
"""

from __future__ import annotations

import enum
import inspect
import itertools
import math
import re
import types
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Union, get_args, get_origin

import annotated_types
from pydantic import BaseModel, ConfigDict

from .errors import RefineryError, RefineryTimeoutError
from .models import JsonOutput, JsonOutputFormat, OutputFormats, SourceOutputFormat

if TYPE_CHECKING:
    from .client import AsyncRefinery, Refinery

# ---------------------------------------------------------------------------
# Field markers
# ---------------------------------------------------------------------------

ReferenceKind = Literal["refers", "contains", "container"]
Certainty = Literal["must", "may"]


@dataclass(frozen=True, slots=True)
class Ref:
    """Refinery-specific options of a reference field, used with ``Annotated``.

    :param opposite: Name of the opposite reference in the target class.
    :param kind: ``"contains"`` for containment, ``"container"`` for the
        opposite of a containment reference.
    :param multiplicity: Overrides the multiplicity inferred from the type,
        e.g. ``"1..*"``, ``"2"`` or ``(1, None)``.
    """

    opposite: str | None = None
    kind: ReferenceKind = "refers"
    multiplicity: str | tuple[int, int | None] | None = None


def Contains(  # noqa: N802 - reads like a type annotation
    opposite: str | None = None,
    multiplicity: str | tuple[int, int | None] | None = None,
) -> Ref:
    """Marks a containment reference (``contains``)."""
    return Ref(opposite=opposite, kind="contains", multiplicity=multiplicity)


def Container(opposite: str) -> Ref:  # noqa: N802
    """Marks the container side of a containment reference (``container``)."""
    return Ref(opposite=opposite, kind="container")


@dataclass(frozen=True, slots=True)
class ClassOptions:
    """How to map a model class, for classes that can't carry the options themselves.

    Passed to :class:`Metamodel` as ``options={cls: ClassOptions(...)}``; the
    options of the nearest class in the MRO apply, so options given for a
    base class also apply to its subclasses.

    :param key: Identity of an instance, e.g. ``lambda s: s.name``. Instances
        with equal keys (and the same class) are the same node, and fields
        typed as this class are references rather than containment.
    :param references: Fields holding the *keys* of other objects rather than
        the objects, e.g. ``{"depends_on": Service}`` for
        ``depends_on: list[str]``. The target is a keyed class, a sequence of
        them, or the name of a union given to :class:`Metamodel`. Keys that
        match no object (or several) are reported as ``"reference"``
        violations.
    :param node_name: Name of the Refinery node of an instance. Defaults to
        the key, if it is a string or integer, else ``<class>_<n>``.
    :param exclude: Names of fields (declared in this class) to leave out.
    :param abstract: Declare the Refinery class ``abstract``.
    :param exists: Whether an instance certainly exists (``"must"`` or
        ``True``) or only may (``"may"``), e.g. something that happens on
        some paths only. Used by :meth:`Metamodel.query` only.
    :param certainty: Whether the value of a field certainly holds, called as
        ``certainty(instance, field_name, value)`` for each reference target
        and attribute value; ``"may"`` asserts it as unknown, e.g. a
        dependency found by a heuristic.
    """

    key: Callable[[Any], Hashable] | None = None
    references: Mapping[str, type[BaseModel] | Sequence[type[BaseModel]] | str] | None = None
    node_name: Callable[[Any], str] | None = None
    exclude: frozenset[str] | Sequence[str] = ()
    abstract: bool | None = None
    exists: Callable[[Any], Certainty | bool] | None = None
    certainty: Callable[[Any, str, Any], Certainty | bool] | None = None


@dataclass(frozen=True, slots=True)
class Predicate:
    """A relation computed in Python, usable in the rules of :meth:`Metamodel.validate`
    and :meth:`Metamodel.query`.

    For what Refinery can't express, such as regular expressions, globs, or
    relations computed elsewhere. It is declared as ``pred <name>(<types>).``,
    where each parameter type is a model class or the name of a union, and
    its facts come from either:

    - ``evaluate``, called on every combination of nodes of the parameter
      types, returning ``True``/``"must"`` (holds), ``"may"`` (may hold), or
      ``False``/``None`` (doesn't hold); or
    - ``tuples``: the argument tuples that hold, each optionally paired with
      a certainty, e.g. ``[((a, b), "may"), (c, d)]``. Use it when the
      relation is already known, or when combinations would be too many.

    >>> Predicate("validEmail", (User,), lambda u: EMAIL.fullmatch(u.email) is not None)
    >>> Predicate("calls", (Service, Service), tuples=traced_calls)
    """

    name: str
    parameters: tuple[type[BaseModel] | str, ...]
    evaluate: Callable[..., Certainty | bool | None] | None = None
    tuples: Iterable[Any] | None = None

    def __post_init__(self) -> None:
        if (self.evaluate is None) == (self.tuples is None):
            raise ValueError(f"Predicate {self.name!r} needs exactly one of evaluate or tuples")
        if self.tuples is not None:
            # Allow one-shot iterables (e.g. generators) to be read once.
            object.__setattr__(self, "tuples", list(self.tuples))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class RefineryNode(BaseModel):
    """Base class of Refinery classes defined in Python.

    Instances are graph nodes: they compare and hash by identity, and
    references to other nodes are kept as-is (not copied) by Pydantic.

    Class attributes:

    - ``__abstract__ = True`` declares an abstract class (only subclasses
      may be instantiated).
    - ``__node_id__ = "name"`` names nodes after the value of a field
      instead of generated names like ``dir_1``.
    """

    model_config = ConfigDict(revalidate_instances="never")

    __abstract__: ClassVar[bool] = False
    __node_id__: ClassVar[str | None] = None

    def model_post_init(self, context: Any, /) -> None:
        if type(self).__dict__.get("__abstract__", False):
            raise TypeError(f"Cannot instantiate abstract Refinery class {type(self).__name__}")

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr_args__(self):
        # Show referenced nodes by class name only to avoid infinite recursion
        # through opposite references.
        for name, value in super().__repr_args__():
            yield name, _short_repr(value)


class _Short:
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _short_repr(value: Any) -> Any:
    if isinstance(value, RefineryNode):
        return _Short(f"<{type(value).__name__} at {id(value):#x}>")
    if isinstance(value, list | tuple):
        return type(value)(_short_repr(v) for v in value)
    if isinstance(value, set | frozenset):
        return [_short_repr(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Refinery syntax helpers
# ---------------------------------------------------------------------------

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


def _encodable(text: str) -> bool:
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
    if not _encodable(escaped):
        # Refinery reads ``\uXXXX`` as a UTF-16 code unit, as Java does.
        escaped = "".join(c if _encodable(c) else f"\\u{ord(c):04x}" for c in escaped)
    return f'"{escaped}"'


def _attribute_literal(value: Any, data_type: str) -> str:
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


def _multiplicity(lower: int, upper: int | None) -> str:
    if upper == lower:
        return f"[{lower}]"
    return f"[{lower}..{'*' if upper is None else upper}]"


def _parse_multiplicity(value: str | tuple[int, int | None]) -> tuple[int, int | None]:
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


# ---------------------------------------------------------------------------
# Metamodel
# ---------------------------------------------------------------------------

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


class Metamodel:
    """A Refinery metamodel derived from :class:`RefineryNode` subclasses
    (and the plain Pydantic models nested in them).

    >>> metamodel = Metamodel(Filesystem, rules=[
    ...     "error pred emptyDir(Dir d) <-> !contents(d, _).",
    ... ])
    >>> print(metamodel.source)
    >>> report = metamodel.validate([my_filesystem], refinery)

    Classes reachable from the given ones are included automatically: their
    superclasses, the classes their fields refer to, and the already-imported
    subclasses of those (but not every subclass of a shared base class).

    :param options: :class:`ClassOptions` by class.
    :param unions: Names for the abstract superclasses standing for unions,
        e.g. ``{"Component": [Service, Database]}``.
    :param unsupported: ``"skip"`` to leave out fields with no Refinery
        counterpart instead of raising ``TypeError``.
    """

    def __init__(
        self,
        *classes: type[BaseModel],
        rules: Iterable[str] = (),
        options: Mapping[type, ClassOptions] | None = None,
        unions: Mapping[str, Iterable[type[BaseModel]]] | None = None,
        unsupported: Literal["error", "skip"] = "error",
    ) -> None:
        if not classes:
            raise ValueError("Metamodel needs at least one model class")
        self.rules = list(rules)
        self.options = dict(options or {})
        self._union_members = {name: tuple(members) for name, members in (unions or {}).items()}
        self._union_names = {
            frozenset(members): name for name, members in self._union_members.items()
        }
        self._unsupported = unsupported
        self.classes: list[type[BaseModel]] = []
        self.unions: list[UnionClass] = []
        self.enums: list[EnumLike] = []
        self.features: dict[type[BaseModel], list[Feature]] = {}
        self.skipped: list[tuple[type[BaseModel], str, str]] = []
        """Fields left out of the metamodel: ``(class, field, reason)``."""
        self._collect(classes)
        self._check_names()
        self._check_opposites()
        self._check_key_references()
        self._type_names = {self.class_name(c) for c in self.classes}
        self._type_names |= {quote_id(u.name) for u in self.unions}
        self._type_names |= {self.class_name(e) for e in self.enums}
        self.source = self._render()

    # -- discovery ----------------------------------------------------------

    def _collect(self, roots: Iterable[type[BaseModel]]) -> None:
        # (class, whether to include its subclasses): roots and field types are
        # expanded downwards, superclasses only upwards, so that a base class
        # shared by unrelated models doesn't pull all of them in.
        pending: list[tuple[type[BaseModel], bool]] = [(cls, True) for cls in roots]
        expanded: set[type[BaseModel]] = set()
        while pending:
            cls, downwards = pending.pop(0)
            if cls is RefineryNode:
                continue
            if not _is_model_class(cls):
                raise TypeError(f"{cls!r} is not a RefineryNode subclass or Pydantic model")
            if cls not in self.features:
                self._rebuild(cls)
                self.classes.append(cls)
                self.features[cls] = self._features(cls)
                pending.extend((b, False) for b in cls.__bases__ if _is_model_class(b))
                for feature in self.features[cls]:
                    target = feature.target
                    if feature.is_enum:
                        if target not in self.enums:
                            self.enums.append(target)  # type: ignore[arg-type]
                    elif isinstance(target, UnionClass):
                        if target not in self.unions:
                            self.unions.append(target)
                        pending.extend((member, True) for member in target.members)
                    elif not feature.is_attribute:
                        pending.append((target, True))  # type: ignore[arg-type]
            if downwards and cls not in expanded:
                expanded.add(cls)
                pending.extend((sub, True) for sub in cls.__subclasses__())
        # Superclasses must be declared before subclasses for readability.
        self.classes.sort(key=lambda c: (len(c.__mro__), c.__qualname__))

    def _options(self, cls: type) -> ClassOptions:
        for base in cls.__mro__:
            if base in self.options:
                return self.options[base]
        return _DEFAULT_OPTIONS

    def _is_entity(self, cls: type) -> bool:
        """Whether instances are shared nodes (RefineryNode or keyed), not owned values."""
        return issubclass(cls, RefineryNode) or self._options(cls).key is not None

    def _is_abstract(self, cls: type) -> bool:
        abstract = self.options[cls].abstract if cls in self.options else None
        return bool(cls.__dict__.get("__abstract__", False)) if abstract is None else abstract

    def _features(self, cls: type[BaseModel]) -> list[Feature]:
        exclude = set(self._options(cls).exclude)
        features = []
        for name in self._own_fields(cls):
            if name in exclude:
                continue
            try:
                feature = self._feature(cls, name)
            except _Unsupported as e:
                if self._unsupported == "error":
                    raise TypeError(
                        f"{e} (exclude the field with ClassOptions(exclude=...) or pass "
                        "unsupported='skip')"
                    ) from None
                self.skipped.append((cls, name, str(e)))
                continue
            if feature is not None:
                features.append(feature)
        return features

    def _check_names(self) -> None:
        seen: dict[str, object] = {}
        named: list[tuple[str, object]] = [(c.__name__, c) for c in self.classes]
        named += [(u.name, u) for u in self.unions]
        named += [(e.name if isinstance(e, LiteralEnum) else e.__name__, e) for e in self.enums]
        for name, owner in named:
            if seen.setdefault(name, owner) is not owner:
                raise TypeError(
                    f"Two different types are both named {name!r}; Refinery needs unique names"
                )

    @staticmethod
    def _rebuild(node_class: type[BaseModel]) -> None:
        if not node_class.__pydantic_complete__:
            try:
                node_class.model_rebuild()
            except Exception as e:
                raise TypeError(
                    f"Cannot resolve the field types of {node_class.__name__}; define all "
                    f"referenced classes before building the Metamodel ({e})"
                ) from e

    @staticmethod
    def _own_fields(node_class: type[BaseModel]) -> list[str]:
        """Fields first declared in ``node_class`` (inherited ones belong to a superclass)."""
        own = inspect.get_annotations(node_class)
        inherited = {
            name
            for base in node_class.__mro__[1:]
            if _is_model_class(base)
            for name in base.model_fields  # type: ignore[union-attr]
        }
        return [name for name in node_class.model_fields if name in own and name not in inherited]

    def _feature(self, cls: type[BaseModel], name: str) -> Feature | None:
        by_key = self._options(cls).references or {}
        if name in by_key:
            return self._key_reference(cls, name, by_key[name])
        info = cls.model_fields[name]
        where = f"{cls.__name__}.{name}"
        annotation, metadata = _strip_annotated(info.annotation)
        metadata = [*info.metadata, *metadata]
        annotation, optional = _strip_optional(annotation)
        annotation, more = _strip_annotated(annotation)
        metadata += more

        many = False
        origin = get_origin(annotation)
        if origin in _COLLECTIONS or annotation in (list, set, frozenset, tuple):
            args = [a for a in get_args(annotation) if a is not Ellipsis]
            if len(args) != 1:
                raise TypeError(f"{where}: collections must have exactly one item type")
            many = True
            annotation, more = _strip_annotated(args[0])
            metadata += more
        elif origin is not None and origin is not Literal and not _is_union(annotation):
            raise _Unsupported(f"{where}: unsupported field type {info.annotation!r}")

        refs = [m for m in metadata if isinstance(m, Ref)]
        if len(refs) > 1:
            raise TypeError(f"{where}: more than one Ref/Contains/Container marker")
        ref = refs[0] if refs else Ref()

        if annotation in _DATA_TYPES:
            if many:
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
            target = self._literal_enum(f"{cls.__name__}{_camel(name)}", values)
        elif _is_union(annotation):
            target = self._union(where, annotation)
        elif _is_model_class(annotation) or _is_enum_class(annotation):
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
            entities = [self._is_entity(m) for m in members]  # type: ignore[arg-type]
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

        if many:
            lower = next(
                (m.min_length for m in metadata if isinstance(m, annotated_types.MinLen)), 0
            )
            upper = next(
                (m.max_length for m in metadata if isinstance(m, annotated_types.MaxLen)), None
            )
        else:
            lower, upper = (0 if optional else 1), 1
        if ref.multiplicity is not None:
            lower, upper = _parse_multiplicity(ref.multiplicity)
        if ref.kind == "container":
            if many:
                raise TypeError(f"{where}: a container reference must be a single node")
            if ref.opposite is None:
                raise TypeError(f"{where}: a container reference needs an opposite")
        return Feature(cls, name, target, many, lower, upper, kind, ref.opposite)

    def _key_reference(
        self,
        cls: type[BaseModel],
        name: str,
        spec: type[BaseModel] | Sequence[type[BaseModel]] | str,
    ) -> Feature:
        info = cls.model_fields[name]
        where = f"{cls.__name__}.{name}"
        annotation, metadata = _strip_annotated(info.annotation)
        metadata = [*info.metadata, *metadata]
        annotation, optional = _strip_optional(annotation)
        annotation, _ = _strip_annotated(annotation)
        many = get_origin(annotation) in _COLLECTIONS or annotation in (list, set, frozenset, tuple)
        if many:
            args = [a for a in get_args(annotation) if a is not Ellipsis]
            annotation = _strip_annotated(args[0])[0] if len(args) == 1 else None
        if annotation not in (str, int):
            raise TypeError(
                f"{where}: a reference by key must hold str or int keys, or a collection of them"
            )
        target: ClassLike
        if isinstance(spec, str):
            if spec not in self._union_members:
                raise TypeError(f"{where}: no union named {spec!r} was given to Metamodel(unions=)")
            target = self._union_of(self._union_members[spec])
        elif isinstance(spec, type):
            if not _is_model_class(spec):
                raise TypeError(f"{where}: {spec!r} is not a Pydantic model class")
            target = spec
        else:
            target = self._union_of(tuple(spec))
        if many:
            lower = next(
                (m.min_length for m in metadata if isinstance(m, annotated_types.MinLen)), 0
            )
            upper = next(
                (m.max_length for m in metadata if isinstance(m, annotated_types.MaxLen)), None
            )
        else:
            lower, upper = (0 if optional else 1), 1
        return Feature(cls, name, target, many, lower, upper, "refers", None, by_key=True)

    def _check_key_references(self) -> None:
        for features in self.features.values():
            for feature in features:
                if not feature.by_key:
                    continue
                target = feature.target
                members = target.members if isinstance(target, UnionClass) else (target,)
                for member in members:
                    if self._options(member).key is None:  # type: ignore[arg-type]
                        raise TypeError(
                            f"{feature.owner.__name__}.{feature.name}: references by key need "
                            f"ClassOptions(key=...) on {member.__name__}"  # type: ignore[union-attr]
                        )

    def _literal_enum(self, name: str, values: tuple[str, ...]) -> LiteralEnum:
        for existing in self.enums:
            if isinstance(existing, LiteralEnum) and existing.values == values:
                return existing
        return LiteralEnum(name, values)

    def _union(self, where: str, annotation: Any) -> UnionClass:
        members = []
        for arg in get_args(annotation):
            arg, _ = _strip_annotated(arg)
            if not _is_model_class(arg):
                raise _Unsupported(f"{where}: unions must only contain Pydantic model classes")
            members.append(arg)
        return self._union_of(tuple(members))

    def _union_of(self, members: tuple[type[BaseModel], ...]) -> UnionClass:
        key = frozenset(members)
        for existing in self.unions:
            if frozenset(existing.members) == key:
                return existing
        name = self._union_names.get(key) or "Or".join(m.__name__ for m in members)
        return UnionClass(name, members)

    def _find_feature(self, cls: type, name: str) -> Feature | None:
        for base in cls.__mro__:
            for feature in self.features.get(base, []):  # type: ignore[call-overload]
                if feature.name == name:
                    return feature
        return None

    def _check_opposites(self) -> None:
        for features in self.features.values():
            for feature in features:
                if feature.opposite is None:
                    continue
                where = f"{feature.owner.__name__}.{feature.name}"
                if isinstance(feature.target, UnionClass):
                    raise TypeError(f"{where}: references to unions can't have opposites")
                other = self._find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
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

    # -- rendering ----------------------------------------------------------

    @staticmethod
    def class_name(node_class: type | UnionClass | LiteralEnum) -> str:
        if isinstance(node_class, UnionClass | LiteralEnum):
            return quote_id(node_class.name)
        return quote_id(node_class.__name__)

    def _literal(self, enum_type: EnumLike, value: Any) -> str:
        """The Refinery name of an enum value, e.g. ``Color::red``."""
        member = value if isinstance(enum_type, LiteralEnum) else value.name
        return f"{self.class_name(enum_type)}::{quote_id(member)}"

    def _render_feature(self, feature: Feature) -> str:
        name = quote_id(feature.name)
        if feature.is_attribute:
            return f"{feature.target} {name}"
        target = self.class_name(feature.target)  # type: ignore[arg-type]
        opposite = f" opposite {quote_id(feature.opposite)}" if feature.opposite else ""
        if feature.kind == "container":
            return f"container {target} {name}{opposite}"
        prefix = "contains " if feature.kind == "contains" else ""
        return f"{prefix}{target}{_multiplicity(feature.lower, feature.upper)} {name}{opposite}"

    def _render(self) -> str:
        lines: list[str] = []
        for enum_type in self.enums:
            if isinstance(enum_type, LiteralEnum):
                members = enum_type.values
            else:
                members = tuple(member.name for member in enum_type)
            literals = ", ".join(quote_id(member) for member in members)
            lines.append(f"enum {self.class_name(enum_type)} {{ {literals} }}")
        for union in self.unions:
            lines.append(f"abstract class {self.class_name(union)}.")
        for cls in self.classes:
            header = f"class {self.class_name(cls)}"
            if self._is_abstract(cls):
                header = f"abstract {header}"
            supers: list[type | UnionClass] = [b for b in cls.__bases__ if _is_model_class(b)]
            supers += [u for u in self.unions if cls in u.members]
            if supers:
                header += " extends " + ", ".join(self.class_name(b) for b in supers)
            features = self.features[cls]
            if not features:
                lines.append(f"{header}.")
                continue
            lines.append(f"{header} {{")
            lines.extend(f"    {self._render_feature(f)}" for f in features)
            lines.append("}")
        if self.rules:
            lines.append("")
            lines.extend(self.rules)
        return "\n".join(lines) + "\n"

    # -- instances ----------------------------------------------------------

    def collect(self, objects: Iterable[BaseModel]) -> list[BaseModel]:
        """All nodes reachable from ``objects`` through reference fields.

        Only outgoing references are followed: a node that is referenced by
        nothing (e.g., a child that only sets its own ``parent``) must be
        passed in ``objects`` itself.
        """
        result: list[BaseModel] = []
        seen: set[Hashable] = set()
        pending = list(objects)
        while pending:
            node = pending.pop(0)
            if type(node) not in self.features:
                raise TypeError(f"{type(node).__name__} is not part of this metamodel")
            identity = self.identity(node)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(node)
            for _, feature, value in self._values(node):
                if not (feature.is_attribute or feature.is_enum or feature.by_key):
                    pending.extend(value)
        return result

    def identity(self, node: BaseModel) -> Hashable:
        """What makes ``node`` one Refinery node: its key if its class has one, else itself.

        Of several instances with the same key, the first one reached
        represents them all (its fields are the ones asserted).
        """
        key = self._options(type(node)).key
        return (type(node), key(node)) if key is not None else id(node)

    def link_opposites(self, objects: Iterable[BaseModel]) -> int:
        """Fill in the missing side of every ``opposite`` pair in the graph.

        For example, after ``home.contents.append(doc)`` this sets
        ``doc.parent = home``, and after ``doc.parent = home`` it appends
        ``doc`` to ``home.contents``. Refinery infers missing sides on its own,
        so this is only needed to keep the Python objects consistent.

        Single-valued fields that already point at a different node are left
        unchanged; :meth:`validate` reports such conflicts. Returns the number
        of references added.
        """
        added = 0
        for node in self.collect(objects):
            for _, feature, values in self._values(node):
                if feature.opposite is None:
                    continue
                for target in values:
                    other = self._find_feature(type(target), feature.opposite)
                    if other is not None and _add_reference(target, other, node, self.identity):
                        added += 1
        return added

    def _values(self, node: BaseModel) -> Iterator[tuple[type, Feature, list[Any]]]:
        for cls in type(node).__mro__:
            for feature in self.features.get(cls, []):  # type: ignore[call-overload]
                value = getattr(node, feature.name)
                if value is None:
                    continue
                values = list(value) if feature.many else [value]
                yield cls, feature, values

    def node_names(self, nodes: Sequence[BaseModel]) -> dict[Hashable, str]:
        """Refinery node names (as rendered in the problem) by :meth:`identity`."""
        names: dict[Hashable, str] = {}
        used: set[str] = set()
        counters: dict[type, int] = {}
        for node in nodes:
            id_field: str | None = getattr(type(node), "__node_id__", None)
            naming = self._options(type(node)).node_name
            if naming is not None or id_field is not None:
                raw = naming(node) if naming is not None else getattr(node, id_field)  # type: ignore[arg-type]
                name = quote_id(str(raw))
                if name in used:
                    source = "node_name" if naming is not None else f"field {id_field!r}"
                    raise ValueError(f"Duplicate node id {name} (from {source})")
            elif (key := self._name_from_key(node)) is not None and key not in used:
                name = key
            else:
                while True:
                    counters[type(node)] = counters.get(type(node), 0) + 1
                    name = quote_id(f"{type(node).__name__.lower()}_{counters[type(node)]}")
                    if name not in used:
                        break
            used.add(name)
            names[self.identity(node)] = name
        return names

    def _fact_prefix(self, node: BaseModel, feature: Feature, value: Any) -> str:
        """``"?"`` if ``ClassOptions.certainty`` says the value only may hold."""
        decide = self._options(type(node)).certainty
        if decide is None:
            return ""
        answer = decide(node, feature.name, value)
        if answer in ("must", True):
            return ""
        if answer == "may":
            return "?"
        raise ValueError(
            f"ClassOptions.certainty of {type(node).__name__} must return 'must', 'may' or True, "
            f"not {answer!r} (leave values that don't hold out of the object)"
        )

    def _name_from_key(self, node: BaseModel) -> str | None:
        """The key as a node name, if it is a string or integer that isn't a type's name."""
        key_of = self._options(type(node)).key
        if key_of is None:
            return None
        key = key_of(node)
        if isinstance(key, bool) or not isinstance(key, str | int) or str(key) == "":
            return None
        if not _encodable(str(key)):
            # Escaped, Refinery would name the node with "?" for each lone
            # surrogate, so the name couldn't be mapped back to the node.
            return None
        name = quote_id(str(key))
        return None if name in self._type_names or str(key) in _RESERVED_NODE_NAMES else name

    def _key_index(self, nodes: Sequence[BaseModel]) -> dict[Hashable, list[BaseModel]]:
        index: dict[Hashable, list[BaseModel]] = {}
        for node in nodes:
            key_of = self._options(type(node)).key
            if key_of is not None:
                index.setdefault(key_of(node), []).append(node)
        return index

    @staticmethod
    def _resolve(
        feature: Feature, key: Hashable, index: Mapping[Hashable, list[BaseModel]]
    ) -> list[BaseModel]:
        target = feature.target
        members = target.members if isinstance(target, UnionClass) else (target,)
        return [n for n in index.get(key, []) if isinstance(n, members)]  # type: ignore[arg-type]

    def reference_errors(
        self, nodes: Sequence[BaseModel], names: Mapping[Hashable, str]
    ) -> list[Violation]:
        """Keys in reference-by-key fields that match no object, or several."""
        index = self._key_index(nodes)
        errors = []
        for node in nodes:
            for cls, feature, keys in self._values(node):
                if not feature.by_key:
                    continue
                for key in keys:
                    found = self._resolve(feature, key, index)
                    if len(found) == 1:
                        continue
                    problem = "unresolved" if not found else "ambiguous"
                    constraint = f"{self.class_name(cls)}::{quote_id(feature.name)}::{problem}"
                    node_names = (names[self.identity(node)], repr(key))
                    errors.append(Violation("reference", constraint, node_names, (node, None)))
        return errors

    def facts(
        self,
        nodes: Sequence[BaseModel],
        names: Mapping[Hashable, str],
        *,
        canonical_opposites: bool = False,
    ) -> str:
        """Assertions describing ``nodes`` (which must include every referenced node).

        With ``canonical_opposites``, a reference on one side of an opposite
        pair is written as the reverse reference on the other side, so that
        closed-world defaults on that side don't contradict it.
        """
        lines = []
        index = self._key_index(nodes)
        for node in nodes:
            name = names[self.identity(node)]
            lines.append(f"{self.class_name(type(node))}({name}).")
            for cls, feature, values in self._values(node):
                symbol = f"{self.class_name(cls)}::{quote_id(feature.name)}"
                if feature.by_key:
                    for key in values:
                        found = self._resolve(feature, key, index)
                        if len(found) == 1:  # others are reported by reference_errors
                            prefix = self._fact_prefix(node, feature, key)
                            target = names[self.identity(found[0])]
                            lines.append(f"{prefix}{symbol}({name}, {target}).")
                    continue
                if canonical_opposites and not self._canonical(feature):
                    for value in values:
                        other = self._find_feature(type(value), feature.opposite)  # type: ignore[arg-type]
                        assert other is not None
                        prefix = self._fact_prefix(node, feature, value)
                        reverse = f"{self.class_name(other.owner)}::{quote_id(other.name)}"
                        lines.append(f"{prefix}{reverse}({names[self.identity(value)]}, {name}).")
                    continue
                for value in values:
                    prefix = self._fact_prefix(node, feature, value)
                    if feature.is_attribute:
                        if prefix:
                            continue  # an uncertain attribute value is simply unknown
                        literal = _attribute_literal(value, feature.target)  # type: ignore[arg-type]
                        lines.append(f"{symbol}({name}): {literal}.")
                    elif feature.is_enum:
                        literal = self._literal(feature.target, value)  # type: ignore[arg-type]
                        lines.append(f"{prefix}{symbol}({name}, {literal}).")
                    else:
                        target = names[self.identity(value)]
                        lines.append(f"{prefix}{symbol}({name}, {target}).")
        return "\n".join(lines) + "\n"

    def problem(self, objects: Iterable[BaseModel]) -> str:
        """The metamodel, rules, and the facts of the graph reachable from ``objects``."""
        nodes = self.collect(objects)
        return self.source + "\n" + self.facts(nodes, self.node_names(nodes))

    # -- validation ---------------------------------------------------------

    def _format(self, rules: Sequence[str], reported: Iterable[str]) -> OutputFormats:
        """Ask for JSON output without the helper predicates of the rules."""
        excluded = _helper_predicates([*self.rules, *rules], reported)
        return OutputFormats(
            json_=JsonOutputFormat(excluded_relations=excluded or None),
            source=SourceOutputFormat(enabled=False),
        )

    def _prepare(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str] = (),
        predicates: Iterable[Predicate] = (),
    ) -> tuple[str, dict[str, BaseModel], list[Violation]]:
        predicates = list(predicates)
        nodes = self.collect(objects)
        names = self.node_names(nodes)
        lines = [self.source, *rules, *(self._declare_predicate(p) for p in predicates), ""]
        lines.append(self.facts(nodes, names))
        lines += self._predicate_facts(predicates, nodes, names)
        by_name = {names[self.identity(node)]: node for node in nodes}
        return "\n".join(lines) + "\n", by_name, self.reference_errors(nodes, names)

    def validate(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Check the graph reachable from ``objects`` against the metamodel and rules.

        ``rules`` (Refinery source, typically ``error pred`` declarations) are added to
        those given to the :class:`Metamodel`, and ``predicates`` provide
        relations computed in Python for them to use.

        Validation is two-valued: it checks what certainly holds. Facts that
        only ``"may"`` hold (from :class:`Predicate` or
        ``ClassOptions(certainty=...)``) count as not holding, and every
        object exists (``ClassOptions(exists=...)`` is ignored). Use
        :meth:`query` to reason about what may hold.

        Only nodes reachable from ``objects`` through outgoing references are
        checked (see :meth:`collect`). Raises
        :class:`~pyfinery.InvalidProblemError` if the rules don't compile;
        constraint violations are returned in the report.
        """
        rules = list(rules)
        source, by_name, errors = self._prepare(objects, rules, predicates)
        try:
            result = refinery.concretize(source, format=self._format(rules, ()))
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return _report(source, result.json_, by_name, self.identity, errors)

    async def avalidate(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> ValidationReport:
        """Asynchronous variant of :meth:`validate`."""
        rules = list(rules)
        source, by_name, errors = self._prepare(objects, rules, predicates)
        try:
            result = await refinery.concretize(source, format=self._format(rules, ()))
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return _report(source, result.json_, by_name, self.identity, errors)

    # -- queries ------------------------------------------------------------

    def query(
        self,
        objects: Iterable[BaseModel],
        refinery: Refinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Evaluate graph patterns over the graph reachable from ``objects``.

        ``rules`` are Refinery predicates, e.g.
        ``"pred reaches(Service s, Database d) <-> Service::depends_on+(s, d)."``;
        every ``pred`` they declare is reported unless ``select`` names the
        ones to report. Unlike :meth:`validate`, the world is closed by
        asserting that there is nothing else (no other objects, references,
        or subclass memberships) while keeping uncertainty: objects that only
        ``may`` exist and :class:`Predicate` facts that only ``may`` hold make
        the matches that depend on them ``may``.
        """
        prepared = self._prepare_query(objects, rules, predicates, select)
        try:
            result = refinery.semantics(prepared.source, format=prepared.format)
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return prepared.result(result.json_)

    async def aquery(
        self,
        objects: Iterable[BaseModel],
        refinery: AsyncRefinery,
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
        select: Iterable[str] | None = None,
    ) -> QueryResult:
        """Asynchronous variant of :meth:`query`."""
        prepared = self._prepare_query(objects, rules, predicates, select)
        try:
            result = await refinery.semantics(prepared.source, format=prepared.format)
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return prepared.result(result.json_)

    def query_problem(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> str:
        """The problem :meth:`query` would send, e.g. to debug it in the web editor."""
        return self._prepare_query(objects, rules, predicates, None).source

    def _prepare_query(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str],
        predicates: Iterable[Predicate],
        select: Iterable[str] | None,
    ) -> _PreparedQuery:
        rules = list(rules)
        predicates = list(predicates)
        nodes = self.collect(objects)
        names = self.node_names(nodes)
        lines = [self.source, *rules, ""]
        lines += [self._declare_predicate(p) for p in predicates]
        lines.append("")
        lines += self._closed_world(predicates, nodes)
        lines.append("")
        lines.append(self.facts(nodes, names, canonical_opposites=True))
        exists: dict[str, Certainty] = {}
        for node in nodes:
            name = names[self.identity(node)]
            certainty = self._exists(node)
            exists[name] = certainty
            if certainty == "may":
                lines.append(f"?exists({name}).")
            lines += [f"!{self.class_name(sub)}({name})." for sub in self._subclasses(type(node))]
        lines += self._predicate_facts(predicates, nodes, names)
        selected = list(select) if select is not None else _declared_predicates(rules)
        by_name = {names[self.identity(node)]: node for node in nodes}
        return _PreparedQuery(
            "\n".join(lines) + "\n",
            self._format(rules, selected),
            selected,
            by_name,
            exists,
            self.identity,
            self.reference_errors(nodes, names),
        )

    def _exists(self, node: BaseModel) -> Certainty:
        decide = self._options(type(node)).exists
        if decide is None:
            return "must"
        answer = decide(node)
        if answer in ("must", True):
            return "must"
        if answer == "may":
            return "may"
        raise ValueError(
            f"ClassOptions.exists of {type(node).__name__} must return 'must', 'may' or True, "
            f"not {answer!r} (leave out objects that don't exist)"
        )

    def _subclasses(self, cls: type) -> list[type[BaseModel]]:
        """Classes of the metamodel strictly below ``cls``."""
        return [c for c in self.classes if c is not cls and issubclass(c, cls)]

    def _parameter_type(self, parameter: type[BaseModel] | str) -> str:
        if isinstance(parameter, str):
            if not any(u.name == parameter for u in self.unions):
                raise ValueError(f"No union named {parameter!r} in this metamodel")
            return quote_id(parameter)
        if parameter not in self.features:
            raise ValueError(f"{parameter.__name__} is not part of this metamodel")
        return self.class_name(parameter)

    def _declare_predicate(self, predicate: Predicate) -> str:
        parameters = ", ".join(
            f"{self._parameter_type(t)} p{i}" for i, t in enumerate(predicate.parameters)
        )
        return f"pred {quote_id(predicate.name)}({parameters})."

    def _matches_type(self, node: BaseModel, parameter: type[BaseModel] | str) -> bool:
        if isinstance(parameter, str):
            union = next(u for u in self.unions if u.name == parameter)
            return isinstance(node, union.members)
        return isinstance(node, parameter)

    def _predicate_facts(
        self,
        predicates: Sequence[Predicate],
        nodes: Sequence[BaseModel],
        names: Mapping[Hashable, str],
    ) -> list[str]:
        lines = []
        for predicate in predicates:
            for arguments, answer in self._predicate_tuples(predicate, nodes):
                if answer in (None, False):
                    continue
                if answer not in (True, "must", "may"):
                    raise ValueError(
                        f"Predicate {predicate.name!r} returned {answer!r}; expected True, "
                        "False, None, 'must' or 'may'"
                    )
                prefix = "?" if answer == "may" else ""
                rendered = []
                for argument, parameter in zip(arguments, predicate.parameters, strict=True):
                    node_name = names.get(self.identity(argument))
                    if node_name is None or not self._matches_type(argument, parameter):
                        raise ValueError(
                            f"Predicate {predicate.name!r}: {argument!r} is not a node of type "
                            f"{getattr(parameter, '__name__', parameter)} in this graph"
                        )
                    rendered.append(node_name)
                lines.append(f"{prefix}{quote_id(predicate.name)}({', '.join(rendered)}).")
        return lines

    def _predicate_tuples(
        self, predicate: Predicate, nodes: Sequence[BaseModel]
    ) -> Iterator[tuple[tuple[BaseModel, ...], Any]]:
        arity = len(predicate.parameters)
        if predicate.evaluate is not None:
            domains = [[n for n in nodes if self._matches_type(n, t)] for t in predicate.parameters]
            for arguments in itertools.product(*domains):
                yield arguments, predicate.evaluate(*arguments)
            return
        for item in predicate.tuples or ():
            # ((a, b), "may") or (a, b); for arity 1, ((a,), "may"), (a,) or a.
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], tuple)
                and (arity != 2 or isinstance(item[1], str | bool))
            ):
                arguments, answer = item
            else:
                arguments, answer = (item if isinstance(item, tuple) else (item,)), True
            if len(arguments) != arity:
                raise ValueError(
                    f"Predicate {predicate.name!r} takes {arity} argument(s), got {arguments!r}"
                )
            yield arguments, answer

    def _canonical(self, feature: Feature) -> bool:
        """Of an opposite pair, the side facts and closed-world defaults are written on."""
        if feature.opposite is None:
            return True
        if feature.kind != "refers":
            return feature.kind == "contains"
        other = self._find_feature(feature.target, feature.opposite)  # type: ignore[arg-type]
        assert other is not None
        return (feature.owner.__name__, feature.name) <= (other.owner.__name__, other.name)

    def _closed_world(
        self, predicates: Sequence[Predicate], nodes: Sequence[BaseModel]
    ) -> list[str]:
        lines = [
            f"!exists({self.class_name(cls)}::new)."
            for cls in self.classes
            if not self._is_abstract(cls)
        ]
        orphans = self._orphans(nodes)
        for cls in self.classes:
            for feature in self.features[cls]:
                if feature.is_attribute or not self._canonical(feature):
                    continue
                if feature.kind == "contains":
                    target: ClassLike = feature.target  # type: ignore[assignment]
                    members = target.members if isinstance(target, UnionClass) else (target,)
                    if not any(isinstance(node, members) for node in orphans):
                        # Refinery already rules out every other container of
                        # a node that certainly has one. A default would only
                        # cost time: Refinery refines it for every node pair.
                        continue
                symbol = f"{self.class_name(cls)}::{quote_id(feature.name)}"
                lines.append(f"default !{symbol}(*, *).")
        for predicate in predicates:
            wildcards = ", ".join("*" for _ in predicate.parameters)
            lines.append(f"default !{quote_id(predicate.name)}({wildcards}).")
        return lines

    def _orphans(self, nodes: Sequence[BaseModel]) -> list[BaseModel]:
        """Nodes without a container that certainly exists and certainly contains them."""
        owned: set[Hashable] = set()
        for node in nodes:
            for _, feature, values in self._values(node):
                if feature.kind not in ("contains", "container"):
                    continue
                for value in values:
                    owner, child = (node, value) if feature.kind == "contains" else (value, node)
                    certain = not self._fact_prefix(node, feature, value)
                    if certain and self._exists(owner) == "must":
                        owned.add(self.identity(child))
        return [node for node in nodes if self.identity(node) not in owned]


_DEFAULT_OPTIONS = ClassOptions()

# Builtin symbols a node name shouldn't shadow.
_RESERVED_NODE_NAMES = frozenset(
    {
        *("node", "exists", "equals", "count"),
        *("contains", "contained", "container"),
        *("int", "real", "string", "boolean"),
    }
)


def _add_reference(
    node: BaseModel,
    feature: Feature,
    value: BaseModel,
    identity: Callable[[BaseModel], Hashable],
) -> bool:
    """Add ``value`` to ``node.<feature>`` unless already present or the slot is taken."""
    current = getattr(node, feature.name)
    if not feature.many:
        if current is not None:
            return False
        setattr(node, feature.name, value)
        return True
    if current is None:
        setattr(node, feature.name, [value])
        return True
    if any(identity(item) == identity(value) for item in current):
        return False
    if isinstance(current, list):
        current.append(value)
    elif isinstance(current, set):
        current.add(value)
    elif isinstance(current, frozenset):
        setattr(node, feature.name, frozenset((*current, value)))
    else:
        setattr(node, feature.name, (*current, value))
    return True


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------

ViolationKind = Literal["rule", "multiplicity", "containment", "contradiction", "reference"]


@dataclass(frozen=True, slots=True)
class Violation:
    """A constraint that the graph violates.

    ``kind`` is ``"rule"`` for user-defined error predicates,
    ``"multiplicity"`` for reference multiplicities, ``"containment"`` for
    containment hierarchy errors (missing or multiple containers, cycles),
    ``"contradiction"`` for facts that contradict the metamodel (e.g., a
    reference to a node of the wrong class), and ``"reference"`` for keys in
    reference-by-key fields that match no object (``...::unresolved``, with
    the key as second name) or several (``...::ambiguous``).
    """

    kind: ViolationKind
    constraint: str
    """Refinery relation that reported the error, e.g. ``Dir::contents::invalidMultiplicity``."""
    node_names: tuple[str, ...]
    nodes: tuple[BaseModel | None, ...] = field(compare=False)
    """The Python objects, or ``None`` for nodes Refinery added (e.g., enum literals)."""

    def __str__(self) -> str:
        return f"{self.kind}: {self.constraint}({', '.join(self.node_names)})"


class ModelValidationError(RefineryError):
    """Raised by :meth:`ValidationReport.raise_for_violations`."""

    result = "modelValidation"

    def __init__(self, violations: Sequence[Violation]) -> None:
        lines = [f"{len(violations)} Refinery constraint violation(s):"]
        lines += [f"  {violation}" for violation in violations]
        super().__init__("\n".join(lines))
        self.violations = list(violations)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    violations: list[Violation]
    source: str
    """The problem sent to Refinery (metamodel, rules, and facts)."""
    model: JsonOutput | None
    identity: Callable[[BaseModel], Hashable] = field(default=id, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok

    def for_node(self, node: BaseModel) -> list[Violation]:
        """Violations that involve ``node`` (or another instance with the same key)."""
        wanted = self.identity(node)
        return [
            v
            for v in self.violations
            if any(n is not None and self.identity(n) == wanted for n in v.nodes)
        ]

    def raise_for_violations(self) -> None:
        if self.violations:
            raise ModelValidationError(self.violations)


@dataclass(frozen=True, slots=True)
class Match:
    """A way a query predicate holds: its arguments, and whether it ``must`` or only ``may``."""

    predicate: str
    node_names: tuple[str, ...]
    nodes: tuple[BaseModel | None, ...] = field(compare=False)
    """The Python objects, or ``None`` for nodes Refinery added (e.g., enum literals)."""
    certainty: Certainty = "must"
    parameters: tuple[str, ...] = field(default=(), compare=False)
    """The names of the predicate's parameters, as declared."""

    def __str__(self) -> str:
        return f"{self.certainty}: {self.predicate}({', '.join(self.node_names)})"

    @property
    def arguments(self) -> dict[str, BaseModel | None]:
        """The Python objects by the name of the parameter they are bound to.

        Empty if Refinery didn't report the parameters' names.
        """
        return dict(zip(self.parameters, self.nodes, strict=True)) if self.parameters else {}

    def __getitem__(self, parameter: str) -> BaseModel | None:
        """The Python object bound to ``parameter``."""
        try:
            return self.nodes[self.parameters.index(parameter)]
        except ValueError:
            raise KeyError(f"{self.predicate} has no parameter named {parameter!r}") from None


@dataclass(frozen=True, slots=True)
class QueryResult:
    matches: list[Match]
    errors: list[Violation]
    """Constraints of the metamodel the graph violates (as reported by :meth:`Metamodel.validate`),
    and contradictions between facts."""
    source: str
    """The problem sent to Refinery."""
    model: JsonOutput | None

    def __getitem__(self, predicate: str) -> list[Match]:
        """The matches of ``predicate``."""
        return [m for m in self.matches if m.predicate == predicate]

    def must(self, predicate: str | None = None) -> list[Match]:
        """Matches that certainly hold (of ``predicate``, or of any)."""
        return [
            m for m in self.matches if m.certainty == "must" and predicate in (None, m.predicate)
        ]

    def may(self, predicate: str | None = None) -> list[Match]:
        """Matches that only may hold (of ``predicate``, or of any)."""
        return [
            m for m in self.matches if m.certainty == "may" and predicate in (None, m.predicate)
        ]


_PRED_DECLARATION = re.compile(r"\bpred\s+([A-Za-z_][A-Za-z0-9_]*|'(?:[^'\\]|\\.)*')")


def _unquote(name: str) -> str:
    if name.startswith("'"):
        return name[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return name


def _declared_predicates(rules: Iterable[str]) -> list[str]:
    """Names of the predicates declared in ``rules`` (``error`` predicates included)."""
    names = []
    for rule in rules:
        for match in _PRED_DECLARATION.finditer(rule):
            name = _unquote(match.group(1))
            if name not in names:
                names.append(name)
    return names


_HELPER_DEFINITION = re.compile(
    r"(?:\b(?P<kind>error|shadow)\s+)?\bpred\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*|'(?:[^'\\]|\\.)*')"
    # Skips an optional ``subsets`` clause; a declaration without a body ends with ``.``.
    r"\s*\([^)]*\)[^.<]*(?P<body><->)?"
)


def _helper_predicates(rules: Iterable[str], reported: Iterable[str]) -> list[str]:
    """Relations of the helper predicates declared in ``rules``, to leave out of the output.

    A helper is a predicate with a body (``<->``) that isn't an ``error``
    predicate and isn't ``reported``. Refinery computes it from other
    relations, so a contradiction in it also shows as an ``error`` in a
    relation that is returned. Its ``::computed`` relation is left out too.
    """
    reported = set(reported)
    excluded: list[str] = []
    for rule in rules:
        for match in _HELPER_DEFINITION.finditer(rule):
            name = _unquote(match.group("name"))
            if match.group("kind") == "error" or match.group("body") is None or name in reported:
                continue
            for relation in (name, f"{name}::computed"):
                if relation not in excluded:
                    excluded.append(relation)
    return excluded


@dataclass(frozen=True, slots=True)
class _PreparedQuery:
    source: str
    format: OutputFormats
    select: list[str]
    by_name: dict[str, BaseModel]
    exists: dict[str, Certainty]
    identity: Callable[[BaseModel], Hashable]
    reference_errors: list[Violation]

    def result(self, model: JsonOutput | None) -> QueryResult:
        if model is None:
            raise RefineryError("Refinery did not return a model to query")
        relations = {r.name: r for r in model.relations}
        matches = []
        for predicate in self.select:
            relation = relations.get(predicate)
            if relation is None:
                raise ValueError(f"Refinery returned no relation named {predicate!r}")
            for raw in model.partial_interpretation.get(relation.name, []):
                value = raw[-1]
                if value not in ("true", "unknown"):
                    continue
                names = tuple(model.nodes[i].name for i in raw[: relation.arity])  # type: ignore[index]
                if any(name.endswith("::new") for name in names):
                    continue  # objects that don't exist in a closed world
                certainty: Certainty = "must" if value == "true" else "may"
                # Refinery doesn't weaken a predicate for the existence of
                # its arguments (only of nodes quantified in its body).
                if any(self.exists.get(name) == "may" for name in names):
                    certainty = "may"
                nodes = tuple(self.by_name.get(name) for name in names)
                parameters = tuple(relation.parameter_names or ())
                matches.append(Match(predicate, names, nodes, certainty, parameters))
        errors = [
            error
            for error in _report(
                self.source, model, self.by_name, self.identity, self.reference_errors
            ).violations
            # Refinery still evaluates constraints on the C::new objects the
            # closed world says don't exist.
            if not any(name.endswith("::new") for name in error.node_names)
        ]
        return QueryResult(matches, errors, self.source, model)


def _timeout_hint(error: RefineryTimeoutError) -> RefineryTimeoutError:
    return RefineryTimeoutError(
        f"{error.message}. Validation runs under the server's semantics timeout (1 second by "
        "default); raise it with RefineryServer(semantics_timeout=<ms>) or the "
        "REFINERY_SEMANTICS_TIMEOUT_MS environment variable of the server. Refinery computes "
        "every relation in full, so helper predicates with many tuples (e.g., with three "
        "parameters) are often what makes it slow, and the upstream Refinery image is much "
        "slower than pyfinery's default on large graphs; see Performance in pyfinery's README."
    )


_MULTIPLICITY_SUFFIX = "::invalidMultiplicity"


def _classify(relation: str, is_error_predicate: bool) -> ViolationKind | None:
    if relation == "builtin::invalidContainer":
        return "containment"
    if relation.startswith("builtin::"):
        # Consequences of other errors (e.g., builtin::contains for a cycle).
        return None
    if is_error_predicate:
        return "multiplicity" if relation.endswith(_MULTIPLICITY_SUFFIX) else "rule"
    return "contradiction"


def _report(
    source: str,
    model: JsonOutput | None,
    by_name: dict[str, BaseModel],
    identity: Callable[[BaseModel], Hashable] = id,
    errors: Sequence[Violation] = (),
) -> ValidationReport:
    if model is None:
        raise RefineryError("Refinery did not return a model to validate")
    violations = list(errors)
    for relation in model.relations:
        is_error_predicate = relation.detail.type == "pred" and relation.detail.kind == "error"
        kind = _classify(relation.name, is_error_predicate)
        if kind is None:
            continue
        for raw in model.partial_interpretation.get(relation.name, []):
            if raw[-1] != "error":
                continue
            names = tuple(model.nodes[i].name for i in raw[: relation.arity])  # type: ignore[index]
            nodes = tuple(by_name.get(name) for name in names)
            violations.append(Violation(kind, relation.name, names, nodes))
    return ValidationReport(violations, source, model, identity)
