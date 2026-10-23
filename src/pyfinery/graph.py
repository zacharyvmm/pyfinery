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
  reports each :class:`Match` with its value in Refinery's four-valued
  logic: ``"true"``, or ``"unknown"`` if it only may hold.

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

This module holds :class:`Metamodel` and re-exports the rest, which lives in
private modules: the options in ``_options``, field descriptions in
``_features``, the graph of nodes built from objects in ``_instances``,
Refinery syntax in ``_syntax``, and reports in ``_results``.
"""

from __future__ import annotations

import inspect
import itertools
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

import annotated_types
from pydantic import BaseModel

from ._features import (
    _COLLECTIONS,
    _DATA_TYPES,
    ClassLike,
    EnumLike,
    Feature,
    LiteralEnum,
    UnionClass,
    _camel,
    _is_enum_class,
    _is_model_class,
    _is_union,
    _strip_annotated,
    _strip_optional,
    _Unsupported,
)
from ._instances import _Graph, _Node, _Value
from ._options import (
    ClassOptions,
    Container,
    Contains,
    LogicValue,
    Predicate,
    Ref,
    ReferenceKind,
    RefineryNode,
)
from ._results import (
    Match,
    ModelValidationError,
    QueryResult,
    ValidationReport,
    Violation,
    ViolationKind,
    _declared_predicates,
    _PreparedQuery,
    _report,
    _timeout_hint,
)
from ._syntax import (
    _RESERVED_NODE_NAMES,
    _attribute_literal,
    _encodable,
    _multiplicity,
    _parse_multiplicity,
    quote_id,
)
from .errors import RefineryTimeoutError
from .models import JsonOutputFormat, OutputFormats, SourceOutputFormat

if TYPE_CHECKING:
    from .client import AsyncRefinery, Refinery

__all__ = [
    "ClassLike",
    "ClassOptions",
    "Container",
    "Contains",
    "EnumLike",
    "Feature",
    "LiteralEnum",
    "LogicValue",
    "Match",
    "Metamodel",
    "ModelValidationError",
    "Predicate",
    "QueryResult",
    "Ref",
    "ReferenceKind",
    "RefineryNode",
    "UnionClass",
    "ValidationReport",
    "Violation",
    "ViolationKind",
    "quote_id",
]


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
        return [node.obj for node, _ in self._reach(objects, {})]

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

    # -- the instance graph -------------------------------------------------

    def _graph(self, objects: Iterable[BaseModel], *, existence: bool = False) -> _Graph:
        """The Refinery nodes of the graph reachable from ``objects``, with their values.

        The ``key``, ``node_name``, ``logic_value`` and (with ``existence``)
        ``exists`` options are evaluated here, once; everything sent to
        Refinery is rendered from the result. Without ``existence``, every
        node exists, as :meth:`validate` assumes.
        """
        identities: dict[int, tuple[BaseModel, Hashable]] = {}
        reached = self._reach(objects, identities)
        nodes = [node for node, _ in reached]
        graph = _Graph(nodes, {node.identity: node for node in nodes}, identities)
        self._name(graph.nodes)
        index: dict[Hashable, list[_Node]] = {}
        for node in graph.nodes:
            if self._options(type(node.obj)).key is not None:
                index.setdefault(node.identity[1], []).append(node)  # type: ignore[index]

        def node_of(obj: BaseModel) -> _Node:
            # Every followed reference was reached, by the same identity.
            return graph.by_identity[self._identity(obj, graph.identities)]

        for node, fields in reached:
            for _, feature, values in fields:
                for value in values:
                    node.values.append(self._value(node, feature, value, index, node_of))
        if existence:
            for node in graph.nodes:
                node.exists = self._exists(node.obj)
        return graph

    def _reach(
        self, objects: Iterable[BaseModel], identities: dict[int, tuple[BaseModel, Hashable]]
    ) -> list[tuple[_Node, list[tuple[type, Feature, list[Any]]]]]:
        """The nodes reachable from ``objects``, each with the values of its fields.

        ``identities`` memoizes :meth:`identity` by object, keeping the
        objects alive so that their ``id`` isn't reused.
        """
        reached = []
        seen: set[Hashable] = set()
        pending = deque(objects)
        while pending:
            obj = pending.popleft()
            if type(obj) not in self.features:
                raise TypeError(f"{type(obj).__name__} is not part of this metamodel")
            identity = self._identity(obj, identities)
            if identity in seen:
                continue
            seen.add(identity)
            fields = list(self._values(obj))
            reached.append((_Node(obj, identity), fields))
            for _, feature, values in fields:
                if not (feature.is_attribute or feature.is_enum or feature.by_key):
                    pending.extend(values)
        return reached

    def _identity(
        self, obj: BaseModel, identities: dict[int, tuple[BaseModel, Hashable]]
    ) -> Hashable:
        entry = identities.get(id(obj))
        if entry is None:
            entry = identities[id(obj)] = (obj, self.identity(obj))
        return entry[1]

    def _name(self, nodes: Sequence[_Node]) -> None:
        """Give every node the name it has in the problem."""
        used: set[str] = set()
        counters: dict[type, int] = {}
        for node in nodes:
            obj = node.obj
            id_field: str | None = getattr(type(obj), "__node_id__", None)
            naming = self._options(type(obj)).node_name
            if naming is not None or id_field is not None:
                raw = naming(obj) if naming is not None else getattr(obj, id_field)  # type: ignore[arg-type]
                name = quote_id(str(raw))
                if name in used:
                    source = "node_name" if naming is not None else f"field {id_field!r}"
                    raise ValueError(f"Duplicate node id {name} (from {source})")
            elif (key := self._name_from_key(node)) is not None and key not in used:
                name = key
            else:
                while True:
                    counters[type(obj)] = counters.get(type(obj), 0) + 1
                    name = quote_id(f"{type(obj).__name__.lower()}_{counters[type(obj)]}")
                    if name not in used:
                        break
            used.add(name)
            node.name = name

    def _name_from_key(self, node: _Node) -> str | None:
        """The key as a node name, if it is a string or integer that isn't a type's name."""
        if self._options(type(node.obj)).key is None:
            return None
        key = node.identity[1]  # type: ignore[index]
        if isinstance(key, bool) or not isinstance(key, str | int) or str(key) == "":
            return None
        if not _encodable(str(key)):
            # Escaped, Refinery would name the node with "?" for each lone
            # surrogate, so the name couldn't be mapped back to the node.
            return None
        name = quote_id(str(key))
        return None if name in self._type_names or str(key) in _RESERVED_NODE_NAMES else name

    def _value(
        self,
        node: _Node,
        feature: Feature,
        value: Any,
        index: Mapping[Hashable, list[_Node]],
        node_of: Callable[[BaseModel], _Node],
    ) -> _Value:
        if feature.by_key:
            target = feature.target
            members = target.members if isinstance(target, UnionClass) else (target,)
            found = [n for n in index.get(value, []) if isinstance(n.obj, members)]  # type: ignore[arg-type]
            if len(found) != 1:
                problem = "unresolved" if not found else "ambiguous"
                return _Value(feature, value, unresolved=problem)
            return _Value(feature, value, self._certain(node.obj, feature, value), found[0])
        certain = self._certain(node.obj, feature, value)
        if feature.is_attribute or feature.is_enum:
            return _Value(feature, value, certain)
        return _Value(feature, value, certain, node_of(value))

    def _certain(self, node: BaseModel, feature: Feature, value: Any) -> bool:
        """Whether ``value`` certainly holds, as ``ClassOptions.logic_value`` says."""
        decide = self._options(type(node)).logic_value
        if decide is None:
            return True
        answer = decide(node, feature.name, value)
        if answer in ("true", True):
            return True
        if answer == "unknown":
            return False
        raise ValueError(
            f"ClassOptions.logic_value of {type(node).__name__} must return 'true', 'unknown' or "
            f"True, "
            f"not {answer!r} (leave values that don't hold out of the object)"
        )

    # -- rendering instances ------------------------------------------------

    def _facts(self, graph: _Graph, *, canonical_opposites: bool = False) -> str:
        """Assertions describing the nodes of ``graph`` and their values.

        With ``canonical_opposites``, a reference on one side of an opposite
        pair is written as the reverse reference on the other side, so that
        closed-world defaults on that side don't contradict it.
        """
        lines = []
        for node in graph.nodes:
            name = node.name
            lines.append(f"{self.class_name(type(node.obj))}({name}).")
            for value in node.values:
                feature = value.feature
                symbol = f"{self.class_name(feature.owner)}::{quote_id(feature.name)}"
                prefix = "" if value.certain else "?"
                if feature.by_key:
                    if value.target is not None:  # others are reported by _reference_errors
                        lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
                elif canonical_opposites and not self._canonical(feature):
                    assert value.target is not None
                    other = self._find_feature(type(value.target.obj), feature.opposite)  # type: ignore[arg-type]
                    assert other is not None
                    reverse = f"{self.class_name(other.owner)}::{quote_id(other.name)}"
                    lines.append(f"{prefix}{reverse}({value.target.name}, {name}).")
                elif feature.is_attribute:
                    if value.certain:  # an uncertain attribute value is simply unknown
                        literal = _attribute_literal(value.value, feature.target)  # type: ignore[arg-type]
                        lines.append(f"{symbol}({name}): {literal}.")
                elif feature.is_enum:
                    literal = self._literal(feature.target, value.value)  # type: ignore[arg-type]
                    lines.append(f"{prefix}{symbol}({name}, {literal}).")
                else:
                    assert value.target is not None
                    lines.append(f"{prefix}{symbol}({name}, {value.target.name}).")
        return "\n".join(lines) + "\n"

    def _reference_errors(self, graph: _Graph) -> list[Violation]:
        """Keys in reference-by-key fields that match no object, or several."""
        errors = []
        for node in graph.nodes:
            for value in node.values:
                if value.unresolved is None:
                    continue
                feature = f"{self.class_name(value.feature.owner)}::{quote_id(value.feature.name)}"
                constraint = f"{feature}::{value.unresolved}"
                node_names = (node.name, repr(value.value))
                errors.append(Violation("reference", constraint, node_names, (node.obj, None)))
        return errors

    def problem(self, objects: Iterable[BaseModel]) -> str:
        """The metamodel, rules, and the facts of the graph reachable from ``objects``."""
        return self.source + "\n" + self._facts(self._graph(objects))

    # -- validation ---------------------------------------------------------

    _FORMAT = OutputFormats(json_=JsonOutputFormat(), source=SourceOutputFormat(enabled=False))

    def _prepare(
        self,
        objects: Iterable[BaseModel],
        rules: Iterable[str] = (),
        predicates: Iterable[Predicate] = (),
    ) -> tuple[str, dict[str, BaseModel], list[Violation]]:
        predicates = list(predicates)
        graph = self._graph(objects)
        lines = [self.source, *rules, *(self._declare_predicate(p) for p in predicates), ""]
        lines.append(self._facts(graph))
        lines += self._predicate_facts(predicates, graph)
        return "\n".join(lines) + "\n", graph.objects_by_name(), self._reference_errors(graph)

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
        are only ``"unknown"`` (from :class:`Predicate` or
        ``ClassOptions(logic_value=...)``) count as not holding, and every
        object exists (``ClassOptions(exists=...)`` is ignored). Use
        :meth:`query` to reason about what may hold.

        Only nodes reachable from ``objects`` through outgoing references are
        checked (see :meth:`collect`). Raises
        :class:`~pyfinery.InvalidProblemError` if the rules don't compile;
        constraint violations are returned in the report.
        """
        source, by_name, errors = self._prepare(objects, rules, predicates)
        try:
            result = refinery.concretize(source, format=self._FORMAT)
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
        source, by_name, errors = self._prepare(objects, rules, predicates)
        try:
            result = await refinery.concretize(source, format=self._FORMAT)
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
        or subclass memberships) while keeping uncertainty: objects whose
        ``exists`` is ``"unknown"`` and :class:`Predicate` facts that are
        ``"unknown"`` make the matches that depend on them ``"unknown"``.
        """
        prepared = self._prepare_query(objects, rules, predicates, select)
        try:
            result = refinery.semantics(prepared.source, format=self._FORMAT)
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
            result = await refinery.semantics(prepared.source, format=self._FORMAT)
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
        graph = self._graph(objects, existence=True)
        lines = [self.source, *rules, ""]
        lines += [self._declare_predicate(p) for p in predicates]
        lines.append("")
        lines += self._closed_world(predicates, graph)
        lines.append("")
        lines.append(self._facts(graph, canonical_opposites=True))
        for node in graph.nodes:
            if node.exists == "unknown":
                lines.append(f"?exists({node.name}).")
            subclasses = self._subclasses(type(node.obj))
            lines += [f"!{self.class_name(sub)}({node.name})." for sub in subclasses]
        lines += self._predicate_facts(predicates, graph)
        selected = list(select) if select is not None else _declared_predicates(rules)
        return _PreparedQuery(
            "\n".join(lines) + "\n",
            selected,
            graph.objects_by_name(),
            {node.name: node.exists for node in graph.nodes},
            self.identity,
            self._reference_errors(graph),
        )

    def _exists(self, node: BaseModel) -> LogicValue:
        decide = self._options(type(node)).exists
        if decide is None:
            return "true"
        answer = decide(node)
        if answer in ("true", True):
            return "true"
        if answer == "unknown":
            return "unknown"
        raise ValueError(
            f"ClassOptions.exists of {type(node).__name__} must return 'true', 'unknown' or True, "
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

    def _predicate_facts(self, predicates: Sequence[Predicate], graph: _Graph) -> list[str]:
        lines = []
        objects = [node.obj for node in graph.nodes]
        for predicate in predicates:
            for arguments, answer in self._predicate_tuples(predicate, objects):
                if answer in (None, False, "false"):
                    continue
                if answer not in (True, "true", "unknown"):
                    raise ValueError(
                        f"Predicate {predicate.name!r} returned {answer!r}; expected True, "
                        "False, None, 'true', 'false' or 'unknown'"
                    )
                prefix = "?" if answer == "unknown" else ""
                rendered = []
                for argument, parameter in zip(arguments, predicate.parameters, strict=True):
                    node = graph.by_identity.get(self._identity(argument, graph.identities))
                    if node is None or not self._matches_type(argument, parameter):
                        raise ValueError(
                            f"Predicate {predicate.name!r}: {argument!r} is not a node of type "
                            f"{getattr(parameter, '__name__', parameter)} in this graph"
                        )
                    rendered.append(node.name)
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
            # ((a, b), "unknown") or (a, b); for arity 1, ((a,), "unknown"), (a,) or a.
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

    def _closed_world(self, predicates: Sequence[Predicate], graph: _Graph) -> list[str]:
        lines = [
            f"!exists({self.class_name(cls)}::new)."
            for cls in self.classes
            if not self._is_abstract(cls)
        ]
        orphans = [node.obj for node in self._orphans(graph)]
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

    @staticmethod
    def _orphans(graph: _Graph) -> list[_Node]:
        """Nodes without a container that certainly exists and certainly contains them."""
        owned: set[_Node] = set()
        for node in graph.nodes:
            for value in node.values:
                kind = value.feature.kind
                if kind not in ("contains", "container"):
                    continue
                assert value.target is not None
                owner, child = (node, value.target) if kind == "contains" else (value.target, node)
                if value.certain and owner.exists == "true":
                    owned.add(child)
        return [node for node in graph.nodes if node not in owned]


_DEFAULT_OPTIONS = ClassOptions()


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
