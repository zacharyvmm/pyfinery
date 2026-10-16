# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Define Refinery metamodels with Pydantic and validate object graphs against them.

Subclass :class:`RefineryNode` to declare classes. Fields become Refinery
features:

=================================  ===========================================
Python annotation                  Refinery declaration
=================================  ===========================================
``Dir``                            ``Dir[1] f``
``Dir | None``                     ``Dir[0..1] f``
``list[Dir]``                      ``Dir[0..*] f``
``Annotated[list[Dir], Contains()]``  ``contains Dir[0..*] f``
``Annotated[Dir | None, Container(opposite="g")]``  ``container Dir f opposite g``
``Annotated[..., Ref(opposite="g")]``  ``... opposite g``
a plain Pydantic ``BaseModel``     ``contains Address[1] f`` (see below)
an :class:`enum.Enum` subclass     ``enum`` declaration and a reference to it
``int`` / ``float`` / ``str`` / ``bool``  ``int`` / ``real`` / ``string`` / ``boolean``
=================================  ===========================================

Narrow list multiplicities with ``Field(min_length=..., max_length=...)`` (also
enforced by Pydantic when constructing objects) or ``Ref(multiplicity="1..3")``
(only checked by Refinery).

Plain Pydantic models (not :class:`RefineryNode`) nested in a field are value
objects owned by the enclosing object: the field becomes an implicit
containment reference, and the nested model becomes a Refinery class whose
fields are mapped the same way. Each nested instance must have exactly one
owner; use a :class:`RefineryNode` for objects referenced from several places.

Set ``__abstract__ = True`` in a class body to declare an abstract class.
Extra constraints are written as Refinery predicates, e.g.
``error pred emptyDir(Dir d) <-> !contents(d, _).``, and passed as ``rules``.

:meth:`Metamodel.validate` asserts the object graph as facts, closes the world
(anything not asserted is false) with Refinery's ``concretize`` endpoint, and
reports every constraint that evaluates to ``error``.
"""

from __future__ import annotations

import enum
import inspect
import math
import re
import types
from collections.abc import Iterable, Iterator, Sequence
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


def _string_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
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
class Feature:
    """A field of a Pydantic model class mapped to Refinery."""

    owner: type[BaseModel]
    name: str
    target: type[BaseModel] | type[enum.Enum] | str
    """Model class, enum class, or data type name for attributes."""
    many: bool
    lower: int
    upper: int | None
    kind: ReferenceKind
    opposite: str | None

    @property
    def is_attribute(self) -> bool:
        return isinstance(self.target, str)

    @property
    def is_enum(self) -> bool:
        return isinstance(self.target, type) and issubclass(self.target, enum.Enum)


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == len(get_args(annotation)) - 1 and len(args) == 1:
            return args[0], True
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


def _is_embedded(value: Any) -> bool:
    """Plain Pydantic models, which are owned by (contained in) their parent."""
    return _is_model_class(value) and not issubclass(value, RefineryNode)


class Metamodel:
    """A Refinery metamodel derived from :class:`RefineryNode` subclasses
    (and the plain Pydantic models nested in them).

    >>> metamodel = Metamodel(Filesystem, rules=[
    ...     "error pred emptyDir(Dir d) <-> !contents(d, _).",
    ... ])
    >>> print(metamodel.source)
    >>> report = metamodel.validate([my_filesystem], refinery)

    Classes reachable from the given ones (through superclasses, references,
    and already-imported subclasses) are included automatically.
    """

    def __init__(self, *classes: type[BaseModel], rules: Iterable[str] = ()) -> None:
        if not classes:
            raise ValueError("Metamodel needs at least one model class")
        self.rules = list(rules)
        self.classes: list[type[BaseModel]] = []
        self.enums: list[type[enum.Enum]] = []
        self.features: dict[type[BaseModel], list[Feature]] = {}
        self._collect(classes)
        self._check_opposites()
        self.source = self._render()

    # -- discovery ----------------------------------------------------------

    def _collect(self, roots: Iterable[type[BaseModel]]) -> None:
        pending = list(roots)
        seen: set[type[BaseModel]] = set()
        while pending:
            cls = pending.pop(0)
            if cls in seen or cls is RefineryNode:
                continue
            if not _is_model_class(cls):
                raise TypeError(f"{cls!r} is not a RefineryNode subclass or Pydantic model")
            seen.add(cls)
            self._rebuild(cls)
            self.classes.append(cls)
            pending.extend(b for b in cls.__bases__ if _is_model_class(b))
            pending.extend(cls.__subclasses__())
            features = [self._feature(cls, name) for name in self._own_fields(cls)]
            self.features[cls] = features
            for feature in features:
                if feature.is_enum:
                    if feature.target not in self.enums:
                        self.enums.append(feature.target)  # type: ignore[arg-type]
                elif not feature.is_attribute:
                    pending.append(feature.target)  # type: ignore[arg-type]
        # Superclasses must be declared before subclasses for readability.
        self.classes.sort(key=lambda c: (len(c.__mro__), c.__qualname__))

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

    def _feature(self, cls: type[BaseModel], name: str) -> Feature:
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
        elif origin is not None:
            raise TypeError(f"{where}: unsupported field type {info.annotation!r}")

        refs = [m for m in metadata if isinstance(m, Ref)]
        if len(refs) > 1:
            raise TypeError(f"{where}: more than one Ref/Contains/Container marker")
        ref = refs[0] if refs else Ref()

        if annotation in _DATA_TYPES:
            if many or refs:
                raise TypeError(f"{where}: attributes cannot be collections or have Ref markers")
            return Feature(cls, name, _DATA_TYPES[annotation], False, 0, 1, "refers", None)

        if not (_is_model_class(annotation) or _is_enum_class(annotation)):
            raise TypeError(
                f"{where}: unsupported field type {annotation!r}; use a RefineryNode subclass, "
                "a Pydantic model, an Enum, int, float, str, or bool"
            )
        if _is_enum_class(annotation) and ref.kind != "refers":
            raise TypeError(f"{where}: enums cannot be containment references")
        kind = ref.kind
        if _is_embedded(annotation):
            # Nested plain models are value objects owned by this object.
            if ref.kind == "container" or ref.opposite is not None:
                raise TypeError(
                    f"{where}: {annotation.__name__} is a plain Pydantic model, which is always "
                    "contained in its owner and cannot have opposites; make it a RefineryNode "
                    "to use opposites or Container()"
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
        return Feature(cls, name, annotation, many, lower, upper, kind, ref.opposite)

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
    def class_name(node_class: type) -> str:
        return quote_id(node_class.__name__)

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
        for enum_class in self.enums:
            literals = ", ".join(quote_id(member.name) for member in enum_class)
            lines.append(f"enum {self.class_name(enum_class)} {{ {literals} }}")
        for cls in self.classes:
            header = f"class {self.class_name(cls)}"
            if cls.__dict__.get("__abstract__", False):
                header = f"abstract {header}"
            supers = [b for b in cls.__bases__ if _is_model_class(b)]
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
        seen: set[int] = set()
        pending = list(objects)
        while pending:
            node = pending.pop(0)
            if id(node) in seen:
                continue
            if type(node) not in self.features:
                raise TypeError(f"{type(node).__name__} is not part of this metamodel")
            seen.add(id(node))
            result.append(node)
            for _, feature, value in self._values(node):
                if not feature.is_attribute and not feature.is_enum:
                    pending.extend(value)
        return result

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
                    if other is not None and _add_reference(target, other, node):
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

    def node_names(self, nodes: Sequence[BaseModel]) -> dict[int, str]:
        """Refinery node names (as rendered in the problem) keyed by ``id(node)``."""
        names: dict[int, str] = {}
        used: set[str] = set()
        counters: dict[type, int] = {}
        for node in nodes:
            id_field: str | None = getattr(type(node), "__node_id__", None)
            if id_field is not None:
                name = quote_id(str(getattr(node, id_field)))
                if name in used:
                    raise ValueError(f"Duplicate node id {name} (from field {id_field!r})")
            else:
                while True:
                    counters[type(node)] = counters.get(type(node), 0) + 1
                    name = quote_id(f"{type(node).__name__.lower()}_{counters[type(node)]}")
                    if name not in used:
                        break
            used.add(name)
            names[id(node)] = name
        return names

    def facts(self, nodes: Sequence[BaseModel], names: dict[int, str]) -> str:
        """Assertions describing ``nodes`` (which must include every referenced node)."""
        lines = []
        for node in nodes:
            name = names[id(node)]
            lines.append(f"{self.class_name(type(node))}({name}).")
            for cls, feature, values in self._values(node):
                symbol = f"{self.class_name(cls)}::{quote_id(feature.name)}"
                for value in values:
                    if feature.is_attribute:
                        literal = _attribute_literal(value, feature.target)  # type: ignore[arg-type]
                        lines.append(f"{symbol}({name}): {literal}.")
                    elif feature.is_enum:
                        enum_name = self.class_name(feature.target)  # type: ignore[arg-type]
                        literal = f"{enum_name}::{quote_id(value.name)}"
                        lines.append(f"{symbol}({name}, {literal}).")
                    else:
                        lines.append(f"{symbol}({name}, {names[id(value)]}).")
        return "\n".join(lines) + "\n"

    def problem(self, objects: Iterable[BaseModel]) -> str:
        """The metamodel, rules, and the facts of the graph reachable from ``objects``."""
        nodes = self.collect(objects)
        return self.source + "\n" + self.facts(nodes, self.node_names(nodes))

    # -- validation ---------------------------------------------------------

    _FORMAT = OutputFormats(json_=JsonOutputFormat(), source=SourceOutputFormat(enabled=False))

    def _prepare(self, objects: Iterable[BaseModel]) -> tuple[str, dict[str, BaseModel]]:
        nodes = self.collect(objects)
        names = self.node_names(nodes)
        source = self.source + "\n" + self.facts(nodes, names)
        return source, {names[id(node)]: node for node in nodes}

    def validate(self, objects: Iterable[BaseModel], refinery: Refinery) -> ValidationReport:
        """Check the graph reachable from ``objects`` against the metamodel and rules.

        Only nodes reachable from ``objects`` through outgoing references are
        checked (see :meth:`collect`). Raises
        :class:`~pyfinery.InvalidProblemError` if the rules don't compile;
        constraint violations are returned in the report.
        """
        source, by_name = self._prepare(objects)
        try:
            result = refinery.concretize(source, format=self._FORMAT)
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return _report(source, result.json_, by_name)

    async def avalidate(
        self, objects: Iterable[BaseModel], refinery: AsyncRefinery
    ) -> ValidationReport:
        """Asynchronous variant of :meth:`validate`."""
        source, by_name = self._prepare(objects)
        try:
            result = await refinery.concretize(source, format=self._FORMAT)
        except RefineryTimeoutError as e:
            raise _timeout_hint(e) from e
        return _report(source, result.json_, by_name)


def _add_reference(node: BaseModel, feature: Feature, value: BaseModel) -> bool:
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
    if any(item is value for item in current):
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

ViolationKind = Literal["rule", "multiplicity", "containment", "contradiction"]


@dataclass(frozen=True, slots=True)
class Violation:
    """A constraint that the graph violates.

    ``kind`` is ``"rule"`` for user-defined error predicates,
    ``"multiplicity"`` for reference multiplicities, ``"containment"`` for
    containment hierarchy errors (missing or multiple containers, cycles),
    and ``"contradiction"`` for facts that contradict the metamodel (e.g., a
    reference to a node of the wrong class).
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

    @property
    def ok(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok

    def for_node(self, node: BaseModel) -> list[Violation]:
        """Violations that involve ``node``."""
        return [v for v in self.violations if any(n is node for n in v.nodes)]

    def raise_for_violations(self) -> None:
        if self.violations:
            raise ModelValidationError(self.violations)


def _timeout_hint(error: RefineryTimeoutError) -> RefineryTimeoutError:
    return RefineryTimeoutError(
        f"{error.message}. Validation runs under the server's semantics timeout (1 second by "
        "default); raise it with RefineryServer(semantics_timeout=<ms>) or the "
        "REFINERY_SEMANTICS_TIMEOUT_MS environment variable of the server."
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
    source: str, model: JsonOutput | None, by_name: dict[str, BaseModel]
) -> ValidationReport:
    if model is None:
        raise RefineryError("Refinery did not return a model to validate")
    violations = []
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
    return ValidationReport(violations, source, model)
