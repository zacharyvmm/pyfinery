# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""How Pydantic classes map to Refinery: field markers, class options, Python
predicates, and :class:`RefineryNode`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import FieldInfo

ReferenceKind = Literal["refers", "contains", "container"]
# The values of Refinery's four-valued logic. Facts are asserted, and matches
# reported, as ``"true"`` or ``"unknown"``; a contradiction is ``"error"``.
LogicValue = Literal["true", "false", "unknown", "error"]


@dataclass(frozen=True, slots=True)
class RefOptions:
    """The Refinery-specific options of a reference field, as :func:`Ref`,
    :func:`Contains` and :func:`Container` record them in the field's metadata."""

    opposite: str | None = None
    kind: ReferenceKind = "refers"
    multiplicity: str | tuple[int, int | None] | None = None


def Ref(  # noqa: N802 - reads like a type annotation
    opposite: str | None = None,
    multiplicity: str | tuple[int, int | None] | None = None,
    **field: Any,
) -> Any:
    """A reference field with Refinery-specific options, as the field's default::

        class Link(RefineryNode):
            backups: list[FSObject] = Ref(multiplicity="0..3")

    Like Pydantic's ``Field``, whose arguments (``description``, ``alias``,
    ``default_factory``...) it also takes. In a :class:`RefineryNode`, a
    collection field left out when creating an object is empty, and an
    optional one ``None``; other fields are required unless given a default.
    It also works inside ``Annotated``: ``Annotated[list[FSObject], Ref()]``.

    :param opposite: Name of the opposite reference in the target class.
    :param multiplicity: Overrides the multiplicity inferred from the type,
        e.g. ``"1..*"``, ``"2"`` or ``(1, None)``.
    """
    return _field(RefOptions(opposite, "refers", multiplicity), field)


def Contains(  # noqa: N802
    opposite: str | None = None,
    multiplicity: str | tuple[int, int | None] | None = None,
    **field: Any,
) -> Any:
    """A containment reference (``contains``): the targets live inside this object.

    Takes the arguments of :func:`Ref`::

        class Dir(RefineryNode):
            contents: list[FSObject] = Contains(multiplicity="0..3")
    """
    return _field(RefOptions(opposite, "contains", multiplicity), field)


def Container(opposite: str | None = None, **field: Any) -> Any:  # noqa: N802
    """The container side of a containment reference (``container``): the object this
    one lives in::

        class FSObject(RefineryNode):
            parent: Dir | None = Container()

    ``opposite`` names the containment reference of the target class it is the
    opposite of; it can be left out when exactly one fits. Takes Pydantic's
    ``Field`` arguments too.
    """
    return _field(RefOptions(opposite, "container"), field)


def _field(options: RefOptions, field: dict[str, Any]) -> FieldInfo:
    info: FieldInfo = Field(**field)
    info.metadata.append(options)
    return info


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
    :param exists: The logic value of ``exists`` for an instance: ``"true"``
        (or ``True``) if it certainly exists, ``"unknown"`` if it only may,
        e.g. something that happens on some paths only. Used by
        :meth:`Metamodel.query` only.
    :param logic_value: The logic value of a field's value, called as
        ``logic_value(instance, field_name, value)`` for each reference
        target and attribute value: ``"true"`` (or ``True``) if it certainly
        holds, ``"unknown"`` if it only may, e.g. a dependency found by a
        heuristic.
    """

    key: Callable[[Any], Hashable] | None = None
    references: Mapping[str, type[BaseModel] | Sequence[type[BaseModel]] | str] | None = None
    node_name: Callable[[Any], str] | None = None
    exclude: frozenset[str] | Sequence[str] = ()
    abstract: bool | None = None
    exists: Callable[[Any], LogicValue | bool] | None = None
    logic_value: Callable[[Any, str, Any], LogicValue | bool] | None = None


@dataclass(frozen=True, slots=True)
class Predicate:
    """A relation computed in Python, usable in the rules of :meth:`Metamodel.validate`
    and :meth:`Metamodel.query`.

    For what Refinery can't express, such as regular expressions, globs, or
    relations computed elsewhere. It is declared as ``pred <name>(<types>).``,
    where each parameter type is a model class or the name of a union, and
    its facts come from either:

    - ``evaluate``, called on every combination of nodes of the parameter
      types, returning its logic value: ``True``/``"true"`` (holds),
      ``"unknown"`` (may hold), or ``False``/``"false"``/``None`` (doesn't
      hold); or
    - ``tuples``: the argument tuples that hold, each optionally paired with
      its logic value, e.g. ``[((a, b), "unknown"), (c, d)]``. Use it when the
      relation is already known, or when combinations would be too many.

    >>> Predicate("validEmail", (User,), lambda u: EMAIL.fullmatch(u.email) is not None)
    >>> Predicate("calls", (Service, Service), tuples=traced_calls)
    """

    name: str
    parameters: tuple[type[BaseModel] | str, ...]
    evaluate: Callable[..., LogicValue | bool | None] | None = None
    tuples: Iterable[Any] | None = None

    def __post_init__(self) -> None:
        if (self.evaluate is None) == (self.tuples is None):
            raise ValueError(f"Predicate {self.name!r} needs exactly one of evaluate or tuples")
        if self.tuples is not None:
            # Allow one-shot iterables (e.g. generators) to be read once.
            object.__setattr__(self, "tuples", list(self.tuples))


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

    @model_validator(mode="before")
    @classmethod
    def _empty_references(cls, data: Any) -> Any:
        """Fill in the reference fields declared with a marker and no default: a collection
        is empty, an optional field ``None``."""
        if not isinstance(data, dict):
            return data
        missing = {
            cls.model_fields[name].alias or name: empty()
            for name, empty in _empty_defaults(cls).items()
            if name not in data and cls.model_fields[name].alias not in data
        }
        return {**data, **missing} if missing else data

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


_EMPTY: dict[type, dict[str, Callable[[], Any]]] = {}


def _empty_defaults(cls: type[BaseModel]) -> dict[str, Callable[[], Any]]:
    """How to make the value of each of ``cls``'s required reference fields declared with a
    marker when left out: an empty collection, or ``None`` if optional."""
    if cls not in _EMPTY:
        from ._schema import empty_factory  # _schema imports this module

        found = {}
        for name, info in cls.model_fields.items():
            if info.is_required() and any(isinstance(m, RefOptions) for m in info.metadata):
                factory = empty_factory(info)
                if factory is not None:
                    found[name] = factory
        _EMPTY[cls] = found
    return _EMPTY[cls]
