"""Reference fields declared with a marker as the default (``contents: list[X] = Contains()``):
what they map to, what they hold when left out, and inferred container opposites."""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from pyfinery import Container, Contains, Metamodel, Ref, RefineryNode

from .graph_models import Dir, File, Filesystem, Link, metamodel

# -- the file system of graph_models, declared with Annotated ----------------------------------


class OldFilesystem(RefineryNode):
    root: Annotated[OldDir, Contains()]


class OldFSObject(RefineryNode):
    __abstract__ = True
    parent: Annotated[OldDir | None, Container(opposite="contents")] = None


class OldDir(OldFSObject):
    name: str
    contents: Annotated[list[OldFSObject], Contains(opposite="parent", multiplicity="0..3")] = (
        Field(default_factory=list)
    )


class OldLink(OldFSObject):
    target: OldFSObject
    backups: Annotated[list[OldFSObject], Ref()] = Field(default_factory=list, min_length=0)


def _features(metamodel: Metamodel) -> set[tuple[str, str, str, str | None, int, int | None]]:
    return {
        (cls.__name__, f.name, f.kind, f.opposite, f.lower, f.upper)
        for cls, features in metamodel.schema.features.items()
        for f in features
    }


def test_maps_as_annotated_markers_do():
    new = _features(metamodel())
    old = {(c.removeprefix("Old"), *rest) for c, *rest in _features(Metamodel(OldFilesystem))}
    assert old <= new
    assert ("FSObject", "parent", "container", "contents", 0, 1) in new
    assert ("Dir", "contents", "contains", "parent", 0, 3) in new


def test_fields_left_out_are_empty():
    a, b = Dir(name="a"), Dir(name="b")
    assert a.contents == [] and a.parent is None
    assert a.contents is not b.contents  # a new list for each object
    assert Link(target=a).backups == []
    given = File()
    assert Dir(name="c", contents=[given]).contents == [given]
    with pytest.raises(ValidationError, match="root"):
        Filesystem()  # a single reference that isn't optional stays required


def test_pydantic_field_arguments_pass_through():
    class Box(RefineryNode):
        items: list[Box] = Contains(alias="things", description="What the box holds")
        tags: set[Box] = Ref()
        pair: tuple[Box, ...] = Ref()
        made: list[Box] = Ref(default_factory=lambda: ["?"])  # an explicit default wins

    assert Box.model_fields["items"].description == "What the box holds"
    box = Box()
    assert (box.items, box.tags, box.pair, box.made) == ([], set(), (), ["?"])
    inner = Box()
    assert Box(things=[inner]).items == [inner]


def test_annotated_markers_still_work():
    class Node(RefineryNode):
        children: Annotated[list[Node], Contains(opposite="up")] = Field(default_factory=list)
        up: Annotated[Node | None, Container(opposite="children")] = None
        peers: Annotated[list[Node], Ref()]  # no default: empty when left out

    assert Node().peers == []
    features = _features(Metamodel(Node))
    assert ("Node", "up", "container", "children", 0, 1) in features


# -- inferred opposites ------------------------------------------------------------------------


class Shelf(RefineryNode):
    books: list[Book] = Contains()
    label: Label | None = Contains()  # holds no Book: not a candidate


class Book(RefineryNode):
    shelf: Shelf | None = Container()


class Label(RefineryNode):
    text: str = ""


def test_container_pairs_with_the_only_fitting_containment():
    features = _features(Metamodel(Shelf))
    assert ("Book", "shelf", "container", "books", 0, 1) in features
    assert ("Shelf", "books", "contains", "shelf", 0, None) in features
    assert ("Shelf", "label", "contains", None, 0, 1) in features


class Library(RefineryNode):
    __abstract__ = True
    holdings: list[Item] = Contains()


class Branch(Library):
    name: str = ""


class Item(RefineryNode):
    __abstract__ = True
    owner: Branch | None = Container()  # holdings is inherited by Branch


class Volume(Item):
    pass


def test_inherited_containment_and_subclass_targets():
    features = _features(Metamodel(Branch, Volume))
    assert ("Item", "owner", "container", "holdings", 0, 1) in features
    assert ("Library", "holdings", "contains", "owner", 0, None) in features


class Desk(RefineryNode):
    drawers: list[Pen] = Contains()
    top: list[Pen] = Contains(opposite="desk")


class Pen(RefineryNode):
    desk: Desk | None = Container()


def test_the_containment_naming_the_container_is_chosen():
    features = _features(Metamodel(Desk))
    assert ("Pen", "desk", "container", "top", 0, 1) in features
    assert ("Desk", "drawers", "contains", None, 0, None) in features


class Bin(RefineryNode):
    fresh: list[Cup] = Contains()
    used: list[Cup] = Contains()


class Cup(RefineryNode):
    bin: Bin | None = Container()


class Room(RefineryNode):
    chairs: list[Table] = Ref()


class Table(RefineryNode):
    room: Room | None = Container()


class Garage(RefineryNode):
    cars: list[Car] = Contains()


class Car(RefineryNode):
    garage: Garage | None = Container()
    keeper: Garage | None = Container()


@pytest.mark.parametrize(
    ("root", "message"),
    [
        (
            Bin,
            r"Cup\.bin: give Container\(opposite=\.\.\.\): it could be the opposite of any "
            r"of Bin\.fresh, Bin\.used",
        ),
        (
            Room,
            r"Table\.room: give Container\(opposite=\.\.\.\): no containment reference of "
            r"Room holds Table objects",
        ),
        (Garage, r"Car\.keeper: Car\.garage could be the opposite of Garage\.cars too"),
    ],
)
def test_opposites_that_cant_be_inferred(root, message):
    with pytest.raises(TypeError, match=message):
        Metamodel(root)
