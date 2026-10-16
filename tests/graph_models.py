# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""File system metamodel from the Refinery tutorial, defined with pyfinery.graph."""

from __future__ import annotations

import enum
from typing import Annotated

from pydantic import Field

from pyfinery import Container, Contains, Metamodel, Ref, RefineryNode


class Permission(enum.Enum):
    read = "r"
    write = "w"


class Filesystem(RefineryNode):
    root: Annotated[Dir, Contains()]


class FSObject(RefineryNode):
    __abstract__ = True
    parent: Annotated[Dir | None, Container(opposite="contents")] = None


class Dir(FSObject):
    __node_id__ = "name"
    name: str
    contents: Annotated[list[FSObject], Contains(opposite="parent", multiplicity="0..3")] = Field(
        default_factory=list
    )


class File(FSObject):
    size: int = 0
    ratio: float = 0.5
    hidden: bool = False
    label: str | None = None
    permission: Permission | None = None


class Link(FSObject):
    target: FSObject
    backups: Annotated[list[FSObject], Ref()] = Field(default_factory=list, min_length=0)


RULES = ["error pred emptyDir(Dir d) <-> !contents(d, _)."]


def metamodel() -> Metamodel:
    return Metamodel(Filesystem, rules=RULES)
