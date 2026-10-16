# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Check a Pydantic object graph against a Refinery metamodel and rules.

Run with: uv run examples/validate_graph.py
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from pyfinery import Container, Contains, Metamodel, RefineryNode, RefineryServer


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


class Link(FSObject):
    target: FSObject


metamodel = Metamodel(
    Filesystem,
    rules=[
        "error pred emptyDir(Dir d) <-> !contents(d, _).",
        "error pred linkToDir(Link l) <-> target(l, d), Dir(d).",
    ],
)
print("Generated metamodel:\n")
print(metamodel.source)

home = Dir(name="home")
docs = Dir(name="docs", contents=[File(size=1)])
notes = File(size=10)
home.contents = [docs, notes, Link(target=notes)]
filesystem = Filesystem(root=home)

with RefineryServer() as server, server.client() as refinery:
    report = metamodel.validate([filesystem], refinery)
    print("Valid graph:", report.ok, [str(v) for v in report.violations])

    # Break some rules.
    orphan = Dir(name="orphan")  # no container, and empty
    home.contents.append(Link(target=docs))  # 4 > [0..3], and links to a directory
    report = metamodel.validate([filesystem, orphan], refinery)
    print("\nAfter breaking it:", report.ok)
    for violation in report.violations:
        print(f"  {violation}")
    print("\nViolations involving 'home':", [str(v) for v in report.for_node(home)])
