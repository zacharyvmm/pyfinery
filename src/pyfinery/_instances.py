# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""The graph of Refinery nodes built from Pydantic objects, which facts are
written from.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from ._features import Feature
from ._options import LogicValue


@dataclass(eq=False, slots=True)
class _Value:
    """One value of a field of a node: an attribute or enum value, or a reference."""

    feature: Feature
    value: Any
    """The attribute or enum value, the key of a reference by key, or the referenced object."""
    certain: bool = True
    """Whether the value certainly holds (``ClassOptions.logic_value``), or only may."""
    target: _Node | None = None
    """The node a reference resolves to."""
    unresolved: Literal["unresolved", "ambiguous"] | None = None
    """Why a key of a reference by key resolves to no node."""


@dataclass(eq=False, slots=True)
class _Node:
    """One Refinery node: a Python object, or the instances sharing a key."""

    obj: BaseModel
    """The object representing the node: of instances sharing a key, the first one reached."""
    identity: Hashable
    name: str = ""
    exists: LogicValue = "true"
    values: list[_Value] = field(default_factory=list)


@dataclass(slots=True)
class _Graph:
    """The nodes reachable from some objects, in the order they were reached."""

    nodes: list[_Node]
    by_identity: dict[Hashable, _Node]
    identities: dict[int, tuple[BaseModel, Hashable]]
    """:meth:`Metamodel.identity` by object, keeping the objects alive so that
    their ``id`` isn't reused."""

    def objects_by_name(self) -> dict[str, BaseModel]:
        return {node.name: node.obj for node in self.nodes}
