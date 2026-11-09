# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""A partial model built into packed bits: evaluate rules on it, and combine it with others.

::

    g1 = PackedModel.build(model_1)  # each relation's must and may, as bits
    g2 = PackedModel.build(model_2)
    both, placed = g1.combine(g2)  # g2's nodes after g1's: more rows and columns
    both = both.connect("edge", [(0, placed[0])])  # an edge from g1's node 0 to g2's
    both.evaluate("pred reach(a, b) <-> edge+(a, b).")

Combining keeps each graph's tuples as they are: a relation's matrix of the
combined graph has the first graph's in the top left block, the second's in
the bottom right, and nothing between them until :meth:`PackedModel.connect`
adds it. The atoms the two graphs name alike (enum literals) are one node.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .errors import RuleError
from .evaluation import PredicateValues
from .kernels import Indices, NodeIndex
from .model import (
    BUILTINS,
    Attribute,
    LogicValue,
    Node,
    PartialModel,
    Relation,
    atom_index,
    lookup,
)
from .rules import Rules
from .syntax import Position
from .values import Modal, NodeTuple, difference, empty, from_tuples, relabel, union


@dataclass(frozen=True)
class PackedRelation:
    """A base relation, its tuples packed into bits."""

    name: str
    arity: int
    values: Modal
    simple_name: str


@dataclass(frozen=True)
class PackedModel:
    """Nodes, and the base relations and attributes rules read, ready to evaluate rules on."""

    nodes: tuple[Node, ...]
    relations: Mapping[str, PackedRelation]
    """By name; ``builtin::exists`` and ``builtin::equals`` always among them."""
    attributes: Mapping[str, Attribute]
    """By name."""

    @classmethod
    def build(cls, model: PartialModel) -> PackedModel:
        """Pack each relation of ``model`` into bits."""
        n = model.size
        relations = {
            r.name: PackedRelation(r.name, r.arity, pack(r, n), r.simple_name)
            for r in model.all_relations
        }
        attributes = {a.name: a for a in model.attributes}
        return cls(tuple(model.nodes), relations, attributes)

    @property
    def size(self) -> int:
        return len(self.nodes)

    # -- evaluating ---------------------------------------------------------------

    def evaluate(
        self, rules: str | Sequence[str] | Rules, select: Sequence[str] | None = None
    ) -> dict[str, PredicateValues]:
        """Evaluate the predicates ``rules`` define (see :func:`evaluate`)."""
        compiled = rules if isinstance(rules, Rules) else Rules.compile(rules)
        return compiled.evaluate(self, select)

    # -- combining ------------------------------------------------------------------

    def combine(self, other: PackedModel) -> tuple[PackedModel, Indices]:
        """This graph and ``other`` side by side, and where each of ``other``'s nodes is
        in it: this graph's nodes keep their places, and ``other``'s come after them,
        but for the atoms both have, which are one node."""
        placement = place(self, other)
        added = [node for node, at in zip(other.nodes, placement, strict=True) if at >= self.size]
        nodes = (*self.nodes, *added)
        n = len(nodes)
        relations = {
            name: combine_relations(
                self.relations.get(name), other.relations.get(name), n, self.size, placement
            )
            for name in {**self.relations, **other.relations}
        }
        attributes = {
            name: combine_attributes(
                self.attributes.get(name), other.attributes.get(name), n, self.size, placement
            )
            for name in {**self.attributes, **other.attributes}
        }
        return PackedModel(nodes, relations, attributes), placement

    def __add__(self, other: PackedModel) -> PackedModel:
        """:meth:`combine`, without where ``other``'s nodes went (after this graph's)."""
        return self.combine(other)[0]

    def connect(
        self, relation: str, tuples: Iterable[NodeTuple], value: LogicValue = "true"
    ) -> PackedModel:
        """This graph, with each of ``tuples`` of ``relation`` (by name or unique simple
        name) set to ``value``: e.g. edges between the graphs :meth:`combine` put together."""
        found = self.relation_named(relation)
        if found is None:
            raise ValueError(f"no relation named {relation!r}")
        n, arity = self.size, found.arity
        facts = from_tuples(arity, n, np.array(list(tuples), np.intp).reshape(-1, arity))
        none = empty(arity, n)
        must = union(
            difference(found.values.must, facts), facts if value in ("true", "error") else none
        )
        may = union(
            difference(found.values.may, facts), facts if value in ("true", "unknown") else none
        )
        relations = {**self.relations, found.name: replace(found, values=Modal(must, may))}
        return replace(self, relations=relations)

    # -- names ------------------------------------------------------------------------

    def relation_named(self, name: str) -> PackedRelation | None:
        """A relation by name, or by simple name if only one has it.

        ``exists`` and ``equals`` are the builtin ones.
        """
        return lookup(self.relations, BUILTINS.get(name, name))

    def attribute_named(self, name: str) -> Attribute | None:
        """An attribute by name, or by simple name if only one has it."""
        return lookup(self.attributes, name)

    def atom_named(self, name: str) -> NodeIndex | None:
        """The index of the atom (or enum literal) ``name``, by name or unique simple name."""
        return atom_index(self.nodes, name)

    def relation(self, name: str, position: Position | None) -> PackedRelation:
        """The relation rules name ``name``, or a :class:`RuleError` at ``position``."""
        try:
            found = self.relation_named(name)
        except ValueError as e:
            raise RuleError(str(e), position) from None
        if found is None:
            raise RuleError(f"no relation named {name!r}", position)
        return found

    def attribute(self, name: str, position: Position | None) -> Attribute:
        """The attribute rules name ``name``, or a :class:`RuleError` at ``position``."""
        try:
            found = self.attribute_named(name)
        except ValueError as e:
            raise RuleError(str(e), position) from None
        if found is None:
            raise RuleError(f"no attribute named {name!r}", position)
        return found


def evaluate(
    rules: str | Sequence[str] | Rules,
    model: PartialModel | PackedModel,
    select: Sequence[str] | None = None,
) -> dict[str, PredicateValues]:
    """Evaluate the predicates ``rules`` define over ``model``.

    ``rules`` is Refinery source (statements other than ``pred`` declarations
    are skipped), or :class:`Rules` compiled from it. The predicates named in
    ``select`` are evaluated, or else every one with a body. A
    :class:`PartialModel` is built into a :class:`PackedModel` first: build it
    yourself, and compile the rules, to evaluate them on it again.
    """
    packed = model if isinstance(model, PackedModel) else PackedModel.build(model)
    return packed.evaluate(rules, select)


def pack(relation: Relation, n: int) -> Modal:
    """A relation's must and may tuples as bits: one value if they're the same tuples."""
    must = from_tuples(relation.arity, n, relation.must)
    if np.array_equal(relation.must, relation.may):
        return Modal(must, must)
    return Modal(must, from_tuples(relation.arity, n, relation.may))


# -- combining ------------------------------------------------------------------


def place(first: PackedModel, second: PackedModel) -> Indices:
    """Where each of ``second``'s nodes goes when combined with ``first``: on the atom
    of ``first`` named alike, or else after ``first``'s nodes, in order."""
    atoms = {
        node.name: i for i, node in reversed(list(enumerate(first.nodes))) if node.kind == "atom"
    }
    placement = np.empty(second.size, np.intp)
    next_place = first.size
    for j, node in enumerate(second.nodes):
        if node.kind == "atom" and node.name in atoms:
            placement[j] = atoms[node.name]
        else:
            placement[j] = next_place
            next_place += 1
    return placement


def combine_relations(
    first: PackedRelation | None,
    second: PackedRelation | None,
    n: int,
    first_size: int,
    placement: Indices,
) -> PackedRelation:
    """A relation of the combined graph: the tuples of both, where each put them."""
    parts = []
    if first is not None:
        parts.append((first, np.arange(first_size)))
    if second is not None:
        parts.append((second, placement))
    named, _ = parts[0]
    if any(relation.arity != named.arity for relation, _ in parts):
        raise ValueError(f"{named.name} has a different arity in each graph")
    must, may = empty(named.arity, n), empty(named.arity, n)
    for relation, places in parts:
        must = union(must, relabel(relation.values.must, n, places))
        may = union(may, relabel(relation.values.may, n, places))
    return PackedRelation(named.name, named.arity, Modal(must, may), named.simple_name)


def combine_attributes(
    first: Attribute | None,
    second: Attribute | None,
    n: int,
    first_size: int,
    placement: Indices,
) -> Attribute:
    """An attribute of the combined graph: the values of both, where each put them, but
    on an atom both graphs have, the first's (if it has one). Nodes neither gives a
    value have none (an error), as outside the attribute's class."""
    given = [a for a in (first, second) if a is not None]
    named = given[0]
    if any(a.data_type != named.data_type for a in given):
        raise ValueError(f"{named.name} has a different type in each graph")
    exact = any(a.lower.dtype == object for a in given)
    dtype = object if exact else np.float64
    lower = np.full(n, math.inf, dtype)
    upper = np.full(n, -math.inf, dtype)
    strings: list[str | None] = [None] * n
    if first is not None:
        lower[:first_size], upper[:first_size] = first.lower, first.upper
        strings[:first_size] = first.strings or [None] * first_size
    if second is not None:
        free = lower[placement] > upper[placement]  # no value yet
        lower[placement[free]], upper[placement[free]] = second.lower[free], second.upper[free]
        for j in np.flatnonzero(free):
            strings[placement[j]] = second.strings[j] if second.strings else None
    kept = tuple(strings) if named.data_type == "string" else ()
    return Attribute(named.name, named.data_type, lower, upper, kept, named.simple_name)
