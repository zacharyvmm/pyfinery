# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""A batch of graphs built into packed bits, to evaluate rules on every graph at once.

::

    batch = PackedBatch.build([model_1, model_2, model_3])  # PartialModel or PackedModel
    batch["edge"].must  # shape (3, n, words(n)): each graph's matrix of edges
    values = batch.evaluate("pred reach(a, b) <-> edge+(a, b).")["reach"]
    values[1]  # graph 1's PredicateValues, in its own node indices

Each graph keeps its tuples to itself: evaluating rules on a batch gives each
graph what evaluating them on that graph alone does. The arrays have a leading
axis of graphs (see :mod:`.kernels`), and every graph the same number ``n`` of
node *slots*: a graph's own nodes (those that aren't atoms) take the first
slots, in their order, and the atoms of every graph take the last, one slot per
atom name, so that a rule naming an atom reads the same slot in every graph.
Slots a graph has no node in are in none of its relations. A batch of one graph
keeps the graph's own node indices.

The graphs may have different relations and attributes; a graph without one has
no tuples of it (an attribute, no value on any node), but a name must have one
arity (or data type) throughout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from . import kernels as k
from .kernels import Device, Indices, NodeIndex, Vector
from .model import BUILTINS, Attribute, DataType, Node, PartialModel, lookup
from .packed import PackedModel
from .values import BatchBits, Size, Tuples, Value, from_tuples, of_graph, to_tuples, with_graph

if TYPE_CHECKING:
    from ..evaluation.pipeline import BatchValues
    from ..rules.compiled import Rules

UNKNOWN_STRING, ERROR_STRING = -1, -2
"""The codes of a string attribute's value that is unknown, and of one that is an error."""


@dataclass(frozen=True)
class BatchRelation:
    """A base relation of every graph: the tuples that must hold and those that may, as
    batched packed bits (:class:`~.values.Value`'s ``data``)."""

    name: str
    arity: int
    must: BatchBits
    may: BatchBits
    simple_name: str


@dataclass(frozen=True)
class BatchAttribute:
    """An attribute of every graph: each slot's value as an interval, as in
    :class:`~.model.Attribute`, of shape ``(graphs, n)``. A string's value is a code
    (``codes``), the same for equal strings throughout the batch, or
    :data:`UNKNOWN_STRING` or :data:`ERROR_STRING`."""

    name: str
    data_type: DataType
    lower: NDArray[np.generic]
    upper: NDArray[np.generic]
    codes: NDArray[np.intp] | None
    simple_name: str


@dataclass(frozen=True)
class PackedBatch:
    """Graphs, and the base relations and attributes rules read, ready to evaluate rules on
    every graph at once."""

    size: Size
    """How many graphs, and how many node slots each."""
    sizes: tuple[int, ...]
    """How many nodes each graph has."""
    placements: tuple[Indices, ...]
    """Each graph's nodes' slots, by the node's index in the graph."""
    nodes: Vector
    """The slots that are nodes of each graph."""
    atoms: tuple[tuple[Node, NodeIndex], ...]
    """The atoms (and enum literals) rules may name, and their slots."""
    relations: Mapping[str, BatchRelation]
    """By name; ``builtin::exists`` and ``builtin::equals`` always among them."""
    attributes: Mapping[str, BatchAttribute]
    """By name."""
    strings: Mapping[str, int] = field(default_factory=dict)
    """The code of each string an attribute has."""
    device: Device = "cpu"
    """Where the relations' arrays (and :attr:`nodes`) are, and rules are evaluated:
    ``"cpu"`` (NumPy arrays) or ``"cuda"`` (CuPy arrays). Attributes stay on the CPU."""

    @classmethod
    def build(
        cls, models: Sequence[PartialModel | PackedModel], device: Device = "cpu"
    ) -> PackedBatch:
        """Pack ``models`` into one batch, on ``device``: graph ``g`` is ``models[g]``."""
        if not models:
            raise ValueError("a batch needs at least one graph")
        if len(models) == 1:
            only = models[0]
            batch = of_model(only if isinstance(only, PackedModel) else PackedModel.build(only))
        else:
            batch = of_models(models)
        return batch.to(device)

    def to(self, device: Device) -> PackedBatch:
        """This batch, with its relations on ``device`` (copied there, unless already)."""
        k.module_of(device)  # raises if the device's library is missing
        if device == self.device:
            return self

        def moved(bits: BatchBits) -> BatchBits:
            return k.to_device(bits, device)  # type: ignore[no-any-return]

        relations = {}
        for name, r in self.relations.items():
            must = moved(r.must)
            may = must if r.may is r.must else moved(r.may)  # one array for both, if one
            relations[name] = replace(r, must=must, may=may)
        nodes: Vector = k.to_device(self.nodes, device)
        return replace(self, relations=relations, nodes=nodes, device=device)

    def __len__(self) -> int:
        return self.size.graphs

    def __getitem__(self, name: str) -> BatchRelation:
        """A relation by name, or by simple name if only one has it; ``exists`` and
        ``equals`` are the builtin ones."""
        found = self.relation_named(name)
        if found is None:
            raise KeyError(name)
        return found

    def evaluate(
        self, rules: str | Sequence[str] | Rules, select: Sequence[str] | None = None
    ) -> dict[str, BatchValues]:
        """Evaluate ``rules`` over every graph: :func:`~pyfinery.bits.evaluate_batch`."""
        from ..evaluation.pipeline import evaluate_batch  # evaluation reads this module

        return evaluate_batch(rules, self, select)

    @cached_property
    def owners(self) -> Indices:
        """Each slot's node in each graph, by its index in the graph; ``-1`` for none:
        shape ``(graphs, n)``."""
        owners = np.full(self.size, -1, np.intp)
        for graph, placement in enumerate(self.placements):
            owners[graph, placement] = np.arange(len(placement))
        return owners

    @property
    def keeps_indices(self) -> bool:
        """Whether every node's slot is its index in its graph."""
        return len(self) == 1

    def value(self, relation: BatchRelation, modality: str) -> Value:
        bits = relation.must if modality == "must" else relation.may
        return Value(relation.arity, self.size, bits)

    # -- names ------------------------------------------------------------------------

    def relation_named(self, name: str) -> BatchRelation | None:
        """A relation by name, or by simple name if only one has it.

        ``exists`` and ``equals`` are the builtin ones.
        """
        return lookup(self.relations, BUILTINS.get(name, name))

    def attribute_named(self, name: str) -> BatchAttribute | None:
        """An attribute by name, or by simple name if only one has it."""
        return lookup(self.attributes, name)

    def atom_named(self, name: str) -> NodeIndex | None:
        """The slot of the atom (or enum literal) ``name``, by name or unique simple name."""
        found = [slot for node, slot in self.atoms if node.name == name]
        if not found:
            found = [slot for node, slot in self.atoms if node.simple_name == name]
        if len(found) > 1:
            raise ValueError(f"atom {name!r} is ambiguous")
        return found[0] if found else None


# -- building -------------------------------------------------------------------------


def of_model(model: PackedModel) -> PackedBatch:
    """A batch of one graph, whose nodes keep their indices: the model's arrays as they are."""
    n = model.size
    size = Size(1, n)
    relations = {
        name: BatchRelation(
            name,
            r.arity,
            of_graph(r.arity, n, r.must).data,
            of_graph(r.arity, n, r.may).data,
            r.simple_name,
        )
        for name, r in model.relations.items()
    }
    placement = np.arange(n, dtype=np.intp)
    atoms = tuple((node, i) for i, node in enumerate(model.nodes) if node.kind == "atom")
    strings: dict[str, int] = {}
    attributes = {
        name: batch_attribute([(0, a)], size, [placement], strings)
        for name, a in model.attributes.items()
    }
    nodes = k.full_vector(1, n)
    return PackedBatch(size, (n,), (placement,), nodes, atoms, relations, attributes, strings)


def of_models(models: Sequence[PartialModel | PackedModel]) -> PackedBatch:
    """A batch of several graphs: each graph's own nodes first, then a slot per atom name."""
    graph_nodes = [tuple(m.nodes) for m in models]
    atom_slots: dict[str, int] = {}  # by name, in the order they come
    atom_nodes: dict[str, Node] = {}
    own_counts = []
    for nodes in graph_nodes:
        own = 0
        for node in nodes:
            if node.kind == "atom":
                atom_nodes.setdefault(node.name, node)
                atom_slots.setdefault(node.name, len(atom_slots))
            else:
                own += 1
        own_counts.append(own)
    first_atom = max(own_counts)
    n = first_atom + len(atom_slots)
    size = Size(len(models), n)
    placements = []
    for nodes in graph_nodes:
        placement = np.empty(len(nodes), np.intp)
        own = 0
        for i, node in enumerate(nodes):
            if node.kind == "atom":
                placement[i] = first_atom + atom_slots[node.name]
            else:
                placement[i] = own
                own += 1
        placements.append(placement)
    atoms = tuple((atom_nodes[name], first_atom + slot) for name, slot in atom_slots.items())
    graphs = np.repeat(np.arange(len(models)), [len(p) for p in placements])
    slots = k.vector_of(*size, graphs, np.concatenate(placements))
    relations = batch_relations(models, size, placements)
    strings: dict[str, int] = {}
    attributes: dict[str, list[tuple[int, Attribute]]] = {}
    for graph, model in enumerate(models):
        for attribute in model_attributes(model):
            attributes.setdefault(attribute.name, []).append((graph, attribute))
    batched = {
        name: batch_attribute(given, size, placements, strings)
        for name, given in attributes.items()
    }
    sizes = tuple(len(nodes) for nodes in graph_nodes)
    return PackedBatch(size, sizes, tuple(placements), slots, atoms, relations, batched, strings)


def batch_relations(
    models: Sequence[PartialModel | PackedModel], size: Size, placements: list[Indices]
) -> dict[str, BatchRelation]:
    """Each relation of any graph, with the tuples of every graph that has it."""
    found: dict[str, list[tuple[int, Tuples, Tuples]]] = {}
    named: dict[str, tuple[int, str]] = {}
    for graph, model in enumerate(models):
        for name, arity, simple_name, must, may in relation_tuples(model):
            if named.setdefault(name, (arity, simple_name))[0] != arity:
                raise ValueError(f"{name} has a different arity in each graph")
            found.setdefault(name, []).append((graph, must, may))
    relations = {}
    for name, given in found.items():
        arity, simple_name = named[name]
        musts = [(graph, m) for graph, m, _ in given]
        must_bits = from_tuples(arity, size, placed(musts, arity, placements)).data
        if all(m is y or np.array_equal(m, y) for _, m, y in given):
            may_bits = must_bits
        else:
            mays = [(graph, y) for graph, _, y in given]
            may_bits = from_tuples(arity, size, placed(mays, arity, placements)).data
        relations[name] = BatchRelation(name, arity, must_bits, may_bits, simple_name)
    return relations


def placed(given: list[tuple[int, Tuples]], arity: int, placements: list[Indices]) -> Tuples:
    """The tuples of each graph (``(graph, tuples)``), at their slots, after the graph."""
    parts = []
    for graph, tuples in given:
        at = placements[graph][tuples]
        parts.append(with_graph(at.reshape(len(at), arity), graph))
    return np.concatenate(parts) if parts else np.empty((0, 1 + arity), np.intp)


def relation_tuples(
    model: PartialModel | PackedModel,
) -> list[tuple[str, int, str, Tuples, Tuples]]:
    """Each relation of a graph: its name, arity and simple name, and the tuples that must
    and may hold."""
    if isinstance(model, PartialModel):
        return [(r.name, r.arity, r.simple_name, r.must, r.may) for r in model.all_relations]
    found = []
    for r in model.relations.values():
        must = to_tuples(of_graph(r.arity, model.size, r.must))[:, 1:]
        may = must if r.may is r.must else to_tuples(of_graph(r.arity, model.size, r.may))[:, 1:]
        found.append((r.name, r.arity, r.simple_name, must, may))
    return found


def model_attributes(model: PartialModel | PackedModel) -> Sequence[Attribute]:
    if isinstance(model, PackedModel):
        return list(model.attributes.values())
    return model.attributes


def batch_attribute(
    given: list[tuple[int, Attribute]],
    size: Size,
    placements: Sequence[Indices],
    strings: dict[str, int],
) -> BatchAttribute:
    """An attribute of the batch, from the graphs that have it (``(graph, attribute)``):
    slots without a value have none, an error. Strings get codes from ``strings``,
    which grows."""
    named = given[0][1]
    if any(a.data_type != named.data_type for _, a in given):
        raise ValueError(f"{named.name} has a different type in each graph")
    exact = any(a.lower.dtype == object for _, a in given)
    dtype = object if exact else np.float64
    if len(given) == 1 and len(placements) == 1 and given[0][0] == 0:
        lower, upper = named.lower[None].astype(dtype), named.upper[None].astype(dtype)
    else:
        lower = np.full(size, math.inf, dtype)
        upper = np.full(size, -math.inf, dtype)
        for graph, attribute in given:
            lower[graph, placements[graph]] = attribute.lower
            upper[graph, placements[graph]] = attribute.upper
    codes = None
    if named.data_type == "string":
        codes = np.full(size, ERROR_STRING, np.intp)
        for graph, attribute in given:
            placement = placements[graph]
            texts = attribute.strings or (None,) * len(attribute.lower)
            for i, text in enumerate(texts):
                if attribute.lower[i] > attribute.upper[i]:
                    continue  # an error
                code = UNKNOWN_STRING if text is None else strings.setdefault(text, len(strings))
                codes[graph, placement[i]] = code
    return BatchAttribute(named.name, named.data_type, lower, upper, codes, named.simple_name)


# -- batching -------------------------------------------------------------------------

BATCH_WORDS = 1 << 22
"""The most words a matrix of a batch :func:`group_by_size` makes may hold: 32 MiB."""


def group_by_size(sizes: Sequence[int], limit: int = BATCH_WORDS) -> list[list[int]]:
    """Graphs (by index, with ``sizes`` nodes) in batches of about the same size: a batch's
    largest graph has at most twice the nodes of its smallest (or 64, a word's worth),
    and a matrix of it holds at most ``limit`` words (but for a batch of one graph)."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    batches: list[list[int]] = []
    smallest = 0
    for i in order:
        n = sizes[i]
        if batches:
            batch = batches[-1]
            words = (len(batch) + 1) * n * k.words(n)
            if n <= max(64, 2 * smallest) and words <= limit:
                batch.append(i)
                continue
        batches.append([i])
        smallest = n
    return batches
