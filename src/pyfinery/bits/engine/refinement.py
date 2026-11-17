# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Where Refinery would refine the graph's facts, which the bits engine refuses.

Refinery refines uncertain facts from what must hold; the bits engine evaluates
the facts as given. These find where the two would differ, to refuse rather
than answer differently. A graph without uncertainty is refined only where it is
contradictory (an object in two containers, a containment cycle).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..._instances import Graph
from ..._schema import Feature, Schema, UnionClass
from ..errors import UnsupportedRuleError
from ..matrices import kernels as k
from ..matrices.kernels import Indices, NodeIndex
from ..matrices.model import LogicValue, PartialModel
from ..matrices.values import Tuples
from ..rules.syntax import Atom, Call, Source
from ..rules.syntax import Predicate as RulePredicate

if TYPE_CHECKING:
    pass

from .build import Membership, feature_name, opposite_name


class Refinement(NamedTuple):
    """Where Refinery would refine the graph's facts."""

    reason: str
    changes: set[str] | None
    """The relations it would change; ``None`` if it would change what exists."""


class Tally(NamedTuple):
    """The values of each reference on each object, and the possible containers of each
    object: arrays over the model's nodes."""

    certain: dict[str, Indices]
    """How many values each reference (by name) certainly has on each node."""
    uncertain: dict[str, Indices]
    """How many it only may have."""
    parents: Indices
    """The possible containment links: each one's container, ..."""
    children: Indices
    """... what it contains, ..."""
    links_certain: NDArray[np.bool_]
    """... and whether it certainly holds."""


def refuse_assertions(parsed: Source) -> None:
    if parsed.asserted:
        name, position = next(iter(parsed.asserted.items()))
        raise UnsupportedRuleError(
            f"assertions in rules ({name}): give facts as Predicate objects", position
        )


def refuse_abstract_instances(schema: Schema, graph: Graph) -> None:
    """An object of an abstract class: Refinery finds its class memberships contradictory."""
    for node in graph.nodes:
        if schema.is_abstract(type(node.obj)):
            raise UnsupportedRuleError(
                f"{node.name}, an instance of the abstract class {type(node.obj).__name__}: "
                "Refinery reports its memberships as contradictions"
            )


def refuse_containment_refinement(
    schema: Schema, graph: Graph, model: PartialModel, used: set[str]
) -> None:
    """Refuse where Refinery would refine a reference the query uses (``used``), or what
    exists, from multiplicities or containment."""
    references = [f for fs in schema.features.values() for f in fs if not f.is_attribute]
    contains = {feature_name(f) for f in references if f.kind == "contains"}
    every_containment = contains | {
        opposite_name(schema, f) for f in references if f.kind == "contains" and f.opposite
    }
    counts = tally(model, contains)
    refinements = chain(
        refined_by_multiplicity(schema, graph, references, counts),
        refined_by_containers(graph, references, counts, every_containment),
        refined_by_cycles(graph, model, contains, every_containment),
    )
    for refinement in refinements:
        if refinement.changes is None or refinement.changes & used:
            raise UnsupportedRuleError(refinement.reason)


def tally(model: PartialModel, contains: set[str]) -> Tally:
    """The values of each reference on each object, and the possible containers of each
    object (``contains`` names the containment references): counted in arrays."""
    n = max(model.size, 1)
    certain: dict[str, Indices] = {}
    uncertain: dict[str, Indices] = {}
    links: list[tuple[Indices, NDArray[np.bool_]]] = []
    for relation in model.relations:
        if relation.arity != 2:
            continue
        must = np.unique(relation.must[:, 0] * n + relation.must[:, 1])
        keys = np.union1d(must, relation.may[:, 0] * n + relation.may[:, 1])
        sure = np.isin(keys, must)
        sources = keys // n
        certain[relation.name] = np.bincount(sources[sure], minlength=n)
        uncertain[relation.name] = np.bincount(sources[~sure], minlength=n)
        if relation.name in contains:
            links.append((keys, sure))
    keys = np.concatenate([k for k, _ in links]) if links else np.empty(0, np.intp)
    sure = np.concatenate([c for _, c in links]) if links else np.empty(0, bool)
    return Tally(certain, uncertain, keys // n, keys % n, sure)


def refined_by_multiplicity(
    schema: Schema,
    graph: Graph,
    references: list[Feature],
    counts: Tally,
) -> Iterator[Refinement]:
    """With ``c`` values certain and ``u`` uncertain, against ``[lower..upper]``:
    the uncertain ones must hold if ``c + u <= lower``, and can't if ``c >= upper``;
    an object that only may exist can't if ``c + u < lower`` or ``c > upper``."""
    size = len(graph.nodes)
    types = Membership(graph)
    maybe = np.array([node.exists == "unknown" for node in graph.nodes], bool)
    none = np.zeros(size, np.intp)
    for feature in references:
        name = feature_name(feature)
        changed = {name} | ({opposite_name(schema, feature)} if feature.opposite else set())
        certain = counts.certain.get(name, none)[:size]
        uncertain = counts.uncertain.get(name, none)[:size]
        lower, upper = feature.lower, feature.upper
        owned = types.mask(feature.owner)
        refined = owned & (uncertain > 0) & (certain + uncertain <= lower)
        too_few = owned & (certain + uncertain < lower) & maybe
        too_many = np.zeros(size, bool)
        if upper is not None:
            refined |= owned & (uncertain > 0) & (certain >= upper)
            too_many = owned & (certain > upper) & maybe
        for i in np.flatnonzero(refined | too_few | too_many).tolist():
            node = graph.nodes[i]
            if refined[i]:
                yield Refinement(
                    f"uncertain values of {name} on {node.name}, which Refinery would refine "
                    f"from its multiplicity [{lower}..{'*' if upper is None else upper}]",
                    changed,
                )
            if too_few[i]:
                yield Refinement(
                    f"{node.name}, which only may exist and lacks the values {name} needs: "
                    "Refinery would conclude it doesn't exist",
                    None,
                )
            if too_many[i]:
                yield Refinement(
                    f"{node.name}, which only may exist and has more values of {name} than "
                    f"[{lower}..{upper}] allows: Refinery would conclude it doesn't exist",
                    None,
                )


def refined_by_containers(
    graph: Graph,
    references: list[Feature],
    counts: Tally,
    every_containment: set[str],
) -> Iterator[Refinement]:
    """An object has one container at most, and exactly one if a containment can hold
    its class."""
    held = tuple(target_classes(f.target) for f in references if f.kind == "contains")
    contained = tuple(c for classes in held for c in classes)
    size = len(graph.nodes)
    children, sure = counts.children, counts.links_certain
    certain = np.bincount(children[sure], minlength=size)[:size]
    uncertain = np.bincount(children[~sure], minlength=size)[:size]
    maybe = np.array([node.exists == "unknown" for node in graph.nodes], bool)
    needs = Membership(graph).mask(contained) if contained else np.zeros(size, bool)
    flagged = (certain > 1) | ((uncertain > 0) & (certain > 0))
    flagged |= needs & (certain == 0) & ((uncertain == 1) | ((uncertain == 0) & maybe))
    for child in np.flatnonzero(flagged).tolist():
        node = graph.nodes[child]
        if certain[child] > 1:
            # A contradiction: Refinery makes the containment links errors, and rules
            # out the objects involved that only may exist.
            holders = counts.parents[(children == child) & sure].tolist()
            uncertain_objects = any(only_may_exist(graph, i) for i in [child, *holders])
            yield Refinement(
                f"{node.name}, in {certain[child]} containers at once: Refinery contradicts it",
                None if uncertain_objects else every_containment,
            )
        if uncertain[child] and certain[child]:
            yield Refinement(
                f"uncertain containers of {node.name}, which has a certain one: "
                "Refinery would rule them out",
                every_containment,
            )
        if not needs[child]:
            continue
        if uncertain[child] == 1 and not certain[child]:
            yield Refinement(
                f"the one possible container of {node.name}, uncertain: Refinery would "
                "conclude it holds, as the object needs a container",
                every_containment,
            )
        if not certain[child] and not uncertain[child] and node.exists == "unknown":
            yield Refinement(
                f"{node.name}, which only may exist and has no possible container: "
                "Refinery would conclude it doesn't exist",
                None,
            )


def refined_by_cycles(
    graph: Graph, model: PartialModel, contains: set[str], every_containment: set[str]
) -> Iterator[Refinement]:
    """Containment has no cycles: Refinery rules out the objects of one that only may
    exist, or else contradicts it."""
    n = len(model.nodes)
    edges = [r for r in model.relations if r.name in contains]
    if not edges:
        return
    rows = np.concatenate([r.may[:, 0] for r in edges] + [r.must[:, 0] for r in edges])
    cols = np.concatenate([r.may[:, 1] for r in edges] + [r.must[:, 1] for r in edges])
    edges_of = k.matrix_of(1, n, np.zeros(len(rows), np.intp), rows, cols)
    _, cyclic = k.indices(k.diagonal(k.closure(edges_of, n), n))
    if len(cyclic):
        uncertain = any(only_may_exist(graph, int(i)) for i in cyclic)
        names = ", ".join(model.nodes[int(i)].name for i in cyclic)
        yield Refinement(
            f"a containment cycle through {names}: Refinery rules out or contradicts it",
            None if uncertain else every_containment,
        )


def only_may_exist(graph: Graph, i: NodeIndex) -> bool:
    """Whether node ``i`` is an object that only may exist (enum literals do exist)."""
    return i < len(graph.nodes) and graph.nodes[i].exists == "unknown"


def target_classes(target: object) -> tuple[type, ...]:
    """The model classes a reference's target stands for."""
    if isinstance(target, UnionClass):
        return target.members
    return (target,) if isinstance(target, type) else ()


def refuse_error_refinement(
    graph: Graph,
    model: PartialModel,
    rule_predicates: dict[str, RulePredicate],
    tuples: Mapping[str, Mapping[LogicValue, Tuples]],
    used: set[str],
) -> None:
    """Refuse where Refinery would refine facts from an error predicate in a way that
    could change the query: to make it false where it may hold, it refines the facts
    it depends on (those the query depends on too matter), or rules out objects that
    only may exist."""
    for name, by_value in tuples.items():
        if rule_predicates[name].kind != "error":
            continue
        shared = dependencies(rule_predicates, [name], model) & used
        found = [(key, value) for value in ("unknown", "error") for key in by_value[value].tolist()]
        for key, value in sorted(found):
            if any(only_may_exist(graph, i) for i in key) or (value == "unknown" and shared):
                names = ", ".join(model.nodes[i].name for i in key)
                raise UnsupportedRuleError(
                    f"error predicate {name}({names}) is {value} where facts are "
                    "uncertain: Refinery would refine them to rule it out"
                )


def dependencies(
    predicates: dict[str, RulePredicate], names: Iterable[str], model: PartialModel
) -> set[str]:
    """The relations and attributes the predicates ``names`` depend on, through the
    predicates they call, by their names in the model."""
    found: set[str] = set()
    seen: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        predicate = predicates.get(name)
        if predicate is None or predicate.body is None:
            found.add(model_name(model, name))
            continue
        for parameter in predicate.parameters:
            if parameter.type is not None:
                pending.append(parameter.type)
        for clause in predicate.body:
            for literal in clause:
                if isinstance(literal, Atom):
                    pending.append(literal.relation)
                    continue
                for side in (literal.left, literal.right):
                    if isinstance(side, Call):
                        found.add(model_name(model, side.relation))
    return found


def model_name(model: PartialModel, name: str) -> str:
    try:
        found = model.relation_named(name) or model.attribute_named(name)
    except ValueError:
        return name
    return found.name if found is not None else name
