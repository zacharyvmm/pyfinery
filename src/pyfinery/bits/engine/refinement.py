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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..._instances import Graph
from ..._schema import Feature, Schema, UnionClass
from ..errors import UnsupportedRuleError
from ..evaluation.pipeline import PredicateValues
from ..matrices import kernels as k
from ..matrices.kernels import NodeIndex
from ..matrices.model import PartialModel
from ..rules.syntax import Atom, Call, Source
from ..rules.syntax import Predicate as RulePredicate

if TYPE_CHECKING:
    pass

from .build import feature_name, opposite_name


class Refinement(NamedTuple):
    """Where Refinery would refine the graph's facts."""

    reason: str
    changes: set[str] | None
    """The relations it would change; ``None`` if it would change what exists."""


@dataclass
class ValueCount:
    """How many values a reference has on an object, certain and uncertain."""

    certain: int = 0
    uncertain: int = 0


type Counts = dict[tuple[str, NodeIndex], ValueCount]
"""The values of each reference (by name) on each object."""


class PossibleContainer(NamedTuple):
    parent: NodeIndex
    certain: bool


type Containers = dict[NodeIndex, list[PossibleContainer]]
"""The possible containers of each contained object."""


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
    counts, containers = tally(model, contains)
    refinements = chain(
        refined_by_multiplicity(schema, graph, references, counts),
        refined_by_containers(graph, references, containers, every_containment),
        refined_by_cycles(graph, model, contains, every_containment),
    )
    for refinement in refinements:
        if refinement.changes is None or refinement.changes & used:
            raise UnsupportedRuleError(refinement.reason)


def tally(model: PartialModel, contains: set[str]) -> tuple[Counts, Containers]:
    """The values of each reference on each object, and the possible containers of each
    object (``contains`` names the containment references)."""
    counts: Counts = {}
    containers: Containers = {}
    for relation in model.relations:
        if relation.arity != 2:
            continue
        must = {tuple(t) for t in relation.must.tolist()}
        for key in {tuple(t) for t in relation.may.tolist()} | must:
            certain = key in must
            count = counts.setdefault((relation.name, key[0]), ValueCount())
            if certain:
                count.certain += 1
            else:
                count.uncertain += 1
            if relation.name in contains:
                containers.setdefault(key[1], []).append(PossibleContainer(key[0], certain))
    return counts, containers


def refined_by_multiplicity(
    schema: Schema,
    graph: Graph,
    references: list[Feature],
    counts: Counts,
) -> Iterator[Refinement]:
    """With ``c`` values certain and ``u`` uncertain, against ``[lower..upper]``:
    the uncertain ones must hold if ``c + u <= lower``, and can't if ``c >= upper``;
    an object that only may exist can't if ``c + u < lower`` or ``c > upper``."""
    for feature in references:
        name = feature_name(feature)
        changed = {name} | ({opposite_name(schema, feature)} if feature.opposite else set())
        for i, node in enumerate(graph.nodes):
            if not isinstance(node.obj, feature.owner):
                continue
            count = counts.get((name, i), ValueCount())
            certain, uncertain = count.certain, count.uncertain
            upper = feature.upper
            if uncertain and (
                certain + uncertain <= feature.lower or (upper is not None and certain >= upper)
            ):
                yield Refinement(
                    f"uncertain values of {name} on {node.name}, which Refinery would refine "
                    f"from its multiplicity [{feature.lower}..{'*' if upper is None else upper}]",
                    changed,
                )
            if certain + uncertain < feature.lower and node.exists == "unknown":
                yield Refinement(
                    f"{node.name}, which only may exist and lacks the values {name} needs: "
                    "Refinery would conclude it doesn't exist",
                    None,
                )
            if upper is not None and certain > upper and node.exists == "unknown":
                yield Refinement(
                    f"{node.name}, which only may exist and has more values of {name} than "
                    f"[{feature.lower}..{upper}] allows: Refinery would conclude it doesn't exist",
                    None,
                )


def refined_by_containers(
    graph: Graph,
    references: list[Feature],
    containers: Containers,
    every_containment: set[str],
) -> Iterator[Refinement]:
    """An object has one container at most, and exactly one if a containment can hold
    its class."""
    held = tuple(target_classes(f.target) for f in references if f.kind == "contains")
    contained = tuple(c for classes in held for c in classes)
    for child, node in enumerate(graph.nodes):
        found = containers.get(child, [])
        holders = [container.parent for container in found if container.certain]
        certain, uncertain = len(holders), len(found) - len(holders)
        if certain > 1:
            # A contradiction: Refinery makes the containment links errors, and rules
            # out the objects involved that only may exist.
            involved = [child, *holders]
            uncertain_objects = any(only_may_exist(graph, i) for i in involved)
            yield Refinement(
                f"{node.name}, in {certain} containers at once: Refinery contradicts it",
                None if uncertain_objects else every_containment,
            )
        if uncertain and certain:
            yield Refinement(
                f"uncertain containers of {node.name}, which has a certain one: "
                "Refinery would rule them out",
                every_containment,
            )
        if not contained or not isinstance(node.obj, contained):
            continue
        if uncertain == 1 and not certain:
            yield Refinement(
                f"the one possible container of {node.name}, uncertain: Refinery would "
                "conclude it holds, as the object needs a container",
                every_containment,
            )
        if not certain and not uncertain and node.exists == "unknown":
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
    cyclic = k.indices(k.diagonal(k.closure(k.matrix_of(n, rows, cols), n), n), n)
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
    values: dict[str, PredicateValues],
    used: set[str],
) -> None:
    """Refuse where Refinery would refine facts from an error predicate in a way that
    could change the query: to make it false where it may hold, it refines the facts
    it depends on (those the query depends on too matter), or rules out objects that
    only may exist."""
    for name, result in values.items():
        if rule_predicates[name].kind != "error":
            continue
        shared = dependencies(rule_predicates, [name], model) & used
        for key, value in result.items():
            if value not in ("unknown", "error"):
                continue
            if any(only_may_exist(graph, i) for i in key) or (value == "unknown" and shared):
                names = ", ".join(model.nodes[i].name for i in key)
                raise UnsupportedRuleError(
                    f"error predicate {result.name}({names}) is {value} where facts are "
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
