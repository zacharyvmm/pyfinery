"""The metamodel's own constraints, as rules the direct path compiles with the user's.

Refinery checks a model against its metamodel with its reasoning layer:
multiplicities, one container per contained object, no containment cycles. A
plain store has no reasoning layer, so pyfinery states these checks as Refinery
predicates over the metamodel's relations, compiled and matched like any rule.
They are named ``pyfinery_...``; :class:`Check` says what each one reports, as
:meth:`~pyfinery.Metamodel.validate` with a server reports it.

A multiplicity is checked by joining as many distinct targets as the bound
needs, since the direct path has no ``count`` yet: bounds up to
:data:`MAX_JOINED` are checked; a feature with a larger bound is listed in
:attr:`Structure.unchecked`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._mapping import Catalogue
from ..._results import ViolationKind
from ..._schema import Feature, Schema, class_name

MAX_JOINED = 5
"""The most distinct targets a multiplicity check joins: an upper bound of at most
``MAX_JOINED - 1``, a lower bound of at most ``MAX_JOINED``."""

CONTAINS = "pyfinery_contains"
"""The containment links: every containment reference, as one relation."""
INVALID_CONTAINER = "pyfinery_invalidContainer"


@dataclass(frozen=True, slots=True)
class Check:
    """A structural predicate and what its tuples are reported as."""

    predicate: str
    """The predicate's name in the compiled problem."""
    kind: ViolationKind
    constraint: str
    """The relation a server names in the violation, e.g. ``Dir::contents::invalidMultiplicity``."""


@dataclass(slots=True)
class Structure:
    text: str
    """The predicates, in Refinery syntax."""
    checks: list[Check] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    """Features whose multiplicity needs ``count``, e.g. ``Dir::contents [0..10]``."""
    containment: bool = False
    """Whether the metamodel has a containment reference (and :data:`CONTAINS` is defined)."""


def structure(schema: Schema, catalogue: Catalogue) -> Structure:
    """The structural predicates of ``schema``."""
    lines: list[str] = []
    found = Structure("")
    features = [f for cls in schema.classes for f in schema.features[cls]]
    containment = [f for f in features if f.kind == "contains" and not f.is_attribute]
    containers = [f for f in features if f.kind == "container"]

    if containment:
        found.containment = True
        body = " ; ".join(f"{catalogue.name(f)}(x, y)" for f in containment)
        lines.append(f"pred {CONTAINS}(node x, node y) <-> {body}.")
        # As Refinery's builtin::invalidContainer: no container, several, or a cycle.
        lines.append(
            f"pred {INVALID_CONTAINER}(contained y) <-> !{CONTAINS}(_, y) ; "
            f"{CONTAINS}(a, y), {CONTAINS}(b, y), a != b ; {CONTAINS}+(y, y)."
        )
        found.checks.append(Check(INVALID_CONTAINER, "containment", "builtin::invalidContainer"))
        # A link on a cycle, and its opposite, contradict the containment hierarchy.
        for i, feature in enumerate(containment):
            name = catalogue.name(feature)
            predicate = f"pyfinery_cycle{i}"
            lines.append(f"pred {predicate}(node x, node y) <-> {name}(x, y), {CONTAINS}+(y, x).")
            found.checks.append(Check(predicate, "contradiction", name))
        for i, feature in enumerate(containers):
            name = catalogue.name(feature)
            predicate = f"pyfinery_cycleOpposite{i}"
            lines.append(f"pred {predicate}(node x, node y) <-> {name}(x, y), {CONTAINS}+(x, y).")
            found.checks.append(Check(predicate, "contradiction", name))

    for i, feature in enumerate(features):
        if feature.is_attribute or feature.kind == "container":
            continue
        clauses = _multiplicity(feature, catalogue, f"pyfinery_multiplicity{i}", lines)
        if clauses is None:
            found.unchecked.append(_describe(feature, catalogue))
            continue
        if not clauses:
            continue
        predicate = f"pyfinery_multiplicity{i}"
        owner = class_name(feature.owner)
        lines.append(f"pred {predicate}({owner} x) <-> {' ; '.join(clauses)}.")
        constraint = f"{catalogue.name(feature)}::invalidMultiplicity"
        found.checks.append(Check(predicate, "multiplicity", constraint))

    found.text = "\n".join(lines)
    return found


def _multiplicity(
    feature: Feature, catalogue: Catalogue, predicate: str, lines: list[str]
) -> list[str] | None:
    """The clauses of ``feature``'s multiplicity check (empty if it can't be broken), or
    ``None`` if a bound needs more than :data:`MAX_JOINED` distinct targets."""
    name = catalogue.name(feature)
    lower, upper = feature.lower, feature.upper
    if lower > MAX_JOINED or (upper is not None and upper + 1 > MAX_JOINED):
        return None
    clauses = []
    if lower == 1:
        clauses.append(f"!{name}(x, _)")
    elif lower > 1:
        helper = f"{predicate}AtLeast"
        owner = class_name(feature.owner)
        lines.append(f"pred {helper}({owner} x) <-> {_distinct(name, lower)}.")
        clauses.append(f"!{helper}(x)")
    if upper is not None:
        clauses.append(_distinct(name, upper + 1))
    return clauses


def _distinct(relation: str, count: int) -> str:
    """``count`` distinct targets of ``x`` in ``relation``."""
    targets = [f"y{i}" for i in range(count)]
    literals = [f"{relation}(x, {t})" for t in targets]
    literals += [f"{a} != {b}" for i, a in enumerate(targets) for b in targets[i + 1 :]]
    return ", ".join(literals)


def _describe(feature: Feature, catalogue: Catalogue) -> str:
    upper = "*" if feature.upper is None else str(feature.upper)
    return f"{catalogue.name(feature)} [{feature.lower}..{upper}]"
