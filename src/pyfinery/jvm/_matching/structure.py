"""The metamodel's own constraints, as rules the direct path compiles with the user's.

Refinery checks a model against its metamodel with its reasoning layer:
multiplicities, one container per contained object, no containment cycles. A
plain store has no reasoning layer, so pyfinery states these checks as Refinery
predicates over the metamodel's relations, compiled and matched like any rule.
They are named ``pyfinery_...``; :class:`Check` says what each one reports, as
:meth:`~pyfinery.Metamodel.validate` with a server reports it.

A multiplicity is checked by counting the targets of each object (``count``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._mapping import Catalogue
from ..._results import ViolationKind
from ..._schema import Feature, Schema, class_name

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
        clauses = _multiplicity(feature, catalogue)
        if not clauses:
            continue
        predicate = f"pyfinery_multiplicity{i}"
        owner = class_name(feature.owner)
        lines.append(f"pred {predicate}({owner} x) <-> {' ; '.join(clauses)}.")
        constraint = f"{catalogue.name(feature)}::invalidMultiplicity"
        found.checks.append(Check(predicate, "multiplicity", constraint))

    found.text = "\n".join(lines)
    return found


def _multiplicity(feature: Feature, catalogue: Catalogue) -> list[str]:
    """The clauses of ``feature``'s multiplicity check: empty if it can't be broken."""
    targets = f"count {{ {catalogue.name(feature)}(x, _) }}"
    clauses = []
    if feature.lower > 0:
        clauses.append(f"{targets} < {feature.lower}")
    if feature.upper is not None:
        clauses.append(f"{targets} > {feature.upper}")
    return clauses
