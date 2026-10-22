# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Validate and query a deployment configuration loaded from JSON.

The models are ordinary Pydantic classes that know nothing about Refinery;
components refer to each other by name.

Run with: uv run examples/validate_deployment.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from pyfinery import ClassOptions, Metamodel, Predicate, RefineryServer


class Port(BaseModel):
    number: int
    public: bool = False


class Service(BaseModel):
    kind: Literal["service"] = "service"
    name: str
    image: str
    tier: Literal["frontend", "backend", "data"]
    ports: list[Port] = []
    depends_on: list[str] = []  # names of other components


class Database(BaseModel):
    kind: Literal["database"] = "database"
    name: str
    engine: Literal["postgres", "redis"]
    encrypted: bool


class Deployment(BaseModel):
    components: list[Annotated[Service | Database, Field(discriminator="kind")]]


deployment = Deployment.model_validate_json(Path(__file__).with_name("deployment.json").read_text())

metamodel = Metamodel(
    Deployment,
    options={
        Service: ClassOptions(key=lambda s: s.name, references={"depends_on": "Component"}),
        Database: ClassOptions(key=lambda d: d.name),
    },
    unions={"Component": [Service, Database]},
    rules=[
        # A public service must not talk to a database directly.
        "error pred publicDependsOnData(Service s, Database d) <-> "
        "Service::ports(s, p), Port::public(p), Service::depends_on(s, d).",
        # Images must be pinned (checked in Python, below).
        "error pred unpinned(Service s) <-> floatingTag(s).",
    ],
)
print("Generated metamodel:\n")
print(metamodel.source)

# What Refinery can't express (here, a regular expression) is computed in Python.
floating_tag = Predicate(
    "floatingTag", (Service,), lambda s: re.search(r":(latest)?$", s.image) is not None
)

with RefineryServer() as server, server.client() as refinery:
    report = metamodel.validate([deployment], refinery, predicates=[floating_tag])
    print("Violations:")
    for violation in report.violations:
        print(f"  {violation}")

    # Which services can reach an unencrypted database, directly or not?
    # The worker -> db dependency was guessed from logs: it only may exist.
    guessed = Metamodel(
        Deployment,
        options={
            Service: ClassOptions(
                key=lambda s: s.name,
                references={"depends_on": "Component"},
                logic_value=lambda s, field, value: "unknown" if s.name == "worker" else True,
            ),
            Database: ClassOptions(key=lambda d: d.name),
        },
        unions={"Component": [Service, Database]},
    )
    result = guessed.query(
        [deployment],
        refinery,
        [
            "pred reachesUnencrypted(Service s, Database d) <-> "
            "Service::depends_on+(s, d), !Database::encrypted(d)."
        ],
    )
    print("\nServices reaching an unencrypted database:")
    for match in result.matches:
        print(f"  {match}")
