# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""A deployment configuration loaded from JSON: components refer to each other by name."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from pyfinery import ClassOptions, Metamodel


class Port(BaseModel):
    number: int
    public: bool = False


class Service(BaseModel):
    kind: Literal["service"] = "service"
    name: str
    image: str
    tier: Literal["frontend", "backend", "data"]
    ports: list[Port] = []
    depends_on: list[str] = []
    replicas: int | None = None


class Database(BaseModel):
    kind: Literal["database"] = "database"
    name: str
    engine: Literal["postgres", "redis"]
    encrypted: bool


class Deployment(BaseModel):
    components: list[Annotated[Service | Database, Field(discriminator="kind")]]


RULES = [
    "error pred publicDependsOnData(Service s, Database d) <-> "
    "Service::ports(s, p), Port::public(p), Service::depends_on(s, d).",
]


def metamodel(**kwargs) -> Metamodel:
    return Metamodel(
        Deployment,
        options={
            Service: ClassOptions(key=lambda s: s.name, references={"depends_on": "Component"}),
            Database: ClassOptions(key=lambda d: d.name),
        },
        unions={"Component": [Service, Database]},
        rules=RULES,
        **kwargs,
    )


DATA = {
    "components": [
        {
            "kind": "service",
            "name": "web",
            "image": "web:1",
            "tier": "frontend",
            "ports": [{"number": 443, "public": True}],
            "depends_on": ["api", "cache"],
        },
        {
            "kind": "service",
            "name": "api",
            "image": "api:latest",
            "tier": "backend",
            "depends_on": ["db", "billing"],
        },
        {"kind": "database", "name": "db", "engine": "postgres", "encrypted": False},
        {"kind": "database", "name": "cache", "engine": "redis", "encrypted": True},
    ]
}


def deployment() -> Deployment:
    return Deployment.model_validate(DATA)
