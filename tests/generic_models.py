# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Plain Pydantic models shaped like generated ones: a shared base, tagged unions,
Literal fields, value-equal copies of resources, and payload fields."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pyfinery import ClassOptions, Metamodel


class Model(BaseModel):
    """Shared base of every generated class, including unrelated ones."""

    model_config = ConfigDict(extra="forbid")


class Unrelated(Model):
    """Must not be pulled into a metamodel just for sharing the base class."""

    value: int = 0


class Path(Model):
    anchor: Literal["root", "home", "unknown"]
    segments: list[str] = []


class File(Model):
    type: Literal["file"] = "file"
    path: Path

    @property
    def key(self) -> str:
        return "file:" + "/".join(self.path.segments)


class Endpoint(Model):
    type: Literal["endpoint"] = "endpoint"
    url: str

    @property
    def key(self) -> str:
        return "endpoint:" + self.url


Resource = Annotated[File | Endpoint, Field(discriminator="type")]
Op = Literal["read", "write", "remove"]


class Effect(Model):
    type: Literal["effect"] = "effect"
    op: Op
    target: Resource | None = None
    sources: list[Resource] = []
    definite: bool
    index: int
    command: str = ""
    trace: list[str] = []


class Graph(Model):
    resources: list[Resource] = []
    effects: list[Effect] = []


OPTIONS = {
    File: ClassOptions(key=lambda r: r.key),
    Endpoint: ClassOptions(key=lambda r: r.key),
    Effect: ClassOptions(exclude=["command", "trace"]),
}


def metamodel(**kwargs) -> Metamodel:
    return Metamodel(
        Graph,
        options=OPTIONS,
        unions={"Resource": [File, Endpoint]},
        unsupported="skip",
        **kwargs,
    )


def secret() -> File:
    return File(path=Path(anchor="home", segments=[".ssh", "id_rsa"]))


def graph() -> Graph:
    """Reads a key and sends it; effects hold equal *copies* of the resources."""
    net = Endpoint(url="https://example.com")
    effects = [
        Effect(op="read", target=secret(), definite=True, index=0, command="cat ~/.ssh/id_rsa"),
        Effect(op="write", target=net.model_copy(), sources=[secret()], definite=True, index=1),
    ]
    return Graph(resources=[secret(), net], effects=effects)
