# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Models mixing RefineryNode classes with nested plain Pydantic models."""

from __future__ import annotations

import enum
from typing import Annotated

from pydantic import BaseModel, Field

from pyfinery import Metamodel, Ref, RefineryNode


class Country(enum.Enum):
    CA = "CA"
    FR = "FR"


class Address(BaseModel):
    street: str
    country: Country


class Contact(BaseModel):
    value: str


class Phone(Contact):
    pass


class Email(Contact):
    verified: bool = False


class Person(RefineryNode):
    __node_id__ = "name"
    name: str
    home: Address
    work: Address | None = None
    contacts: list[Contact] = Field(default_factory=list, max_length=2)
    manager: Person | None = None


class Line(BaseModel):
    sku: str
    qty: int


class Order(BaseModel):
    lines: Annotated[list[Line], Ref(multiplicity="1..*")]


def people() -> Metamodel:
    return Metamodel(Person, rules=["error pred selfManaged(Person p) <-> manager(p, p)."])
