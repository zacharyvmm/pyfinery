# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Request and response models of the Refinery HTTP API (``/api/v1``).

Ported from the zod schemas in ``dto/`` and ``chat/dto/`` of the official
JavaScript client ``@tools.refinery/client``:
https://github.com/graphs4value/refinery/tree/main/subprojects/client-js/src

Field names are snake_case in Python and camelCase on the wire. Models can be
constructed with either spelling.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


def _json_field(**kwargs):
    """``json`` on the wire, ``json_`` in Python (``json`` would shadow a BaseModel method)."""
    return Field(
        validation_alias=AliasChoices("json", "json_"), serialization_alias="json", **kwargs
    )


class RefineryModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

PartialInterpretationPreservation = Literal["keep", "discard"]


class ProblemInput(RefineryModel):
    source: str


class JsonOutputFormat(RefineryModel):
    enabled: bool = True
    non_existing_objects: PartialInterpretationPreservation = "discard"
    shadow_predicates: PartialInterpretationPreservation = "discard"
    relations: list[str] | None = None
    """Names of the relations to return; all of them if ``None``."""
    excluded_relations: list[str] | None = None
    """Names of the relations to leave out, even if ``relations`` lists them.

    ``relations`` and ``excluded_relations`` need the server of
    :class:`~pyfinery.RefineryServer`'s default image; other servers ignore
    them and return every relation.
    """


class SourceOutputFormat(RefineryModel):
    enabled: bool = True


class OutputFormats(RefineryModel):
    """Which outputs the server should return (both are enabled by default)."""

    json_: JsonOutputFormat = _json_field(default_factory=JsonOutputFormat)
    source: SourceOutputFormat = Field(default_factory=SourceOutputFormat)


_SCOPE_PATTERN = re.compile(
    r"^\s*(?P<relation>[^\s+=]+)\s*(?P<incremental>\+?)=\s*"
    r"(?P<lower>\d+)\s*(?:\.\.\s*(?P<upper>\d+|\*))?\s*$"
)


class Scope(RefineryModel):
    """Bounds on the number of objects of a type (or tuples of a relation).

    Equivalent to a ``scope`` declaration in the problem source, e.g.
    ``Scope.parse("File=10..*")`` corresponds to ``scope File = 10..*.``.
    """

    relation: str
    override: bool = False
    incremental: bool = False
    lower_bound: int = Field(default=0, ge=0)
    upper_bound: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> Scope:
        if self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be less than or equal to upper_bound")
        return self

    @classmethod
    def parse(cls, constraint: str, *, override: bool = False) -> Scope:
        """Parse ``relation=lower..upper`` syntax.

        Accepted forms: ``node=10..20``, ``File=5`` (exactly 5), ``Dir=2..*``
        (no upper bound) and ``Dir+=1..3`` (incremental, i.e., on top of the
        objects already present in the problem).
        """
        match = _SCOPE_PATTERN.match(constraint)
        if match is None:
            raise ValueError(f"Invalid scope constraint: {constraint!r}")
        lower = int(match["lower"])
        upper_text = match["upper"]
        if upper_text is None:
            upper: int | None = lower
        elif upper_text == "*":
            upper = None
        else:
            upper = int(upper_text)
        return cls(
            relation=match["relation"],
            override=override,
            incremental=bool(match["incremental"]),
            lower_bound=lower,
            upper_bound=upper,
        )


class GenerateRequest(RefineryModel):
    input: ProblemInput
    format: OutputFormats = Field(default_factory=OutputFormats)
    scopes: list[Scope] | None = None
    random_seed: int = 1


class GenerateManyRequest(GenerateRequest):
    count: int = Field(default=1, gt=0, le=2**31 - 1)


class SemanticsRequest(RefineryModel):
    input: ProblemInput
    format: OutputFormats = Field(default_factory=OutputFormats)


class TextToModelRequest(RefineryModel):
    metamodel: ProblemInput
    text: str
    format: OutputFormats = Field(default_factory=OutputFormats)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

Severity = Literal["error", "warning", "info", "ignore"]


class Issue(RefineryModel):
    description: str
    severity: Severity
    line: int
    column: int
    offset: int
    length: int

    def __str__(self) -> str:
        return f"{self.line}:{self.column}: {self.severity}: {self.description}"


class NodeMetadata(RefineryModel):
    name: str
    simple_name: str
    color: str | None = None
    kind: Literal["default", "atom", "multi"]


class ClassDetail(RefineryModel):
    type: Literal["class"]
    is_abstract: bool
    color: str | None = None


class ComputedDetail(RefineryModel):
    type: Literal["computed"]
    of: str


class ReferenceDetail(RefineryModel):
    type: Literal["reference"]
    is_containment: bool


class AttributeDetail(RefineryModel):
    type: Literal["attribute"]


class OppositeDetail(RefineryModel):
    type: Literal["opposite"]
    of: str
    is_container: bool


class PredicateDetail(RefineryModel):
    type: Literal["pred"]
    kind: Literal["defined", "base", "error", "shadow"]


class FunctionDetail(RefineryModel):
    type: Literal["function"]
    kind: Literal["defined", "base", "shadow"]


class DomainDetail(RefineryModel):
    type: Literal["domain"]
    of: str


RelationDetail = Annotated[
    ClassDetail
    | ComputedDetail
    | ReferenceDetail
    | AttributeDetail
    | OppositeDetail
    | PredicateDetail
    | FunctionDetail
    | DomainDetail,
    Field(discriminator="type"),
]

Visibility = Literal["all", "must", "none"]


class RelationMetadata(RefineryModel):
    name: str
    simple_name: str
    arity: int
    parameter_names: list[str] | None = None
    detail: RelationDetail
    visibility: Visibility | None = None
    data_type: str | None = None


class UnknownValue(RefineryModel):
    unknown: str


class ErrorValue(RefineryModel):
    error: str


TupleElement = int | float | str | UnknownValue | ErrorValue


class JsonOutput(RefineryModel):
    """A (partial or candidate) model in Refinery's JSON output format.

    ``partial_interpretation`` maps each relation name to a list of tuples.
    The first ``arity`` elements of a tuple are indices into ``nodes`` and the
    last element is the logic value (e.g., ``"true"``, ``"unknown"``,
    ``"error"``) or the value of an attribute.
    """

    concreteness: Literal["partial", "candidate"]
    nodes: list[NodeMetadata]
    relations: list[RelationMetadata]
    partial_interpretation: dict[str, list[list[TupleElement]]]

    def relation(self, name: str) -> RelationMetadata:
        """Look up a relation by its qualified or simple name."""
        for relation in self.relations:
            if relation.name == name:
                return relation
        matches = [r for r in self.relations if r.simple_name == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(name)
        candidates = ", ".join(r.name for r in matches)
        raise KeyError(f"Ambiguous relation name {name!r}: {candidates}")

    def tuples(self, relation: str) -> list[tuple[TupleElement, ...]]:
        """Tuples of ``relation`` with node indices replaced by node names."""
        metadata = self.relation(relation)
        result: list[tuple[TupleElement, ...]] = []
        for raw in self.partial_interpretation.get(metadata.name, []):
            indices = cast(list[int], raw[: metadata.arity])
            nodes = tuple(self.nodes[index].name for index in indices)
            result.append(nodes + tuple(raw[metadata.arity :]))
        return result

    def true_tuples(self, relation: str) -> list[tuple[str, ...]]:
        """Node names of the tuples for which ``relation`` is definitely true."""
        return [cast(tuple[str, ...], t[:-1]) for t in self.tuples(relation) if t[-1] == "true"]

    def tuple_counts(self) -> dict[str, int]:
        """The number of tuples of each relation, largest first.

        Refinery computes every relation in full, so a large count, e.g. of a
        helper predicate with three parameters, usually explains a slow
        request. :meth:`~pyfinery.Metamodel.validate` and
        :meth:`~pyfinery.Metamodel.query` don't ask for the helper predicates
        of their rules, so those are missing here.
        """
        counts = {name: len(tuples) for name, tuples in self.partial_interpretation.items()}
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


class GenerateResult(RefineryModel):
    """A generated model. ``source`` is the model in Refinery syntax."""

    json_: JsonOutput | None = _json_field(default=None)
    source: str | None = None


class GenerateStatus(RefineryModel):
    message: str


StopReason = Literal["requestFulfilled", "noMoreSolutions", "unsatisfiable", "timeout"]


class GenerateManyResult(RefineryModel):
    stop_reason: StopReason
    models: list[GenerateResult] = Field(default_factory=list, exclude=True)
    """Models received as status updates (filled in by the client)."""


class SemanticsResult(RefineryModel):
    issues: list[Issue]
    json_: JsonOutput | None = _json_field(default=None)


class ConcretizationResult(SemanticsResult):
    source: str | None = None


class ChatMessage(RefineryModel):
    """Status update of the text-to-model chat service."""

    role: Literal["refinery", "assistant"]
    content: str


TextToModelResult = ConcretizationResult


# ---------------------------------------------------------------------------
# Error results
# ---------------------------------------------------------------------------


class RequestErrorDetail(RefineryModel):
    property_path: str
    message: str


class ErrorResult(RefineryModel):
    result: Literal[
        "timeout", "cancelled", "requestError", "serverError", "invalidProblem", "unsatisfiable"
    ]
    message: str
    details: list[RequestErrorDetail] | None = None
    issues: list[Issue] | None = None
