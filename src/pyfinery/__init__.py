# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Python client for Refinery, the graph solver for partial models.

See https://refinery.tools for the problem specification language.
"""

from .errors import (
    BadRequestError,
    InvalidProblemError,
    ProtocolError,
    RefineryError,
    RefineryTimeoutError,
    RequestCancelledError,
    ServerError,
    UnsatisfiableError,
)
from .http import (
    DEFAULT_BASE_URL,
    AsyncRefinery,
    AsyncRefineryChat,
    ChatMessage,
    ConcretizationResult,
    GenerateManyResult,
    GenerateResult,
    GenerateStatus,
    Issue,
    JsonOutput,
    JsonOutputFormat,
    NodeMetadata,
    OutputFormats,
    Refinery,
    RefineryChat,
    RefineryServer,
    RelationMetadata,
    Scope,
    SemanticsResult,
    ServerStartError,
    SourceOutputFormat,
    TextToModelResult,
)

# Load http before graph: _results imports http.models at runtime, and http imports http.engine.
# isort: split

from .graph import (
    ClassOptions,
    Container,
    Contains,
    LogicValue,
    Match,
    Metamodel,
    ModelValidationError,
    Predicate,
    QueryResult,
    Ref,
    RefineryNode,
    ValidationReport,
    Violation,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "AsyncRefinery",
    "AsyncRefineryChat",
    "BadRequestError",
    "ChatMessage",
    "ClassOptions",
    "ConcretizationResult",
    "Container",
    "Contains",
    "GenerateManyResult",
    "GenerateResult",
    "GenerateStatus",
    "InvalidProblemError",
    "Issue",
    "JsonOutput",
    "JsonOutputFormat",
    "LogicValue",
    "Match",
    "Metamodel",
    "ModelValidationError",
    "NodeMetadata",
    "OutputFormats",
    "Predicate",
    "ProtocolError",
    "QueryResult",
    "Ref",
    "Refinery",
    "RefineryChat",
    "RefineryError",
    "RefineryNode",
    "RefineryServer",
    "RefineryTimeoutError",
    "RelationMetadata",
    "RequestCancelledError",
    "Scope",
    "SemanticsResult",
    "ServerError",
    "ServerStartError",
    "SourceOutputFormat",
    "TextToModelResult",
    "UnsatisfiableError",
    "ValidationReport",
    "Violation",
]
