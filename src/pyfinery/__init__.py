# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Python client for Refinery, the graph solver for partial models.

See https://refinery.tools for the problem specification language.
"""

from .client import (
    DEFAULT_BASE_URL,
    AsyncRefinery,
    AsyncRefineryChat,
    Refinery,
    RefineryChat,
)
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
from .models import (
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
    RelationMetadata,
    Scope,
    SemanticsResult,
    SourceOutputFormat,
    TextToModelResult,
)
from .server import RefineryServer, ServerStartError

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
