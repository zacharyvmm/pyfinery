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
    "ConcretizationResult",
    "GenerateManyResult",
    "GenerateResult",
    "GenerateStatus",
    "InvalidProblemError",
    "Issue",
    "JsonOutput",
    "JsonOutputFormat",
    "NodeMetadata",
    "OutputFormats",
    "ProtocolError",
    "Refinery",
    "RefineryChat",
    "RefineryError",
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
]
