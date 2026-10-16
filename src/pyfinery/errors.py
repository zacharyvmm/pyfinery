# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Exceptions raised by the Refinery client.

Ported from ``RefineryError.ts`` of the official JavaScript client
``@tools.refinery/client``:
https://github.com/graphs4value/refinery/tree/main/subprojects/client-js/src
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ErrorResult, Issue, RequestErrorDetail


class RefineryError(Exception):
    """Base class of all errors reported by Refinery or this client."""

    #: The ``result`` field reported by the server (e.g., ``"unsatisfiable"``).
    result: str = "error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RefineryTimeoutError(RefineryError, TimeoutError):
    """The server ran out of time (see ``REFINERY_*_TIMEOUT_*`` server settings)."""

    result = "timeout"


class RequestCancelledError(RefineryError):
    """The server cancelled the operation."""

    result = "cancelled"


class BadRequestError(RefineryError):
    """The server rejected the request as malformed."""

    result = "requestError"

    def __init__(self, message: str, details: list[RequestErrorDetail] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


class ServerError(RefineryError):
    """Internal server error."""

    result = "serverError"


class InvalidProblemError(RefineryError):
    """The problem source has syntax or validation errors (see ``issues``)."""

    result = "invalidProblem"

    def __init__(self, message: str, issues: list[Issue] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []

    def __str__(self) -> str:
        if not self.issues:
            return self.message
        return "\n".join([self.message, *(f"  {issue}" for issue in self.issues)])


class UnsatisfiableError(RefineryError):
    """The model generation problem has no solution."""

    result = "unsatisfiable"


class ProtocolError(RefineryError):
    """The server sent a response this client could not understand."""

    result = "protocolError"


def from_error_result(error: ErrorResult) -> RefineryError:
    match error.result:
        case "timeout":
            return RefineryTimeoutError(error.message)
        case "cancelled":
            return RequestCancelledError(error.message)
        case "requestError":
            return BadRequestError(error.message, error.details)
        case "serverError":
            return ServerError(error.message)
        case "invalidProblem":
            return InvalidProblemError(error.message, error.issues)
        case "unsatisfiable":
            return UnsatisfiableError(error.message)
