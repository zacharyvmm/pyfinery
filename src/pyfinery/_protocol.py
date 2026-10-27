# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Transport-independent parts of the Refinery wire protocol.

The server-sent event parsing is ported from ``GenericRefinery.ts`` of the
official JavaScript client ``@tools.refinery/client``:
https://github.com/graphs4value/refinery/tree/main/subprojects/client-js/src

Every endpoint answers with JSON envelopes of the form
``{"result": "success" | "status" | <error>, ...}``. Streaming responses send
one envelope per server-sent event: zero or more ``status`` updates followed by
either a ``success`` or an error envelope.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from .errors import ProtocolError, RefineryError, from_error_result
from .models import ErrorResult, GenerateResult, GenerateStatus

type ProblemSource = str | os.PathLike[str]
type StatusParser = Callable[[Any], Any]

DATA_PREFIX = "data:"

DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=60.0)
"""The server sends a heartbeat every second while streaming, so a finite read
timeout only trips if the connection is actually stuck."""


def read_source(problem: ProblemSource) -> str:
    """Strings are problem sources; path-like objects are read from disk."""
    if isinstance(problem, str):
        return problem
    return Path(problem).read_text(encoding="utf-8")


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


@dataclass(frozen=True, slots=True)
class Endpoint:
    """An endpoint of the API, and how to read the events it streams."""

    path: str
    success_model: type[BaseModel]
    parse_status: StatusParser | None = None
    """How to read status updates; ``None`` to drop them."""


@dataclass(frozen=True, slots=True)
class Event:
    kind: Literal["status", "success"]
    value: Any


class SSEDecoder:
    """Assembles the ``data:`` lines of server-sent events into payloads."""

    def __init__(self) -> None:
        self._data: list[str] | None = None

    def feed(self, line: str) -> str | None:
        """Process one line; return the event payload when an event ends."""
        if line == "":
            return self.flush()
        if line.startswith(DATA_PREFIX):
            value = line[len(DATA_PREFIX) :]
            if value.startswith(" "):
                value = value[1:]
            if self._data is None:
                self._data = []
            self._data.append(value)
        # Comments (heartbeats) and other fields are ignored.
        return None

    def flush(self) -> str | None:
        data, self._data = self._data, None
        return None if data is None else "\n".join(data)


class EventParser:
    """Turns the lines of an endpoint's server-sent event stream into events."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._decoder = SSEDecoder()

    def feed(self, line: str) -> Event | None:
        """Process one line; return the event it ends, unless that is a dropped status."""
        data = self._decoder.feed(line)
        return None if data is None else self._parse(data)

    def finish(self) -> Any:
        """The result in an event left unterminated at the end of the stream, if any."""
        data = self._decoder.flush()
        event = None if data is None else self._parse(data)
        return event.value if event is not None and event.kind == "success" else None

    def _parse(self, data: str) -> Event | None:
        return parse_event_data(data, self._endpoint.success_model, self._endpoint.parse_status)


def _validate[T](parse: Callable[[Any], T], value: Any) -> T:
    try:
        return parse(value)
    except ValidationError as e:
        raise ProtocolError(f"Unexpected response from Refinery: {e}") from e


def parse_envelope(
    envelope: Any,
    success_model: type[BaseModel],
    parse_status: StatusParser | None,
) -> Event | None:
    """Turn an envelope into an :class:`Event`, raising for error results.

    Status updates are dropped (``None``) if the endpoint has no status parser.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("result"), str):
        raise ProtocolError(f"Unexpected response from Refinery: {envelope!r}")
    match envelope["result"]:
        case "success":
            return Event("success", _validate(success_model.model_validate, envelope.get("value")))
        case "status":
            if parse_status is None:
                return None
            return Event("status", _validate(parse_status, envelope.get("value")))
        case _:
            raise from_error_result(_validate(ErrorResult.model_validate, envelope))


def parse_event_data(
    data: str,
    success_model: type[BaseModel],
    parse_status: StatusParser | None,
) -> Event | None:
    try:
        envelope = json.loads(data)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"Malformed event from Refinery: {data!r}") from e
    return parse_envelope(envelope, success_model, parse_status)


def error_from_response(response: httpx.Response) -> RefineryError:
    """Exception for a response with a non-2xx HTTP status code.

    ``response`` must already have been read.
    """
    try:
        envelope = response.json()
    except ValueError:
        envelope = None
    if isinstance(envelope, dict) and envelope.get("result") not in (None, "success", "status"):
        try:
            parse_envelope(envelope, GenerateResult, None)
        except RefineryError as e:
            return e
    snippet = response.text[:200]
    return ProtocolError(f"HTTP {response.status_code} from {response.request.url}: {snippet}")


def unexpected_end_of_stream() -> ProtocolError:
    return ProtocolError("Refinery closed the stream without sending a result")


def parse_generate_many_status(value: Any) -> GenerateStatus | GenerateResult:
    """``generateMany`` statuses are either progress messages or generated models."""
    if isinstance(value, dict):
        if value.keys() == {"message"}:
            return GenerateStatus.model_validate(value)
        if value.keys() == {"model"}:
            return GenerateResult.model_validate(value["model"])
    raise ProtocolError(f"Unexpected generateMany status: {value!r}")
