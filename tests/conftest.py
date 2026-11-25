# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def sse(*envelopes: dict, heartbeat: bool = True, line_ending: str = "\n") -> bytes:
    """Encode envelopes as a server-sent event stream like Refinery does."""
    parts = []
    for envelope in envelopes:
        if heartbeat:
            parts.append(":" + line_ending * 2)
        parts.append(f"data: {json.dumps(envelope)}" + line_ending * 2)
    return "".join(parts).encode()


def sse_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)


class Recorder:
    """Mock transport that records requests and answers with a fixed handler."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)


@pytest.fixture
def filesystem_problem() -> Path:
    return EXAMPLES / "filesystem.problem"
