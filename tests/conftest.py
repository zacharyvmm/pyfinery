# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import json
import os
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


GPU_TESTS = os.environ.get("PYFINERY_GPU_TESTS") == "1"

DEVICES = [
    pytest.param("cpu"),
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not GPU_TESTS, reason="set PYFINERY_GPU_TESTS=1"),
        ],
    ),
]


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    """Where the bits engine evaluates: the CPU, and with PYFINERY_GPU_TESTS=1, a GPU."""
    return str(request.param)


@pytest.fixture
def filesystem_problem() -> Path:
    return EXAMPLES / "filesystem.problem"
