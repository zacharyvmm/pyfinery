# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import time

import httpx
import pytest

import pyfinery as pf

from .conftest import sse, sse_response

TIMEOUT = {"result": "timeout", "message": "Request timed out"}
SUCCESS = {"result": "success", "value": {"issues": []}}
INVALID = {"result": "invalidProblem", "message": "Validation errors detected", "issues": []}


def _server_with_responses(monkeypatch, responses):
    """A RefineryServer whose client() answers with ``responses`` in order."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[-1])
        return sse_response(sse(responses.pop(0)))

    server = pf.RefineryServer()
    server.port = 1
    monkeypatch.setattr(server, "logs", lambda: "")
    monkeypatch.setattr(
        server,
        "client",
        lambda: pf.Refinery(
            server.base_url, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )
    return server, calls


def test_warm_up_retries_after_timeouts(monkeypatch):
    responses = [TIMEOUT, SUCCESS, TIMEOUT, SUCCESS, SUCCESS, INVALID]
    server, calls = _server_with_responses(monkeypatch, responses)
    server._warm_up(time.monotonic() + 60)
    assert calls == ["semantics", "semantics", "concretize", "semantics", "concretize", "semantics"]
    assert responses == []


def test_warm_up_gives_up(monkeypatch):
    server, _ = _server_with_responses(monkeypatch, [TIMEOUT] * 100)
    with pytest.raises(pf.ServerStartError, match="semantics_timeout"):
        server._warm_up(time.monotonic() + 60)
