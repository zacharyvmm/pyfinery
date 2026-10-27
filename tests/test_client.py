# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import asyncio
import json

import httpx
import pytest

import pyfinery as pf
from pyfinery._protocol import SSEDecoder

from .conftest import Recorder, sse, sse_response

BASE = "http://refinery.test/api/v1"

MODEL_JSON = {
    "concreteness": "candidate",
    "nodes": [
        {"name": "dir1", "simpleName": "dir1", "kind": "default", "color": "7"},
        {"name": "file1", "simpleName": "file1", "kind": "default"},
    ],
    "relations": [
        {
            "name": "Dir",
            "simpleName": "Dir",
            "arity": 1,
            "detail": {"type": "class", "isAbstract": False},
        },
        {
            "name": "Dir::contents",
            "simpleName": "contents",
            "arity": 2,
            "parameterNames": ["Dir", "FSObject"],
            "detail": {"type": "reference", "isContainment": True},
        },
    ],
    "partialInterpretation": {
        "Dir": [[0, "true"], [1, "false"]],
        "Dir::contents": [[0, 1, "true"], [1, 0, {"unknown": "?"}]],
    },
}

GENERATE_SUCCESS = {"result": "success", "value": {"json": MODEL_JSON, "source": "Dir(dir1)."}}


def sync_client(handler) -> tuple[pf.Refinery, Recorder]:
    recorder = Recorder(handler)
    http = httpx.Client(transport=httpx.MockTransport(recorder))
    return pf.Refinery(BASE + "/", http_client=http), recorder


def async_client(handler) -> tuple[pf.AsyncRefinery, Recorder]:
    recorder = Recorder(handler)
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return pf.AsyncRefinery(BASE, http_client=http), recorder


def status(value) -> dict:
    return {"result": "status", "value": value}


def test_generate_sends_camel_case_request_and_reports_status():
    body = sse(
        status({"message": "Initializing"}), status({"message": "Generating"}), GENERATE_SUCCESS
    )
    client, recorder = sync_client(lambda _: sse_response(body))
    messages = []

    result = client.generate(
        "class A.",
        scopes=["node=10..20", pf.Scope(relation="A", lower_bound=2)],
        random_seed=5,
        on_status=lambda s: messages.append(s.message),
    )

    request = recorder.requests[0]
    assert str(request.url) == f"{BASE}/generate"
    assert request.headers["accept"] == "text/event-stream"
    assert recorder.last_json == {
        "input": {"source": "class A."},
        "format": {
            "json": {
                "enabled": True,
                "nonExistingObjects": "discard",
                "shadowPredicates": "discard",
            },
            "source": {"enabled": True},
        },
        "scopes": [
            {
                "relation": "node",
                "override": False,
                "incremental": False,
                "lowerBound": 10,
                "upperBound": 20,
            },
            {"relation": "A", "override": False, "incremental": False, "lowerBound": 2},
        ],
        "randomSeed": 5,
    }
    assert messages == ["Initializing", "Generating"]
    assert result.source == "Dir(dir1)."
    assert result.json_ is not None and result.json_.concreteness == "candidate"


def test_problem_can_be_read_from_path(filesystem_problem):
    client, recorder = sync_client(lambda _: sse_response(sse(GENERATE_SUCCESS)))
    client.generate(filesystem_problem)
    assert recorder.last_json["input"]["source"] == filesystem_problem.read_text()


def test_stream_generate_yields_statuses_then_result():
    body = sse(status({"message": "a"}), GENERATE_SUCCESS)
    client, _ = sync_client(lambda _: sse_response(body))
    events = list(client.stream_generate("class A."))
    assert [type(e) for e in events] == [pf.GenerateStatus, pf.GenerateResult]


def test_generate_many_collects_models():
    model = {"json": MODEL_JSON, "source": "m"}
    body = sse(
        status({"message": "Generating 1/2"}),
        status({"model": model}),
        status({"model": model}),
        {"result": "success", "value": {"stopReason": "requestFulfilled"}},
    )
    client, recorder = sync_client(lambda _: sse_response(body))
    seen = []
    result = client.generate_many("class A.", 2, on_status=seen.append)
    assert recorder.last_json["count"] == 2
    assert str(recorder.requests[0].url) == f"{BASE}/generateMany"
    assert result.stop_reason == "requestFulfilled"
    assert [m.source for m in result.models] == ["m", "m"]
    assert [type(s) for s in seen] == [pf.GenerateStatus, pf.GenerateResult, pf.GenerateResult]


def test_generate_many_rejects_ambiguous_status():
    body = sse(status({"message": "x", "model": {}}))
    client, _ = sync_client(lambda _: sse_response(body))
    with pytest.raises(pf.ProtocolError):
        client.generate_many("class A.", 1)


def test_semantics_and_concretize():
    body = sse({"result": "success", "value": {"issues": [], "json": MODEL_JSON, "source": "s"}})
    client, recorder = sync_client(lambda _: sse_response(body))
    assert client.semantics("class A.").issues == []
    assert client.concretize("class A.").source == "s"
    assert [r.url.path for r in recorder.requests] == ["/api/v1/semantics", "/api/v1/concretize"]
    assert "scopes" not in recorder.last_json


def test_error_at_end_of_stream():
    body = sse(status({"message": "Generating"}), {"result": "unsatisfiable", "message": "Nope"})
    client, _ = sync_client(lambda _: sse_response(body))
    with pytest.raises(pf.UnsatisfiableError, match="Nope"):
        client.generate("class A.")


def test_result_in_unterminated_last_event():
    # The stream ends without the blank line that ends the last event.
    body = sse(status({"message": "Generating"}), GENERATE_SUCCESS).rstrip(b"\n")
    statuses = []
    client, _ = sync_client(lambda _: sse_response(body))
    assert client.generate("class A.", on_status=statuses.append).source == "Dir(dir1)."
    assert [s.message for s in statuses] == ["Generating"]

    async_refinery, _ = async_client(lambda _: sse_response(body))
    result = asyncio.run(async_refinery.generate("class A."))
    assert result.source == "Dir(dir1)."


def test_status_in_unterminated_last_event():
    body = sse(status({"message": "Generating"})).rstrip(b"\n")
    client, _ = sync_client(lambda _: sse_response(body))
    with pytest.raises(pf.ProtocolError, match="without sending a result"):
        client.generate("class A.")

    async_refinery, _ = async_client(lambda _: sse_response(body))
    with pytest.raises(pf.ProtocolError, match="without sending a result"):
        asyncio.run(async_refinery.generate("class A."))


def test_error_with_http_status():
    issue = {
        "description": "Bad",
        "severity": "error",
        "line": 1,
        "column": 2,
        "offset": 0,
        "length": 1,
    }
    response = httpx.Response(
        400, json={"result": "invalidProblem", "message": "Invalid", "issues": [issue]}
    )
    client, _ = sync_client(lambda _: response)
    with pytest.raises(pf.InvalidProblemError) as info:
        client.generate("class A {")
    assert info.value.issues[0].description == "Bad"
    assert "1:2: error: Bad" in str(info.value)


def test_request_error_details():
    response = httpx.Response(
        400,
        json={
            "result": "requestError",
            "message": "Bad request",
            "details": [{"propertyPath": "input", "message": "must not be null"}],
        },
    )
    client, _ = sync_client(lambda _: response)
    with pytest.raises(pf.BadRequestError) as info:
        client.semantics("")
    assert info.value.details[0].property_path == "input"


def test_timeout_error_is_builtin_timeout():
    client, _ = sync_client(
        lambda _: httpx.Response(408, json={"result": "timeout", "message": "Request timed out"})
    )
    with pytest.raises(TimeoutError):
        client.semantics("class A.")


def test_non_json_error_page():
    client, _ = sync_client(lambda _: httpx.Response(404, text="<html>Not found</html>"))
    with pytest.raises(pf.ProtocolError, match="HTTP 404"):
        client.generate("class A.")


def test_stream_without_result():
    client, _ = sync_client(lambda _: sse_response(sse(status({"message": "x"}))))
    with pytest.raises(pf.ProtocolError, match="without sending a result"):
        client.generate("class A.")


def test_crlf_and_missing_final_blank_line():
    body = (
        sse(status({"message": "x"}), line_ending="\r\n")
        + f"data: {json.dumps(GENERATE_SUCCESS)}".encode()
    )
    client, _ = sync_client(lambda _: sse_response(body))
    assert client.generate("class A.").source == "Dir(dir1)."


def test_sse_decoder_joins_multiline_data():
    decoder = SSEDecoder()
    assert decoder.feed(": heartbeat") is None
    assert decoder.feed("event: ignored") is None
    assert decoder.feed("data: {") is None
    assert decoder.feed("data:}") is None
    assert decoder.feed("") == "{\n}"
    assert decoder.feed("") is None


def test_async_client():
    model = {"json": MODEL_JSON}
    body = sse(
        status({"model": model}),
        {"result": "success", "value": {"stopReason": "noMoreSolutions"}},
    )

    async def main():
        client, recorder = async_client(lambda _: sse_response(body))
        statuses = []

        async def on_status(s):
            statuses.append(s)

        async with client:
            result = await client.generate_many("class A.", 5, on_status=on_status)
        assert recorder.last_json["count"] == 5
        assert result.stop_reason == "noMoreSolutions"
        assert len(result.models) == 1 and len(statuses) == 1

        client, _ = async_client(
            lambda _: sse_response(sse({"result": "serverError", "message": "Boom"}))
        )
        with pytest.raises(pf.ServerError):
            await client.generate("class A.")

    asyncio.run(main())


def test_chat_client():
    body = sse(
        status({"role": "assistant", "content": "Thinking"}),
        {"result": "success", "value": {"issues": [], "source": "A(a)."}},
    )
    recorder = Recorder(lambda _: sse_response(body))
    chat = pf.RefineryChat(
        "http://chat.test", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )
    messages = []
    result = chat.text_to_model("class A.", "one A", on_status=messages.append)
    assert recorder.requests[0].url.path == "/textToModel"
    assert recorder.last_json["metamodel"] == {"source": "class A."}
    assert messages[0].role == "assistant"
    assert result.source == "A(a)."


def test_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("REFINERY_API_BASE", "http://env.test/api/v1/")
    with pf.Refinery() as client:
        assert client.base_url == "http://env.test/api/v1"
