"""How :class:`Metamodel` turns what it is passed into an engine, and what it gives it."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

import pyfinery as pf
from pyfinery import Predicate
from pyfinery._engines import (
    AsyncClientEngine,
    ClientEngine,
    Engine,
    Problems,
    QueryEngine,
    async_engine,
    async_query_engine,
    engine,
    query_engine,
)
from pyfinery._instances import build_graph
from pyfinery._render import render_query_problem, render_validation_problem
from pyfinery._results import QueryResult, ValidationReport

from .conftest import Recorder, sse, sse_response
from .graph_models import Dir, Filesystem, metamodel

EXTRA = ["pred named(Dir d) <-> Dir::name(d, _)."]
NAMED = Predicate("special", (Dir,), lambda d: d.name == "home")


def objects() -> list[Filesystem]:
    return [Filesystem(root=Dir(name="home"))]


def _model_json() -> dict:
    return {
        "concreteness": "candidate",
        "nodes": [{"name": "home", "simpleName": "home", "kind": "default"}],
        "relations": [],
        "partialInterpretation": {},
    }


def _client(recorder: Recorder) -> pf.Refinery:
    return pf.Refinery(
        "http://x", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )


def _async_client(recorder: Recorder) -> pf.AsyncRefinery:
    transport = httpx.MockTransport(recorder)
    return pf.AsyncRefinery("http://x", http_client=httpx.AsyncClient(transport=transport))


def _recorder() -> Recorder:
    body = sse({"result": "success", "value": {"issues": [], "json": _model_json()}})
    return Recorder(lambda _: sse_response(body))


class QueryOnly:
    """An engine of one's own that only queries, recording what it is given."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def query_graph(self, schema, graph, rules, predicates, select) -> QueryResult:
        self.calls.append(("query", schema, graph, rules, predicates, select))
        return QueryResult([], [], "mine", None)


class Validating(QueryOnly):
    def validate_graph(self, schema, graph, rules, predicates) -> ValidationReport:
        self.calls.append(("validate", schema, graph, rules, predicates))
        return ValidationReport([], "mine", None, graph.identity)


class AsyncValidating:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def query_graph(self, schema, graph, rules, predicates, select) -> QueryResult:
        self.calls.append(("query", schema, graph, rules, predicates, select))
        return QueryResult([], [], "mine", None)

    async def validate_graph(self, schema, graph, rules, predicates) -> ValidationReport:
        self.calls.append(("validate", schema, graph, rules, predicates))
        return ValidationReport([], "mine", None, graph.identity)


# -- clients ------------------------------------------------------------------


def test_validate_wraps_a_client():
    mm, recorder = metamodel(), _recorder()
    report = mm.validate(objects(), _client(recorder), EXTRA, predicates=[NAMED])
    graph = build_graph(mm.schema, objects())
    expected = render_validation_problem(mm.schema, mm.source, graph, EXTRA, [NAMED])
    assert recorder.requests[0].url.path.endswith("/concretize")
    assert recorder.last_json["input"]["source"] == expected == report.source
    assert report.ok


def test_query_wraps_a_client():
    mm, recorder = metamodel(), _recorder()
    result = mm.query(objects(), _client(recorder), EXTRA, predicates=[NAMED], select=[])
    graph = build_graph(mm.schema, objects(), existence=True)
    expected = render_query_problem(mm.schema, mm.source, graph, EXTRA, [NAMED])
    assert recorder.requests[0].url.path.endswith("/semantics")
    assert recorder.last_json["input"]["source"] == expected == result.source
    assert mm.query_problem(objects(), EXTRA, predicates=[NAMED]) == expected


def test_async_clients_are_wrapped():
    mm, recorder = metamodel(), _recorder()

    async def main():
        async with _async_client(recorder) as client:
            report = await mm.avalidate(objects(), client, EXTRA)
            result = await mm.aquery(objects(), client, EXTRA, select=[])
            return report, result

    report, result = asyncio.run(main())
    paths = [r.url.path.rsplit("/", 1)[-1] for r in recorder.requests]
    assert paths == ["concretize", "semantics"]
    assert report.ok and result.source.startswith(mm.source)


def test_client_engines_are_engines():
    problems = Problems("", [])
    assert isinstance(engine(_client(_recorder()), problems), ClientEngine)
    assert isinstance(async_engine(_async_client(_recorder()), problems), AsyncClientEngine)
    assert isinstance(ClientEngine(_client(_recorder()), problems), Engine)


def test_problems_render_rules_not_of_the_metamodel_in_full():
    mm = metamodel()
    graph = build_graph(mm.schema, objects(), existence=True)
    problems = Problems(mm.source, mm.rules)
    assert problems.query(mm.schema, graph, [*mm.rules, *EXTRA], []) == render_query_problem(
        mm.schema, mm.source, graph, EXTRA, []
    )
    assert problems.query(mm.schema, graph, EXTRA, []).count("emptyDir") == 0


def test_the_wrong_kind_of_client_is_refused():
    mm = metamodel()
    with pytest.raises(TypeError, match=r"query\(\) needs a synchronous engine"):
        mm.query(objects(), _async_client(_recorder()))
    with pytest.raises(TypeError, match=r"validate\(\) needs a synchronous engine"):
        mm.validate(objects(), _async_client(_recorder()))
    with pytest.raises(TypeError, match=r"aquery\(\) needs an AsyncRefinery client"):
        asyncio.run(mm.aquery(objects(), _client(_recorder())))
    with pytest.raises(TypeError, match=r"avalidate\(\) needs an AsyncRefinery client"):
        asyncio.run(mm.avalidate(objects(), _client(_recorder())))
    with pytest.raises(TypeError, match=r"engine with query_graph\(\), not object"):
        mm.query(objects(), object())
    with pytest.raises(TypeError, match=r"engine with query_graph\(\), not object"):
        asyncio.run(mm.aquery(objects(), object()))


# -- engines of one's own -----------------------------------------------------


def test_a_custom_engine_queries():
    mm, mine = metamodel(), Validating()
    result = mm.query(objects(), mine, EXTRA, predicates=[NAMED])
    assert result.source == "mine"
    [(kind, schema, graph, rules, predicates, select)] = mine.calls
    assert kind == "query" and schema is mm.schema
    assert rules == [*mm.rules, *EXTRA]
    assert predicates == [NAMED]
    assert select == ["named"]
    assert [n.name for n in graph.nodes] == [
        n.name for n in build_graph(mm.schema, objects()).nodes
    ]
    mm.query(objects(), mine, EXTRA, select=iter(["emptyDir"]))
    assert mine.calls[-1][-1] == ["emptyDir"]


def test_a_custom_engine_validates():
    mm, mine = metamodel(), Validating()
    report = mm.validate(objects(), mine, EXTRA, predicates=iter([NAMED]))
    assert report.source == "mine"
    [(kind, schema, _graph, rules, predicates)] = mine.calls
    assert (kind, schema, rules, predicates) == (
        "validate",
        mm.schema,
        [*mm.rules, *EXTRA],
        [NAMED],
    )


def test_validate_needs_an_engine_that_validates():
    mm = metamodel()
    with pytest.raises(TypeError, match=r"validate\(\) needs an engine that validates.*QueryOnly"):
        mm.validate(objects(), QueryOnly())
    with pytest.raises(TypeError, match=r"avalidate\(\) needs an engine that validates"):
        asyncio.run(mm.avalidate(objects(), QueryOnly()))
    assert mm.query(objects(), QueryOnly()).source == "mine"


def test_async_engines_are_awaited():
    mm, mine = metamodel(), AsyncValidating()
    assert asyncio.run(mm.aquery(objects(), mine, EXTRA)).source == "mine"
    assert asyncio.run(mm.avalidate(objects(), mine, EXTRA)).source == "mine"
    assert [c[0] for c in mine.calls] == ["query", "validate"]
    assert mine.calls[0][3] == [*mm.rules, *EXTRA] and mine.calls[0][5] == ["named"]
    assert async_query_engine(mine, Problems("", [])) is mine
    with pytest.raises(TypeError, match=r"query\(\) needs a synchronous engine"):
        mm.query(objects(), mine)


def test_sync_engines_run_in_async_methods():
    mm, mine = metamodel(), Validating()
    assert asyncio.run(mm.aquery(objects(), mine, EXTRA)).source == "mine"
    assert asyncio.run(mm.avalidate(objects(), mine)).source == "mine"
    assert [c[0] for c in mine.calls] == ["query", "validate"]


def test_engines_are_passed_through():
    mine = Validating()
    assert query_engine(mine, Problems("", [])) is mine
    assert engine(mine, Problems("", [])) is mine
    assert isinstance(mine, QueryEngine) and isinstance(mine, Engine)


# -- the bits engine ----------------------------------------------------------


def test_the_bits_engine_queries_and_does_not_validate():
    pytest.importorskip("numpy")
    from pyfinery.bits import BitsEngine

    bits, mm = BitsEngine(), metamodel()
    assert isinstance(bits, QueryEngine) and not isinstance(bits, Engine)
    assert query_engine(bits, Problems(mm.source, mm.rules)) is bits
    with pytest.raises(TypeError, match="the bits engine only queries"):
        mm.validate(objects(), bits)
    with pytest.raises(TypeError, match="the bits engine only queries"):
        asyncio.run(mm.avalidate(objects(), bits))


def test_choosing_an_engine_does_not_import_the_bits_engine():
    code = (
        "import sys, pyfinery\n"
        "from tests.graph_models import Dir, Filesystem, metamodel\n"
        "class Mine:\n"
        "    def query_graph(self, *args): return None\n"
        "metamodel().query([Filesystem(root=Dir(name='a'))], Mine())\n"
        "assert 'pyfinery.bits' not in sys.modules and 'numpy' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


def test_objects_shaped_like_a_client_are_still_clients():
    """Before engines, anything with concretize() and semantics() was used as a client."""

    class Shaped:
        def __init__(self, client):
            self.client = client

        def concretize(self, *args, **kwargs):
            return self.client.concretize(*args, **kwargs)

        def semantics(self, *args, **kwargs):
            return self.client.semantics(*args, **kwargs)

    class AsyncShaped(Shaped):
        async def concretize(self, *args, **kwargs):
            return await self.client.concretize(*args, **kwargs)

        async def semantics(self, *args, **kwargs):
            return await self.client.semantics(*args, **kwargs)

    problems = Problems("", [])
    assert isinstance(engine(Shaped(None), problems), ClientEngine)
    assert isinstance(async_engine(AsyncShaped(None), problems), AsyncClientEngine)
    mm, recorder = metamodel(), _recorder()
    assert mm.validate(objects(), Shaped(_client(recorder))).ok
    assert recorder.requests[0].url.path.endswith("/concretize")
    with pytest.raises(TypeError, match=r"query\(\) needs a synchronous engine"):
        mm.query(objects(), AsyncShaped(None))
