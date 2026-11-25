# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

from __future__ import annotations

import asyncio

import httpx
import pytest

import pyfinery as pf
from pyfinery import ClassOptions, Metamodel, Predicate
from pyfinery._results import declared_predicates

from .conftest import Recorder, sse, sse_response
from .generic_models import OPTIONS, Effect, Endpoint, File, Graph, Path, graph, secret

RULES = [
    "pred writes(Resource a, Resource b) <-> Effect::sources(e, a), Effect::target(e, b), "
    "Effect::op(e, EffectOp::write).",
    "pred exfiltration(File key, Endpoint net) <-> secretFile(key), writes+(key, net).",
    "pred removes(Effect e, File f) <-> Effect::target(e, f), Effect::op(e, EffectOp::remove).",
]


def is_secret(file: File):
    if file.path.anchor == "unknown":
        return "unknown"
    return file.path.anchor == "home" and file.path.segments[:1] == [".ssh"]


PREDICATES = [Predicate("secretFile", (File,), is_secret)]


def metamodel() -> Metamodel:
    options = dict(OPTIONS)
    options[Effect] = ClassOptions(
        exclude=["command", "trace"], exists=lambda e: "true" if e.definite else "unknown"
    )
    return Metamodel(
        Graph, options=options, unions={"Resource": [File, Endpoint]}, unsupported="skip"
    )


def uncertain_graph() -> Graph:
    g = graph()
    unknown = File(path=Path(anchor="unknown", segments=["x"]))
    g.effects += [
        Effect(
            op="write",
            target=g.resources[1].model_copy(),
            sources=[unknown],
            definite=True,
            index=2,
        ),
        Effect(op="remove", target=secret(), definite=False, index=3),
    ]
    return g


def testdeclared_predicates():
    rules = [*RULES, "error pred 'odd name'(File f) <-> !secretFile(f).", "pred writes(a, b)."]
    rules.append("pred 'it\\'s \\\\'(File f).")
    rules.append(r"pred '\u0061'(File f).")
    rules.append("pred a(File f).")
    assert declared_predicates(rules) == [
        "writes",
        "exfiltration",
        "removes",
        "odd name",
        "it's \\",
        "a",
    ]


def test_quoted_predicate_resolution_never_collides_with_serialized_names():
    from types import SimpleNamespace

    from pyfinery.jvm._matching.engine import problem_name

    # Metamodel declares 'a b' (key "'a b'"), query declares '\u0027a b\u0027' (key "'\'a b\''").
    declared = declared_predicates([r"pred '\u0027a b\u0027'(File f) <-> File(f), !File(f)."])
    assert declared == ["'a b'"]

    # JVM resolver: resolves '\u000a' to canonical '\n' and keeps declared '\'a b\'' separate.
    both = SimpleNamespace(
        queries={
            "'a b'": 1,
            r"'\'a b\''": 2,
            r"'\n'": 3,
            "builtin::exists": 4,
            "'builtin::exists'": 5,
            "atom": 6,
            "'he said \\\"yes\\\"'": 7,
            r"'ready\n'": 8,
            "'My Class'::'my field'": 9,
        },
        unsupported={},
    )
    assert problem_name(both, declared[0]) == r"'\'a b\''"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, r"'\u000a'") == r"'\n'"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, "'a b'") == "'a b'"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, "a b") == "'a b'"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, "builtin::exists") == "builtin::exists"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, "'builtin::exists'") == "'builtin::exists'"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, "'My Class'::'my field'") == "'My Class'::'my field'"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, declared_predicates(["pred 'builtin::exists'(x)."])[0]) == (
        "'builtin::exists'"
    )  # pyright: ignore[reportArgumentType]
    assert problem_name(both, declared_predicates(["pred atom(x)."])[0]) == "atom"  # pyright: ignore[reportArgumentType]
    assert problem_name(both, declared_predicates(["pred 'he said \"yes\"'(x)."])[0]) == (
        "'he said \\\"yes\\\"'"
    )  # pyright: ignore[reportArgumentType]
    assert problem_name(both, declared_predicates([r"pred 'ready\n'(x)."])[0]) == r"'ready\n'"  # pyright: ignore[reportArgumentType]

    only_first = SimpleNamespace(queries={"'a b'": 1}, unsupported={})
    with pytest.raises(ValueError, match="No predicate named"):
        problem_name(only_first, declared[0])  # pyright: ignore[reportArgumentType]
    # Explicit quoted selector still resolves "'a b'" -> 'a b' when no "'a b'" predicate exists.
    assert problem_name(only_first, "'a b'") == "'a b'"  # pyright: ignore[reportArgumentType]

    # HTTP read_matches: automatic selection matches '\'a b\'' (empty), not "'a b'" (true),
    # and qualified builtin::exists stays distinct from quoted 'builtin::exists'.
    model_both = {
        "concreteness": "partial",
        "nodes": [{"name": "keys", "simpleName": "keys", "kind": "default"}],
        "relations": [
            {
                "name": "'a b'",
                "simpleName": "'a b'",
                "arity": 1,
                "detail": {"type": "pred", "kind": "defined"},
                "parameterNames": ["f"],
            },
            {
                "name": r"'\'a b\''",
                "simpleName": r"'\'a b\''",
                "arity": 1,
                "detail": {"type": "pred", "kind": "defined"},
                "parameterNames": ["f"],
            },
            {
                "name": "builtin::exists",
                "simpleName": "exists",
                "arity": 1,
                "detail": {"type": "pred", "kind": "base"},
                "parameterNames": ["n"],
            },
            {
                "name": "'builtin::exists'",
                "simpleName": "'builtin::exists'",
                "arity": 1,
                "detail": {"type": "pred", "kind": "defined"},
                "parameterNames": ["f"],
            },
        ],
        "partialInterpretation": {
            "'a b'": [[0, "true"]],
            r"'\'a b\''": [],
            "builtin::exists": [[0, "true"]],
            "'builtin::exists'": [],
        },
    }

    def run_http(
        model_json: dict,
        rules: list[str] | None = None,
        select: list[str] | None = None,
    ) -> pf.QueryResult:
        body = sse({"result": "success", "value": {"issues": [], "json": model_json}})
        recorder = Recorder(lambda _: sse_response(body))
        client = pf.Refinery(
            "http://x", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
        )
        return metamodel().query(
            [uncertain_graph()],
            client,
            rules or [r"pred '\u0027a b\u0027'(File f) <-> File(f), !File(f)."],
            select=select,
        )

    assert run_http(model_both).matches == []
    assert [m.node_names for m in run_http(model_both, select=["builtin::exists"]).matches] == [
        ("keys",)
    ]
    assert (
        run_http(
            model_both, rules=["pred 'builtin::exists'(File f) <-> File(f), !File(f)."]
        ).matches
        == []
    )

    model_only_first = {
        **model_both,
        "relations": [model_both["relations"][0]],
        "partialInterpretation": {"'a b'": [[0, "true"]]},
    }
    with pytest.raises(ValueError, match="Refinery returned no relation named"):
        run_http(model_only_first)


def test_query_problem_closes_the_world():
    source = metamodel().query_problem([uncertain_graph()], RULES, predicates=PREDICATES)
    lines = source.splitlines()
    # Declarations, then no other objects, references, or predicate facts.
    assert "pred secretFile(File p0)." in lines
    assert "!exists(Effect::new)." in lines and "!exists(File::new)." in lines
    assert "!exists(Resource::new)." not in lines  # abstract: no new object
    assert "default !Effect::sources(*, *)." in lines
    assert "default !secretFile(*)." in lines
    # Uncertainty is kept.
    assert "?exists(effect_4)." in lines
    assert "secretFile('file:.ssh/id_rsa')." in lines
    assert "?secretFile('file:x')." in lines


def test_query_canonical_opposites_and_subclasses():
    from .graph_models import Dir, Filesystem
    from .graph_models import File as FsFile
    from .graph_models import metamodel as fs_metamodel

    home = Dir(name="home")
    doc = FsFile(parent=home)  # only the container side is set
    source = fs_metamodel().query_problem([Filesystem(root=home), doc])
    lines = source.splitlines()
    assert "Dir::contents(home, file_1)." in lines  # written on the containment side
    assert not any(line.startswith("FSObject::parent(") for line in lines)
    assert "default !FSObject::parent(*, *)." not in lines


def _containment_defaults(metamodel, objects):
    lines = metamodel.query_problem(objects).splitlines()
    containment = ["default !Dir::contents(*, *).", "default !Filesystem::root(*, *)."]
    return [line for line in containment if line in lines]


def test_query_closes_containment_only_for_orphans():
    from .graph_models import Dir, Filesystem
    from .graph_models import File as FsFile
    from .graph_models import metamodel as fs_metamodel

    def graph():
        doc = FsFile()
        return doc, Filesystem(root=Dir(name="home", contents=[doc]))

    doc, fs = graph()
    # Every object certainly has a container: Refinery rules out the others.
    assert _containment_defaults(fs_metamodel(), [fs]) == []
    # An object with no container could be in any directory, unless closed.
    assert _containment_defaults(fs_metamodel(), [fs, FsFile()]) == [
        "default !Dir::contents(*, *)."
    ]
    assert _containment_defaults(fs_metamodel(), [fs, Dir(name="tmp")]) == [
        "default !Dir::contents(*, *).",
        "default !Filesystem::root(*, *).",
    ]
    # A container that only may contain it, or only may exist, doesn't count.
    doc, fs = graph()
    uncertain = Metamodel(
        Filesystem,
        options={
            Dir: ClassOptions(
                logic_value=lambda d, field, value: "unknown" if value is doc else True
            )
        },
    )
    assert _containment_defaults(uncertain, [fs]) == ["default !Dir::contents(*, *)."]
    doc, fs = graph()
    ghost = Metamodel(Filesystem, options={Dir: ClassOptions(exists=lambda d: "unknown")})
    assert _containment_defaults(ghost, [fs]) == ["default !Dir::contents(*, *)."]


def _query_with_response(model_json: dict, **kwargs):
    body = sse({"result": "success", "value": {"issues": [], "json": model_json}})
    recorder = Recorder(lambda _: sse_response(body))
    client = pf.Refinery(
        "http://x", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )
    return metamodel().query(
        [uncertain_graph()], client, RULES, predicates=PREDICATES, **kwargs
    ), recorder


def _pred(name, arity, parameters=None):
    return {
        "name": name,
        "simpleName": name,
        "arity": arity,
        "parameterNames": parameters,
        "detail": {"type": "pred", "kind": "defined"},
    }


MODEL = {
    "concreteness": "partial",
    "nodes": [
        {"name": n, "simpleName": n, "kind": "default"}
        for n in ["File::new", "file_1", "file_2", "endpoint_1", "effect_4"]
    ],
    "relations": [
        _pred("writes", 2),
        _pred("exfiltration", 2, ["key", "net"]),
        _pred("removes", 2, ["e", "f"]),
    ],
    "partialInterpretation": {
        "writes": [[1, 3, "true"], [2, 3, "false"]],
        "exfiltration": [[1, 3, "true"], [2, 3, "unknown"], [0, 3, "true"], [2, 2, "error"]],
        "removes": [[4, 1, "true"]],
    },
}


def test_query_result_logic_values():
    result, recorder = _query_with_response(MODEL)
    assert recorder.requests[0].url.path == "/semantics"
    assert [str(m) for m in result.matches] == [
        "true: writes(file_1, endpoint_1)",
        "true: exfiltration(file_1, endpoint_1)",
        "unknown: exfiltration(file_2, endpoint_1)",  # File::new is skipped, error is not a match
        "unknown: removes(effect_4, file_1)",  # effect_4 only may exist
    ]
    assert [m.node_names for m in result.true("exfiltration")] == [("file_1", "endpoint_1")]
    assert [m.predicate for m in result.unknown()] == ["exfiltration", "removes"]
    assert isinstance(result["removes"][0].nodes[0], Effect)
    assert [str(e) for e in result.errors] == ["contradiction: exfiltration(file_2, file_2)"]


def test_query_matches_by_parameter_name():
    result, _ = _query_with_response(MODEL)
    exfiltration = result.true("exfiltration")[0]
    assert exfiltration.parameters == ("key", "net")
    assert exfiltration["key"] is exfiltration.nodes[0]
    assert list(exfiltration.arguments) == ["key", "net"]
    assert isinstance(result["removes"][0]["e"], Effect)
    with pytest.raises(KeyError, match="no parameter named 'secret'"):
        exfiltration["secret"]
    # Without parameter names from Refinery, there is nothing to look up by name.
    assert result["writes"][0].arguments == {}


def test_query_select():
    result, _ = _query_with_response(MODEL, select=["removes"])
    assert [m.predicate for m in result.matches] == ["removes"]
    with pytest.raises(ValueError, match="no relation named 'missing'"):
        _query_with_response(MODEL, select=["missing"])


def test_query_input_errors():
    mm = metamodel()
    with pytest.raises(ValueError, match="returned 'maybe'"):
        mm.query_problem([graph()], predicates=[Predicate("p", (File,), lambda f: "maybe")])
    with pytest.raises(ValueError, match="No union named 'Nope'"):
        mm.query_problem([graph()], predicates=[Predicate("p", ("Nope",), lambda f: True)])
    bad = Metamodel(
        Graph,
        options={
            **OPTIONS,
            Effect: ClassOptions(exclude=["command", "trace"], exists=lambda e: False),
        },
        unsupported="skip",
    )
    with pytest.raises(ValueError, match="must return 'true', 'unknown' or True"):
        bad.query_problem([graph()])


def test_union_parameter_predicate():
    lines = (
        metamodel()
        .query_problem([graph()], predicates=[Predicate("known", ("Resource",), lambda r: True)])
        .splitlines()
    )
    assert "pred known(Resource p0)." in lines
    assert {"known('file:.ssh/id_rsa').", "known('endpoint:https://example.com')."} <= set(lines)


def test_aquery():
    body = sse({"result": "success", "value": {"issues": [], "json": MODEL}})

    async def main():
        transport = httpx.MockTransport(lambda _: sse_response(body))
        async with pf.AsyncRefinery(
            "http://x", http_client=httpx.AsyncClient(transport=transport)
        ) as client:
            return await metamodel().aquery(
                [uncertain_graph()], client, RULES, predicates=PREDICATES
            )

    assert len(asyncio.run(main()).matches) == 4


def test_inputs_are_read_in_order():
    # Generators may depend on each other, e.g. predicates that load objects,
    # so the order they are read in, and the request is sent, is part of the API.
    order: list[str] = []

    def read(name, items):
        order.append(name)
        yield from items

    def respond(request):
        order.append("request")
        return sse_response(sse({"result": "success", "value": {"issues": [], "json": MODEL}}))

    client = pf.Refinery(
        "http://x", http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    mm = metamodel()

    def query(**kwargs):
        order.clear()
        mm.query(
            read("objects", [uncertain_graph()]),
            client,
            read("rules", RULES),
            predicates=read("predicates", PREDICATES),
            **kwargs,
        )
        return order

    assert query() == ["rules", "predicates", "objects", "request"]
    assert query(select=read("select", ["writes"])) == [
        *("rules", "predicates", "objects", "select", "request")
    ]
    order.clear()
    mm.query_problem(
        read("objects", [uncertain_graph()]),
        read("rules", RULES),
        predicates=read("predicates", PREDICATES),
    )
    assert order == ["rules", "predicates", "objects"]
    order.clear()
    mm.validate(
        read("objects", [uncertain_graph()]),
        client,
        read("rules", RULES),
        predicates=read("predicates", PREDICATES),
    )
    assert order == ["predicates", "objects", "rules", "request"]
