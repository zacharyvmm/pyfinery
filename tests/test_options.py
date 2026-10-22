# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""validate() and query() share rules, Python predicates, and logic-value options."""

from __future__ import annotations

import re

import httpx
import pytest

import pyfinery as pf
from pyfinery import ClassOptions, Metamodel, Predicate

from .conftest import Recorder, sse, sse_response
from .deployment_models import RULES, Database, Deployment, Service, deployment

FLOATING = Predicate(
    "floatingTag", (Service,), lambda s: re.search(r":(latest)?$", s.image) is not None
)


def metamodel(logic_value=None) -> Metamodel:
    return Metamodel(
        Deployment,
        options={
            Service: ClassOptions(
                key=lambda s: s.name,
                references={"depends_on": "Component"},
                logic_value=logic_value,
            ),
            Database: ClassOptions(key=lambda d: d.name),
        },
        unions={"Component": [Service, Database]},
        rules=RULES,
    )


def components():
    d = deployment()
    web, api, db, cache = d.components
    return d, web, api, db, cache


def test_predicate_needs_evaluate_or_tuples():
    with pytest.raises(ValueError, match="exactly one of evaluate or tuples"):
        Predicate("p", (Service,))
    with pytest.raises(ValueError, match="exactly one of evaluate or tuples"):
        Predicate("p", (Service,), lambda s: True, tuples=[])


def test_predicate_tuples_forms():
    d, web, api, db, cache = components()
    unary = Predicate("u", (Service,), tuples=(s for s in [web, (api,), ((web,), "unknown")]))
    never = Predicate("v", (Service,), tuples=[((web,), "false")])
    binary = Predicate(
        "b",
        (Service, "Component"),
        tuples=[(web, api), ((api, db), "unknown"), ((web, cache), True)],
    )
    lines = metamodel().query_problem([d], predicates=[unary, binary, never]).splitlines()
    assert "pred u(Service p0)." in lines and "pred b(Service p0, Component p1)." in lines
    assert {
        "u(web).",
        "u(api).",
        "?u(web).",
        "b(web, api).",
        "?b(api, db).",
        "b(web, cache).",
    } <= set(lines)
    assert "v(web)." not in lines and "?v(web)." not in lines  # "false" is not a fact


def test_predicate_tuple_errors():
    d, web, api, db, cache = components()
    mm = metamodel()
    with pytest.raises(ValueError, match="takes 2 argument"):
        mm.query_problem([d], predicates=[Predicate("b", (Service, Service), tuples=[(web,)])])
    with pytest.raises(ValueError, match="is not a node of type Service"):
        mm.query_problem([d], predicates=[Predicate("u", (Service,), tuples=[db])])
    stranger = Service(name="stranger", image="x", tier="backend")
    with pytest.raises(ValueError, match="in this graph"):
        mm.query_problem([d], predicates=[Predicate("u", (Service,), tuples=[stranger])])


def test_logic_value_option():
    def heuristic(service, field, value):
        if field == "depends_on" and value == "db":
            return "unknown"
        if field == "replicas":
            return "unknown"
        return True

    d = deployment()
    d.components[1].replicas = 3
    lines = metamodel(heuristic).query_problem([d]).splitlines()
    assert "?Service::depends_on(api, db)." in lines
    assert "Service::depends_on(web, api)." in lines
    assert not any(line.startswith("Service::replicas(api)") for line in lines)  # unknown

    with pytest.raises(ValueError, match="must return 'true', 'unknown' or True"):
        metamodel(lambda s, f, v: False).query_problem([deployment()])


def _mock(model_json):
    body = sse({"result": "success", "value": {"issues": [], "json": model_json}})
    recorder = Recorder(lambda _: sse_response(body))
    return pf.Refinery(
        "http://x", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    ), recorder


EMPTY = {"concreteness": "candidate", "nodes": [], "relations": [], "partialInterpretation": {}}


def test_validate_accepts_rules_and_predicates():
    client, recorder = _mock(EMPTY)
    metamodel().validate(
        [deployment()],
        client,
        ["error pred unpinned(Service s) <-> floatingTag(s)."],
        predicates=[FLOATING],
    )
    source = recorder.last_json["input"]["source"]
    assert "error pred unpinned(Service s) <-> floatingTag(s)." in source
    assert "pred floatingTag(Service p0)." in source
    assert "floatingTag(api)." in source.splitlines()
    assert "floatingTag(web)." not in source.splitlines()


def test_query_errors_skip_objects_that_do_not_exist():
    model = {
        "concreteness": "partial",
        "nodes": [{"name": n, "simpleName": n, "kind": "default"} for n in ["Service::new", "web"]],
        "relations": [
            {
                "name": "Service::tier::invalidMultiplicity",
                "simpleName": "invalidMultiplicity",
                "arity": 1,
                "detail": {"type": "pred", "kind": "error"},
            },
        ],
        "partialInterpretation": {
            "Service::tier::invalidMultiplicity": [[0, "error"], [1, "error"]]
        },
    }
    client, _ = _mock(model)
    result = metamodel().query([deployment()], client)
    assert [str(e) for e in result.errors] == [
        "reference: Service::depends_on::unresolved(api, 'billing')",
        "multiplicity: Service::tier::invalidMultiplicity(web)",
    ]
