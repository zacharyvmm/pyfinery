# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

import pyfinery as pf
from pyfinery import ClassOptions, Metamodel

from .conftest import sse, sse_response
from .deployment_models import Database, Deployment, Service, deployment, metamodel


def test_reference_by_key_source():
    source = metamodel().source
    assert "abstract class Component." in source
    assert "    Component[0..*] depends_on\n" in source
    assert "class Service extends Component {" in source


def test_reference_by_key_facts_and_names():
    mm = metamodel()
    nodes = mm.collect([deployment()])
    names = mm.node_names(nodes)
    assert sorted(names.values()) == ["api", "cache", "db", "deployment_1", "port_1", "web"]
    facts = mm.facts(nodes, names).splitlines()
    assert "Service::depends_on(web, api)." in facts
    assert "Service::depends_on(web, cache)." in facts
    assert "Service::depends_on(api, db)." in facts
    assert not any("billing" in line for line in facts)  # unresolved: reported, not asserted


def test_reference_errors():
    d = deployment()
    d.components.append(Service(name="db", image="x", tier="backend"))  # clashes with the database
    mm = metamodel()
    nodes = mm.collect([d])
    errors = mm.reference_errors(nodes, mm.node_names(nodes))
    assert [str(e) for e in errors] == [
        "reference: Service::depends_on::ambiguous(api, 'db')",
        "reference: Service::depends_on::unresolved(api, 'billing')",
    ]
    assert isinstance(errors[0].nodes[0], Service) and errors[0].nodes[1] is None


def test_names_fall_back_when_keys_clash_with_types_or_each_other():
    d = deployment()
    d.components += [
        Service(name="Service", image="x", tier="backend"),  # a class name
        Service(name="exists", image="x", tier="backend"),  # a builtin
        Database(name="web", engine="redis", encrypted=True),  # another node's name
    ]
    mm = metamodel()
    names = mm.node_names(mm.collect([d]))
    assert sorted(names.values()) == [
        "api",
        "cache",
        "database_1",
        "db",
        "deployment_1",
        "port_1",
        "service_1",
        "service_2",
        "web",
    ]


def test_validate_reports_reference_errors():
    model = {"concreteness": "candidate", "nodes": [], "relations": [], "partialInterpretation": {}}
    body = sse({"result": "success", "value": {"issues": [], "json": model}})
    transport = httpx.MockTransport(lambda _: sse_response(body))
    client = pf.Refinery("http://x", http_client=httpx.Client(transport=transport))
    report = metamodel().validate([deployment()], client)
    assert [str(v) for v in report.violations] == [
        "reference: Service::depends_on::unresolved(api, 'billing')"
    ]
    api = next(c for c in deployment().components if c.name == "api")
    assert report.for_node(api) == report.violations  # same key, different instance


def test_reference_definition_errors():
    class Thing(BaseModel):
        name: str
        parent: str | None = None
        tags: dict[str, str] = {}

    with pytest.raises(TypeError, match="need ClassOptions\\(key=...\\) on Thing"):
        Metamodel(
            Thing,
            options={Thing: ClassOptions(references={"parent": Thing})},
            unsupported="skip",
        )
    with pytest.raises(TypeError, match="must hold str or int keys"):
        Metamodel(
            Thing,
            options={Thing: ClassOptions(key=lambda t: t.name, references={"tags": Thing})},
        )
    with pytest.raises(TypeError, match="no union named 'Nope'"):
        Metamodel(
            Thing,
            options={Thing: ClassOptions(key=lambda t: t.name, references={"parent": "Nope"})},
            unsupported="skip",
        )
    mm = Metamodel(
        Thing,
        options={Thing: ClassOptions(key=lambda t: t.name, references={"parent": Thing})},
        unsupported="skip",
    )
    assert "Thing[0..1] parent" in mm.source


def test_deployment_classes_are_plain_pydantic():
    # Nothing in the models knows about Refinery.
    assert not any(issubclass(c, pf.RefineryNode) for c in (Service, Database, Deployment))


def test_unencodable_keys_and_strings():
    """A lone surrogate, e.g. from bytes decoded with ``surrogateescape``, can't be sent as JSON."""
    d = deployment()
    d.components.append(Service(name="w\udcffb", image="i\udcff\u00e9", tier="backend"))
    mm = metamodel()
    nodes = mm.collect([d])
    names = mm.node_names(nodes)
    # Refinery would name the node "'w?b'", which couldn't be mapped back.
    assert "service_1" in names.values()
    source = mm.problem([d])
    source.encode("utf-8")
    assert 'Service::image(service_1): "i\\udcff\u00e9".' in source
