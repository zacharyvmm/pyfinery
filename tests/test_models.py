# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import pytest
from pydantic import ValidationError

import pyfinery as pf

from .test_client import MODEL_JSON


@pytest.mark.parametrize(
    ("text", "incremental", "lower", "upper"),
    [
        ("node=10..20", False, 10, 20),
        ("File = 5", False, 5, 5),
        ("Dir=2..*", False, 2, None),
        ("Dir += 1..3", True, 1, 3),
        ("fs::Dir=0..*", False, 0, None),
    ],
)
def test_scope_parse(text, incremental, lower, upper):
    scope = pf.Scope.parse(text)
    assert (scope.incremental, scope.lower_bound, scope.upper_bound) == (incremental, lower, upper)


@pytest.mark.parametrize("text", ["node", "node=", "node=5..2", "node=-1..2", "node=a..b"])
def test_scope_parse_rejects_invalid(text):
    with pytest.raises(ValueError):
        pf.Scope.parse(text)


def test_scope_bounds_validated():
    with pytest.raises(ValidationError):
        pf.Scope(relation="A", lower_bound=3, upper_bound=1)


def test_json_output_helpers():
    model = pf.JsonOutput.model_validate(MODEL_JSON)
    assert model.relation("contents").name == "Dir::contents"
    assert model.tuples("Dir") == [("dir1", "true"), ("file1", "false")]
    assert model.true_tuples("contents") == [("dir1", "file1")]
    assert model.tuples("Dir::contents")[1][-1] == pf.models.UnknownValue(unknown="?")
    with pytest.raises(KeyError):
        model.relation("missing")


def test_json_output_tuple_counts():
    interpretation = {**MODEL_JSON["partialInterpretation"], "File": [[1, "true"]]}
    interpretation["Dir::contents"] = [*interpretation["Dir::contents"], [0, 0, "false"]]
    model = pf.JsonOutput.model_validate({**MODEL_JSON, "partialInterpretation": interpretation})
    assert list(model.tuple_counts().items()) == [("Dir::contents", 3), ("Dir", 2), ("File", 1)]


def test_output_formats_accept_python_names():
    formats = pf.OutputFormats(
        json_=pf.JsonOutputFormat(non_existing_objects="keep"), source={"enabled": False}
    )
    assert formats.model_dump(mode="json") == {
        "json": {
            "enabled": True,
            "nonExistingObjects": "keep",
            "shadowPredicates": "discard",
            "relations": None,
            "excludedRelations": None,
        },
        "source": {"enabled": False},
    }
    formats = pf.OutputFormats(json_=pf.JsonOutputFormat(excluded_relations=["helper"]))
    assert formats.model_dump(mode="json", exclude_none=True)["json"] == {
        "enabled": True,
        "nonExistingObjects": "discard",
        "shadowPredicates": "discard",
        "excludedRelations": ["helper"],
    }
