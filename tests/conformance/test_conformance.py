# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Refinery's answers to the conformance cases meet their expectations.

Without Docker, the expectations are checked against the recordings (see
``recording``). With ``PYFINERY_DOCKER_TESTS=1``, the recordings are checked
against the server of ``recording.IMAGE``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from pyfinery import RefineryServer

from .interpretation import Interpretation, check, errors
from .problems import Case, Expectation, load
from .recording import IMAGE, ROOT, Outcome, ask, digest, load_recording, problem_files

FILES = problem_files()
CASES = [(path, case) for path in FILES for case in load(path)]


def _id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _outcome(path: Path, case: Case) -> Outcome:
    return load_recording(path)[1][case.index]


def _model(outcome: Outcome, expectation: Expectation) -> Interpretation:
    model = outcome.candidate if expectation.concreteness == "candidate" else outcome.partial
    assert model is not None, "the recording has no model for this expectation"
    return Interpretation(model)


def failures(case: Case, outcome: Outcome) -> list[str]:
    """Where Refinery's answer for ``case`` breaks what the case expects."""
    if case.kind in ("propagation failure", "concretization failure"):
        return [f"{case.kind} isn't checked: the HTTP API doesn't say which step failed"]
    if outcome.error is not None:
        return [f"Refinery answered with {outcome.error}"]
    found = []
    if case.kind == "no errors":
        assert outcome.partial is not None
        found += [f"issue: {issue['description']}" for issue in outcome.issues]
        found += [f"inconsistent: {error}" for error in errors(Interpretation(outcome.partial))]
    for expectation in case.expectations:
        found += check(expectation, _model(outcome, expectation))
    return found


def test_cases_are_found():
    assert len(FILES) >= 30
    assert len(CASES) >= 90


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_recording_is_of_the_current_file(path):
    recorded_digest, outcomes = load_recording(path)
    assert recorded_digest == digest(path), "re-record: python -m tests.conformance.recording"
    assert len(outcomes) == len(load(path))


@pytest.mark.parametrize(("path", "case"), CASES, ids=[f"{_id(p)}::{c.label}" for p, c in CASES])
def test_refinery_meets_expectations(path, case):
    assert failures(case, _outcome(path, case)) == []


@pytest.mark.parametrize(("path", "case"), CASES, ids=[f"{_id(p)}::{c.label}" for p, c in CASES])
def test_checks_fail_for_other_values(path, case):
    """Each truth-valued assertion fails when it expects any other value, so a check
    that passes can't be one that reads the wrong tuple, or nothing."""
    outcome = _outcome(path, case)
    for expectation in case.expectations:
        model = _model(outcome, expectation)
        for assertion in expectation.assertions:
            if assertion.value not in ("true", "false", "unknown") or None in assertion.arguments:
                continue
            exact = dataclasses.replace(expectation, exact=True, assertions=(assertion,))
            if check(exact, model):
                continue  # the actual value only refines the expected one
            for other in {"true", "false", "unknown", "error"} - {assertion.value}:
                mutated = dataclasses.replace(assertion, value=other)
                assert check(dataclasses.replace(exact, assertions=(mutated,)), model), (
                    f"{assertion.source} also passed as {other}"
                )


# -- against a live server ------------------------------------------------------


@pytest.fixture(scope="module")
def refinery():
    with (
        RefineryServer(image=IMAGE, semantics_timeout=60_000) as server,
        server.client(timeout=None) as client,
    ):
        yield client


def _normalized(outcome: Outcome) -> dict:
    data = outcome.to_json()
    for key in ("partial", "candidate"):
        if data[key] is not None:
            interpretation = data[key]["partialInterpretation"]
            data[key]["partialInterpretation"] = {
                name: sorted(tuples, key=repr) for name, tuples in interpretation.items()
            }
    return data


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("PYFINERY_DOCKER_TESTS") != "1", reason="set PYFINERY_DOCKER_TESTS=1"
)
@pytest.mark.parametrize(("path", "case"), CASES, ids=[f"{_id(p)}::{c.label}" for p, c in CASES])
def test_refinery_answers_as_recorded(refinery, path, case):
    assert _normalized(ask(refinery, case)) == _normalized(_outcome(path, case))
