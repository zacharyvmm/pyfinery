# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Refinery's answers for the conformance cases, recorded so tests can run without it.

For each case, the recording holds what Refinery's ``semantics`` endpoint
returns (the partial model, with objects that don't exist and shadow
predicates kept), and for cases with ``EXPECT CANDIDATE`` what ``concretize``
returns. Re-record after changing a ``.problem`` file::

    uv run python -m tests.conformance.recording            # starts IMAGE in Docker
    REFINERY_API_BASE=http://localhost:8888/api/v1 uv run python -m tests.conformance.recording

``tests/conformance/test_conformance.py`` checks that every recording is of the
current file, and with ``PYFINERY_DOCKER_TESTS=1``, that Refinery still answers
the same.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyfinery import Refinery, RefineryError, RefineryServer
from pyfinery.models import JsonOutput, JsonOutputFormat, OutputFormats, SourceOutputFormat

from .problems import Case, load

ROOT = Path(__file__).parent
SUITES = ("upstream", "queries", "generated")
RECORDED = ROOT / "recorded"

# The image the recordings are of: pyfinery's default image, pinned.
IMAGE = (
    "ghcr.io/zacharyvmm/refinery@sha256:"
    "783dffb3baaea555344b617718745f10f7b164224a53d028054bf16186947400"
)

FORMAT = OutputFormats(
    json_=JsonOutputFormat(non_existing_objects="keep", shadow_predicates="keep"),
    source=SourceOutputFormat(enabled=False),
)


def problem_files() -> list[Path]:
    return sorted(p for suite in SUITES for p in (ROOT / suite).rglob("*.problem"))


def recording_path(problem: Path) -> Path:
    return RECORDED / problem.relative_to(ROOT).with_suffix(".json")


def digest(problem: Path) -> str:
    return hashlib.sha256(problem.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Outcome:
    """What Refinery answered for a case."""

    partial: JsonOutput | None
    candidate: JsonOutput | None
    issues: list[dict[str, Any]]
    error: str | None
    """The type and message of the error Refinery answered with, if it did."""

    def to_json(self) -> dict[str, Any]:
        return {
            "partial": _dump(self.partial),
            "candidate": _dump(self.candidate),
            "issues": self.issues,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Outcome:
        return cls(_load(data["partial"]), _load(data["candidate"]), data["issues"], data["error"])


def _dump(model: JsonOutput | None) -> Any:
    return None if model is None else model.model_dump(mode="json", by_alias=True)


def _load(data: Any) -> JsonOutput | None:
    return None if data is None else JsonOutput.model_validate(data)


def ask(refinery: Refinery, case: Case) -> Outcome:
    """Refinery's answer for ``case``."""
    try:
        semantics = refinery.semantics(case.source, format=FORMAT)
        candidate = None
        if case.needs_candidate:
            candidate = refinery.concretize(case.source, format=FORMAT)
    except RefineryError as e:
        return Outcome(None, None, [], f"{type(e).__name__}: {e}")
    issues = [issue.model_dump(mode="json", by_alias=True) for issue in semantics.issues]
    return Outcome(semantics.json_, candidate.json_ if candidate else None, issues, None)


def load_recording(problem: Path) -> tuple[str, list[Outcome]]:
    """The digest of the file recorded, and an outcome per case."""
    data = json.loads(recording_path(problem).read_text(encoding="utf-8"))
    return data["sha256"], [Outcome.from_json(case) for case in data["cases"]]


def record(refinery: Refinery, problem: Path) -> None:
    cases = load(problem)
    data = {
        "file": problem.relative_to(ROOT).as_posix(),
        "sha256": digest(problem),
        "image": IMAGE,
        "cases": [{"name": case.name, **ask(refinery, case).to_json()} for case in cases],
    }
    path = recording_path(problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def main(argv: list[str]) -> None:
    problems = [Path(arg).resolve() for arg in argv] or problem_files()
    base_url = os.environ.get("REFINERY_API_BASE")
    if base_url:
        with Refinery(base_url, timeout=None) as refinery:
            for problem in problems:
                record(refinery, problem)
                print(f"recorded {problem.relative_to(ROOT)}")
        return
    with (
        RefineryServer(image=IMAGE, semantics_timeout=60_000) as server,
        server.client(timeout=None) as refinery,
    ):
        for problem in problems:
            record(refinery, problem)
            print(f"recorded {problem.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
