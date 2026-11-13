"""End-to-end tests that start Refinery's JVM, each in a subprocess (a process has one JVM,
started once).

Opt in with ``PYFINERY_JVM_TESTS=1``, with a JDK 25 in ``PYFINERY_JAVA_HOME`` or ``JAVA_HOME``. The
first run fetches Refinery's jars (about 31 MB, and 26 MB more for generation) into the user cache
directory, or ``PYFINERY_CACHE_DIR``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.jvm,
    pytest.mark.skipif(
        os.environ.get("PYFINERY_JVM_TESTS") != "1", reason="set PYFINERY_JVM_TESTS=1"
    ),
]


def run(script: str) -> dict:
    """Run ``script`` in a fresh Python process; it prints one JSON object as its last line."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_starts_for_validation_and_moves_tuples_in_bulk():
    out = run(
        """
        import json
        import jpype
        import pyfinery.jvm as jvm

        status = jvm.start()
        Symbol = jpype.JClass("tools.refinery.store.representation.Symbol")
        ModelStore = jpype.JClass("tools.refinery.store.model.ModelStore")
        Tuple = jpype.JClass("tools.refinery.store.tuple.Tuple")
        # Tuple.of(int...) is the one overload that is never ambiguous
        ids = jpype.JArray(jpype.JInt)
        edge = Symbol.of("edge", 2)
        store = ModelStore.builder().symbols(edge).build()
        model = store.createEmptyModel()
        edges = model.getInterpretation(edge)
        Bulk = jvm.helpers().Bulk
        Bulk.put(edges, jpype.JArray(jpype.JInt)([1, 2, 2, 3, 3, 1]), 2, True)
        Bulk.put(edges, jpype.JArray(jpype.JInt)([2, 3]), 2, False)
        try:
            jvm.require("generation")
            refused = False
        except jvm.JvmError:
            refused = True
        print(json.dumps({
            "features": list(status.settings.features),
            "java": status.jdk.version,
            "edges": int(edges.getSize()),
            "has_1_2": bool(edges.get(Tuple.of(ids([1, 2])))),
            "has_2_3": bool(edges.get(Tuple.of(ids([2, 3])))),
            "generation_refused": refused,
            "seconds": status.seconds,
        }))
        """
    )
    assert out["features"] == ["validation"]
    assert out["java"] >= 25
    assert out["edges"] == 2
    assert out["has_1_2"] and not out["has_2_3"]
    assert out["generation_refused"]


def test_generates_with_ortools_loaded_from_the_cache():
    script = """
        import json
        import jpype
        import pyfinery.jvm as jvm

        jvm.configure(generation=True)
        status = jvm.start()
        StandaloneRefinery = jpype.JClass("tools.refinery.generator.standalone.StandaloneRefinery")
        ArrayList = jpype.JClass("java.util.ArrayList")
        injector = StandaloneRefinery.getInjector()
        loader = injector.getInstance(jpype.JClass("tools.refinery.generator.ProblemLoader"))
        problem = loader.loadString("class Person { Person[0..*] friend }")
        problem = loader.loadScopeConstraints(problem, ArrayList(["Person = 3"]), ArrayList())
        factory_class = jpype.JClass("tools.refinery.generator.ModelGeneratorFactory")
        factory = injector.getInstance(factory_class)
        with factory.createGenerator(problem) as generator:
            generator.setRandomSeed(1)
            result = str(generator.tryGenerate())
            person = generator.getProblemTrace().getPartialRelation("Person")
            cursor = generator.getPartialInterpretation(person).getAll()
            people = 0
            while cursor.move():
                people += 1
        print(json.dumps({"features": list(status.settings.features), "result": result,
                          "people": people, "seconds": status.seconds}))
        """
    run(
        script
    )  # the first run may fetch, unpack, and have the system scan the new native libraries
    out = run(script)
    assert out["features"] == ["validation", "generation"]
    assert out["result"] == "SUCCESS"
    assert out["people"] == 3
    # Loaded from the unpacked cache, not unpacked again by OR-tools' own loader (about 14 s on
    # macOS each start).
    assert out["seconds"]["natives"] < 2.0
