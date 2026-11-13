"""Sessions (:class:`pyfinery.jvm.Session`): a graph validated as it changes, in one live model.

Opt in with ``PYFINERY_JVM_TESTS=1`` (see ``test_jvm_engine``); each test runs in a subprocess.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from . import test_jvm_engine

pytestmark = [
    pytest.mark.jvm,
    pytest.mark.skipif(
        os.environ.get("PYFINERY_JVM_TESTS") != "1", reason="set PYFINERY_JVM_TESTS=1"
    ),
]

HELPERS = """
from pyfinery import Predicate
from pyfinery.jvm import Changes, Session


def names(violations):
    return sorted(str(v) for v in violations)


def changed(changes):
    return {"added": names(changes.added), "removed": names(changes.removed),
            "any": bool(changes)}
"""


def run(script: str) -> dict:
    """Run ``script`` after the engine tests' prelude and :data:`HELPERS`."""
    return test_jvm_engine.run(HELPERS + textwrap.dedent(script))


def test_validates_what_changed():
    out = run(
        """
        root = Dir(name="r")
        fs = Filesystem(root=root)
        out = {}
        with Session(mm) as session:
            session.add(fs)
            out["first"] = names(session.validate().violations)
            out["nothing"] = changed(session.changes())

            root.contents.append(File())  # changed in place: seen by the next call
            out["filled"] = changed(session.changes())

            orphan = Dir(name="orphan")
            session.add(orphan)
            session.add(orphan)  # once only
            out["objects"] = len(session.objects)
            out["orphan"] = changed(session.changes())
            out["validate"] = names(session.validate().violations)

            session.remove(orphan)
            out["removed"] = changed(session.changes())
            session.remove(orphan)  # not there: nothing happens
            out["objects_after"] = len(session.objects)

            root.contents.clear()
            out["emptied"] = changed(session.changes())
            out["same_as_engine"] = names(session.validate().violations) == names(
                mm.validate([fs], engine).violations)
        print(json.dumps(out))
        """
    )
    assert out["first"] == ["rule: emptyDir(r)"]
    assert out["nothing"] == {"added": [], "removed": [], "any": False}
    assert out["filled"] == {"added": [], "removed": ["rule: emptyDir(r)"], "any": True}
    assert out["objects"] == 2
    assert out["orphan"]["added"] == [
        "containment: builtin::invalidContainer(orphan)",
        "rule: emptyDir(orphan)",
    ]
    assert out["orphan"]["removed"] == []
    assert out["validate"] == out["orphan"]["added"]
    assert out["removed"] == {"added": [], "removed": out["orphan"]["added"], "any": True}
    assert out["objects_after"] == 1
    assert out["emptied"] == {"added": ["rule: emptyDir(r)"], "removed": [], "any": True}
    assert out["same_as_engine"]


def test_attempt_checkpoint_and_rollback():
    out = run(
        """
        root = Dir(name="r", contents=[File()])
        out = {}
        session = Session(mm, [Filesystem(root=root)])
        out["start"] = names(session.validate().violations)

        with session.attempt() as same:
            out["same"] = same is session
            same.add(Dir(name="extra"))
            out["inside"] = changed(session.changes())
            out["inside_objects"] = len(session.objects)
        out["after_objects"] = len(session.objects)
        out["after"] = names(session.validate().violations)
        out["after_changes"] = changed(session.changes())

        # Rolled back on an error too.
        try:
            with session.attempt():
                session.add(Dir(name="doomed"))
                session.validate()
                raise KeyError("stop")
        except KeyError:
            pass
        out["after_error"] = names(session.validate().violations)

        checkpoint = session.checkpoint()
        orphan = Dir(name="orphan")
        session.add(orphan)
        out["added"] = changed(session.changes())
        session.rollback(checkpoint)
        out["rolled_back"] = changed(session.changes())
        out["rolled_back_objects"] = len(session.objects)
        # The same checkpoint, twice; and the session still works after it.
        session.add(orphan)
        session.validate()
        session.rollback(checkpoint)
        out["twice"] = names(session.validate().violations)
        session.add(orphan)
        out["again"] = changed(session.changes())
        session.close()
        print(json.dumps(out))
        """
    )
    assert out["start"] == []
    assert out["same"]
    assert out["inside"] == {
        "added": ["containment: builtin::invalidContainer(extra)", "rule: emptyDir(extra)"],
        "removed": [],
        "any": True,
    }
    assert out["inside_objects"] == 2
    assert out["after_objects"] == 1
    assert out["after"] == []
    assert out["after_changes"]["any"] is False
    assert out["after_error"] == []
    assert out["added"]["added"] == [
        "containment: builtin::invalidContainer(orphan)",
        "rule: emptyDir(orphan)",
    ]
    assert out["rolled_back"]["any"] is False
    assert out["rolled_back_objects"] == 1
    assert out["twice"] == []
    assert out["again"]["added"] == out["added"]["added"]


def test_predicates_lifetime_and_unsupported_rules():
    out = run(
        """
        root = Dir(name="r", contents=[File()])
        fs = Filesystem(root=root)
        rules = ["error pred m(Dir d) <-> marked(d)."]
        out = {}
        session = Session(mm, [fs], rules, predicates=[Predicate("marked", (Dir,), tuples=[])])
        out["unmarked"] = names(session.validate().violations)
        session.set_predicates([Predicate("marked", (Dir,), tuples=[(root,)])])
        out["marked"] = changed(session.changes())
        session.set_predicates([Predicate("marked", (Dir,), evaluate=lambda d: False)])
        out["unmarked_again"] = changed(session.changes())
        out["other_name"] = raises(ValueError, lambda: session.set_predicates(
            [Predicate("other", (Dir,), tuples=[])]))
        out["other_parameters"] = raises(ValueError, lambda: session.set_predicates(
            [Predicate("marked", (File,), tuples=[])]))
        out["none"] = raises(ValueError, lambda: session.set_predicates([]))
        out["predicates"] = [p.name for p in session.predicates]

        session.close()
        session.close()  # twice is fine
        out["closed"] = [
            raises(RuntimeError, call) for call in (
                session.validate, session.changes, session.checkpoint)
        ]

        with Session(mm, [fs]) as managed:
            managed.validate()
        out["managed"] = raises(RuntimeError, managed.validate)

        out["unsupported"] = raises(UnsupportedRuleError, lambda: Session(
            mm, rules=["error pred big(File f) <-> File::size(f) > 3."]))
        # Unsupported, but no error predicate: fine.
        with Session(mm, [fs], ["pred big(File f) <-> File::size(f) > 3."]) as fine:
            out["fine"] = names(fine.validate().violations)
        print(json.dumps(out))
        """
    )
    assert out["unmarked"] == []
    assert out["marked"] == {"added": ["rule: m(r)"], "removed": [], "any": True}
    assert out["unmarked_again"] == {"added": [], "removed": ["rule: m(r)"], "any": True}
    assert "same names and parameters" in out["other_name"]
    assert out["other_parameters"] is not None
    assert out["none"] is not None
    assert out["predicates"] == ["marked"]
    assert out["closed"] == ["This session is closed"] * 3
    assert out["managed"] == "This session is closed"
    assert "big: uses" in out["unsupported"]
    assert out["fine"] == []


def test_negation_follows_what_is_added_and_removed():
    out = run(
        """
        root = Dir(name="w")
        fs = Filesystem(root=root)
        out = {}
        with Session(mm, [fs], ["error pred noLink(Dir d) <-> Dir(d), !Link(_)."]) as session:
            out["none"] = names(session.validate().violations)
            link = Link(target=root)
            root.contents.append(link)
            link.parent = root
            out["with"] = names(session.validate().violations)
            root.contents.remove(link)
            out["again"] = names(session.validate().violations)
        out = {key: [v for v in found if "noLink" in v] for key, found in out.items()}
        print(json.dumps(out))
        """
    )
    assert out["none"] == ["rule: noLink(w)"]
    assert out["with"] == []
    assert out["again"] == ["rule: noLink(w)"]
