"""A graph validated as it changes, in one live Refinery model.

A :class:`Session` keeps the graph reachable from some objects in a plain
Refinery model whose rules Refinery's incremental (Rete) matcher keeps up to
date. Each :meth:`Session.validate` writes into the model only what changed
since the last one, so the matcher updates what those changes affect rather
than matching everything again::

    from pyfinery.jvm import Session

    with Session(metamodel, rules=RULES) as session:
        session.add(graph)
        report = session.validate()
        with session.attempt():          # undone on leaving the block
            session.add(candidate)
            if session.changes().added:  # violations the candidate brings
                ...

The objects are read when the session validates: change them (or add and
remove objects) between calls, and the next call sees the change.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..._instances import build_graph
from ..._options import Predicate
from ..._results import ValidationReport, Violation, reference_errors
from .cache import compiled
from .engine import require_supported
from .model import Key, PlainModel

if TYPE_CHECKING:
    from ...graph import Metamodel

__all__ = ["Changes", "Checkpoint", "Session"]

type ViolationKey = tuple[str, str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class Changes:
    """How the violations changed since the session last reported them."""

    added: list[Violation] = field(default_factory=list)
    removed: list[Violation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A state of a :class:`Session` to :meth:`~Session.rollback` to."""

    version: Any
    roots: dict[int, BaseModel]
    predicates: tuple[Predicate, ...]
    facts: frozenset[Key]
    known: dict[Any, int]
    next_id: int
    reported: dict[ViolationKey, Violation]


class Session:
    """The graph reachable from some objects, validated incrementally (see the module).

    ``rules`` are added to the metamodel's, and ``predicates`` give relations
    computed in Python, as in :meth:`~pyfinery.Metamodel.validate`; validation is
    two-valued in the same way. Raises :class:`~pyfinery.jvm.UnsupportedRuleError`
    if an error predicate uses what the direct path can't match yet.
    """

    def __init__(
        self,
        metamodel: Metamodel,
        objects: Iterable[BaseModel] = (),
        rules: Iterable[str] = (),
        *,
        predicates: Iterable[Predicate] = (),
    ) -> None:
        self.metamodel = metamodel
        self._predicates = tuple(predicates)
        self.problem = compiled(metamodel.schema, [*metamodel.rules, *rules], self._predicates)
        require_supported(self.problem, self.problem.rule_errors)
        self._model = PlainModel(self.problem)
        self._roots: dict[int, BaseModel] = {}
        self._reported: dict[ViolationKey, Violation] = {}
        self._closed = False
        self.add(*objects)

    # -- the graph ---------------------------------------------------------------------------------

    @property
    def objects(self) -> list[BaseModel]:
        """The objects the graph is reached from, in the order they were added."""
        return list(self._roots.values())

    def add(self, *objects: BaseModel) -> None:
        """Add the graph reachable from ``objects``."""
        for obj in objects:
            self._roots.setdefault(id(obj), obj)

    def remove(self, *objects: BaseModel) -> None:
        """Stop reaching the graph from ``objects``; what other objects reach stays."""
        for obj in objects:
            self._roots.pop(id(obj), None)

    @property
    def predicates(self) -> tuple[Predicate, ...]:
        return self._predicates

    def set_predicates(self, predicates: Iterable[Predicate]) -> None:
        """Replace the :class:`~pyfinery.Predicate` objects by others of the same names and
        parameters, e.g. with new tuples."""
        new = tuple(predicates)
        if [(p.name, tuple(p.parameters)) for p in new] != [
            (p.name, tuple(p.parameters)) for p in self._predicates
        ]:
            raise ValueError(
                "set_predicates() takes predicates of the same names and parameters as the "
                "session's; start another session to declare others"
            )
        self._predicates = new

    # -- validating --------------------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """The constraints the graph violates now, as :meth:`~pyfinery.Metamodel.validate`
        reports them; also what :meth:`changes` compares with next."""
        violations = self._current()
        return ValidationReport(
            violations, self.problem.source, None, self.metamodel.schema.identity
        )

    def changes(self) -> Changes:
        """The violations added and removed since the session last reported (by
        :meth:`validate`, :meth:`changes` or :meth:`rollback`)."""
        before = self._reported
        self._current()
        after = self._reported
        return Changes(
            [v for key, v in after.items() if key not in before],
            [v for key, v in before.items() if key not in after],
        )

    def _current(self) -> list[Violation]:
        self._check_open()
        graph = build_graph(self.metamodel.schema, self._roots.values())
        self._model.sync(graph, self._predicates)
        found = self._model.violations()
        keys = {}
        for violation in found:
            ids = tuple(
                self._model.known.get(graph.identity_of(n)) if n is not None else name
                for n, name in zip(violation.nodes, violation.node_names, strict=True)
            )
            keys[(violation.kind, violation.constraint, ids)] = violation
        self._reported = keys
        return reference_errors(graph) + found

    # -- undoing -----------------------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        """The session's state, to :meth:`rollback` to. Objects changed in place since are not
        changed back: only what the session holds is."""
        self._check_open()
        model = self._model
        return Checkpoint(
            model.commit(),
            dict(self._roots),
            self._predicates,
            frozenset(model.facts),
            dict(model.known),
            model.next_id,
            dict(self._reported),
        )

    def rollback(self, checkpoint: Checkpoint) -> None:
        """Go back to ``checkpoint``: the objects, predicates, model and reported violations."""
        self._check_open()
        model = self._model
        model.restore(checkpoint.version)
        model.facts = set(checkpoint.facts)
        model.known = dict(checkpoint.known)
        model.next_id = checkpoint.next_id
        self._roots = dict(checkpoint.roots)
        self._predicates = checkpoint.predicates
        self._reported = dict(checkpoint.reported)

    @contextmanager
    def attempt(self) -> Iterator[Session]:
        """Undo what is done in the block on leaving it, as :meth:`rollback`."""
        checkpoint = self.checkpoint()
        try:
            yield self
        finally:
            self.rollback(checkpoint)

    # -- lifetime ----------------------------------------------------------------------------------

    def close(self) -> None:
        """Free the model; the session can't be used afterwards."""
        if not self._closed:
            self._closed = True
            self._model.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("This session is closed")
