# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Evaluating rules on a batch of graphs at once: each graph gets what evaluating the
rules on it alone gives it, which is the reference."""

from __future__ import annotations

import random

import numpy as np
import pytest

from pyfinery import ClassOptions, Metamodel
from pyfinery.bits import (
    Attribute,
    BitsEngine,
    Node,
    PackedBatch,
    PackedModel,
    PartialModel,
    Relation,
    RuleError,
    UnsupportedRuleError,
    evaluate,
    evaluate_batch,
)
from pyfinery.bits.matrices import kernels as k
from pyfinery.bits.matrices.batch import group_by_size

from .test_bits_metamodel import FAMILIES, Shape, Shapes, Square
from .test_bits_packed import RULES, graph

MORE_RULES = """
pred named(p) <-> Person(p), Person::name(p) == "a".
pred unnamed(p) <-> Person(p), Person::name(p) != "b".
pred trusted(p, q).
pred maybeTrusted(p, q) <-> trusted(p, q), Person(p).
pred anyPair(p, q) <-> trusted(p, q).
pred unary(p).
pred anyNode(p) <-> unary(p).
pred loop(p) <-> Person::friend+(p, p).
pred taggedX() <-> tag(_, E::x).
pred triple(a, b, c) <-> link(a, b, c).
pred notLinked(a, b, c) <-> link(a, b, c), !link(c, b, a).
pred far(a, c) <-> link(a, b, c), friend(b, c), !friend(a, c).
"""


def named(model: PartialModel, seed: int) -> PartialModel:
    """``model``, with names for its people, and an atom ``E::y`` in some graphs."""
    rng = random.Random(seed)
    nodes = list(model.nodes)
    n = len(nodes)
    people = [i for i, node in enumerate(nodes) if node.kind != "atom"]
    names = [(i, rng.choice([None, "a", "b"])) for i in people]
    attributes = [*model.attributes, Attribute.of("Person::name", "string", n, names)]
    if seed % 2:  # an atom only some graphs have, before the others' own nodes
        nodes.insert(0, Node("E::y", "y", "atom"))
        shift = np.r_[0, np.arange(1, n + 1)]
        relations = [
            Relation(r.name, r.arity, shift[1:][r.must], shift[1:][r.may], r.simple_name)
            for r in model.relations
        ]
        attributes = [
            Attribute(
                a.name,
                a.data_type,
                np.r_[np.asarray([np.inf], a.lower.dtype), a.lower],
                np.r_[np.asarray([-np.inf], a.upper.dtype), a.upper],
                (None, *a.strings) if a.strings else (),
            )
            for a in attributes
        ]
        return PartialModel(nodes, relations, attributes)
    return PartialModel(nodes, model.relations, attributes)


def answers(values) -> dict[str, list]:
    return {name: list(result.items()) for name, result in values.items()}


@pytest.mark.parametrize("seed", range(40))
def test_each_graph_gets_what_it_gets_alone(seed):
    rng = random.Random(seed)
    models = [
        named(graph(100 * seed + i, rng.randint(1, 25)), 100 * seed + i)
        for i in range(rng.randint(1, 6))
    ]
    given = [PackedModel.build(m) if rng.random() < 0.3 else m for m in models]
    rules = RULES + MORE_RULES
    batch = evaluate_batch(rules, given)
    for g, model in enumerate(models):
        alone = answers(evaluate(rules, model))
        assert answers({name: values[g] for name, values in batch.items()}) == alone, g


def test_graphs_without_a_relation_have_no_tuples_of_it():
    one = PartialModel([Node("a", "a")], [Relation.of("edge", 2, [((0, 0), "true")])])
    other = PartialModel([Node("b", "b"), Node("c", "c")], [])
    values = evaluate_batch("pred loop(x) <-> edge(x, x).", [one, other])["loop"]
    assert [dict(values[g].items()) for g in range(2)] == [{(0,): "true"}, {}]
    with pytest.raises(ValueError, match="different arity"):
        PackedBatch.build([one, PartialModel([Node("d", "d")], [Relation.of("edge", 1, [])])])


def test_untyped_parameters_range_over_each_graphs_own_nodes():
    # Smaller graphs leave slots empty: an unknown predicate isn't unknown on them.
    small = PartialModel([Node("a", "a")])
    large = PartialModel([Node(f"n{i}", f"n{i}") for i in range(5)])
    values = evaluate_batch("pred p(x, y).", [small, large], select=["p"])["p"]
    assert len(list(values[0].items())) == 1
    assert len(list(values[1].items())) == 25
    triples = evaluate_batch("pred t(x, y, z).", [small, large], select=["t"])["t"]
    assert [len(list(triples[g].items())) for g in range(2)] == [1, 125]


def test_a_batch_holds_every_graph_in_one_array_per_relation():
    first = PartialModel(
        [Node("a", "a"), Node("E::x", "x", "atom")], [Relation.of("edge", 2, [((0, 1), "true")])]
    )
    second = PartialModel(
        [Node("E::x", "x", "atom"), Node("b", "b"), Node("c", "c")],
        [Relation.of("edge", 2, [((1, 2), "unknown"), ((2, 0), "true")])],
    )
    batch = PackedBatch.build([first, second])
    # Own nodes first (two slots, the most a graph has), then E::x, in one slot.
    assert batch.size == (2, 3) and len(batch) == 2
    assert [p.tolist() for p in batch.placements] == [[0, 2], [2, 0, 1]]
    assert batch.atom_named("x") == 2
    edge = batch["edge"]
    assert edge.must.shape == (2, 3, 1)
    assert k.unpack(edge.must, 3).tolist() == [
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    ]
    assert k.unpack(batch.nodes, 3).tolist() == [[1, 0, 1], [1, 1, 1]]
    values = batch.evaluate("pred toX(p) <-> edge(p, E::x).")["toX"]
    # Rows: the graph, then each node by its index in its own graph.
    assert values.tuples()["true"].tolist() == [[0, 0], [1, 2]]
    assert dict(values[1].items()) == {(2,): "true"}
    assert values[-1].n == 3
    with pytest.raises(IndexError):
        values[2]


def test_a_batch_of_one_graph_keeps_its_arrays():
    packed = PackedModel.build(graph(3, 10))
    batch = PackedBatch.build([packed])
    assert np.shares_memory(batch["Person::friend"].must, packed["Person::friend"].must)
    assert batch.placements[0].tolist() == list(range(packed.size))


def test_rule_errors_are_the_batchs():
    with pytest.raises(RuleError, match="no relation named 'missing'"):
        evaluate_batch("pred p(x) <-> missing(x).", [graph(1, 3), graph(2, 4)])
    with pytest.raises(ValueError, match="at least one graph"):
        PackedBatch.build([])


def test_graphs_are_batched_by_size():
    sizes = [10, 300, 12, 70, 64, 1, 140, 129]
    batches = group_by_size(sizes)
    assert sorted(i for b in batches for i in b) == list(range(len(sizes)))
    for batch in batches:
        found = [sizes[i] for i in batch]
        assert max(found) <= max(64, 2 * min(found))
    assert group_by_size([5000], limit=1) == [[0]]  # one graph, over the limit
    assert len(group_by_size([100] * 10, limit=100 * 2 * 3)) == 4  # three a batch


# -- Metamodel.query_batch ------------------------------------------------------------


def _matches(result):
    return sorted((m.predicate, m.node_names, m.logic_value) for m in result.matches)


def _errors(result):
    return sorted(str(e) for e in result.errors)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_query_batch_answers_each_graph_as_query_does(family):
    drawn = [FAMILIES[family](random.Random(seed)) for seed in range(25)]
    metamodel, _, rules, _ = drawn[0]
    graphs = [objects for _, objects, _, _ in drawn]
    own = [predicates for _, _, _, predicates in drawn]
    results = metamodel.query_batch(
        graphs, BitsEngine(), rules, graph_predicates=own, return_exceptions=True
    )
    answered = 0
    for objects, predicates, result in zip(graphs, own, results, strict=True):
        try:
            alone = metamodel.query(objects, BitsEngine(), rules, predicates=predicates)
        except UnsupportedRuleError as e:
            assert isinstance(result, UnsupportedRuleError) and str(result) == str(e)
            continue
        assert _matches(result) == _matches(alone)
        assert _errors(result) == _errors(alone)
        assert result.source == alone.source
        answered += 1
    assert answered >= 15


def test_query_batch_raises_for_a_refused_graph_unless_told_to_return_it():
    metamodel = Metamodel(Shapes, Square, options={Shape: ClassOptions(abstract=True)})
    rules = ["pred p(Shape x) <-> Shape(x)."]
    graphs = [[Shapes(shapes=[Square()])], [Shapes(shapes=[Shape()])], [Shapes(shapes=[])]]
    with pytest.raises(UnsupportedRuleError, match="abstract class Shape"):
        metamodel.query_batch(graphs, BitsEngine(), rules)
    first, refused, last = metamodel.query_batch(
        graphs, BitsEngine(), rules, return_exceptions=True
    )
    assert isinstance(refused, UnsupportedRuleError)
    assert [m.node_names for m in first.matches] == [("square_1",)]
    assert last.matches == []


def test_query_batch_queries_one_graph_at_a_time_with_other_engines():
    metamodel, objects, rules, predicates = FAMILIES["deployment"](random.Random(1))
    asked = []

    class OneAtATime:
        def query_graph(self, schema, graph, rules, predicates, select):
            asked.append(graph)
            return BitsEngine().query_graph(schema, graph, rules, predicates, select)

    results = metamodel.query_batch([objects, objects], OneAtATime(), rules)
    assert len(asked) == 2
    expected = metamodel.query(objects, BitsEngine(), rules)
    assert [_matches(r) for r in results] == [_matches(expected)] * 2


def test_query_batch_needs_one_list_of_predicates_per_graph():
    metamodel, objects, rules, _ = FAMILIES["deployment"](random.Random(1))
    with pytest.raises(ValueError, match="one per graph"):
        metamodel.query_batch([objects, objects], BitsEngine(), rules, graph_predicates=[[]])
    assert metamodel.query_batch([], BitsEngine(), rules) == []
