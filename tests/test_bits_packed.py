# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Building a :class:`PackedModel`, evaluating rules on it, and combining graphs.

The reference for combining and connecting is the :class:`PartialModel` with the same
tuples, written out: the second graph's nodes after the first's, atoms shared.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pyfinery.bits import (
    Attribute,
    Node,
    PackedModel,
    PartialModel,
    Relation,
    RuleError,
    Rules,
    UnsupportedRuleError,
    evaluate,
)

RULES = """
pred reach(a, b) <-> Person::friend+(a, b).
pred lonely(p) <-> Person(p), !Person::friend(p, _).
pred older(a, b) <-> Person::friend(a, b), age(a) > age(b).
pred mutual(a, b) <-> friend(a, b), friend(b, a).
pred tagged(p) <-> tag(p, E::x).
pred anyone() <-> Person(_).
pred trio(a, b, c) <-> link(a, b, c), !friend(a, c).
"""


def graph(seed: int, n: int) -> PartialModel:
    """``n`` people with random friends, links and ages, some that only may exist, and the
    atom ``E::x``, which every graph has."""
    rng = random.Random(seed)
    logic = ["true", "true", "unknown"]
    nodes = [Node(f"p{seed}_{i}", f"p{seed}_{i}") for i in range(n)] + [Node("E::x", "x", "atom")]
    x = n
    pairs = {(rng.randrange(n), rng.randrange(n)) for _ in range(2 * n)}
    triples = {(rng.randrange(n), rng.randrange(n), rng.randrange(n)) for _ in range(n)}
    ages = [
        (i, rng.choice([None, rng.randrange(5), f"{rng.randrange(3)}..{rng.randrange(3, 6)}"]))
        for i in range(n)
    ]
    return PartialModel(
        nodes,
        [
            Relation.of("Person", 1, [((i,), "true") for i in range(n)]),
            Relation.of("Person::friend", 2, [(p, rng.choice(logic)) for p in pairs]),
            Relation.of(
                "tag", 2, [((i, x), rng.choice(logic)) for i in range(n) if rng.random() < 0.3]
            ),
            Relation.of("link", 3, [(t, rng.choice(logic)) for t in triples]),
            Relation.of(
                "builtin::exists",
                1,
                [((i,), rng.choice(logic)) for i in range(n)] + [((x,), "true")],
            ),
        ],
        [Attribute.of("Person::age", "int", n + 1, ages)],
    )


def written_out(
    first: PartialModel, second: PartialModel, cross: list[tuple[int, int]], value: str
) -> PartialModel:
    """``first`` and ``second`` as one model, the shared atom once, and ``Person::friend``
    holding ``cross`` (indices of ``first``'s node, then ``second``'s) with ``value``."""
    atoms = {node.name: i for i, node in enumerate(first.nodes) if node.kind == "atom"}
    placement, nodes = [], list(first.nodes)
    for node in second.nodes:
        if node.kind == "atom" and node.name in atoms:
            placement.append(atoms[node.name])
        else:
            placement.append(len(nodes))
            nodes.append(node)
    at = np.array(placement)
    relations = []
    for mine in first.all_relations:
        theirs = second.relation_named(mine.name)
        assert theirs is not None
        must = np.concatenate([mine.must, at[theirs.must]])
        may = np.concatenate([mine.may, at[theirs.may]])
        if mine.name == "Person::friend":
            added = np.array([(i, at[j]) for i, j in cross], np.intp).reshape(-1, 2)
            may = np.concatenate([may, added])
            if value == "true":
                must = np.concatenate([must, added])
        relations.append(Relation(mine.name, mine.arity, must, may, mine.simple_name))
    age, other_age = first.attribute_named("age"), second.attribute_named("age")
    assert age is not None and other_age is not None
    lower = np.full(len(nodes), np.inf)
    upper = np.full(len(nodes), -np.inf)
    lower[: first.size], upper[: first.size] = age.lower, age.upper
    lower[at[:-1]], upper[at[:-1]] = other_age.lower[:-1], other_age.upper[:-1]  # not E::x
    return PartialModel(nodes, relations, [Attribute("Person::age", "int", lower, upper)])


def answers(model: PartialModel | PackedModel) -> dict[str, list]:
    return {name: list(result.items()) for name, result in evaluate(RULES, model).items()}


def test_a_built_model_evaluates_as_the_partial_model_does():
    model = graph(1, 12)
    packed = PackedModel.build(model)
    assert answers(packed) == answers(model)
    assert answers(packed) == answers(packed)  # built once, evaluated again


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("value", ["true", "unknown"])
def test_combined_and_connected_graphs_evaluate_as_written_out(seed, value):
    first, second = graph(2 * seed, 7), graph(2 * seed + 1, 5)
    rng = random.Random(seed)
    cross = [(rng.randrange(7), rng.randrange(5)) for _ in range(3)]
    both, placed = PackedModel.build(first).combine(PackedModel.build(second))
    connected = both.connect("Person::friend", [(i, placed[j]) for i, j in cross], value)
    assert answers(connected) == answers(written_out(first, second, cross, value))


def test_combining_puts_the_second_graphs_nodes_after_the_firsts():
    g1, g2 = PackedModel.build(graph(1, 3)), PackedModel.build(graph(2, 2))
    both, placed = g1.combine(g2)
    # g1: p1_0 p1_1 p1_2 E::x; g2's people after them, and its E::x on g1's.
    assert [node.name for node in both.nodes] == ["p1_0", "p1_1", "p1_2", "E::x", "p2_0", "p2_1"]
    assert placed.tolist() == [4, 5, 3]
    assert (g1 + g2).nodes == both.nodes


def test_connecting_joins_the_graphs_and_leaves_them_as_they_were():
    def chain(name: str) -> PartialModel:
        nodes = [Node(f"{name}{i}", f"{name}{i}") for i in range(2)]
        return PartialModel(nodes, [Relation.of("edge", 2, [((0, 1), "true")])])

    g1, g2 = PackedModel.build(chain("a")), PackedModel.build(chain("b"))
    both, placed = g1.combine(g2)
    rules = "pred reach(x, y) <-> edge+(x, y)."
    apart = dict(evaluate(rules, both)["reach"].items())
    assert apart == {(0, 1): "true", (2, 3): "true"}  # two blocks, nothing between them
    joined = both.connect("edge", [(1, placed[0])])  # a1 -> b0
    assert dict(evaluate(rules, joined)["reach"].items()) == {
        (0, 1): "true",
        (0, 2): "true",
        (0, 3): "true",
        (1, 2): "true",
        (1, 3): "true",
        (2, 3): "true",
    }
    assert dict(evaluate(rules, both)["reach"].items()) == apart  # both is unchanged
    unknown = both.connect("edge", [(1, placed[0])], "unknown")
    assert evaluate(rules, unknown)["reach"].value((0, 3)) == "unknown"


def test_combining_refuses_relations_of_different_arities():
    one = PartialModel([Node("a", "a")], [Relation.of("r", 1, [((0,), "true")])])
    two = PartialModel([Node("b", "b")], [Relation.of("r", 2, [((0, 0), "true")])])
    with pytest.raises(ValueError, match="different arity"):
        PackedModel.build(one) + PackedModel.build(two)


def test_connecting_needs_a_relation_of_the_model():
    packed = PackedModel.build(graph(1, 2))
    with pytest.raises(ValueError, match="no relation named 'missing'"):
        packed.connect("missing", [(0, 1)])


def test_compiled_rules_evaluate_any_built_model():
    rules = Rules.compile(RULES)
    for seed in range(5):
        model = graph(seed, 9)
        packed = PackedModel.build(model)
        expected = answers(model)
        assert {n: list(r.items()) for n, r in rules.evaluate(packed).items()} == expected
        assert {n: list(r.items()) for n, r in packed.evaluate(rules).items()} == expected
    assert list(rules.evaluate(packed, select=["anyone"])) == ["anyone"]


def test_compiling_checks_what_doesnt_depend_on_a_model():
    with pytest.raises(RuleError, match="defined twice"):
        Rules.compile("pred p(x) <-> q(x).\npred p(x) <-> q(x).")
    with pytest.raises(UnsupportedRuleError, match="assertions about a predicate with a body"):
        Rules.compile("pred p(x) <-> q(x).\np(a).")
