<!--
  SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>

  SPDX-License-Identifier: EPL-2.0
-->

# Conformance suite

Refinery `.problem` files with expectations, and Refinery's recorded answers to them. They pin down Refinery's partial-model semantics, so that code computing it in Python can be checked against them without a server.

| Directory | What |
|---|---|
| `upstream/` | Refinery's own file-based semantics tests, unchanged, from [`generator-core/src/test/resources`](https://github.com/graphs4value/refinery/tree/747071004a969288113bb09d85299bbd28c9f9df/subprojects/generator-core/src/test/resources/tools/refinery/generator) at commit `7470710` |
| `queries/` | pyfinery's cases for evaluating predicates: negation, closure, attributes, equality, arity, modal operators and error predicates |
| `generated/` | 100 random problems from `generate.py`: small partial models and random predicates, without expectations: what Refinery answered, recorded, is the expectation |
| `recorded/` | what the server of `recording.IMAGE` answered for each case (generated) |

## Format

The format is Refinery's, read as its `ProblemSplitter` and `SemanticsTestBuilder` do (`problems.py`):

```
class Person { Person[] friend }   % common to every case
Person(a).
% TEST: a case: the common part, then these statements
friend(a, a).
% EXPECT:
friend(a, a).
% EXPECT EXACTLY: equal, rather than at least as precise
!friend(a, a).
```

- `% TEST[ WITH ERRORS][: name]` starts a case. Without `WITH ERRORS`, the model must be consistent: no tuple may be `error`.
- `% EXPECT[ CANDIDATE][ EXACTLY][: description]` starts assertions about the partial model, or with `CANDIDATE`, its concretization. A value passes if it refines the expected one: `unknown` is refined by anything, `true` and `false` by themselves and `error`, and an interval by the intervals in it.
- `% EXPECT:` before any `% TEST` makes the whole file one case.
- A tuple of an object that doesn't exist has its relation's default value, as in Refinery's `FilteredInterpretation`: false, or for an attribute, error. Refinery lists an attribute's value on the objects of its class (`unknown` if nothing is asserted); elsewhere it has none, and comparing it is an error.

## Running

```sh
uv run pytest tests/conformance                              # against the recordings
PYFINERY_DOCKER_TESTS=1 uv run pytest tests/conformance      # and the recordings against Refinery
```

To draw the random problems again (it starts `recording.IMAGE` in Docker, or uses the server at `REFINERY_API_BASE`, to redraw those Refinery rejects):

```sh
uv run python -m tests.conformance.generate [COUNT]
```

After adding or changing a `.problem` file, record Refinery's answers again:

```sh
uv run python -m tests.conformance.recording [FILE.problem ...]
```

This starts `recording.IMAGE` in Docker, or uses the server at `REFINERY_API_BASE`. Each recording holds the digest of its file, and a test fails if they don't match.

Names in assertions are compared as Refinery writes them in its output: quoted only where they must be, escapes decoded (`'a\u0041'` is `aA`). A string holding a lone UTF-16 surrogate (`"\uD83D"`) comes back from the server as `"?"`, so it can't be expected as written.

`PROPAGATION FAILURE` and `CONCRETIZATION FAILURE` cases can't be checked over the HTTP API, which doesn't say which step failed; no file uses them.

## Known Refinery issues

On the pinned image, a call that repeats a variable no other literal of its clause has is evaluated as true: `pred cyc() <-> r+(z, z).` holds even when `r` has no cycles, and `pred loop() <-> r(z, z).` is true when every loop is only unknown. A negated atom's variables are its own, so `!t(y, y)` is false even beside `!r(x, y)`. With a parameter instead (`pred cycP(A x) <-> r+(x, x).`), the answer is right. `generate.py` doesn't draw such clauses, and no other file has one.
