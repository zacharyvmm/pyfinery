<!--
  SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>

  SPDX-License-Identifier: EPL-2.0
-->

# pyfinery

Unofficial Python client for [Refinery](https://refinery.tools), the graph solver that generates consistent graph models from partial models.

It talks to a Refinery server over its HTTP API (`/api/v1`) and mirrors the official JavaScript client [`@tools.refinery/client`](https://github.com/graphs4value/refinery/tree/main/subprojects/client-js):

- sync (`Refinery`) and asyncio (`AsyncRefinery`) clients
- `generate`, `generate_many`, `semantics` and `concretize`, plus `stream_*` variants that yield progress updates as they arrive
- typed request and response models (pydantic), with helpers for reading the generated graph
- typed exceptions for every error the server reports
- `RefineryServer`, which starts a local Refinery server in Docker for you
- `pyfinery.graph`: define a Refinery metamodel as Pydantic classes, then check that your object graphs follow its rules

## Installation

```sh
uv add git+https://github.com/zacharyvmm/pyfinery
# or: pip install git+https://github.com/zacharyvmm/pyfinery
```

You also need a Refinery server. Either start one yourself:

```sh
docker run --rm -p 8888:8888 ghcr.io/graphs4value/refinery
```

or let `RefineryServer` manage the container (requires Docker).

## Quick start

```python
from pyfinery import Refinery

problem = """
class Filesystem {
    contains Dir[1] root
}

abstract class FSObject {
    container Dir parent opposite contents
}

class Dir extends FSObject {
    contains FSObject[] contents opposite parent
}

class File extends FSObject.

scope node = 5..8.
"""

with Refinery("http://localhost:8888/api/v1") as refinery:
    result = refinery.generate(problem, scopes=["File=2..*"], random_seed=42)

print(result.source)                              # the model in Refinery syntax
print(result.json_.true_tuples("contents"))       # [('dir1', 'file1'), ...]
```

If you leave out the base URL, the client uses `$REFINERY_API_BASE`, and falls back to `http://localhost:8888/api/v1` when that isn't set. Problems can be source strings or `pathlib.Path`s to `.problem` files.

### Managed Docker server

```python
from pyfinery import RefineryServer

with RefineryServer(model_generation_timeout=1200, library_paths=["./lib"]) as server:
    with server.client() as refinery:
        result = refinery.generate_many("model.problem", 10)
```

The container listens on a random free port of `127.0.0.1` and is removed on exit. `library_paths` are mounted read-only, so problems can `import` the `.refinery` modules in them. Any other [server setting](https://refinery.tools/learn/docker/#environmental-variables) can be passed via `env={...}`. Before returning, `RefineryServer` sends a few small warm-up requests, so the first `semantics`/`concretize`/`validate` call doesn't time out while the JVM warms up. Pass `warmup=False` to skip this.

## API overview

| Method | Endpoint | Returns |
| --- | --- | --- |
| `generate(problem, *, scopes, random_seed, format, on_status)` | `generate` | `GenerateResult` (`.json_`, `.source`) |
| `generate_many(problem, count, ...)` | `generateMany` | `GenerateManyResult` (`.stop_reason`, `.models`) |
| `semantics(problem, *, format)` | `semantics` | `SemanticsResult` (`.issues`, `.json_`): the partial model |
| `concretize(problem, *, format)` | `concretize` | `ConcretizationResult`: unknowns resolved to false |
| `stream_generate(...)`, `stream_generate_many(...)` | | iterator of status updates, then the result |

`AsyncRefinery` has the same methods as coroutines and async iterators. `RefineryChat` / `AsyncRefineryChat` wrap the text-to-model chat service (`text_to_model(metamodel, text)`).

**Scopes** can be strings (`"node=10..20"`, `"File=5"`, `"Dir=2..*"`, `"Dir+=1..3"`) or `Scope(...)` objects. Use `Scope(..., override=True)` to replace a `scope` declaration from the problem.

**Output formats:** both the JSON and the source outputs are on by default. Use `OutputFormats` to turn either off or to keep non-existing objects and shadow predicates:

```python
from pyfinery import OutputFormats, JsonOutputFormat
fmt = OutputFormats(json_=JsonOutputFormat(non_existing_objects="keep"), source={"enabled": False})
```

**Reading models:** `JsonOutput.tuples(relation)` gives the tuples of a relation with node names resolved, ending with the truth value (`"true"`, `"false"`, `"unknown"`, `"error"`) or the attribute value. `true_tuples(relation)` keeps only the tuples that are definitely true. Relations can be looked up by qualified name (`Dir::contents`) or simple name (`contents`).

**Streaming and cancellation:**

```python
for event in refinery.stream_generate_many(problem, 5):
    match event:
        case GenerateStatus(message=msg): print(msg)
        case GenerateResult():            handle(event)
        case GenerateManyResult():        print("done:", event.stop_reason)
```

If you close a stream early (`break` followed by `stream.close()`, or `contextlib.closing`/`aclosing`), or cancel an asyncio task, the client disconnects and the server cancels the generation.

## Validating Pydantic object graphs

`pyfinery.graph` lets you declare Refinery classes as Pydantic models. Build an ordinary object graph from them, then ask Refinery whether it follows the metamodel and your rules: multiplicities, the containment hierarchy, opposites, types, and custom `error pred` constraints.

```python
from __future__ import annotations
from typing import Annotated
from pydantic import Field
from pyfinery import Container, Contains, Metamodel, RefineryNode, RefineryServer

class Filesystem(RefineryNode):
    root: Annotated[Dir, Contains()]                      # contains Dir[1] root

class FSObject(RefineryNode):
    __abstract__ = True
    parent: Annotated[Dir | None, Container(opposite="contents")] = None

class Dir(FSObject):
    __node_id__ = "name"                                   # name nodes after this field
    name: str
    contents: Annotated[list[FSObject], Contains(opposite="parent", multiplicity="0..3")] = Field(default_factory=list)

class File(FSObject):
    size: int = 0

metamodel = Metamodel(Filesystem, rules=[
    "error pred emptyDir(Dir d) <-> !contents(d, _).",
])
print(metamodel.source)          # the generated Refinery metamodel

home = Dir(name="home", contents=[File(size=10)])
orphan = Dir(name="orphan")      # not contained anywhere, and empty

with RefineryServer() as server, server.client() as refinery:
    report = metamodel.validate([Filesystem(root=home), orphan], refinery)

for violation in report.violations:
    print(violation)
# containment: builtin::invalidContainer(orphan)
# rule: emptyDir(orphan)
report.raise_for_violations()    # raises ModelValidationError
```

How Python annotations map to Refinery:

| Python field | Refinery |
| --- | --- |
| `x: Dir` / `x: Dir \| None` / `x: list[Dir]` | `Dir[1] x` / `Dir[0..1] x` / `Dir[0..*] x` |
| `Field(min_length=1, max_length=3)` on a list, or `Ref(multiplicity="1..3")` | `Dir[1..3] x` |
| `Annotated[..., Contains()]` | `contains Dir[...] x` |
| `Annotated[Dir \| None, Container(opposite="y")]` | `container Dir x opposite y` |
| `Annotated[..., Ref(opposite="y")]` | `Dir[...] x opposite y` |
| a plain Pydantic `BaseModel` (`home: Address`) | `contains Address[1] home`, with `Address` as a class |
| an `enum.Enum` subclass | an `enum` and a reference to its literals |
| `int`, `float`, `str`, `bool` | `int`, `real`, `string`, `boolean` attributes |
| Python subclassing, `__abstract__ = True` | `extends`, `abstract class` |

### Nested Pydantic models (composition)

Ordinary Pydantic models nested inside a node are treated as **value objects owned by their parent**. The field becomes a containment reference, and the nested model becomes a Refinery class whose fields are mapped the same way. Lists, optionals and subclasses of nested models all work:

```python
class Address(BaseModel):          # plain Pydantic, not a RefineryNode
    street: str
    country: Country               # an Enum

class Person(RefineryNode):
    name: str
    home: Address                  # contains Address[1] home
    work: Address | None = None    # contains Address[0..1] work
    manager: Person | None = None  # ordinary reference
```

You can even validate a graph made only of plain models, such as an `Order` containing `Line`s. Since each nested object has exactly one owner, a single `Address` *instance* reused for two people is reported as `containment: builtin::invalidContainer(address_1)`. Use `address.model_copy()` to give each person their own, or make the class a `RefineryNode` if it should be shared. Plain models can't take part in `opposite` pairs or use `Container()`. Make them `RefineryNode`s if they need a back-reference.

### Existing and generated Pydantic classes

pyfinery also works with classes you can't (or don't want to) change, such as models generated at runtime from a grammar. You configure them from outside:

```python
from pyfinery import ClassOptions, Metamodel

metamodel = Metamodel(
    EffectGraph,
    options={
        File: ClassOptions(key=lambda f: f.key),        # equal keys -> one node
        Endpoint: ClassOptions(key=lambda e: e.key),
        Effect: ClassOptions(exclude=["command", "trace"]),
    },
    unions={"Resource": [File, Variable, Endpoint]},   # name for File | Variable | Endpoint
    unsupported="skip",                                 # leave out e.g. list[str] fields
)
print(metamodel.skipped)   # [(Path, 'segments', '...')]
```

- **`key`** makes a class an *entity*. Instances with equal keys are one node, so a field holding an equal **copy** of a resource refers to that resource instead of owning a new node. Fields typed as keyed classes are plain references, not containment. Classes without a key stay owned values, as described above.
- **Unions** of model classes become an abstract superclass that their members extend. This works with or without a Pydantic `discriminator`. The superclass is named after its members (`FileOrEndpoint`) unless you name it in `unions`.
- **`Literal` fields** with several string values become Refinery enums (`op: Literal["read", "write"]` → `enum EffectOp { read, write }`). Single-value `Literal`s, such as discriminator tags (`type: Literal["file"]`), are skipped, since the class already says it.
- **Exclusion:** `exclude` drops payload fields. `unsupported="skip"` drops fields that have no Refinery counterpart, such as `list[str]` or `dict`, and records them in `metamodel.skipped`. By default, such fields raise `TypeError`.
- **Subclass discovery** only goes *down* from the classes your fields refer to. A base class shared by many unrelated generated classes doesn't pull them all in.
- **Other options:** `node_name` names nodes (for example after their key), and `abstract` declares an abstract class.

How validation works:

- **Building the metamodel:** `Metamodel(...)` collects every `RefineryNode` class reachable from the ones you pass: superclasses, referenced classes, and subclasses that have already been imported. It checks that each `opposite` is declared on both sides.
- **Sending the graph:** `validate()` walks the graph from the objects you pass and turns it into Refinery facts. It sends them with the server's `concretize` call, which treats everything you didn't assert as false.
- **Reading the result:** it returns every constraint that evaluates to `error` as a `Violation`. Each has a `kind` (`rule`, `multiplicity`, `containment` or `contradiction`), the Refinery `constraint` name, and the offending Python `nodes`.
- **Async:** use `await metamodel.avalidate(...)` with an `AsyncRefinery`.

A few things to keep in mind:

- **Identity:** nodes compare and hash by identity, so two `Dir(name="a")` objects are different nodes.
- **Opposites:** you only need to set one side of an `opposite` pair. Refinery infers the other side. If you want your Python objects to agree too, call `metamodel.link_opposites(objects)`; it fills in each missing side (for example, setting `doc.parent` after `home.contents.append(doc)`) and leaves conflicting values for `validate()` to report.
- **Reachability:** `validate()` walks only *outgoing* references from the objects you pass. A node that nothing points to, such as a child that only sets its own `parent`, isn't found unless you pass it too (or run `link_opposites` with it first).
- **Where multiplicities are enforced:** Pydantic enforces `Field(min_length=...)` when you construct an object. `Ref(multiplicity=...)` is only checked by Refinery, which is handy when you build graphs step by step.
- **Graph size and timeouts:** validation runs under the server's *semantics* timeout, which defaults to 1 second (enough for a few hundred nodes). For larger graphs, start the server with `RefineryServer(semantics_timeout=10_000)`, or set `REFINERY_SEMANTICS_TIMEOUT_MS`.
- **Inspecting what's sent:** `metamodel.problem(objects)` returns the exact problem that would be sent, so you can paste it into the Refinery web editor.

## Querying graph patterns with must/may

`metamodel.query()` answers *which parts of the graph match a pattern*, rather than *does the graph obey the metamodel*. Patterns are Refinery predicates. Uncertainty in your data carries through Refinery's four-valued logic, so each match either **must** hold or only **may** hold:

```python
from pyfinery import ClassOptions, Metamodel, Predicate

metamodel = Metamodel(
    EffectGraph,
    options={
        File: ClassOptions(key=lambda f: f.key),
        Endpoint: ClassOptions(key=lambda e: e.key),
        # An effect that happens on some paths only may not exist.
        Effect: ClassOptions(exists=lambda e: "must" if e.definite else "may"),
    },
    unions={"Resource": [File, Endpoint]},
    unsupported="skip",
)
rules = [
    "pred writes(Resource a, Resource b) <-> Effect::sources(e, a), Effect::target(e, b), Effect::op(e, EffectOp::write).",
    "pred exfiltration(File key, Endpoint net) <-> underSsh(key), writes+(key, net).",
]
# What Refinery can't do (globs, path logic) is computed in Python: True, "may", or False.
under_ssh = Predicate("underSsh", (File,), lambda f: "may" if f.path_unknown else f.path.startswith("~/.ssh/"))

result = metamodel.query([graph], refinery, rules, predicates=[under_ssh])
for match in result.matches:
    print(match)          # must: exfiltration(file_1, endpoint_1)
result.must("exfiltration"), result.may("exfiltration"), result["writes"]
```

- **Closed world:** `query()` asserts that the graph holds nothing beyond what you pass: no other objects, references, subclass memberships or `Predicate` facts. What's uncertain stays uncertain.
  - **Objects:** an object that only *may* exist (`ClassOptions(exists=...)`) is asserted as `?exists(...)`.
  - **Facts:** a `Predicate` returning `"may"` becomes an unknown fact.
- **How certainty is decided:** a match is `must` when Refinery evaluates the predicate to true, and `may` when it's unknown. It's also `may` when one of its arguments only may exist, because Refinery weakens a predicate for nodes quantified in its body but not for its arguments.
- **Which predicates are reported:** every `pred` declared in `rules`, unless you choose with `select=[...]`.
- **Errors:** metamodel violations and contradictions found along the way are in `result.errors`.
- **Arguments:** `Predicate` parameters are model classes or union names. Each predicate is evaluated on every combination of nodes of those types, so keep arity low on large graphs.
- **Debugging:** `metamodel.query_problem(...)` returns the exact problem sent, for pasting into the web editor. `aquery()` is the async variant.

I checked this against [bash-validation](https://github.com/zacharyvmm/bash-validation)'s own matcher for its `exfiltration` rule on six sessions (a variable, a copy, a pipe, a path known only at run time, a branch, and a benign case). Both gave the same matches and certainties. The rule above ignores the *order* of writes, so it also flags a key that is sent before it's copied. Order-aware data flow is better computed in Python and passed as a two-argument `Predicate`.

## Errors

All errors subclass `RefineryError`:

| Exception | Server result | Meaning |
| --- | --- | --- |
| `InvalidProblemError` | `invalidProblem` | syntax or validation errors; see `.issues` |
| `UnsatisfiableError` | `unsatisfiable` | no consistent model exists |
| `RefineryTimeoutError` (also a `TimeoutError`) | `timeout` | the server ran out of time |
| `BadRequestError` | `requestError` | malformed request; see `.details` |
| `ServerError` | `serverError` | internal server error |
| `RequestCancelledError` | `cancelled` | the server cancelled the operation |
| `ProtocolError` | | the response couldn't be understood |
| `ServerStartError` | | `RefineryServer` failed to start |
| `ModelValidationError` | | `ValidationReport.raise_for_violations()` found violations |

Network failures are raised as `httpx` exceptions.

`semantics` and `concretize` run under the server's semantics timeout (`REFINERY_SEMANTICS_TIMEOUT_MS`, 1 second by default). On a server you started yourself, the first requests may hit it while the JVM warms up, and then report `timeout` even for problems that are invalid. `RefineryServer` warms the server up for you.

## Development

```sh
uv run pytest                              # unit tests (mocked HTTP)
PYFINERY_DOCKER_TESTS=1 uv run pytest      # plus end-to-end tests against a real server in Docker
uv run examples/generate_filesystems.py
uv run examples/validate_graph.py
```

## License and attribution

pyfinery is licensed under the [Eclipse Public License 2.0](LICENSE) (`EPL-2.0`), the same license as Refinery.

The client code is a Python port of Refinery's official JavaScript client, [`@tools.refinery/client`](https://github.com/graphs4value/refinery/tree/main/subprojects/client-js), which is Copyright (c) 2021-2025 [The Refinery Authors](https://github.com/graphs4value/refinery/blob/main/CONTRIBUTORS.md). The example problem comes from the [Refinery file system tutorial](https://refinery.tools/learn/tutorials/file-system). [NOTICE.md](NOTICE.md) lists the details, and every file has [REUSE](https://reuse.software)-style SPDX headers.

Refinery itself was developed by The Refinery Authors with the support of the 2022 Amazon Research Award. If you use it in research, please cite the [Refinery publications](https://github.com/graphs4value/refinery#related-publications), for example:

> K. Marussy, A. Ficsor, O. Semeráth, D. Varró: "Refinery: Graph Solver as a Service." *ICSE 2024 Demonstrations*. [doi:10.1145/3639478.3640045](https://doi.org/10.1145/3639478.3640045)

pyfinery isn't affiliated with or endorsed by the Refinery project.
