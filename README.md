<!--
  SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>

  SPDX-License-Identifier: EPL-2.0
-->

# pyfinery

Unofficial Python library for [Refinery](https://refinery.tools), the graph solver for partial models. pyfinery covers two use cases:

- **Validate and query Pydantic models with Refinery.** Load your data (JSON, YAML, an API response, a parse tree) into Pydantic objects, as you already do. pyfinery derives a Refinery metamodel from the classes and checks the objects against it, along with any rules you write in Refinery's logic language. It reports violations, or pattern matches that **must** or only **may** hold.
- **Call Refinery from Python.** This is a typed client for Refinery's HTTP API, ported from the official JavaScript client [`@tools.refinery/client`](https://github.com/graphs4value/refinery/tree/main/subprojects/client-js). It generates models, computes partial-model semantics, and runs a local server in Docker.

## Installation

```sh
uv add git+https://github.com/zacharyvmm/pyfinery
# or: pip install git+https://github.com/zacharyvmm/pyfinery
```

pyfinery needs Python 3.12+ and a Refinery server. Either let `RefineryServer` start one in Docker for you, or run one yourself:

```sh
docker run --rm -p 8888:8888 ghcr.io/zacharyvmm/refinery
```

This image is built from [a fork of Refinery](https://github.com/zacharyvmm/refinery). The fork loads large problems in linear rather than quadratic time, and lets clients leave relations out of the response. On a graph of 3,000 objects, `validate()` takes 1.6 seconds with it and 37 seconds with the upstream image `ghcr.io/graphs4value/refinery`. pyfinery works with either; pass `RefineryServer(image=...)` to choose.

## Validating Pydantic models

Take ordinary Pydantic models that know nothing about Refinery. Here, a deployment configuration where components refer to each other by name:

```python
class Port(BaseModel):
    number: int
    public: bool = False

class Service(BaseModel):
    kind: Literal["service"] = "service"
    name: str
    image: str
    tier: Literal["frontend", "backend", "data"]
    ports: list[Port] = []
    depends_on: list[str] = []          # names of other components

class Database(BaseModel):
    kind: Literal["database"] = "database"
    name: str
    engine: Literal["postgres", "redis"]
    encrypted: bool

class Deployment(BaseModel):
    components: list[Annotated[Service | Database, Field(discriminator="kind")]]

deployment = Deployment.model_validate_json(Path("deployment.json").read_text())
```

Describe how they map to Refinery, write rules, and validate:

```python
from pyfinery import ClassOptions, Metamodel, Predicate, RefineryServer

metamodel = Metamodel(
    Deployment,
    options={
        # Components are identified by name; depends_on holds names.
        Service: ClassOptions(key=lambda s: s.name, references={"depends_on": "Component"}),
        Database: ClassOptions(key=lambda d: d.name),
    },
    unions={"Component": [Service, Database]},
    rules=[
        "error pred publicDependsOnData(Service s, Database d) <-> "
        "Service::ports(s, p), Port::public(p), Service::depends_on(s, d).",
        "error pred unpinned(Service s) <-> floatingTag(s).",
    ],
)
print(metamodel.source)   # the generated Refinery metamodel

# What Refinery can't express, such as a regular expression, is computed in Python.
floating_tag = Predicate("floatingTag", (Service,), lambda s: re.search(r":(latest)?$", s.image) is not None)

with RefineryServer() as server, server.client() as refinery:
    report = metamodel.validate([deployment], refinery, predicates=[floating_tag])

for violation in report.violations:
    print(violation)
# reference: Service::depends_on::unresolved(api, 'billing')
# rule: publicDependsOnData(web, cache)
# rule: unpinned(api)
report.raise_for_violations()   # raises ModelValidationError
```

Each `Violation` has a `kind`, the Refinery `constraint` that failed, and the offending Python objects (`nodes`):

| `kind` | Meaning |
| --- | --- |
| `rule` | one of your `error pred` rules matched |
| `multiplicity` | too few or too many references, such as a required field with nothing in it |
| `containment` | an owned object with no owner, two owners, or an ownership cycle |
| `reference` | a key in a reference-by-key field matched no object (`::unresolved`) or several (`::ambiguous`) |
| `contradiction` | facts that contradict the metamodel, such as a reference to an object of the wrong class |

`validate()` asserts the objects as facts and closes the world with Refinery's `concretize` operation: anything not asserted is false. `metamodel.problem(objects)` shows exactly what is sent, so you can paste it into the [Refinery web editor](https://refinery.services). `avalidate()` is the async variant.

## Querying with must and may

`query()` asks *which parts of the graph match a pattern*, rather than *does the graph follow the rules*. Patterns are Refinery predicates. When parts of your data are uncertain, Refinery's four-valued logic keeps track of it, and each match either **must** hold (true) or only **may** hold (unknown):

```python
guessed = Metamodel(
    Deployment,
    options={
        Service: ClassOptions(
            key=lambda s: s.name,
            references={"depends_on": "Component"},
            # The worker's dependencies were guessed from logs: they only may exist.
            certainty=lambda s, field, value: "may" if s.name == "worker" else True,
        ),
        Database: ClassOptions(key=lambda d: d.name),
    },
    unions={"Component": [Service, Database]},
)
result = guessed.query([deployment], refinery, [
    "pred reachesUnencrypted(Service s, Database d) <-> "
    "Service::depends_on+(s, d), !Database::encrypted(d).",
])
for match in result.matches:
    print(match)
# must: reachesUnencrypted(web, db)
# must: reachesUnencrypted(api, db)
# may: reachesUnencrypted(worker, db)
result.must("reachesUnencrypted"), result.may(), result["reachesUnencrypted"]
match = result["reachesUnencrypted"][0]
match["s"], match.arguments   # by parameter name: the Service, {"s": ..., "d": ...}
```

Uncertainty can come from three places:

| Source | How |
| --- | --- |
| an object that may not exist | `ClassOptions(exists=lambda obj: "must" or "may")` |
| a reference or attribute value that may not hold | `ClassOptions(certainty=lambda obj, field, value: "must" or "may")` |
| a fact computed in Python that may hold | a `Predicate` returning `"may"`, or `tuples=[((a, b), "may")]` |

- **Closed world:** `query()` asserts that there is nothing beyond what you pass: no other objects, references, subclass memberships or predicate facts. What's uncertain stays unknown.
- **Certainty:** a match is `must` when Refinery evaluates it to true, and `may` when it's unknown. pyfinery also weakens it to `may` when one of its arguments only may exist, because Refinery doesn't do that for predicate parameters.
- **What's reported:** every `pred` declared in the query's rules, unless you choose with `select=[...]`. Metamodel violations found along the way are in `result.errors`.
- **Debugging and async:** `query_problem(...)` shows what is sent, and `aquery()` is the async variant.
- **`validate()` is two-valued:** facts that only *may* hold count as not holding there, and every object exists.

## Mapping Pydantic classes to Refinery

`Metamodel(*classes, ...)` collects the given classes, their superclasses, the classes their fields refer to, and the already-imported subclasses of those. It doesn't collect every subclass of a shared base class.

| Python | Refinery |
| --- | --- |
| `int`, `float`, `str`, `bool` (optionally `\| None`) | `int`, `real`, `string`, `boolean` attributes; `None` is unknown |
| `Literal["a", "b"]` | `enum OwnerField { a, b }`, shared by fields with the same values |
| `Literal["tag"]` (a single value) | skipped: discriminator tags add nothing to the class |
| an `enum.Enum` subclass | an `enum` and a reference to its literals |
| `x: Model` / `x: Model \| None` / `x: list[Model]` | a reference: `Model[1] x` / `Model[0..1] x` / `Model[0..*] x` |
| `Field(min_length=1, max_length=3)` on a list | `[1..3]` (also enforced by Pydantic on construction) |
| `A \| B`, optionally with a `discriminator` | an abstract superclass that `A` and `B` extend, named `AOrB` or as given in `unions=` |
| Python subclassing | `extends` |

**Identity decides containment.** Each class is one of these:

- **Owned value** (a plain Pydantic model, the default): each instance is its own node, owned (`contains`) by the object whose field holds it. Reusing one instance in two places is a `containment` violation. Use `model_copy()`, or make the class an entity.
- **Entity, by key** (`ClassOptions(key=...)`): instances with equal keys are *one* node, so equal copies merge. Fields typed as the class are plain references. Nodes are named after the key when it's a string or integer.
- **Entity, by object** (subclass `RefineryNode`): instances are nodes by identity. Use this for hand-built graphs; see below.

**`ClassOptions`** configure classes from outside, so generated or third-party models can be used unchanged. Options given for a base class apply to its subclasses.

| Option | Purpose |
| --- | --- |
| `key` | entity identity, such as `lambda s: s.name` |
| `references={"field": Target}` | the field holds *keys* of `Target` objects: a keyed class, a list of them, or a union name |
| `exclude=[...]` | leave payload fields out |
| `node_name` | name nodes explicitly |
| `abstract` | declare the Refinery class `abstract` |
| `exists`, `certainty` | uncertainty, for `query()` |

Fields with no Refinery counterpart (`list[str]`, `dict`, …) raise `TypeError` by default. Pass `unsupported="skip"` to leave them out instead; they're listed in `metamodel.skipped`.

### Hand-built graphs with `RefineryNode`

When you build the graph in Python yourself, subclass `RefineryNode`. You get identity semantics, and markers for Refinery's containment and opposite references:

```python
from pyfinery import Container, Contains, Metamodel, RefineryNode

class Filesystem(RefineryNode):
    root: Annotated[Dir, Contains()]                      # contains Dir[1] root

class FSObject(RefineryNode):
    __abstract__ = True
    parent: Annotated[Dir | None, Container(opposite="contents")] = None

class Dir(FSObject):
    __node_id__ = "name"                                   # name nodes after this field
    name: str
    contents: Annotated[list[FSObject], Contains(opposite="parent", multiplicity="0..3")] = []

class File(FSObject):
    size: int = 0

metamodel = Metamodel(Filesystem, rules=["error pred emptyDir(Dir d) <-> !contents(d, _)."])
home = Dir(name="home", contents=[File(size=10)])
report = metamodel.validate([Filesystem(root=home), Dir(name="orphan")], refinery)
# containment: builtin::invalidContainer(orphan)
# rule: emptyDir(orphan)
```

- **Markers:** `Contains(opposite=..., multiplicity=...)`, `Container(opposite=...)` and `Ref(opposite=..., multiplicity=...)` go in `Annotated[...]`. `Ref(multiplicity=...)` is only checked by Refinery, unlike Pydantic's `min_length`, which is handy when you build graphs step by step.
- **Opposites:** you only need to set one side of an `opposite` pair. Refinery infers the other. `metamodel.link_opposites(objects)` fills in the missing sides on your Python objects too.

### Things to know

- **Reachability:** `validate()` and `query()` follow only *outgoing* references from the objects you pass. An object that nothing refers to, such as a child that only sets its own `parent`, must be passed too.
- **Timeouts:** `validate()` and `query()` run under the server's semantics timeout, which is 1 second by default and enough for about a thousand objects. Raise it with `RefineryServer(semantics_timeout=10_000)` or `REFINERY_SEMANTICS_TIMEOUT_MS`. `RefineryServer` warms the server up, so the first call doesn't time out while the JVM starts. See [Performance](#performance) for what makes calls slow.
- **Rule syntax** is Refinery's. See its [language reference](https://refinery.tools/learn/language/). Inside rules, features are written `Class::field`. Boolean attributes are used as literals (`Port::public(p)`), and other attributes are compared with operators (`Service::replicas(s) > 1`).

### Performance

Each `validate()` or `query()` call sends the whole problem. Refinery parses it and computes every relation over every object, both what must and what may hold. It doesn't limit the work to the predicates you report. With the default image, a `validate()` or `query()` over 3,000 objects takes about 2 seconds.

- **Keep helper predicates small.** A predicate with three parameters, such as `between(Resource r, Effect a, Effect b)`, can hold for hundreds of thousands of tuples over a few hundred objects, and Refinery computes them all. `validate()` and `query()` ask the server not to send back helper predicates: predicates declared in rules with a body (`<->`) that aren't `error` predicates and that `query()` doesn't report. Sending them used to take as long as computing them. Servers other than the default image ignore the request and send every relation. `result.model.tuple_counts()` lists the largest relations that were sent.
- **Pass fewer objects.** The number of objects matters more than the number of attribute values. Leave out fields that rules don't need with `exclude=[...]`.
- **Give every owned value its owner in `query()`.** To close the world, `query()` rules out the containers an object isn't in. When each owned value certainly has its owner, Refinery infers that by itself. An object passed without one makes pyfinery close every containment field that could hold it, which takes time proportional to the square of the number of objects.

## Calling Refinery directly

```python
from pyfinery import Refinery

with Refinery("http://localhost:8888/api/v1") as refinery:
    result = refinery.generate(Path("filesystem.problem"), scopes=["File=2..*"], random_seed=42)

print(result.source)                          # the generated model in Refinery syntax
print(result.json_.true_tuples("contents"))   # [('dir1', 'file1'), ...]
```

| Method | Returns |
| --- | --- |
| `generate(problem, *, scopes, random_seed, format, on_status)` | `GenerateResult` (`.json_`, `.source`) |
| `generate_many(problem, count, ...)` | `GenerateManyResult` (`.stop_reason`, `.models`) |
| `semantics(problem, *, format)` | `SemanticsResult` (`.issues`, `.json_`): the partial model |
| `concretize(problem, *, format)` | `ConcretizationResult`: unknowns resolved to false |
| `stream_generate(...)`, `stream_generate_many(...)` | an iterator of status updates, then the result |

- **Inputs:** problems are source strings or `pathlib.Path`s. Without a base URL, the client uses `$REFINERY_API_BASE`, or `http://localhost:8888/api/v1`.
- **Async:** `AsyncRefinery` has the same methods as coroutines and async iterators. `RefineryChat` and `AsyncRefineryChat` wrap Refinery's text-to-model chat service.
- **Scopes** are strings (`"node=10..20"`, `"File=5"`, `"Dir=2..*"`, `"Dir+=1..3"`) or `Scope(...)` objects.
- **Output:** `OutputFormats` turns the JSON or source output off, or keeps non-existing objects and shadow predicates. `JsonOutput.tuples(relation)` and `true_tuples(relation)` read a model with node names resolved.
- **Streaming and cancellation:** closing a stream early (`contextlib.closing`/`aclosing`) or cancelling an asyncio task disconnects, and the server stops working on the request.

### Managed Docker server

```python
from pyfinery import RefineryServer

with RefineryServer(model_generation_timeout=1200, semantics_timeout=5000, library_paths=["./lib"]) as server:
    with server.client() as refinery:
        ...
```

The container runs `ghcr.io/zacharyvmm/refinery:latest` unless you pass `image=...`, and pulls it only if it's missing or you pass `pull=True`. It listens on a random free port of `127.0.0.1` and is removed on exit. `library_paths` are mounted read-only, so problems can `import` the `.refinery` modules in them. Other [server settings](https://refinery.tools/learn/docker/#environmental-variables) go in `env={...}`. Pass `warmup=False` to skip the warm-up requests.

## Errors

All errors subclass `RefineryError`:

| Exception | Meaning |
| --- | --- |
| `InvalidProblemError` | the problem (or a rule) has syntax or validation errors; see `.issues` |
| `UnsatisfiableError` | no consistent model exists |
| `RefineryTimeoutError` (also a `TimeoutError`) | the server ran out of time |
| `BadRequestError` | malformed request; see `.details` |
| `ServerError` | internal server error |
| `RequestCancelledError` | the server cancelled the operation |
| `ProtocolError` | the response couldn't be understood |
| `ServerStartError` | `RefineryServer` failed to start |
| `ModelValidationError` | `ValidationReport.raise_for_violations()` found violations |

Mistakes in a metamodel's definition, such as an unsupported field or a one-sided `opposite`, raise `TypeError` when the `Metamodel` is built. Network failures are raised as `httpx` exceptions.

## Examples

```sh
uv run examples/validate_deployment.py    # Pydantic models from JSON: validate and query
uv run examples/validate_graph.py         # a hand-built RefineryNode graph
uv run examples/generate_filesystems.py   # model generation with the client
```

## Development

```sh
uv run pytest                              # unit tests (mocked HTTP)
PYFINERY_DOCKER_TESTS=1 uv run pytest      # plus end-to-end tests against a real server in Docker
```

## License and attribution

pyfinery is licensed under the [Eclipse Public License 2.0](LICENSE) (`EPL-2.0`), the same license as Refinery.

The client code is a Python port of Refinery's official JavaScript client, [`@tools.refinery/client`](https://github.com/graphs4value/refinery/tree/main/subprojects/client-js), which is Copyright (c) 2021-2025 [The Refinery Authors](https://github.com/graphs4value/refinery/blob/main/CONTRIBUTORS.md). The file system example problem comes from the [Refinery tutorial](https://refinery.tools/learn/tutorials/file-system). [NOTICE.md](NOTICE.md) lists the details, and every file has [REUSE](https://reuse.software)-style SPDX headers.

Refinery itself was developed by The Refinery Authors with the support of the 2022 Amazon Research Award. If you use it in research, please cite the [Refinery publications](https://github.com/graphs4value/refinery#related-publications), for example:

> K. Marussy, A. Ficsor, O. Semeráth, D. Varró: "Refinery: Graph Solver as a Service." *ICSE 2024 Demonstrations*. [doi:10.1145/3639478.3640045](https://doi.org/10.1145/3639478.3640045)

pyfinery isn't affiliated with or endorsed by the Refinery project.
