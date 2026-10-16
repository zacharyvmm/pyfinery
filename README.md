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

The container listens on a random free port of `127.0.0.1` and is removed on exit. `library_paths` are mounted read-only, so problems can `import` the `.refinery` modules in them. Any other [server setting](https://refinery.tools/learn/docker/#environmental-variables) can be passed via `env={...}`.

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

Network failures are raised as `httpx` exceptions.

Known server behaviour: in Refinery 0.3, `semantics`/`concretize` on a problem with errors returns `timeout` rather than `invalidProblem`. Use `generate` (or the web editor) to see the validation issues.

## Development

```sh
uv run pytest                              # unit tests (mocked HTTP)
PYFINERY_DOCKER_TESTS=1 uv run pytest      # plus end-to-end tests against a real server in Docker
uv run examples/generate_filesystems.py
```

## License and attribution

pyfinery is licensed under the [Eclipse Public License 2.0](LICENSE) (`EPL-2.0`), the same license as Refinery.

The client code is a Python port of Refinery's official JavaScript client, [`@tools.refinery/client`](https://github.com/graphs4value/refinery/tree/main/subprojects/client-js), which is Copyright (c) 2021-2025 [The Refinery Authors](https://github.com/graphs4value/refinery/blob/main/CONTRIBUTORS.md). The example problem comes from the [Refinery file system tutorial](https://refinery.tools/learn/tutorials/file-system). [NOTICE.md](NOTICE.md) lists the details, and every file has [REUSE](https://reuse.software)-style SPDX headers.

Refinery itself was developed by The Refinery Authors with the support of the 2022 Amazon Research Award. If you use it in research, please cite the [Refinery publications](https://github.com/graphs4value/refinery#related-publications), for example:

> K. Marussy, A. Ficsor, O. Semeráth, D. Varró: "Refinery: Graph Solver as a Service." *ICSE 2024 Demonstrations*. [doi:10.1145/3639478.3640045](https://doi.org/10.1145/3639478.3640045)

pyfinery isn't affiliated with or endorsed by the Refinery project.
