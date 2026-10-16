<!--
  SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>

  SPDX-License-Identifier: EPL-2.0
-->

# Notices for pyfinery

pyfinery is an unofficial Python client for [Refinery](https://refinery.tools). It isn't affiliated with or endorsed by the Refinery project.

## Copyright

- Copyright (c) 2021-2025 [The Refinery Authors](https://github.com/graphs4value/refinery/blob/main/CONTRIBUTORS.md) <https://refinery.tools/>
- Copyright (c) 2026 zacharyvmm

## License

This program and the accompanying materials are made available under the terms of the Eclipse Public License 2.0, which is available in [LICENSE](LICENSE) and at <https://www.eclipse.org/legal/epl-2.0/>.

SPDX-License-Identifier: EPL-2.0

## Third-party content

### Refinery JavaScript client (`@tools.refinery/client`)

- Source: <https://github.com/graphs4value/refinery/tree/main/subprojects/client-js>, ported from commit [`7470710`](https://github.com/graphs4value/refinery/tree/747071004a969288113bb09d85299bbd28c9f9df/subprojects/client-js)
- Copyright (c) 2021-2025 The Refinery Authors
- License: EPL-2.0

`src/pyfinery/client.py`, `src/pyfinery/models.py`, `src/pyfinery/errors.py` and `src/pyfinery/_protocol.py` are Python ports of this client. They were translated from TypeScript to Python and adapted to Python idioms (pydantic models instead of zod schemas, httpx instead of `fetch`, iterators and callbacks instead of the `onStatus` modes).

### Refinery file system tutorial

- Source: <https://refinery.tools/learn/tutorials/file-system>
- Copyright (c) 2023-2024 The Refinery Authors
- License: EPL-2.0

`examples/filesystem.problem` is the tutorial's example problem.

### Refinery documentation

The API behaviour, Docker image (`ghcr.io/graphs4value/refinery`) and server environment variables used by `src/pyfinery/server.py` are described in the [Refinery documentation](https://refinery.tools/learn/docker/). This project doesn't redistribute the Docker image.
