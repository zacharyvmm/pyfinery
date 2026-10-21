# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Run a local Refinery server in Docker."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import httpx

from .client import AsyncRefinery, Refinery
from .errors import InvalidProblemError, RefineryError, RefineryTimeoutError

DEFAULT_IMAGE = "ghcr.io/zacharyvmm/refinery:latest"
CONTAINER_PORT = 8888
LIBRARY_MOUNT_ROOT = "/refinery-library"

# Exercises the code paths of semantics, concretization and validation errors so
# the JVM has loaded and compiled them before the first real request.
_WARMUP_PROBLEM = """\
enum Color { red, green }
abstract class Item { container Box box opposite items }
class Box extends Item {
    contains Item[0..3] items opposite box
    Color[0..1] color
    string label
    int size
    real weight
    boolean open
}
error pred emptyBox(Box b) <-> !items(b, _).
Box(b1). Box(b2). Box::items(b1, b2). Box::color(b1, Color::red).
Box::label(b1): "warm". Box::size(b1): 1. Box::weight(b1): 1.5. Box::open(b1): true.
"""
_WARMUP_INVALID_PROBLEM = "class Box.\nmissing(b1).\n"
_WARMUP_MAX_ROUNDS = 10


class ServerStartError(RefineryError):
    """The Refinery container failed to start or become healthy."""

    result = "serverStartError"


class RefineryServer:
    """A Refinery server running in a Docker container.

    >>> with RefineryServer(model_generation_timeout=1200) as server:
    ...     with server.client() as refinery:
    ...         refinery.generate(source)

    The container is started with ``--rm`` and is removed when the server is
    stopped. By default, it listens on a random free port of ``127.0.0.1``.

    :param library_paths: Host directories with ``.refinery`` modules to make
        importable from problems (mounted read-only, exposed via
        ``REFINERY_LIBRARY_PATH``).
    :param model_generation_timeout: Seconds before model generation times out
        (``REFINERY_MODEL_GENERATION_TIMEOUT_SEC``, server default 600).
    :param semantics_timeout: Milliseconds before partial model semantics
        times out (``REFINERY_SEMANTICS_TIMEOUT_MS``, server default 1000).
    :param threads: Number of concurrent model generation tasks
        (``REFINERY_MODEL_GENERATION_THREAD_COUNT``).
    :param env: Additional environment variables for the container, see
        https://refinery.tools/learn/docker/#environmental-variables
    :param warmup: Send a few small requests after startup, retrying until they
        finish within the semantics timeout. Without this, the first
        ``semantics``/``concretize`` requests on a fresh server (including
        graph validation) may fail with ``RefineryTimeoutError`` while the JVM
        warms up.
    """

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        port: int | None = None,
        host: str = "127.0.0.1",
        library_paths: Sequence[str | os.PathLike[str]] = (),
        model_generation_timeout: int | None = None,
        semantics_timeout: int | None = None,
        threads: int | None = None,
        env: Mapping[str, str] | None = None,
        pull: bool = False,
        startup_timeout: float = 120.0,
        docker: str = "docker",
        extra_args: Sequence[str] = (),
        warmup: bool = True,
    ) -> None:
        self.image = image
        self.host = host
        self.startup_timeout = startup_timeout
        self.warmup = warmup
        self._requested_port = port
        self._pull = pull
        self._docker = docker
        self._extra_args = list(extra_args)
        self._env = self._build_env(model_generation_timeout, semantics_timeout, threads, env)
        self._mounts = [Path(p).resolve() for p in library_paths]
        self.container_id: str | None = None
        self.port: int | None = None

    @staticmethod
    def _build_env(
        model_generation_timeout: int | None,
        semantics_timeout: int | None,
        threads: int | None,
        env: Mapping[str, str] | None,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        if model_generation_timeout is not None:
            result["REFINERY_MODEL_GENERATION_TIMEOUT_SEC"] = str(model_generation_timeout)
        if semantics_timeout is not None:
            result["REFINERY_SEMANTICS_TIMEOUT_MS"] = str(semantics_timeout)
        if threads is not None:
            result["REFINERY_MODEL_GENERATION_THREAD_COUNT"] = str(threads)
        result.update(env or {})
        return result

    @property
    def url(self) -> str:
        """Root URL of the server (also serves the web editor)."""
        if self.port is None:
            raise RuntimeError("Refinery server is not running")
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def base_url(self) -> str:
        """API base URL to pass to :class:`~pyfinery.Refinery`."""
        return f"{self.url}/api/v1"

    def client(self, **kwargs) -> Refinery:
        return Refinery(self.base_url, **kwargs)

    def async_client(self, **kwargs) -> AsyncRefinery:
        return AsyncRefinery(self.base_url, **kwargs)

    def _docker_cmd(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run([self._docker, *args], capture_output=True, text=True, check=check)

    def _run_args(self) -> list[str]:
        port = "" if self._requested_port is None else str(self._requested_port)
        args = ["run", "--detach", "--rm", "--publish", f"{self.host}:{port}:{CONTAINER_PORT}"]
        env = dict(self._env)
        if self._mounts:
            targets = [f"{LIBRARY_MOUNT_ROOT}/{i}" for i in range(len(self._mounts))]
            for source, target in zip(self._mounts, targets, strict=True):
                args += ["--volume", f"{source}:{target}:ro"]
            env.setdefault("REFINERY_LIBRARY_PATH", ":".join(targets))
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        return [*args, *self._extra_args, self.image]

    def start(self) -> Self:
        if self.container_id is not None:
            return self
        if shutil.which(self._docker) is None:
            raise ServerStartError(f"{self._docker!r} executable not found; is Docker installed?")
        for path in self._mounts:
            if not path.is_dir():
                raise ServerStartError(f"Library path is not a directory: {path}")
        if self._pull:
            self._checked("pull", self.image)
        self.container_id = self._checked(*self._run_args()).stdout.strip()
        try:
            deadline = time.monotonic() + self.startup_timeout
            self.port = self._published_port()
            self._wait_until_healthy(deadline)
            if self.warmup:
                self._warm_up(deadline)
        except BaseException:
            self.stop()
            raise
        return self

    def _checked(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return self._docker_cmd(*args)
        except subprocess.CalledProcessError as e:
            raise ServerStartError(f"docker {args[0]} failed: {e.stderr.strip()}") from e

    def _published_port(self) -> int:
        assert self.container_id is not None
        output = self._checked("port", self.container_id, f"{CONTAINER_PORT}/tcp").stdout
        # One line per binding, e.g. "127.0.0.1:49153".
        return int(output.splitlines()[0].rsplit(":", 1)[1])

    def _is_running(self) -> bool:
        assert self.container_id is not None
        inspect = self._docker_cmd(
            "inspect", "--format", "{{.State.Running}}", self.container_id, check=False
        )
        return inspect.returncode == 0 and inspect.stdout.strip() == "true"

    def logs(self) -> str:
        """Combined stdout and stderr of the container."""
        if self.container_id is None:
            return ""
        result = self._docker_cmd("logs", self.container_id, check=False)
        return result.stdout + result.stderr

    def _wait_until_healthy(self, deadline: float) -> None:
        with httpx.Client(timeout=5.0) as http:
            while time.monotonic() < deadline:
                try:
                    if http.get(f"{self.url}/health").is_success:
                        return
                except httpx.TransportError:
                    pass
                if not self._is_running():
                    raise ServerStartError(f"Refinery container exited:\n{self.logs()}")
                time.sleep(0.5)
        raise ServerStartError(
            f"Refinery did not become healthy within {self.startup_timeout}s:\n{self.logs()}"
        )

    def _warm_up(self, deadline: float) -> None:
        """Repeat warm-up requests until a full round finishes without a timeout."""
        with self.client() as refinery:
            for _ in range(_WARMUP_MAX_ROUNDS):
                if time.monotonic() >= deadline:
                    break
                try:
                    refinery.semantics(_WARMUP_PROBLEM)
                    refinery.concretize(_WARMUP_PROBLEM)
                    with contextlib.suppress(InvalidProblemError):
                        refinery.semantics(_WARMUP_INVALID_PROBLEM)
                except RefineryTimeoutError:
                    continue
                return
        raise ServerStartError(
            "Refinery kept timing out on small warm-up requests; consider raising "
            f"semantics_timeout:\n{self.logs()}"
        )

    def stop(self) -> None:
        if self.container_id is None:
            return
        self._docker_cmd("stop", "--time", "5", self.container_id, check=False)
        self.container_id = None
        self.port = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
