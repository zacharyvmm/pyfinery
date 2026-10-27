# SPDX-FileCopyrightText: 2021-2025 The Refinery Authors <https://refinery.tools/>
# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Synchronous and asynchronous clients for the Refinery HTTP API.

Ported from ``Refinery.ts``, ``GenericRefinery.ts`` and ``chat/RefineryChat.ts`` of
the official JavaScript client ``@tools.refinery/client``:
https://github.com/graphs4value/refinery/tree/main/subprojects/client-js/src
"""

from __future__ import annotations

import atexit
import inspect
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from typing import Any, Self

import httpx
from pydantic import BaseModel

from ._protocol import (
    DEFAULT_TIMEOUT,
    ProblemSource,
    SSEDecoder,
    StatusParser,
    error_from_response,
    normalize_base_url,
    parse_event_data,
    parse_generate_many_status,
    read_source,
    unexpected_end_of_stream,
)
from .models import (
    ChatMessage,
    ConcretizationResult,
    GenerateManyRequest,
    GenerateManyResult,
    GenerateRequest,
    GenerateResult,
    GenerateStatus,
    OutputFormats,
    ProblemInput,
    Scope,
    SemanticsRequest,
    SemanticsResult,
    TextToModelRequest,
    TextToModelResult,
)

DEFAULT_BASE_URL = "http://localhost:8888/api/v1"
"""API base of a local server, e.g., ``docker run -p 8888:8888 ghcr.io/zacharyvmm/refinery``."""

BASE_URL_ENV = "REFINERY_API_BASE"

_shared_lock = threading.Lock()
_shared: Refinery | None = None

type ScopeLike = Scope | str
type StatusCallback[S] = Callable[[S], object]
type AsyncStatusCallback[S] = Callable[[S], object | Awaitable[object]]


def _scopes(scopes: Iterable[ScopeLike] | None) -> list[Scope] | None:
    if scopes is None:
        return None
    return [Scope.parse(s) if isinstance(s, str) else s for s in scopes]


def _generate_request(
    problem: ProblemSource,
    scopes: Iterable[ScopeLike] | None,
    random_seed: int,
    format: OutputFormats | None,
) -> GenerateRequest:
    return GenerateRequest(
        input=ProblemInput(source=read_source(problem)),
        format=format or OutputFormats(),
        scopes=_scopes(scopes),
        random_seed=random_seed,
    )


def _generate_many_request(
    problem: ProblemSource,
    count: int,
    scopes: Iterable[ScopeLike] | None,
    random_seed: int,
    format: OutputFormats | None,
) -> GenerateManyRequest:
    request = _generate_request(problem, scopes, random_seed, format)
    return GenerateManyRequest(**dict(request), count=count)


def _semantics_request(problem: ProblemSource, format: OutputFormats | None) -> SemanticsRequest:
    return SemanticsRequest(
        input=ProblemInput(source=read_source(problem)), format=format or OutputFormats()
    )


def _text_to_model_request(
    metamodel: ProblemSource, text: str, format: OutputFormats | None
) -> TextToModelRequest:
    return TextToModelRequest(
        metamodel=ProblemInput(source=read_source(metamodel)),
        text=text,
        format=format or OutputFormats(),
    )


def _with_models(result: GenerateManyResult, models: list[GenerateResult]) -> GenerateManyResult:
    return result.model_copy(update={"models": models})


class _BaseClient:
    def __init__(
        self,
        base_url: str,
        headers: Mapping[str, str] | None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self._headers = dict(headers or {})

    def _request_args(self, endpoint: str, request: BaseModel) -> dict[str, Any]:
        return {
            "method": "POST",
            "url": f"{self.base_url}/{endpoint}",
            "json": request.model_dump(mode="json", exclude_none=True),
            "headers": {**self._headers, "Accept": "text/event-stream"},
        }


# ---------------------------------------------------------------------------
# Synchronous API
# ---------------------------------------------------------------------------


class _SyncClient(_BaseClient):
    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(base_url, headers)
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _stream(
        self,
        endpoint: str,
        request: BaseModel,
        success_model: type[BaseModel],
        parse_status: StatusParser | None,
    ) -> Iterator[Any]:
        """Yield status updates followed by the final result.

        Closing the iterator early disconnects, which cancels the server-side task.
        """
        result: Any = None
        with self._http.stream(**self._request_args(endpoint, request)) as response:
            if not response.is_success:
                response.read()
                raise error_from_response(response)
            decoder = SSEDecoder()
            for line in response.iter_lines():
                data = decoder.feed(line)
                if data is None:
                    continue
                event = parse_event_data(data, success_model, parse_status)
                if event is None:
                    continue
                if event.kind == "success":
                    result = event.value
                    break
                yield event.value
            else:
                data = decoder.flush()
                if data is not None:
                    event = parse_event_data(data, success_model, parse_status)
                    if event is not None and event.kind == "success":
                        result = event.value
        # Yield the result only after the connection was released.
        if result is None:
            raise unexpected_end_of_stream()
        yield result

    def _run(
        self,
        stream: Iterator[Any],
        on_status: StatusCallback[Any] | None,
    ) -> Any:
        last: Any = None
        for item in stream:
            if last is not None and on_status is not None:
                on_status(last)
            last = item
        return last


class Refinery(_SyncClient):
    """Client for a Refinery server.

    >>> with Refinery("http://localhost:8888/api/v1") as refinery:
    ...     result = refinery.generate(problem_source, scopes=["node=10..20"])
    ...     print(result.source)

    ``base_url`` defaults to the ``REFINERY_API_BASE`` environment variable or
    a local server on port 8888.

    All methods take the problem either as source text (``str``) or as a path
    to a ``.problem`` file (:class:`os.PathLike`), and raise a subclass of
    :class:`~pyfinery.RefineryError` if the server reports an error (e.g.,
    :class:`~pyfinery.UnsatisfiableError`).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
            headers=headers,
            timeout=timeout,
            http_client=http_client,
        )

    @classmethod
    def shared(
        cls,
        *,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        **server_options: Any,
    ) -> Refinery:
        """A client shared by the whole process, for code that just needs a server.

        >>> Refinery.shared(semantics_timeout=600_000, timeout=None).semantics(source)

        The server at ``REFINERY_API_BASE`` if it is set; otherwise a
        :class:`~pyfinery.RefineryServer` started with ``server_options`` (e.g.,
        ``image=...``, ``semantics_timeout=...``) the first time it is needed,
        and stopped when the process exits. Set ``REFINERY_API_BASE`` to have
        processes, such as parallel test workers, share one server.

        Every call returns the same client, from any thread. The first call's
        arguments win: later calls ignore theirs. Don't close the client.
        """
        global _shared
        with _shared_lock:
            if _shared is None:
                base_url = os.environ.get(BASE_URL_ENV)
                if not base_url:
                    from .server import RefineryServer  # which imports this module

                    server = RefineryServer(**server_options).start()
                    atexit.register(server.stop)
                    base_url = server.base_url
                _shared = Refinery(base_url, timeout=timeout)
            return _shared

    def stream_generate(
        self,
        problem: ProblemSource,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
    ) -> Iterator[GenerateStatus | GenerateResult]:
        """Generate a model, yielding progress messages and finally the model."""
        request = _generate_request(problem, scopes, random_seed, format)
        return self._stream("generate", request, GenerateResult, GenerateStatus.model_validate)

    def generate(
        self,
        problem: ProblemSource,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
        on_status: StatusCallback[GenerateStatus] | None = None,
    ) -> GenerateResult:
        """Generate a consistent concrete model from a partial model.

        ``scopes`` add (or with ``Scope(override=True)`` replace) ``scope``
        declarations, e.g. ``["node=10..20", "File=3..*"]``. Generation is
        deterministic for a given ``random_seed``.
        """
        stream = self.stream_generate(
            problem, scopes=scopes, random_seed=random_seed, format=format
        )
        return self._run(stream, on_status)

    def stream_generate_many(
        self,
        problem: ProblemSource,
        count: int,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
    ) -> Iterator[GenerateStatus | GenerateResult | GenerateManyResult]:
        """Generate up to ``count`` models, yielding each model as it is found.

        Yields :class:`GenerateStatus` progress messages, a
        :class:`GenerateResult` per model, and finally a
        :class:`GenerateManyResult` with the stop reason (its ``models`` list
        is left empty when streaming).
        """
        request = _generate_many_request(problem, count, scopes, random_seed, format)
        return self._stream("generateMany", request, GenerateManyResult, parse_generate_many_status)

    def generate_many(
        self,
        problem: ProblemSource,
        count: int,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
        on_status: StatusCallback[GenerateStatus | GenerateResult] | None = None,
    ) -> GenerateManyResult:
        """Generate up to ``count`` distinct models.

        Fewer models are returned if the server runs out of solutions or time;
        check ``stop_reason``.
        """
        models: list[GenerateResult] = []

        def collect(status: GenerateStatus | GenerateResult) -> None:
            if isinstance(status, GenerateResult):
                models.append(status)
            if on_status is not None:
                on_status(status)

        stream = self.stream_generate_many(
            problem, count, scopes=scopes, random_seed=random_seed, format=format
        )
        return _with_models(self._run(stream, collect), models)

    def semantics(
        self, problem: ProblemSource, *, format: OutputFormats | None = None
    ) -> SemanticsResult:
        """Compute the partial model described by the problem, without generation."""
        request = _semantics_request(problem, format)
        return self._run(self._stream("semantics", request, SemanticsResult, None), None)

    def concretize(
        self, problem: ProblemSource, *, format: OutputFormats | None = None
    ) -> ConcretizationResult:
        """Turn the partial model into a concrete one by resolving unknowns to false."""
        request = _semantics_request(problem, format)
        return self._run(self._stream("concretize", request, ConcretizationResult, None), None)


class RefineryChat(_SyncClient):
    """Client for Refinery's text-to-model chat service (``chatURL`` of the server)."""

    def stream_text_to_model(
        self,
        metamodel: ProblemSource,
        text: str,
        *,
        format: OutputFormats | None = None,
    ) -> Iterator[ChatMessage | TextToModelResult]:
        request = _text_to_model_request(metamodel, text, format)
        return self._stream("textToModel", request, TextToModelResult, ChatMessage.model_validate)

    def text_to_model(
        self,
        metamodel: ProblemSource,
        text: str,
        *,
        format: OutputFormats | None = None,
        on_status: StatusCallback[ChatMessage] | None = None,
    ) -> TextToModelResult:
        """Build an instance model of ``metamodel`` from a natural language description."""
        return self._run(self.stream_text_to_model(metamodel, text, format=format), on_status)


# ---------------------------------------------------------------------------
# Asynchronous API
# ---------------------------------------------------------------------------


class _AsyncClient(_BaseClient):
    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url, headers)
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _stream(
        self,
        endpoint: str,
        request: BaseModel,
        success_model: type[BaseModel],
        parse_status: StatusParser | None,
    ) -> AsyncIterator[Any]:
        result: Any = None
        async with self._http.stream(**self._request_args(endpoint, request)) as response:
            if not response.is_success:
                await response.aread()
                raise error_from_response(response)
            decoder = SSEDecoder()
            async for line in response.aiter_lines():
                data = decoder.feed(line)
                if data is None:
                    continue
                event = parse_event_data(data, success_model, parse_status)
                if event is None:
                    continue
                if event.kind == "success":
                    result = event.value
                    break
                yield event.value
            else:
                data = decoder.flush()
                if data is not None:
                    event = parse_event_data(data, success_model, parse_status)
                    if event is not None and event.kind == "success":
                        result = event.value
        if result is None:
            raise unexpected_end_of_stream()
        yield result

    async def _run(
        self,
        stream: AsyncIterator[Any],
        on_status: AsyncStatusCallback[Any] | None,
    ) -> Any:
        last: Any = None
        async for item in stream:
            if last is not None and on_status is not None:
                outcome = on_status(last)
                if inspect.isawaitable(outcome):
                    await outcome
            last = item
        return last


class AsyncRefinery(_AsyncClient):
    """Asynchronous variant of :class:`Refinery` with the same methods.

    Cancelling the awaiting task (or closing a ``stream_*`` iterator)
    disconnects, which cancels the server-side task. Use
    :func:`contextlib.aclosing` when breaking out of a ``stream_*`` loop early.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
            headers=headers,
            timeout=timeout,
            http_client=http_client,
        )

    def stream_generate(
        self,
        problem: ProblemSource,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
    ) -> AsyncIterator[GenerateStatus | GenerateResult]:
        request = _generate_request(problem, scopes, random_seed, format)
        return self._stream("generate", request, GenerateResult, GenerateStatus.model_validate)

    async def generate(
        self,
        problem: ProblemSource,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
        on_status: AsyncStatusCallback[GenerateStatus] | None = None,
    ) -> GenerateResult:
        stream = self.stream_generate(
            problem, scopes=scopes, random_seed=random_seed, format=format
        )
        return await self._run(stream, on_status)

    def stream_generate_many(
        self,
        problem: ProblemSource,
        count: int,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
    ) -> AsyncIterator[GenerateStatus | GenerateResult | GenerateManyResult]:
        request = _generate_many_request(problem, count, scopes, random_seed, format)
        return self._stream("generateMany", request, GenerateManyResult, parse_generate_many_status)

    async def generate_many(
        self,
        problem: ProblemSource,
        count: int,
        *,
        scopes: Iterable[ScopeLike] | None = None,
        random_seed: int = 1,
        format: OutputFormats | None = None,
        on_status: AsyncStatusCallback[GenerateStatus | GenerateResult] | None = None,
    ) -> GenerateManyResult:
        models: list[GenerateResult] = []

        async def collect(status: GenerateStatus | GenerateResult) -> None:
            if isinstance(status, GenerateResult):
                models.append(status)
            if on_status is not None:
                outcome = on_status(status)
                if inspect.isawaitable(outcome):
                    await outcome

        stream = self.stream_generate_many(
            problem, count, scopes=scopes, random_seed=random_seed, format=format
        )
        return _with_models(await self._run(stream, collect), models)

    async def semantics(
        self, problem: ProblemSource, *, format: OutputFormats | None = None
    ) -> SemanticsResult:
        request = _semantics_request(problem, format)
        return await self._run(self._stream("semantics", request, SemanticsResult, None), None)

    async def concretize(
        self, problem: ProblemSource, *, format: OutputFormats | None = None
    ) -> ConcretizationResult:
        request = _semantics_request(problem, format)
        return await self._run(
            self._stream("concretize", request, ConcretizationResult, None), None
        )


class AsyncRefineryChat(_AsyncClient):
    """Asynchronous variant of :class:`RefineryChat`."""

    def stream_text_to_model(
        self,
        metamodel: ProblemSource,
        text: str,
        *,
        format: OutputFormats | None = None,
    ) -> AsyncIterator[ChatMessage | TextToModelResult]:
        request = _text_to_model_request(metamodel, text, format)
        return self._stream("textToModel", request, TextToModelResult, ChatMessage.model_validate)

    async def text_to_model(
        self,
        metamodel: ProblemSource,
        text: str,
        *,
        format: OutputFormats | None = None,
        on_status: AsyncStatusCallback[ChatMessage] | None = None,
    ) -> TextToModelResult:
        return await self._run(self.stream_text_to_model(metamodel, text, format=format), on_status)
