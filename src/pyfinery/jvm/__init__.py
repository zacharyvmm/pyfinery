"""Refinery in this process: its Java library, driven through JPype.

Install the ``jvm`` extra (``pyfinery[jvm]``) and a JDK 25 or later. The
first use fetches Refinery's jars from Maven Central into a cache directory
and checks them against the hashes pyfinery pins; later uses start from the
cache.

A process has one JVM, and what it can do is fixed when it starts: by
default, it validates and queries. To generate models too, say so before
anything starts it::

    import pyfinery.jvm
    pyfinery.jvm.configure(generation=True)

:func:`start` starts it explicitly; the engines that need it start it
themselves on first use.

To validate and query on Refinery's own matcher, pass a :class:`JvmEngine`
where :meth:`~pyfinery.Metamodel.validate` takes a client; to validate a graph
as it changes, keep it in a :class:`Session`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ._jars import FEATURES, Feature, default_cache_dir, fetch, host_platform, manifest
from ._java import MINIMUM_VERSION, Jdk, find_jdk
from ._setup import compile_helpers, ortools_libraries, unpack_natives
from .errors import JvmError, UnsupportedRuleError

__all__ = [
    "Changes",
    "Checkpoint",
    "JvmEngine",
    "JvmError",
    "MINIMUM_VERSION",
    "Session",
    "Settings",
    "Status",
    "UnsupportedRuleError",
    "configure",
    "helpers",
    "require",
    "settings",
    "start",
    "started",
    "status",
]

logger = logging.getLogger("pyfinery.jvm")

# Options for every JVM pyfinery starts. Refinery recurses deeply on large queries; JPype and Guice
# use APIs newer JDKs warn about on standard error.
_JVM_OPTIONS = (
    "-Xss16m",
    "--enable-native-access=ALL-UNNAMED",
    "--sun-misc-unsafe-memory-access=allow",
)


@dataclass(frozen=True, slots=True)
class Settings:
    """How the JVM is to be started (see :func:`configure`)."""

    generation: bool = False
    java_home: Path | None = None
    cache_dir: Path | None = None
    jvm_options: tuple[str, ...] = ()

    @property
    def features(self) -> tuple[Feature, ...]:
        return ("validation", "generation") if self.generation else ("validation",)


@dataclass(frozen=True, slots=True)
class Status:
    """The running JVM: what it was started with and how long that took."""

    settings: Settings
    jdk: Jdk
    cache_dir: Path
    classpath: tuple[Path, ...]
    seconds: dict[str, float] = field(default_factory=dict)
    # Why generation can't be used although it was asked for, e.g. OR-tools failed to load.
    generation_error: JvmError | None = None

    @property
    def generation(self) -> bool:
        """Whether this JVM can generate."""
        return self.settings.generation and self.generation_error is None


_lock = threading.RLock()
_settings = Settings()
_status: Status | None = None
_helpers: dict[str, Any] = {}


def configure(
    *,
    generation: bool | None = None,
    java_home: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    jvm_options: tuple[str, ...] | list[str] | None = None,
) -> Settings:
    """Set how the JVM will start; arguments left out keep their value. Returns the settings.

    :param generation: also load what model generation needs (OR-tools, about 57 MB
        more to fetch once). A JVM started without it cannot generate.
    :param java_home: the JDK to run on; by default ``$PYFINERY_JAVA_HOME``, else
        ``$JAVA_HOME``, else (on macOS) the one ``/usr/libexec/java_home`` finds.
    :param cache_dir: where jars, compiled helpers and native libraries are kept; by
        default ``$PYFINERY_CACHE_DIR``, else the platform's user cache directory.
    :param jvm_options: extra options for the JVM, e.g. ``["-Xmx4g"]``.

    Raises :class:`JvmError` once the JVM has started with other settings.
    """
    global _settings
    with _lock:
        changes: dict[str, Any] = {}
        if generation is not None:
            changes["generation"] = generation
        if java_home is not None:
            changes["java_home"] = Path(java_home)
        if cache_dir is not None:
            changes["cache_dir"] = Path(cache_dir)
        if jvm_options is not None:
            changes["jvm_options"] = tuple(jvm_options)
        wanted = replace(_settings, **changes)
        if _status is not None and wanted != _status.settings:
            raise JvmError(
                "The JVM has already started, and its settings can't change in this process"
                + (
                    ": it was started without generation"
                    if wanted.generation and not _status.generation
                    else ""
                )
                + ". Call pyfinery.jvm.configure(...) before anything uses the JVM."
            )
        _settings = wanted
        return _settings


def settings() -> Settings:
    """The settings the JVM starts, or started, with."""
    return _status.settings if _status is not None else _settings


def started() -> bool:
    """Whether the JVM runs in this process (started by pyfinery or by anything else using
    JPype)."""
    try:
        jpype = _jpype()
    except JvmError:
        return False  # without JPype, nothing runs a JVM here
    return jpype.isJVMStarted()


def status() -> Status | None:
    """What pyfinery started the JVM with, or None if pyfinery hasn't started it."""
    return _status


def start() -> Status:
    """Start the JVM with the current settings, fetching and preparing what it needs first; if it
    runs, return.

    Raises :class:`JvmError` if the JDK, the jars or the helpers can't be had, or if the
    JVM was started by something other than pyfinery.
    """
    global _status
    with _lock:
        if _status is not None:
            return _status
        jpype = _jpype()
        if jpype.isJVMStarted():
            raise JvmError(
                "A JVM was started in this process by something other than pyfinery, "
                "without Refinery's jars. "
                "A process has one JVM: let pyfinery start it."
            )
        chosen = _settings
        seconds: dict[str, float] = {}
        clock = time.perf_counter()
        jdk = find_jdk(chosen.java_home)
        cache_dir = chosen.cache_dir or default_cache_dir()
        jars: dict[str, Any] = {}
        for feature in chosen.features:
            for jar in manifest().jars(feature):
                jars.setdefault(jar.sha256, jar)
        paths = fetch(list(jars.values()), cache_dir)
        seconds["fetch"] = time.perf_counter() - clock

        clock = time.perf_counter()
        helper_classes = compile_helpers(jdk, paths, cache_dir)
        natives = None
        if chosen.generation:
            native_jar = next(jar for jar in jars.values() if jar.platform == host_platform())
            natives = unpack_natives(native_jar, native_jar.path(cache_dir), cache_dir)
        seconds["prepare"] = time.perf_counter() - clock

        clock = time.perf_counter()
        classpath = (*paths, helper_classes)
        try:
            jpype.startJVM(
                str(jdk.libjvm),
                *_JVM_OPTIONS,
                *chosen.jvm_options,
                classpath=[str(entry) for entry in classpath],
                convertStrings=False,
            )
        except (OSError, RuntimeError) as error:
            raise JvmError(f"The JVM at {jdk.libjvm} did not start: {error}") from error
        seconds["start"] = time.perf_counter() - clock

        # The JVM runs from here on, whatever happens next: record it, so that a failure below
        # leaves validation usable and is reported again to whoever asks to generate.
        _status = Status(chosen, jdk, cache_dir, classpath, seconds)
        if natives is not None:
            clock = time.perf_counter()
            try:
                libraries = ortools_libraries(natives, host_platform())
                paths = jpype.JArray(jpype.JString)([str(path) for path in libraries])
                jpype.JClass("pyfinery.jvm.Natives").loadOrTools(paths)
            except Exception as error:  # a JvmError, or a Java exception from System.load
                failure = (
                    error
                    if isinstance(error, JvmError)
                    else JvmError(f"OR-tools' native libraries in {natives} did not load: {error}")
                )
                _status = replace(_status, generation_error=failure)
                raise failure from error
            seconds["natives"] = time.perf_counter() - clock
        logger.info(
            "Started Refinery %s on Java %d (%s) in %.2f s",
            manifest().refinery,
            jdk.version,
            "+".join(chosen.features),
            sum(seconds.values()),
        )
        return _status


def helpers() -> SimpleNamespace:
    """The helper classes, ``Bulk`` and ``Natives``, starting the JVM if it hasn't started."""
    start()
    if not _helpers:
        jpype = _jpype()
        _helpers.update(
            Bulk=jpype.JClass("pyfinery.jvm.Bulk"), Natives=jpype.JClass("pyfinery.jvm.Natives")
        )
    return SimpleNamespace(**_helpers)


def require(feature: Feature) -> Status:
    """Start the JVM with ``feature`` if it hasn't started, or check it was started with it; for
    engines to call before using it."""
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}")
    global _settings
    with _lock:
        if feature == "generation" and _status is None:
            # Not started yet: start it with what is asked for, and if that fails before the
            # JVM runs, leave the settings as they were, so that validation still can start.
            before = _settings
            configure(generation=True)
            try:
                running = start()
            except JvmError:
                if _status is None:
                    _settings = before
                raise
        else:
            running = start()
        if feature == "generation" and running.generation_error is not None:
            raise running.generation_error
        if feature not in running.settings.features:
            raise JvmError(
                "This process's JVM was started without generation, and a JVM can't be restarted. "
                "Call pyfinery.jvm.configure(generation=True) before anything uses the JVM."
            )
        return running


def _jpype() -> Any:
    try:
        import jpype
    except ImportError as error:
        raise JvmError(
            "pyfinery.jvm needs JPype: install the jvm extra, e.g. pip install 'pyfinery[jvm]'."
        ) from error
    return jpype


from .engine import JvmEngine  # noqa: E402  (after the functions it uses)
from .session import Changes, Checkpoint, Session  # noqa: E402
