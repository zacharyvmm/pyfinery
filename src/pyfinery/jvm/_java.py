"""Finding the JDK the JVM runs on.

Refinery 0.3.0 is compiled for Java 25 (class file version 69), and the
helpers in ``java/`` are compiled on first use with its ``javac``, so a JDK
25 or later is needed, not only a runtime.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import JvmError

MINIMUM_VERSION = 25

_HOW_TO = (
    "pyfinery.jvm needs a JDK {minimum} or later. Install one (e.g. from https://adoptium.net), "
    "then set PYFINERY_JAVA_HOME or JAVA_HOME to it, or call pyfinery.jvm.configure(java_home=...)."
)


@dataclass(frozen=True, slots=True)
class Jdk:
    """A JDK: its home directory, its major version, and the files pyfinery uses in it."""

    home: Path
    version: int
    libjvm: Path
    javac: Path


def find_jdk(explicit: str | os.PathLike[str] | None = None) -> Jdk:
    """The JDK to use: ``explicit``, else ``$PYFINERY_JAVA_HOME``, else ``$JAVA_HOME``. Nothing is
    searched for: raises :class:`JvmError` if none is set, or if it isn't a JDK 25 or later.
    """
    if explicit is not None:
        return inspect_jdk(Path(explicit), "configure(java_home=...)")
    for variable in ("PYFINERY_JAVA_HOME", "JAVA_HOME"):
        if os.environ.get(variable):
            return inspect_jdk(Path(os.environ[variable]), f"${variable}")
    raise JvmError(
        "Neither PYFINERY_JAVA_HOME nor JAVA_HOME is set. "
        + _HOW_TO.format(minimum=MINIMUM_VERSION)
    )


def inspect_jdk(home: Path, source: str = "the JDK") -> Jdk:
    """The :class:`Jdk` at ``home``, checked: it exists, is recent enough, and has a JVM library and
    ``javac``."""
    if not home.is_dir():
        raise JvmError(
            f"{source} is {home}, which is not a directory. "
            + _HOW_TO.format(minimum=MINIMUM_VERSION)
        )
    version = read_version(home)
    if version is None:
        raise JvmError(
            f"{source} is {home}, which has no 'release' file: is it a JDK home directory?"
        )
    if version < MINIMUM_VERSION:
        raise JvmError(
            f"{source} is {home}, a Java {version}; Refinery needs Java {MINIMUM_VERSION} or "
            "later. " + _HOW_TO.format(minimum=MINIMUM_VERSION)
        )
    libjvm = _libjvm(home)
    if libjvm is None:
        raise JvmError(
            f"{source} is {home}, which has no JVM library "
            "(lib/server/libjvm.*, bin/server/jvm.dll)."
        )
    javac = home / "bin" / ("javac.exe" if sys.platform == "win32" else "javac")
    if not javac.exists():
        raise JvmError(
            f"{source} is {home}, which has no javac: it is a runtime, not a JDK. pyfinery "
            "compiles two small helpers with javac on first use. "
            + _HOW_TO.format(minimum=MINIMUM_VERSION)
        )
    return Jdk(home=home, version=version, libjvm=libjvm, javac=javac)


def read_version(home: Path) -> int | None:
    """The major Java version in ``home/release`` (``JAVA_VERSION="25.0.3"`` is 25), or None without
    one."""
    release = home / "release"
    if not release.is_file():
        return None
    for line in release.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("JAVA_VERSION="):
            value = line.partition("=")[2].strip().strip('"')
            try:
                return int(value.split(".")[0].split("-")[0].split("+")[0])
            except ValueError:
                return None
    return None


def _libjvm(home: Path) -> Path | None:
    for pattern in ("lib/server/libjvm.*", "bin/server/jvm.dll"):
        for path in sorted(home.glob(pattern)):
            if path.suffix in (".dylib", ".so", ".dll"):
                return path
    return None
