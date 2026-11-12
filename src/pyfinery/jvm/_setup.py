"""What is prepared in the cache directory once, besides the jars: the helpers compiled, and
OR-tools' native libraries unpacked."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path

from ._jars import Jar
from ._java import Jdk
from .errors import JvmError

logger = logging.getLogger("pyfinery.jvm")

_SOURCES = ("Bulk.java", "Natives.java")
_DONE = ".complete"


def helper_sources() -> dict[str, bytes]:
    """The helpers' Java sources shipped with pyfinery, by file name."""
    package = resources.files(__package__).joinpath("java", "pyfinery", "jvm")
    return {name: package.joinpath(name).read_bytes() for name in _SOURCES}


def compile_helpers(jdk: Jdk, classpath: list[Path], cache_dir: Path) -> Path:
    """The directory of the helpers' classes, compiled with ``jdk``'s ``javac`` the first time.

    Kept under a hash of their sources, the Java version and the class path, so that upgrading
    pyfinery, Refinery or the JDK compiles them again.
    """
    sources = helper_sources()
    key = hashlib.sha256()
    for name, text in sorted(sources.items()):
        key.update(name.encode() + b"\0" + text + b"\0")
    key.update(f"java {jdk.version}\0".encode())
    for entry in classpath:
        # Jars sit in a folder named after their hash: key on both.
        key.update(f"{entry.parent.name}/{entry.name}".encode() + b"\0")
    target = cache_dir / "helpers" / key.hexdigest()[:16]
    if (target / _DONE).exists():
        return target
    work = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    shutil.rmtree(work, ignore_errors=True)
    (work / "src").mkdir(parents=True)
    files = []
    for name, text in sources.items():
        path = work / "src" / name
        path.write_bytes(text)
        files.append(str(path))
    logger.info("Compiling pyfinery's Java helpers into %s", target)
    result = subprocess.run(
        [
            str(jdk.javac),
            "-nowarn",
            "--release",
            str(jdk.version),
            "-cp",
            os.pathsep.join(map(str, classpath)),
            "-d",
            str(work / "classes"),
            *files,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        raise JvmError(f"javac could not compile pyfinery's helpers:\n{result.stderr.strip()}")
    (work / "classes" / _DONE).touch()
    _publish(work / "classes", target)
    shutil.rmtree(work, ignore_errors=True)
    return target


def unpack_natives(jar: Jar, jar_path: Path, cache_dir: Path) -> Path:
    """The directory OR-tools' native libraries of ``jar`` are unpacked into, the first time it is
    needed."""
    target = cache_dir / "natives" / jar.sha256[:16]
    if (target / _DONE).exists():
        return target
    work = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    logger.info("Unpacking OR-tools' native libraries into %s", target)
    folder = f"ortools-{jar.platform}/"
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(jar_path) as archive:
            for member in archive.infolist():
                # The libraries sit in one folder named after the platform; nothing else is needed.
                if member.is_dir() or not member.filename.startswith(folder):
                    continue
                name = member.filename[len(folder) :]
                if not name or "/" in name or "\\" in name or name in (".", "..") or name in seen:
                    raise JvmError(f"{jar_path} has an unexpected entry {member.filename!r}")
                seen.add(name)
                with archive.open(member) as source, (work / name).open("wb") as out:
                    shutil.copyfileobj(source, out)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if not seen:
        shutil.rmtree(work, ignore_errors=True)
        raise JvmError(f"{jar_path} has no native libraries in {folder}")
    (work / _DONE).touch()
    _publish(work, target)
    return target


# What OR-tools' Loader (9.15) loads on Windows before jniortools, in its order: Windows does not
# look beside a DLL for the DLLs it needs. Elsewhere jniortools finds them through its run path.
_WINDOWS_ORDER = (
    "zlib1",
    "bz2",
    "abseil_dll",
    "re2",
    "libutf8_validity",
    "libprotobuf",
    "highs",
    "libscip",
    "ortools",
)


def ortools_libraries(natives: Path, platform: str) -> list[Path]:
    """The native libraries in ``natives`` to load, in order, for OR-tools on ``platform``."""
    jni = [
        path
        for path in sorted(natives.iterdir())
        if path.name.startswith(("libjniortools", "jniortools"))
        and path.suffix in (".dylib", ".so", ".dll")
    ]
    if not jni:
        raise JvmError(
            f"{natives} has no OR-tools JNI library (libjniortools.*); "
            "delete it to unpack it again."
        )
    if not platform.startswith("win32"):
        return jni[:1]
    needed = [natives / f"{name}.dll" for name in _WINDOWS_ORDER]
    missing = [path.name for path in needed if not path.exists()]
    if missing:
        raise JvmError(f"{natives} lacks {', '.join(missing)}; delete it to unpack it again.")
    return [*needed, jni[0]]


def _publish(work: Path, target: Path) -> None:
    """Move ``work`` to ``target``; if another process got there first, keep theirs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        work.replace(target)
    except OSError:
        if not (target / _DONE).exists():
            shutil.rmtree(target, ignore_errors=True)
            work.replace(target)
        else:
            shutil.rmtree(work, ignore_errors=True)
