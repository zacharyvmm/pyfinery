"""Refinery's jars: which ones each feature needs, and fetching them on first use.

``jars.json`` pins every jar by its Maven Central URL and SHA-256 (it is
written by ``tools/jars``). A jar is downloaded the first time a feature
needs it, checked against its hash, and kept in the cache directory; jars of
native libraries are fetched for this platform only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Literal

from ..errors import JvmError

logger = logging.getLogger("pyfinery.jvm")

type Feature = Literal["validation", "generation"]
FEATURES: tuple[Feature, ...] = ("validation", "generation")

_TIMEOUT = 120.0


@dataclass(frozen=True, slots=True)
class Jar:
    """One jar of the manifest."""

    file: str
    url: str
    size: int
    sha256: str
    platform: str | None  # e.g. "darwin-aarch64" for a jar of native libraries, else None

    def path(self, cache_dir: Path) -> Path:
        """Where the jar is kept: under its hash, so two versions of one file name never collide."""
        return cache_dir / "jars" / self.sha256[:16] / self.file


@dataclass(frozen=True, slots=True)
class Manifest:
    refinery: str
    java: int
    features: dict[str, tuple[Jar, ...]]

    def jars(self, feature: Feature, host: str | None = None) -> list[Jar]:
        """The jars ``feature`` needs on ``host`` (this platform by default): native libraries only
        for it."""
        host = host or host_platform()
        jars = [jar for jar in self.features[feature] if jar.platform in (None, host)]
        if any(jar.platform for jar in self.features[feature]) and not any(
            jar.platform == host for jar in jars
        ):
            raise JvmError(
                f"Refinery's {feature} has no native libraries for this platform ({host})."
            )
        return jars


@cache
def manifest() -> Manifest:
    """The manifest shipped with pyfinery."""
    data = json.loads(
        resources.files(__package__).joinpath("jars.json").read_text(encoding="utf-8")
    )
    return Manifest(
        refinery=data["refinery"],
        java=data["java"],
        features={
            name: tuple(Jar(**jar) for jar in jars) for name, jars in data["features"].items()
        },
    )


def host_platform() -> str:
    """This platform as OR-tools names its jars of native libraries, e.g. ``darwin-aarch64``."""
    system = {"darwin": "darwin", "linux": "linux", "win32": "win32"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86-64", "amd64": "x86-64"}.get(
        machine
    )
    if system is None or arch is None:
        return f"{sys.platform}-{machine}"
    return f"{system}-{arch}"


def default_cache_dir() -> Path:
    """``$PYFINERY_CACHE_DIR``, else the platform's user cache directory, under ``pyfinery``."""
    if os.environ.get("PYFINERY_CACHE_DIR"):
        return Path(os.environ["PYFINERY_CACHE_DIR"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pyfinery"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "pyfinery" / "Cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pyfinery"


def fetch(jars: list[Jar], cache_dir: Path) -> list[Path]:
    """The paths of ``jars`` in ``cache_dir``, downloading and checking each one not there yet."""
    missing = [jar for jar in jars if not jar.path(cache_dir).is_file()]
    if missing:
        total = sum(jar.size for jar in missing) / 1e6
        logger.info(
            "Fetching %d of Refinery's jars (%.1f MB) into %s", len(missing), total, cache_dir
        )
    for jar in missing:
        _download(jar, jar.path(cache_dir))
    return [jar.path(cache_dir) for jar in jars]


def _download(jar: Jar, path: Path) -> None:
    try:
        with urllib.request.urlopen(jar.url, timeout=_TIMEOUT) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise JvmError(
            f"Could not download {jar.file} from {jar.url}: {error}. pyfinery fetches Refinery's "
            "jars on first "
            "use; on a machine without network access, copy a filled cache directory to "
            f"{path.parents[2]} "
            "or set PYFINERY_CACHE_DIR to one."
        ) from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != jar.sha256:
        raise JvmError(
            f"{jar.file} from {jar.url} has SHA-256 {digest}, not the pinned {jar.sha256}; "
            "not using it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and renamed, so a process that is interrupted, or one running beside it, never
    # sees half a jar.
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        partial.write_bytes(data)
        partial.replace(path)
    except OSError as error:
        partial.unlink(missing_ok=True)
        # Another process may have put the same jar there first, and Windows refuses to replace a
        # file that is open: theirs will do if it is the pinned one.
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == jar.sha256:
            return
        raise JvmError(f"Could not write {jar.file} into {path.parent}: {error}") from error
