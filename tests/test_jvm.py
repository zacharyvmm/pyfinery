"""pyfinery.jvm without a JVM: finding the JDK, the manifest, fetching jars, unpacking natives,
settings."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

import pyfinery.jvm as jvm
from pyfinery.errors import RefineryError
from pyfinery.jvm import _jars, _java, _setup
from pyfinery.jvm._jars import Jar, Manifest
from pyfinery.jvm.errors import JvmError

# -- finding the JDK ----------------------------------------------------------


def fake_jdk(
    root: Path, version: str = "25.0.3", *, javac: bool = True, libjvm: bool = True
) -> Path:
    root.mkdir(parents=True)
    (root / "release").write_text(f'IMPLEMENTOR="Eclipse Adoptium"\nJAVA_VERSION="{version}"\n')
    (root / "bin").mkdir()
    if javac:
        (root / "bin" / "javac").touch()
    if libjvm:
        (root / "lib" / "server").mkdir(parents=True)
        (root / "lib" / "server" / "libjvm.so").touch()
    return root


@pytest.mark.parametrize(
    ("value", "major"),
    [("25.0.3", 25), ("26", 26), ("21.0.2+13", 21), ("1.8.0_402", 8), ("25-ea", 25)],
)
def test_reads_major_version(tmp_path, value, major):
    (tmp_path / "release").write_text(f'JAVA_VERSION="{value}"\n')
    assert _java.read_version(tmp_path) == major


def test_accepts_jdk_25(tmp_path):
    jdk = _java.inspect_jdk(fake_jdk(tmp_path / "jdk"))
    assert jdk.version == 25
    assert jdk.libjvm.name == "libjvm.so"
    assert jdk.javac.name == "javac"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": "21.0.2"}, "a Java 21; Refinery needs Java 25"),
        ({"javac": False}, "no javac"),
        ({"libjvm": False}, "no JVM library"),
    ],
)
def test_rejects_unusable_jdks(tmp_path, kwargs, message):
    with pytest.raises(JvmError, match=message):
        _java.inspect_jdk(fake_jdk(tmp_path / "jdk", **kwargs))


def test_rejects_a_directory_that_is_not_a_jdk(tmp_path):
    with pytest.raises(JvmError, match="no 'release' file"):
        _java.inspect_jdk(tmp_path)
    with pytest.raises(JvmError, match="not a directory"):
        _java.inspect_jdk(tmp_path / "missing")


def test_jdk_precedence(tmp_path, monkeypatch):
    explicit = fake_jdk(tmp_path / "explicit")
    ours = fake_jdk(tmp_path / "ours", "26")
    theirs = fake_jdk(tmp_path / "theirs", "25.0.1")
    monkeypatch.setenv("PYFINERY_JAVA_HOME", str(ours))
    monkeypatch.setenv("JAVA_HOME", str(theirs))
    assert _java.find_jdk(explicit).home == explicit
    assert _java.find_jdk().home == ours
    monkeypatch.delenv("PYFINERY_JAVA_HOME")
    assert _java.find_jdk().home == theirs


def test_an_explicit_choice_is_not_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("JAVA_HOME", str(fake_jdk(tmp_path / "good")))
    with pytest.raises(JvmError, match="configure"):
        _java.find_jdk(fake_jdk(tmp_path / "old", "17.0.12"))


def test_no_jdk(monkeypatch):
    monkeypatch.delenv("PYFINERY_JAVA_HOME", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(_java, "_macos_java_home", lambda: None)
    with pytest.raises(JvmError, match="No JDK found"):
        _java.find_jdk()


def test_errors_are_refinery_errors():
    assert issubclass(JvmError, RefineryError)


# -- the manifest -------------------------------------------------------------


def test_manifest_pins_every_jar():
    manifest = _jars.manifest()
    assert manifest.java == 25
    assert set(manifest.features) == {"validation", "generation"}
    for jars in manifest.features.values():
        for jar in jars:
            assert jar.url.startswith("https://repo1.maven.org/maven2/")
            assert jar.url.endswith("/" + jar.file)
            assert len(jar.sha256) == 64
            assert jar.size > 0


def test_validation_needs_no_native_code():
    validation = _jars.manifest().features["validation"]
    assert not [jar for jar in validation if "ortools" in jar.file or jar.platform]
    assert any(jar.file.startswith("refinery-generator-") for jar in validation)


def test_generation_adds_ortools_for_each_platform():
    manifest = _jars.manifest()
    validation = {jar.sha256 for jar in manifest.features["validation"]}
    generation = manifest.features["generation"]
    assert validation <= {jar.sha256 for jar in generation}
    platforms = sorted(jar.platform for jar in generation if jar.platform)
    assert platforms == [
        "darwin-aarch64",
        "darwin-x86-64",
        "linux-aarch64",
        "linux-x86-64",
        "win32-x86-64",
    ]


def test_jars_for_a_platform():
    manifest = _jars.manifest()
    jars = manifest.jars("generation", "linux-x86-64")
    assert [jar.platform for jar in jars if jar.platform] == ["linux-x86-64"]
    with pytest.raises(JvmError, match="no native libraries for this platform"):
        manifest.jars("generation", "sunos-sparc")
    assert manifest.jars("validation", "sunos-sparc")  # matching needs no native code


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("darwin", "arm64", "darwin-aarch64"),
        ("linux", "x86_64", "linux-x86-64"),
        ("win32", "AMD64", "win32-x86-64"),
    ],
)
def test_host_platform(monkeypatch, system, machine, expected):
    monkeypatch.setattr(_jars.sys, "platform", system)
    monkeypatch.setattr(_jars.platform, "machine", lambda: machine)
    assert _jars.host_platform() == expected


def test_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PYFINERY_CACHE_DIR", str(tmp_path))
    assert _jars.default_cache_dir() == tmp_path
    monkeypatch.delenv("PYFINERY_CACHE_DIR")
    monkeypatch.setattr(_jars.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _jars.default_cache_dir() == tmp_path / "xdg" / "pyfinery"


# -- fetching -----------------------------------------------------------------


def jar_of(data: bytes, name: str = "a-1.0.jar") -> Jar:
    return Jar(
        name,
        f"https://repo1.maven.org/maven2/a/{name}",
        len(data),
        hashlib.sha256(data).hexdigest(),
        None,
    )


class FakeUrlopen:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if url not in self.bodies:
            raise urllib.error.URLError("unreachable")
        return io.BytesIO(self.bodies[url])


def test_fetches_once_and_checks_hashes(tmp_path, monkeypatch):
    jar = jar_of(b"jar bytes")
    fake = FakeUrlopen({jar.url: b"jar bytes"})
    monkeypatch.setattr(_jars.urllib.request, "urlopen", fake)
    [path] = _jars.fetch([jar], tmp_path)
    assert path.read_bytes() == b"jar bytes"
    assert path == tmp_path / "jars" / jar.sha256[:16] / jar.file
    _jars.fetch([jar], tmp_path)
    assert fake.calls == [jar.url]  # cached


def test_refuses_a_jar_with_another_hash(tmp_path, monkeypatch):
    jar = jar_of(b"expected")
    monkeypatch.setattr(_jars.urllib.request, "urlopen", FakeUrlopen({jar.url: b"tampered"}))
    with pytest.raises(JvmError, match="not the pinned"):
        _jars.fetch([jar], tmp_path)
    assert not jar.path(tmp_path).exists()


def test_another_process_got_there_first(tmp_path, monkeypatch):
    jar = jar_of(b"jar bytes")
    monkeypatch.setattr(_jars.urllib.request, "urlopen", FakeUrlopen({jar.url: b"jar bytes"}))

    def refuse(self, target):  # as Windows does when the other process has the jar open
        Path(target).write_bytes(b"jar bytes")
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "replace", refuse)
    [path] = _jars.fetch([jar], tmp_path)
    assert path.read_bytes() == b"jar bytes"
    assert not list(path.parent.glob("*.part"))


def test_a_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    jar = jar_of(b"jar bytes")
    monkeypatch.setattr(_jars.urllib.request, "urlopen", FakeUrlopen({jar.url: b"jar bytes"}))

    def full(self, target):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", full)
    with pytest.raises(JvmError, match="Could not write"):
        _jars.fetch([jar], tmp_path)
    assert not list(jar.path(tmp_path).parent.glob("*"))


def test_says_how_to_work_offline(tmp_path, monkeypatch):
    jar = jar_of(b"x")
    monkeypatch.setattr(_jars.urllib.request, "urlopen", FakeUrlopen({}))
    with pytest.raises(JvmError, match="PYFINERY_CACHE_DIR"):
        _jars.fetch([jar], tmp_path)


def test_manifest_dataclass_filters_without_natives():
    jar = jar_of(b"x")
    assert Manifest("0", 25, {"validation": (jar,)}).jars("validation", "anything") == [jar]


# -- natives and helpers ------------------------------------------------------


def test_unpacks_natives_once(tmp_path):
    archive = tmp_path / "ortools-linux-x86-64.jar"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("META-INF/MANIFEST.MF", "x")
        out.writestr("ortools-linux-x86-64/libjniortools.so", b"jni")
        out.writestr("ortools-linux-x86-64/libortools.so.9", b"lib")
    jar = Jar(archive.name, "https://x", 1, "ab" * 32, "linux-x86-64")
    natives = _setup.unpack_natives(jar, archive, tmp_path / "cache")
    assert sorted(p.name for p in natives.iterdir()) == [
        ".complete",
        "libjniortools.so",
        "libortools.so.9",
    ]
    assert [p.name for p in _setup.ortools_libraries(natives, "linux-x86-64")] == [
        "libjniortools.so"
    ]
    archive.unlink()  # a second call uses what is unpacked
    assert _setup.unpack_natives(jar, archive, tmp_path / "cache") == natives


def test_missing_jni_library(tmp_path):
    with pytest.raises(JvmError, match="no OR-tools JNI library"):
        _setup.ortools_libraries(tmp_path, "linux-x86-64")


def test_windows_loads_what_jniortools_needs_first(tmp_path):
    names = [*_setup._WINDOWS_ORDER, "jniortools"]
    for name in names:
        (tmp_path / f"{name}.dll").touch()
    loaded = _setup.ortools_libraries(tmp_path, "win32-x86-64")
    assert [p.stem for p in loaded] == names
    (tmp_path / "re2.dll").unlink()
    with pytest.raises(JvmError, match="lacks re2.dll"):
        _setup.ortools_libraries(tmp_path, "win32-x86-64")


def native_jar(tmp_path, entries: dict[str, bytes]) -> tuple[Jar, Path]:
    archive = tmp_path / "ortools-linux-x86-64.jar"
    with zipfile.ZipFile(archive, "w") as out:
        for name, data in entries.items():
            out.writestr(name, data)
    return Jar(archive.name, "https://x", 1, "cd" * 32, "linux-x86-64"), archive


def test_unpacks_only_the_platform_folder(tmp_path):
    jar, archive = native_jar(
        tmp_path,
        {
            "com/google/ortools/Loader.class": b"class",
            "ortools-linux-x86-64/libjniortools.so": b"jni",
        },
    )
    natives = _setup.unpack_natives(jar, archive, tmp_path / "cache")
    assert sorted(p.name for p in natives.iterdir()) == [".complete", "libjniortools.so"]


@pytest.mark.parametrize(
    "entry", ["ortools-linux-x86-64/../evil.so", "ortools-linux-x86-64/sub/lib.so"]
)
def test_refuses_unexpected_entries(tmp_path, entry):
    jar, archive = native_jar(tmp_path, {entry: b"x"})
    with pytest.raises(JvmError, match="unexpected entry"):
        _setup.unpack_natives(jar, archive, tmp_path / "cache")
    assert not (tmp_path / "cache" / "natives" / jar.sha256[:16]).exists()
    assert not (tmp_path / "evil.so").exists()


def test_a_jar_without_natives_is_refused(tmp_path):
    jar, archive = native_jar(tmp_path, {"META-INF/MANIFEST.MF": b"x"})
    with pytest.raises(JvmError, match="no native libraries"):
        _setup.unpack_natives(jar, archive, tmp_path / "cache")


def test_javac_errors_are_reported(tmp_path):
    jdk = _java.inspect_jdk(fake_jdk(tmp_path / "jdk"))
    javac = jdk.javac
    javac.write_text("#!/bin/sh\necho 'Bulk.java:1: error: broken' >&2\nexit 1\n")
    javac.chmod(0o755)
    with pytest.raises(JvmError, match="broken"):
        _setup.compile_helpers(jdk, [], tmp_path / "cache")
    assert not list((tmp_path / "cache" / "helpers").glob("*"))


def test_helper_sources_ship_with_the_package():
    sources = _setup.helper_sources()
    assert set(sources) == {"Bulk.java", "Natives.java"}
    assert all(b"package pyfinery.jvm;" in text for text in sources.values())


# -- settings -----------------------------------------------------------------


@pytest.fixture
def fresh(monkeypatch):
    """Module state as before any start, restored afterwards."""
    monkeypatch.setattr(jvm, "_settings", jvm.Settings())
    monkeypatch.setattr(jvm, "_status", None)
    return jvm


def test_configure_keeps_what_is_left_out(fresh, tmp_path):
    fresh.configure(generation=True)
    settings = fresh.configure(cache_dir=tmp_path, jvm_options=["-Xmx2g"])
    assert settings.generation
    assert settings.cache_dir == tmp_path
    assert settings.features == ("validation", "generation")
    assert fresh.settings() == settings


def test_settings_cannot_change_once_started(fresh, monkeypatch, tmp_path):
    running = jvm.Status(
        jvm.Settings(), _java.inspect_jdk(fake_jdk(tmp_path / "jdk")), tmp_path, ()
    )
    monkeypatch.setattr(jvm, "_status", running)
    assert fresh.configure() == jvm.Settings()  # nothing asked to change
    with pytest.raises(JvmError, match="started without generation"):
        fresh.configure(generation=True)


def test_generation_before_start_starts_with_it(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(
        jvm,
        "start",
        lambda: seen.append(jvm._settings) or jvm.Status(jvm._settings, None, None, ()),
    )
    assert fresh.require("generation").generation
    assert seen[0].generation


def test_generation_after_a_start_without_it(fresh, monkeypatch, tmp_path):
    running = jvm.Status(jvm.Settings(), None, tmp_path, ())
    monkeypatch.setattr(jvm, "_status", running)
    monkeypatch.setattr(jvm, "start", lambda: running)
    assert fresh.require("validation") is running
    with pytest.raises(JvmError, match="configure\\(generation=True\\)"):
        fresh.require("generation")


class FakeJPype:
    """Just enough of JPype for start(): a JVM that starts, and a Natives helper that may fail."""

    def __init__(self, natives_error: Exception | None = None) -> None:
        self.running = False
        self.natives_error = natives_error
        self.loaded: list[str] = []
        self.JString = str

    def isJVMStarted(self):
        return self.running

    def startJVM(self, *args, **kwargs):
        self.running = True

    def JArray(self, kind):
        return list

    def JClass(self, name):
        fake = self

        class Natives:
            @staticmethod
            def loadOrTools(paths):
                if fake.natives_error is not None:
                    raise fake.natives_error
                fake.loaded.extend(paths)

        return Natives


@pytest.fixture
def stubbed(fresh, monkeypatch, tmp_path):
    """start() with everything but its own logic stubbed out."""
    jdk = _java.inspect_jdk(fake_jdk(tmp_path / "jdk"))
    monkeypatch.setattr(jvm, "find_jdk", lambda home: jdk)
    monkeypatch.setattr(jvm, "fetch", lambda jars, cache: [cache / j.file for j in jars])
    monkeypatch.setattr(jvm, "compile_helpers", lambda jdk, paths, cache: cache / "helpers")
    monkeypatch.setattr(jvm, "unpack_natives", lambda jar, path, cache: cache / "natives")
    monkeypatch.setattr(jvm, "ortools_libraries", lambda natives, host: [natives / "libjni.so"])
    fresh.configure(cache_dir=tmp_path / "cache")
    return fresh


def test_starts_and_loads_natives(stubbed, monkeypatch, tmp_path):
    fake = FakeJPype()
    monkeypatch.setattr(jvm, "_jpype", lambda: fake)
    status = stubbed.require("generation")
    assert status.generation
    assert fake.loaded == [str(tmp_path / "cache" / "natives" / "libjni.so")]


def test_a_natives_failure_leaves_validation_usable(stubbed, monkeypatch):
    fake = FakeJPype(natives_error=RuntimeError("UnsatisfiedLinkError: no ortools"))
    monkeypatch.setattr(jvm, "_jpype", lambda: fake)
    with pytest.raises(JvmError, match="did not load: UnsatisfiedLinkError"):
        stubbed.require("generation")
    assert fake.running
    status = stubbed.require("validation")  # no "started by something other than pyfinery"
    assert not status.generation
    with pytest.raises(JvmError, match="did not load"):
        stubbed.require("generation")


def test_a_failed_start_for_generation_keeps_the_settings(stubbed, monkeypatch):
    def offline(jars, cache):
        if any(jar.platform for jar in jars):
            raise JvmError("Could not download ortools")
        return [cache / j.file for j in jars]

    fake = FakeJPype()
    monkeypatch.setattr(jvm, "_jpype", lambda: fake)
    monkeypatch.setattr(jvm, "fetch", offline)
    with pytest.raises(JvmError, match="Could not download"):
        stubbed.require("generation")
    assert not stubbed.settings().generation
    assert stubbed.require("validation").settings.features == ("validation",)


def test_started_without_jpype(monkeypatch):
    def missing():
        raise JvmError("pyfinery.jvm needs JPype")

    monkeypatch.setattr(jvm, "_jpype", missing)
    assert jvm.started() is False
