# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

import threading
from types import SimpleNamespace

import httpx
import pytest

import pyfinery as pf
from pyfinery import client, server

BASE = "http://refinery.test/api/v1"


class FakeServer:
    """Stands in for RefineryServer: records how it was made, started and stopped."""

    made: list["FakeServer"] = []

    def __init__(self, **options) -> None:
        self.options = options
        self.running = False
        self.base_url = f"http://127.0.0.1:{4000 + len(self.made)}/api/v1"
        self.made.append(self)

    def start(self) -> "FakeServer":
        self.running = True
        return self

    def stop(self) -> None:
        self.running = False


@pytest.fixture(autouse=True)
def at_exit(monkeypatch):
    """No shared client yet and no Docker; what is registered to run at exit, not run."""
    monkeypatch.setattr(client, "_shared", None)
    monkeypatch.setattr(server, "RefineryServer", FakeServer)
    monkeypatch.setattr(FakeServer, "made", [])
    registered = []
    monkeypatch.setattr(client, "atexit", SimpleNamespace(register=registered.append))
    return registered


def test_uses_the_server_at_refinery_api_base(monkeypatch, at_exit):
    monkeypatch.setenv("REFINERY_API_BASE", BASE)
    refinery = pf.Refinery.shared(image="ignored", timeout=None)
    assert refinery is pf.Refinery.shared()
    assert refinery.base_url == BASE
    assert refinery._http.timeout == httpx.Timeout(None)
    assert FakeServer.made == [] and at_exit == []


def test_starts_a_server_once_and_stops_it_at_exit(monkeypatch, at_exit):
    monkeypatch.delenv("REFINERY_API_BASE", raising=False)
    refinery = pf.Refinery.shared(image="example/refinery", semantics_timeout=600_000, timeout=5)
    assert pf.Refinery.shared(image="other", timeout=None) is refinery
    [started] = FakeServer.made
    assert started.options == {"image": "example/refinery", "semantics_timeout": 600_000}
    assert refinery.base_url == started.base_url
    assert refinery._http.timeout == httpx.Timeout(5)
    assert at_exit == [started.stop]
    at_exit[0]()
    assert not started.running


def test_one_client_across_threads(monkeypatch):
    monkeypatch.delenv("REFINERY_API_BASE", raising=False)
    clients = []
    threads = [
        threading.Thread(target=lambda: clients.append(pf.Refinery.shared())) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(FakeServer.made) == 1
    assert len(clients) == 8 and all(c is clients[0] for c in clients)


def test_tries_again_after_a_failed_start(monkeypatch, at_exit):
    monkeypatch.delenv("REFINERY_API_BASE", raising=False)
    start = FakeServer.start

    def fail(self):
        raise pf.ServerStartError("no Docker")

    monkeypatch.setattr(FakeServer, "start", fail)
    with pytest.raises(pf.ServerStartError):
        pf.Refinery.shared()
    assert at_exit == []
    monkeypatch.setattr(FakeServer, "start", start)
    refinery = pf.Refinery.shared()
    assert refinery.base_url == FakeServer.made[-1].base_url
    assert len(FakeServer.made) == 2
