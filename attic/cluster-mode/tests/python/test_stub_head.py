"""Tests for the cluster-mode preflight stub head.

The full preflight (`tests/test-cluster-preflight.sh`) needs the
arbiter binary built, so it gates on go-build availability and
isn't run from `make test-python`. These tests instead exercise
the stub head's HTTP surface directly so the script's helper
contract is covered by the standard Python suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_stub_head_module():
    spec = importlib.util.spec_from_file_location(
        "_stub_head_under_test",
        str(REPO_ROOT / "tests" / "fixtures" / "stub-head.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SH = _load_stub_head_module()


def _free_port() -> int:
    import socket
    s = socket.socket()
    # Bind to loopback only -- the ephemeral port is handed straight to the
    # 127.0.0.1 HTTPServer below, so there is no reason to expose it on every
    # interface (flagged by py/bind-socket-all-network-interfaces).
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _ServerCtx:
    def __init__(self, head: SH.StubHead) -> None:
        self.port = _free_port()
        self.head = head
        self.httpd = HTTPServer(("127.0.0.1", self.port), head.make_handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ServerCtx":
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _post(url: str, body: dict, token: str | None) -> tuple[int, dict | None]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.getcode(), payload
    except urllib.error.HTTPError as e:
        return e.code, None


class TestStubHeadRegister(unittest.TestCase):
    def test_register_assigns_worker_id(self) -> None:
        head = SH.StubHead("the-token", commands=None)
        with _ServerCtx(head) as srv:
            code, body = _post(
                f"http://127.0.0.1:{srv.port}/v1/cluster/register",
                {"name": "worker-a"},
                "the-token",
            )
            self.assertEqual(code, 200)
            assert body is not None
            self.assertTrue(body["worker_id"].startswith("wid-"))
            self.assertEqual(len(head.registrations), 1)
            self.assertEqual(head.registrations[0]["name"], "worker-a")

    def test_register_rejects_missing_token(self) -> None:
        head = SH.StubHead("the-token", commands=None)
        with _ServerCtx(head) as srv:
            code, _ = _post(
                f"http://127.0.0.1:{srv.port}/v1/cluster/register",
                {"name": "worker-a"},
                None,
            )
            self.assertEqual(code, 401)
            self.assertEqual(len(head.registrations), 0)

    def test_register_rejects_wrong_token(self) -> None:
        head = SH.StubHead("the-token", commands=None)
        with _ServerCtx(head) as srv:
            code, _ = _post(
                f"http://127.0.0.1:{srv.port}/v1/cluster/register",
                {"name": "worker-a"},
                "wrong-token",
            )
            self.assertEqual(code, 401)


class TestStubHeadHeartbeat(unittest.TestCase):
    def test_first_heartbeat_returns_queued_commands(self) -> None:
        head = SH.StubHead(
            "the-token",
            commands=[
                {"type": "drain", "backend": "vllm"},
                {"type": "shutdown", "grace_seconds": 5},
            ],
        )
        with _ServerCtx(head) as srv:
            code, body = _post(
                f"http://127.0.0.1:{srv.port}/v1/cluster/heartbeat",
                {"worker_id": "w1", "counter": 1, "queue_depth": 0},
                "the-token",
            )
            self.assertEqual(code, 200)
            assert body is not None
            self.assertEqual(len(body["commands"]), 2)
            self.assertEqual(body["commands"][0]["type"], "drain")
            self.assertEqual(body["commands"][1]["type"], "shutdown")

    def test_subsequent_heartbeats_return_empty_commands(self) -> None:
        head = SH.StubHead(
            "the-token",
            commands=[{"type": "drain", "backend": "vllm"}],
        )
        with _ServerCtx(head) as srv:
            url = f"http://127.0.0.1:{srv.port}/v1/cluster/heartbeat"
            _post(url, {"worker_id": "w1", "counter": 1, "queue_depth": 0}, "the-token")
            code, body = _post(
                url, {"worker_id": "w1", "counter": 2, "queue_depth": 0}, "the-token"
            )
            self.assertEqual(code, 200)
            assert body is not None
            self.assertEqual(body["commands"], [])

    def test_heartbeat_records_counter_and_state(self) -> None:
        head = SH.StubHead("the-token", commands=None)
        with _ServerCtx(head) as srv:
            url = f"http://127.0.0.1:{srv.port}/v1/cluster/heartbeat"
            for i in range(3):
                _post(url, {"worker_id": "w1", "counter": i + 1, "queue_depth": i}, "the-token")
            self.assertEqual(len(head.heartbeats), 3)
            counters = [hb["counter"] for hb in head.heartbeats]
            self.assertEqual(counters, [1, 2, 3])

    def test_heartbeat_rejects_bad_token(self) -> None:
        head = SH.StubHead("the-token", commands=None)
        with _ServerCtx(head) as srv:
            code, _ = _post(
                f"http://127.0.0.1:{srv.port}/v1/cluster/heartbeat",
                {"worker_id": "w1", "counter": 1},
                None,
            )
            self.assertEqual(code, 401)


class TestStubHeadIntrospect(unittest.TestCase):
    def test_introspect_returns_state(self) -> None:
        head = SH.StubHead("the-token", commands=[{"type": "drain"}])
        with _ServerCtx(head) as srv:
            req = urllib.request.Request(
                f"http://127.0.0.1:{srv.port}/_introspect",
                headers={"Authorization": "Bearer the-token"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body["registrations"], [])
            self.assertEqual(body["heartbeats_count"], 0)
            self.assertEqual(body["commands_remaining"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
