#!/usr/bin/env python3
"""Minimal stub head for cluster-mode worker tests.

Implements POST /v1/cluster/register and POST /v1/cluster/heartbeat
with predictable, scriptable behaviour:

  - register: validates the bearer token, returns a worker_id derived
    from the request name. Logs every registration to stderr.
  - heartbeat: validates the bearer token, returns a JSON
    HeartbeatResponse with a Commands list assembled from the
    `--commands FILE` option (a JSON file of [cmd, cmd, ...]).
    Returns the entire list ONCE, then empty thereafter, so a test
    can assert "did the worker execute these commands."

Designed for cluster-mode Phase 1.5 preflight scenarios. Stdlib
only; no external deps. Run with:

  python3 tests/fixtures/stub-head.py --token the-token --port 18080

Then point a worker at it:

  DEVAI_HEAD_URL=http://localhost:18080 \\
    DEVAI_WORKER_TOKEN_FILE=/tmp/the-token \\
    gpu-arbiter --mode=worker
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


class StubHead:
    def __init__(
        self,
        token: str,
        commands: list[dict] | None,
    ) -> None:
        self.token = token
        self._commands_pending: list[dict] = list(commands or [])
        self.registrations: list[dict] = []
        self.heartbeats: list[dict] = []
        self.lock = threading.Lock()

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        head = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                # Keep stderr quiet; tests assert against the
                # registration / heartbeat lists instead of log
                # output.
                return

            def _check_token(self) -> bool:
                auth = self.headers.get("Authorization", "")
                if not auth.lower().startswith("bearer "):
                    self.send_error(401, "missing bearer")
                    return False
                got = auth[7:].strip()
                if got != head.token:
                    self.send_error(401, "bad token")
                    return False
                return True

            def _read_body(self) -> dict | None:
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    n = 0
                raw = self.rfile.read(n) if n > 0 else b""
                try:
                    return json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None

            def _json_response(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if not self._check_token():
                    return
                body = self._read_body()
                if body is None:
                    self.send_error(400, "bad json")
                    return
                if self.path == "/v1/cluster/register":
                    with head.lock:
                        head.registrations.append(body)
                    wid = "wid-" + uuid.uuid4().hex[:8]
                    self._json_response(200, {"worker_id": wid})
                    return
                if self.path == "/v1/cluster/heartbeat":
                    with head.lock:
                        head.heartbeats.append(body)
                        cmds = head._commands_pending
                        head._commands_pending = []
                    self._json_response(200, {"commands": cmds})
                    return
                self.send_error(404, "no such path")

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self._json_response(200, {"status": "ok"})
                    return
                if self.path == "/_introspect":
                    if not self._check_token():
                        return
                    with head.lock:
                        payload = {
                            "registrations": head.registrations,
                            "heartbeats_count": len(head.heartbeats),
                            "commands_remaining": len(head._commands_pending),
                        }
                    self._json_response(200, payload)
                    return
                self.send_error(404, "no such path")

        return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--token", required=True, help="bearer token to accept")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument(
        "--commands",
        default="",
        help="path to JSON file containing a list of Command dicts to "
             "return on the next heartbeat (then empty thereafter)",
    )
    args = ap.parse_args()
    cmds: list[dict] = []
    if args.commands:
        with open(args.commands) as f:
            cmds = json.load(f)
        if not isinstance(cmds, list):
            sys.exit(f"--commands {args.commands} must be a JSON list")
    head = StubHead(args.token, cmds)
    handler = head.make_handler()
    httpd = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"stub-head listening on :{args.port} (token={args.token!r})", file=sys.stderr)
    if cmds:
        print(f"  will return {len(cmds)} command(s) on next heartbeat", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
