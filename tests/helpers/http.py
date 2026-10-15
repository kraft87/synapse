"""A loopback HTTP server for tests that stand in for the Synapse server."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LoopbackServer:
    """Serves ``handler`` on a free 127.0.0.1 port until ``close()``; ``url`` is its base.

    ``serve_forever`` is polled every 50 ms rather than its default 500 ms: ``shutdown()``
    waits for the next poll, so the default cost every test that used a server half a
    second."""

    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
