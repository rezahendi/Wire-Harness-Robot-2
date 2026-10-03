"""A small client for GR00T's policy server (gr00t/eval/run_gr00t_server.py).

The server speaks ZeroMQ REQ/REP with msgpack + msgpack-numpy payloads:
request {"endpoint": "get_action", "data": {"observation": ..., "options": None}},
reply [action_dict, info_dict]. This client only needs pyzmq, msgpack and msgpack-numpy,
so the simulator does not have to share GR00T's Python environment.

    client = GrootClient("127.0.0.1", 5556)
    client.ping()
    actions = client.get_action(observation)     # {"motion": (1, 16, 4), "gripper": (1, 16, 1)}
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

# GR00T's server defaults to 5555, but on NVIDIA's GPU VM images (Nebius included) the DCGM
# host engine (nv-hostengine) already listens there, so everything here uses 5556.
DEFAULT_PORT = 5556


def pack(obj: Any) -> bytes:
    import msgpack
    import msgpack_numpy as mnp
    return msgpack.packb(obj, default=mnp.encode, use_bin_type=True)


def unpack(data: bytes) -> Any:
    import msgpack
    import msgpack_numpy as mnp
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


class GrootError(RuntimeError):
    pass


class GrootClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout_ms: int = 60000,
                 api_token: Optional[str] = None):
        import zmq
        self._zmq = zmq
        self.host, self.port, self.timeout_ms, self.api_token = host, port, timeout_ms, api_token
        self.context = zmq.Context()
        self.socket = None
        self.calls = 0
        self.seconds = 0.0
        self._connect()

    def _connect(self) -> None:
        zmq = self._zmq
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        request: Dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(pack(request))
            reply = self.socket.recv()
        except self._zmq.error.Again as exc:
            self._connect()              # a REQ socket is stuck after a timeout: start over
            raise GrootError(f"no answer from the GR00T server at {self.host}:{self.port} "
                             f"within {self.timeout_ms / 1000:.0f} s") from exc
        if reply == b"ERROR":
            raise GrootError("the GR00T server reported an error (is it the right policy server?)")
        out = unpack(reply)
        if isinstance(out, dict) and "error" in out:
            raise GrootError(f"GR00T server error: {out['error']}")
        return out

    def ping(self) -> bool:
        try:
            return bool(self.call("ping"))
        except GrootError:
            return False

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Any:
        return self.call("reset", {"options": options})

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        out = self.call("get_action", {"observation": observation, "options": None})
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        action = out[0] if isinstance(out, (list, tuple)) else out
        return {(k.split(".", 1)[1] if k.startswith("action.") else k): np.asarray(v) for k, v in action.items()}

    def close(self) -> None:
        try:
            if self.socket is not None:
                self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass
