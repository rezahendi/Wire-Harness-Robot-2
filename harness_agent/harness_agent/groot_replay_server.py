"""A stand-in for GR00T's policy server that replays recorded episodes (no GPU needed).

    python -m harness_agent.groot_replay_server data/harness_route --port 5556
    python -m harness_agent.groot_eval --replay-dataset data/harness_route --seeds 1000 --forks F1 --out eval/replay

It speaks the same protocol as gr00t/eval/run_gr00t_server.py (ZeroMQ REQ/REP, msgpack +
msgpack-numpy), checks every observation the way Gr00tPolicy does (keys, dtypes, shapes),
and answers get_action with the next 16 recorded actions, advancing by the execution
horizon, like GR00T's ReplayPolicy. Replaying the expert's own actions on the same seed
must route the wire: if it does, the observation builder, the client, the chunk execution
and the success check all work, before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from . import groot_features as gf
from .groot_client import DEFAULT_PORT, pack, unpack


class ReplayPolicy:
    def __init__(self, dataset: str, action_horizon: int = 16, execution_horizon: int = 8):
        import pyarrow.parquet as pq
        self._pq = pq
        self.dataset = dataset
        with open(os.path.join(dataset, "meta", "info.json")) as f:
            self.info = json.load(f)
        with open(os.path.join(dataset, "meta", "episodes.jsonl")) as f:
            self.episodes = [json.loads(line) for line in f]
        with open(os.path.join(dataset, "meta", "modality.json")) as f:
            modality = json.load(f)
        self.state_layout = [(k, v["end"] - v["start"]) for k, v in modality["state"].items()]
        self.action_horizon, self.execution_horizon = action_horizon, execution_horizon
        self.episode_index, self.step = 0, 0
        self.observations = 0
        self._load(0)

    def _load(self, i: int) -> None:
        path = os.path.join(self.dataset, self.info["data_path"].format(
            episode_chunk=i // self.info["chunks_size"], episode_index=i))
        self.actions = np.asarray(self._pq.read_table(path).to_pydict()["action"], dtype=np.float32)
        self.episode_index = i

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        options = options or {}
        if "episode_index" in options and int(options["episode_index"]) != self.episode_index:
            self._load(int(options["episode_index"]))
        self.step = int(options.get("step_index", 0))
        return {"episode_index": self.episode_index, "episode_length": len(self.actions)}

    def modality_config(self) -> Dict[str, Any]:
        """What GR00T's server reports, in the same shape: the keys of each modality."""
        return {"video": {"modality_keys": list(gf.VIDEO_KEYS)},
                "state": {"modality_keys": [k for k, _ in self.state_layout]},
                "action": {"modality_keys": [k for k, _ in gf.ACTION_LAYOUT]},
                "language": {"modality_keys": [gf.LANGUAGE_KEY]}}

    def check_observation(self, obs: Dict[str, Any]) -> None:
        for m in ("video", "state", "language"):
            assert isinstance(obs.get(m), dict), f"observation needs a '{m}' dict"
        for k in gf.VIDEO_KEYS:
            v = obs["video"][k]
            assert isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 5 and v.shape[-1] == 3, \
                f"video.{k} must be uint8 (B, T, H, W, 3), got {getattr(v, 'dtype', type(v))} {getattr(v, 'shape', '')}"
            assert v.shape[1] == 1, f"video.{k}: one frame expected"
        extra = sorted(set(obs["state"]) - {k for k, _ in self.state_layout})
        assert not extra, f"state keys the model was not trained with: {extra}"
        for k, n in self.state_layout:
            assert k in obs["state"], f"state.{k} is missing"
            s = obs["state"][k]
            assert isinstance(s, np.ndarray) and s.dtype == np.float32 and s.shape[1:] == (1, n), \
                f"state.{k} must be float32 (B, 1, {n}), got {getattr(s, 'dtype', type(s))} {getattr(s, 'shape', '')}"
        lang = obs["language"][gf.LANGUAGE_KEY]
        assert isinstance(lang, list) and isinstance(lang[0], list) and isinstance(lang[0][0], str), \
            "language must be [[text]]"

    def get_action(self, observation: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        self.check_observation(observation)
        self.observations += 1
        a, h, n = self.actions, self.action_horizon, len(self.actions)
        if self.step >= n:
            chunk = np.tile(a[-1:], (h, 1))
        else:
            chunk = a[self.step:self.step + h]
            if len(chunk) < h:
                chunk = np.concatenate([chunk, np.tile(a[-1:], (h - len(chunk), 1))])
        self.step += self.execution_horizon
        out = {}
        for k, (lo, hi) in gf.layout_slices(gf.ACTION_LAYOUT).items():
            out[k] = chunk[None, :, lo:hi].astype(np.float32)
        return [out, {"episode_index": self.episode_index, "current_step": self.step - self.execution_horizon}]


def serve(policy: ReplayPolicy, host: str = "*", port: int = DEFAULT_PORT, max_requests: Optional[int] = None) -> None:
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    handled = 0
    try:
        while max_requests is None or handled < max_requests:
            req = unpack(sock.recv())
            handled += 1
            ep = req.get("endpoint", "get_action")
            try:
                if ep == "ping":
                    rep = {"status": "ok", "message": "replay server"}
                elif ep == "reset":
                    rep = policy.reset(**req.get("data", {}))
                elif ep == "get_action":
                    rep = policy.get_action(**req.get("data", {}))
                elif ep == "get_modality_config":
                    rep = policy.modality_config()
                elif ep == "kill":
                    sock.send(pack({"status": "ok"}))
                    break
                else:
                    rep = {"error": f"unknown endpoint {ep}"}
            except AssertionError as exc:
                rep = {"error": f"bad observation: {exc}"}
            sock.send(pack(rep))
    finally:
        sock.close(linger=0)
        ctx.term()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--execution-horizon", type=int, default=8)
    args = ap.parse_args(argv)
    policy = ReplayPolicy(args.dataset, execution_horizon=args.execution_horizon)
    print(f"replaying {len(policy.episodes)} episodes from {args.dataset} on port {args.port}", flush=True)
    serve(policy, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
