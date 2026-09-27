"""Expert rollouts and the on-disk demonstration format.

One episode = one compressed ``.npz`` file:

    obs          (T+1, obs_dim) float32   observations, including the final one
    actions      (T, 5)         float32   actions in [-1, 1]
    rewards      (T,)           float32
    terminated   (T,)           bool
    truncated    (T,)           bool
    success      ()             bool
    seed         ()             int64     env seed -> replays the identical episode
    obs_layout   ()             str       JSON {name: [start, stop]}
    meta         ()             str       JSON (expert log, config, versions, ...)

The format maps 1:1 onto robomimic / LeRobot style datasets (obs, actions,
rewards, dones) and can be converted with a few lines of code.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import numpy as np

from harness_core.expert import ExpertParams, HarnessExpert

from .env import HarnessRoutingEnv


def rollout_expert(env: HarnessRoutingEnv, seed: int, expert_params: Optional[ExpertParams] = None,
                   render_every: int = 0) -> Dict[str, Any]:
    """Run the scripted expert for one episode and record everything."""
    t0 = time.time()
    obs, info = env.reset(seed=seed)
    expert = HarnessExpert(env.cfg, env.spec_actions, params=expert_params)
    observations = [obs]
    actions, rewards, terms, truncs, frames = [], [], [], [], []
    terminated = truncated = False
    k = 0
    while not (terminated or truncated):
        a = expert.step(env.last_obs_dict).astype(np.float32)
        obs, r, terminated, truncated, info = env.step(a)
        observations.append(obs)
        actions.append(a)
        rewards.append(r)
        terms.append(terminated)
        truncs.append(truncated)
        if render_every and k % render_every == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        k += 1
        if expert.done and not terminated:
            # expert finished (success or give-up) but the env did not terminate: stop here
            truncated = True
            truncs[-1] = True
    return {
        "obs": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terms, dtype=bool),
        "truncated": np.asarray(truncs, dtype=bool),
        "success": bool(info["is_success"]),
        "seed": int(seed),
        "info": info,
        "expert_log": [(float(t), m) for t, m in expert.log],
        "expert_failed": bool(expert.failed),
        "wall_time": time.time() - t0,
        "frames": frames,
    }


def save_episode(path: str, ep: Dict[str, Any], env: HarnessRoutingEnv) -> None:
    layout = {k: [v.start, v.stop] for k, v in env.obs_layout.items()}
    meta = {
        "expert_log": ep["expert_log"],
        "final_info": {k: v for k, v in ep["info"].items()},
        "sim_time": ep["info"]["sim_time"],
        "wall_time": ep["wall_time"],
        "config": env.cfg.to_dict(),
        "randomize": env.randomize,
        "policy_dt": env.cfg.sim.policy_dt,
    }
    np.savez_compressed(
        path, obs=ep["obs"], actions=ep["actions"], rewards=ep["rewards"],
        terminated=ep["terminated"], truncated=ep["truncated"],
        success=np.array(ep["success"]), seed=np.array(ep["seed"], dtype=np.int64),
        obs_layout=np.array(json.dumps(layout)), meta=np.array(json.dumps(meta, default=str)))


def load_episode(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["obs_layout"] = {k: slice(*v) for k, v in json.loads(str(ep["obs_layout"])).items()}
    ep["meta"] = json.loads(str(ep["meta"]))
    ep["success"] = bool(ep["success"])
    ep["seed"] = int(ep["seed"])
    return ep
