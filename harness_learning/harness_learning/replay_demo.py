"""Replay a recorded episode open-loop and check it reproduces (optionally to video).

    ros2 run harness_learning replay_demo harness_demos/episode_000003.npz --video replay.mp4
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode .npz written by record_demos")
    ap.add_argument("--video", default="", help="write an mp4 of the replay")
    ap.add_argument("--camera", default="overview", help="overview | top | side | wrist")
    args = ap.parse_args(argv)
    if args.video:
        from harness_core.render_util import choose_gl_backend
        choose_gl_backend()

    from harness_core.config import CellConfig
    from harness_learning.demos import load_episode
    from harness_learning.env import HarnessRoutingEnv

    ep = load_episode(args.episode)
    cfg = CellConfig.from_dict(ep["meta"]["config"])
    env = HarnessRoutingEnv(cfg=cfg, randomize=ep["meta"]["randomize"],
                            render_mode="rgb_array" if args.video else None, camera=args.camera)
    obs, _ = env.reset(seed=ep["seed"])
    max_dev = float(np.max(np.abs(obs - ep["obs"][0])))
    frames = []
    info = {}
    for k, a in enumerate(ep["actions"]):
        obs, _, term, trunc, info = env.step(a)
        max_dev = max(max_dev, float(np.max(np.abs(obs - ep["obs"][k + 1]))))
        if args.video and k % 2 == 0:
            frames.append(env.render())
        if term:
            break
    print(f"replayed {k + 1} steps of seed {ep['seed']}: success={info.get('is_success')} "
          f"(recorded {ep['success']}), max |obs - recorded obs| = {max_dev:.2e}")
    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=10, macro_block_size=1)
        print(f"video -> {args.video}")
    env.close()
    return 0 if bool(info.get("is_success")) == ep["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
