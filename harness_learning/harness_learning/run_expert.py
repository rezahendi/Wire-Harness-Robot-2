"""Run the scripted, force-guided expert in simulation without ROS.

    ros2 run harness_learning run_expert --seed 3 --video expert_seed3.mp4
    python3 -m harness_learning.run_expert --episodes 10          # success statistics
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default="", help="mp4 path (first episode only)")
    ap.add_argument("--camera", default="overview", help="overview | top | side | wrist")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.video:
        from harness_core.render_util import choose_gl_backend
        choose_gl_backend()

    from harness_learning.demos import rollout_expert
    from harness_learning.env import HarnessRoutingEnv

    n_ok = 0
    for k in range(args.episodes):
        seed = args.seed + k
        video = bool(args.video) and k == 0
        env = HarnessRoutingEnv(config_path=args.config, randomize=not args.no_randomize,
                                render_mode="rgb_array" if video else None, camera=args.camera)
        t0 = time.time()
        ep = rollout_expert(env, seed, render_every=2 if video else 0)
        n_ok += ep["success"]
        if not args.quiet:
            for t, msg in ep["expert_log"]:
                print(f"  [{t:6.1f} s] {msg}")
        info = ep["info"]
        print(f"seed {seed}: success={ep['success']} forks={info['forks_routed']} "
              f"connector_seated={info['connector_seated']} sim {info['sim_time']:.0f} s, "
              f"wall {time.time() - t0:.0f} s, return {ep['rewards'].sum():.2f}")
        if video and ep["frames"]:
            import imageio
            imageio.mimsave(args.video, ep["frames"], fps=10, macro_block_size=1)
            print(f"video -> {args.video}")
        env.close()
    if args.episodes > 1:
        print(f"success rate: {n_ok}/{args.episodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
