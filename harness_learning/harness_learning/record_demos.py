"""Record expert demonstrations for imitation learning.

    ros2 run harness_learning record_demos --episodes 50 --out ~/harness_demos
    python3 -m harness_learning.record_demos --episodes 50 --out demos --workers 4

Successful episodes are kept (``--keep-failures`` keeps all). Seeds are
consecutive from ``--seed``; every saved episode can be replayed exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from typing import Any, Dict, Optional


def _worker(args) -> Dict[str, Any]:
    seed, out_dir, config_path, randomize, keep_failures, max_time, video = args
    if video:
        from harness_core.render_util import choose_gl_backend
        choose_gl_backend()
    from harness_learning.demos import rollout_expert, save_episode
    from harness_learning.env import HarnessRoutingEnv

    env = HarnessRoutingEnv(config_path=config_path, randomize=randomize, max_episode_time=max_time,
                            render_mode="rgb_array" if video else None)
    try:
        ep = rollout_expert(env, seed, render_every=2 if video else 0)
    except Exception as exc:  # keep recording other seeds
        return {"seed": seed, "success": False, "error": repr(exc)}
    rec = {"seed": seed, "success": ep["success"], "steps": int(len(ep["actions"])),
           "sim_time": float(ep["info"]["sim_time"]), "wall_time": float(ep["wall_time"]),
           "return": float(ep["rewards"].sum()), "n_routed": int(ep["info"]["n_routed"]),
           "file": None}
    if ep["success"] or keep_failures:
        name = f"episode_{seed:06d}.npz"
        save_episode(os.path.join(out_dir, name), ep, env)
        rec["file"] = name
        if video and ep["frames"]:
            import imageio
            imageio.mimsave(os.path.join(out_dir, f"episode_{seed:06d}.mp4"), ep["frames"], fps=10,
                            macro_block_size=1)
    env.close()
    return rec


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=20, help="number of expert rollouts")
    ap.add_argument("--seed", type=int, default=0, help="first episode seed")
    ap.add_argument("--out", default="harness_demos", help="output directory")
    ap.add_argument("--config", default=None, help="cell YAML (default: built-in config)")
    ap.add_argument("--no-randomize", action="store_true", help="nominal layout every episode")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--max-time", type=float, default=150.0, help="episode time limit [s]")
    ap.add_argument("--video", action="store_true", help="also write an mp4 per saved episode")
    args = ap.parse_args(argv)

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    jobs = [(args.seed + i, out, args.config, not args.no_randomize, args.keep_failures,
             args.max_time, args.video) for i in range(args.episodes)]
    t0 = time.time()
    results = []
    print(f"recording {args.episodes} expert episodes into {out} with {args.workers} worker(s)")
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for rec in pool.imap_unordered(_worker, jobs):
            results.append(rec)
            status = "ok  " if rec.get("success") else "fail"
            extra = rec.get("error") or f"{rec['steps']} steps, {rec['sim_time']:.0f}s sim, {rec['wall_time']:.0f}s wall"
            print(f"  [{len(results):3d}/{args.episodes}] seed {rec['seed']:6d} {status} {extra}", flush=True)
    results.sort(key=lambda r: r["seed"])
    n_ok = sum(r.get("success", False) for r in results)
    index = {"episodes": results, "n_success": n_ok, "n_total": len(results),
             "success_rate": n_ok / max(1, len(results)), "wall_time": time.time() - t0}
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    print(f"done: {n_ok}/{len(results)} successful ({100.0 * index['success_rate']:.0f}%), "
          f"{time.time() - t0:.0f}s total -> {out}/index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
