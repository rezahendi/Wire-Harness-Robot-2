"""Summarise a folder of recorded demonstrations.

    ros2 run harness_learning inspect_demos harness_demos
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import numpy as np


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    args = ap.parse_args(argv)
    from harness_learning.demos import load_episode

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.folder), "episode_*.npz")))
    if not files:
        print("no episodes found")
        return 1
    lengths, returns, succ = [], [], []
    acts = []
    for f in files:
        ep = load_episode(f)
        lengths.append(len(ep["actions"]))
        returns.append(float(ep["rewards"].sum()))
        succ.append(ep["success"])
        acts.append(ep["actions"])
    A = np.concatenate(acts)
    layout = load_episode(files[0])["obs_layout"]
    print(f"{len(files)} episodes, {sum(succ)} successful")
    print(f"steps/episode: mean {np.mean(lengths):.0f}, min {np.min(lengths)}, max {np.max(lengths)} "
          f"({np.sum(lengths)} transitions)")
    print(f"return: mean {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print("action mean:", np.round(A.mean(0), 3), " std:", np.round(A.std(0), 3))
    print("observation layout:")
    for k, sl in layout.items():
        print(f"  {k:16s} [{sl.start:3d}:{sl.stop:3d}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
