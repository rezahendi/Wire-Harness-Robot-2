"""Run the benchmark suite in Isaac Sim without having to set PYTHONPATH by hand.

    python run_isaac_benchmarks.py --out results/isaac.json [--quick]

Everything else (which benchmarks exist, how they are scored) lives in
``harness_core.benchmarks`` and ``harness_bench.run``, shared with the MuJoCo run.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "harness_core"), os.path.join(ROOT, "harness_bench"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    from harness_bench.run import main
    argv = sys.argv[1:]
    if not any(a.startswith("--engine") for a in argv):
        argv = ["--engine", "isaac"] + argv
    sys.exit(main(argv))
