"""First-run check: build one wire in Isaac Sim, let it droop, compare with theory.

    python -m harness_isaac.smoke_test

Expected output: a tip droop within a few mm of the elastica reference (it will be on
the stiff side, like every lumped-parameter chain). If this runs, the rest of the
benchmark suite will at least start.
"""

from __future__ import annotations

import sys

from harness_core import benchmarks as B
from harness_core.config import CellConfig


def main() -> int:
    from harness_isaac.rigs_isaac import IsaacRigs
    cfg = CellConfig()
    rigs = IsaacRigs(cfg)
    props = rigs.properties()
    print(f"engine: {rigs.version()}")
    print(f"wire: EI = {props['EI']:.3e} N m^2, {1000 * props['mass_per_length']:.1f} g/m")
    length, seg = 0.09, 0.015
    trace = rigs.cantilever(length, seg, B.CantileverSpec(settle_time=3.0))
    m = B.analyse_cantilever(trace["points"], trace["clamp"], props["EI"],
                             props["weight_per_length"], length)
    print(f"{trace['n_segments']} segments of {1000 * trace['segment_length']:.1f} mm")
    print(f"tip droop      : {1000 * m['tip_drop']:7.2f} mm")
    print(f"elastica       : {1000 * m['elastica_tip_drop']:7.2f} mm")
    print(f"relative error : {100 * m['tip_drop_rel_error']:+7.1f} %")
    ok = bool(trace["finite"]) and abs(m["tip_drop"]) < length
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
