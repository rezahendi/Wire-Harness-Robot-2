"""Record a build in 3D for the browser viewer: what each control level does, frame by frame.

    python -m harness_agent.replay3d runs/showcase_both_s4             # writes runs/.../scene3d.json
    python -m harness_agent.replay3d runs/showcase_both_s4 --html      # and a standalone viewer

It re-runs the planner's recorded tool calls (same spec, seed and disturbance scenario) in
the simulator and records, at the 20 Hz skill rate:

* level 3, the planner: which tool call is running, its result, and the model's reasoning
  (taken from the trace, so it is what Nemotron actually wrote),
* level 2, the skill: the expert's phase, its log messages and the compliance target it
  commands,
* level 1, the admittance controller: the target it follows, where the tool actually is,
  the filtered wrench it reacts to and the twist it commands,

plus the pose of every moving body, so the viewer can redraw the MuJoCo scene with plain
boxes, capsules and cylinders. Every call's outcome is checked against the trace; the file
says whether the replay matched.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .session import CellSession
from .spec import HarnessSpec

POS_SCALE = 20000.0          # int16 units per metre (0.05 mm, range +-1.6 m)
QUAT_SCALE = 32767.0

PHASES = {
    "start": "Waiting for the next skill.",
    "init": "Waiting for the next skill.",
    "done": "Idle: the skill has finished.",
    "pick_approach": "Move above the grasp point on the wire, fingers open.",
    "pick_descend": "Guarded touch-down: lower until the fingertips feel the board (1.5 N), then back off 1.5 mm.",
    "pick_close": "Close on the wire and check the opening matches the wire's diameter.",
    "route_lift": "Lift the held wire so it cannot drag across the fork posts.",
    "route_transit": "Sweep around the last fixed point with the wire nearly taut, over the fork.",
    "route_descend": "Lower past the fork, keeping the wire tension between 2.5 and 6 N: "
                     "move away when it goes slack, give way when it pulls.",
    "route_seat": "Ramp the tension to 10 N, centre the wire over the slot from perception, "
                  "wiggle until it snaps past the barbs.",
    "route_release": "Open, lift, and check the wire stayed in the slot.",
    "connector_tip": "The connector stands on its end: pull it over by its wire.",
    "connector_relocate": "Move the connector to a clear spot where the fingers can reach around it.",
    "connector_approach": "Move above the connector, turned to match it.",
    "connector_grasp": "Descend and grasp the connector.",
    "connector_transit": "Carry it behind the holder so the wire leaves through the back slot.",
    "connector_align": "Line up over the pocket.",
    "connector_descend": "Guarded descent until the connector touches (about 2 N).",
    "connector_search": "Spiral search, pushing down with 3 N, until the connector drops into the pocket.",
    "connector_press": "Press with 6 N and a small wiggle until the latch clicks.",
    "connector_release": "Open, back off, and check the connector is seated.",
    "retreat": "Move the arm out of the camera's view.",
    "disturbance": "Nothing: the robot holds still while the disturbance happens.",
}


def _b64_int16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<i2").tobytes()).decode("ascii")


def find_spec(name: str) -> str:
    """Packaged spec file whose harness name matches ``name``."""
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dirs = [os.path.join(here, "specs")]
    try:
        from ament_index_python.packages import get_package_share_directory
        dirs.insert(0, os.path.join(get_package_share_directory("harness_agent"), "specs"))
    except Exception:            # no ROS environment: the source tree is enough
        pass
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            try:
                if HarnessSpec.from_yaml(path).name == name:
                    return path
            except Exception:
                continue
    raise SystemExit(f"no packaged spec named {name!r}; pass --spec")


class RecordingSession(CellSession):
    """A CellSession that snapshots the cell after every 20 Hz step."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.rec: Optional[Dict[str, Any]] = None
        super().__init__(*args, **kwargs)
        if self.env is not None:
            self._start_recording()

    # ---------------------------------------------------------------- setup
    def _start_recording(self) -> None:
        import mujoco
        sim = self.env.cell.sim
        m, d = sim.model, sim.data
        moving = np.zeros(m.nbody, bool)
        for b in range(1, m.nbody):
            moving[b] = m.body_jntnum[b] > 0 or moving[m.body_parentid[b]]
        self._moving = [int(b) for b in np.flatnonzero(moving)]
        name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or ""   # noqa: E731
        geoms = []
        for g in range(m.ngeom):
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            if m.geom_group[g] > 2 or rgba[3] <= 0.01:
                continue
            geoms.append({"type": int(m.geom_type[g]), "body": int(m.geom_bodyid[g]),
                          "name": name(mujoco.mjtObj.mjOBJ_GEOM, g),
                          "size": [round(float(v), 5) for v in m.geom_size[g]],
                          "rgba": [round(float(v), 3) for v in rgba],
                          "pos": [round(float(v), 5) for v in m.geom_pos[g]],
                          "quat": [round(float(v), 5) for v in m.geom_quat[g]]})
        static = {int(b): [round(float(v), 5) for v in np.concatenate([d.xpos[b], d.xquat[b]])]
                  for b in range(m.nbody) if not moving[b]}
        forks = []
        for i, fid in enumerate(self.route):
            p, R = sim.fork_pose(i)
            forks.append({"id": fid, "pos": [round(float(v), 4) for v in p],
                          "yaw": round(float(np.arctan2(R[1, 0], R[0, 0])), 4)})
        hp, _ = sim.holder_seat_pose()
        cam = {}
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.env.camera) if self.env.camera else -1
        if cid >= 0:
            pos = d.cam_xpos[cid].copy()
            fwd = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
            # look at the board plane
            bz = float(self.obs["board_z"][0])
            s = (bz - pos[2]) / fwd[2] if abs(fwd[2]) > 1e-6 else 1.0
            cam = {"pos": [round(float(v), 4) for v in pos],
                   "lookat": [round(float(v), 4) for v in pos + max(s, 0.3) * fwd],
                   "fovy": float(m.cam_fovy[cid])}
        self.rec = {
            "bodies": [name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)],
            "moving": self._moving, "static": static, "geoms": geoms,
            "forks": forks, "holder": [round(float(v), 4) for v in hp],
            "connector_body": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "connector"),
            "board_z": round(float(self.obs["board_z"][0]), 4), "camera": cam,
            "wire_bodies": [b for b in range(m.nbody) if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("wire_")],
            "t": [], "poses": [], "tcp": [], "target": [], "target_yaw": [], "wrench": [], "twist": [],
            "gripper": [], "phase": [], "fork": [], "call": [], "mode": [], "log_n": [], "truth": [],
            "pokes": {}}
        self._phase_ids: Dict[str, int] = {}
        self.call_index = -1
        self.mode = "skill"
        self._record()

    # ------------------------------------------------------------- snapshot
    def _grab_frame(self, force: bool = False) -> None:
        super()._grab_frame(force)
        if self.rec is not None:
            self._record()

    def _record(self) -> None:
        cell = self.env.cell
        sim = cell.sim
        d = sim.data
        r = self.rec
        t = round(float(sim.time), 3)
        if r["t"] and t <= r["t"][-1]:
            return
        r["t"].append(t)
        mv = self._moving
        pose = np.concatenate([np.clip(np.round(d.xpos[mv] * POS_SCALE), -32767, 32767),
                               np.round(d.xquat[mv] * QUAT_SCALE)], axis=1)
        r["poses"].append(pose.astype(np.int16))
        p, _ = sim.tcp_pose()
        r["tcp"].append([int(round(v * 1e4)) for v in p])                  # 0.1 mm
        r["target"].append([int(round(v * 1e4)) for v in cell.target.position])
        R = cell.target.rotation
        r["target_yaw"].append(round(float(np.arctan2(R[1, 0], R[0, 0])), 3))
        w = cell.ctrl.last_wrench_world
        r["wrench"].append([round(float(v), 2) for v in w])
        tw = cell.ctrl.last_twist
        r["twist"].append([round(float(v) * 1000, 1) for v in tw[:3]])        # mm/s
        r["gripper"].append(round(float(sim.gripper_opening()) * 1000, 1))    # mm
        ph = "disturbance" if self.mode == "disturbance" else self.expert.phase
        if ph not in self._phase_ids:
            self._phase_ids[ph] = len(self._phase_ids)
        r["phase"].append(self._phase_ids[ph])
        r["fork"].append(int(self.expert.current_fork))
        r["call"].append(int(self.call_index))
        r["mode"].append(self.mode)
        r["log_n"].append(len(self.expert.log))
        st = sim.task_status()
        r["truth"].append([int(bool(v)) for v in st["forks_routed"]] + [int(bool(st["connector_seated"])),
                                                                        int(bool(st.get("connector_latched")))])
        pokes = []
        for b in r["wire_bodies"]:
            f = d.xfrc_applied[b, :3]
            if float(np.linalg.norm(f)) > 1e-6:
                pokes.append([round(float(v), 4) for v in d.xpos[b]] + [round(float(v), 3) for v in f])
        if pokes:
            r["pokes"][str(len(r["t"]) - 1)] = pokes


def _short(result: Dict[str, Any]) -> Dict[str, Any]:
    """What the planner was told, trimmed for the viewer."""
    out: Dict[str, Any] = {}
    for k in ("ok", "outcome", "error", "refused", "messages", "arm_clear", "method", "recorded"):
        if k in result:
            out[k] = result[k]
    st = result.get("state") or (result if "forks" in result else None)
    if isinstance(st, dict) and "forks" in st:
        out["forks_in_slot"] = [f for f, v in st["forks"].items() if v.get("wire_in_slot")]
        c = st.get("connector") or {}
        out["connector_in_holder"] = c.get("in_holder")
    if "vision" in result:
        out["vision"] = {k: {"seated": v.get("seated"), "model": v.get("model"), "evidence": v.get("evidence")}
                         for k, v in (result["vision"] or {}).items() if isinstance(v, dict)}
    return out


def record(trace_path: str, spec_path: Optional[str] = None, opening_status: bool = False,
           log=print) -> Dict[str, Any]:
    from .annotate import _reasoning_per_call
    from .disturbances import make_scenario
    from .tools import ToolBox

    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    sess = trace.get("session") or {}
    spec_path = spec_path or find_spec(sess["spec"])
    spec = HarnessSpec.from_yaml(spec_path)
    session = RecordingSession(spec, seed=int(sess.get("seed", 0)), randomize=bool(sess.get("randomize", False)))
    scenario = make_scenario(trace.get("scenario") or "nominal", session.route)
    for trig in scenario.triggers:                       # mark the frames of a disturbance
        act = trig.action

        def wrapped(s, _act=act):
            s.mode = "disturbance"
            try:
                return _act(s)
            finally:
                s.mode = "skill"
        trig.action = wrapped
    box = ToolBox(session, scenario=scenario)
    if opening_status:
        session.status()               # a Nemotron run starts with the state in the opening message
    reasons = _reasoning_per_call(trace)
    calls: List[Dict[str, Any]] = []
    matched = True
    t_wall = time.perf_counter()
    for i, c in enumerate(trace.get("tool_calls") or []):
        session.call_index = i
        t0 = session.sim_time
        n_dist = len(box.disturbances)
        out = box.call(c["name"], c.get("arguments") or {})
        orig = c.get("result") or {}
        same = (out.get("outcome") == orig.get("outcome")) and (("error" in out) == ("error" in orig))
        matched &= same
        log(f"  {i + 1:2d} {c['name']:<18} {json.dumps(c.get('arguments') or {})[:40]:<40} "
            f"trace: {orig.get('outcome', 'ok' if 'error' not in orig else 'error'):<24} "
            f"replay: {out.get('outcome', 'ok' if 'error' not in out else 'error'):<24} "
            f"t {session.sim_time:6.1f} s (trace {c.get('sim_time', 0):6.1f})" + ("" if same else "  MISMATCH"))
        calls.append({"name": c["name"], "arguments": c.get("arguments") or {},
                      "t_start": round(t0, 2), "t_end": round(session.sim_time, 2),
                      "result": _short(orig), "replay_outcome": out.get("outcome"),
                      "reasoning": reasons[i] if i < len(reasons) else "",
                      "disturbances": [{"label": d["label"], "t": d["sim_time"]}
                                       for d in box.disturbances[n_dist:]]})
    session.call_index = len(calls)
    session._hold(1.0)                                   # a second of stillness at the end
    r = session.rec
    phases = sorted(session._phase_ids, key=session._phase_ids.get)
    disturb = [{"label": d["label"], "t": d["sim_time"]} for d in box.disturbances]
    disturb += [{"label": e["label"], "t": round(float(e["sim_time"]), 2)} for e in session.events
                if e.get("type") == "disturbance" and "after" not in e]
    poses = np.stack(r.pop("poses"))                     # (N, M, 7)
    doc = {
        "meta": {"spec": spec.name, "revision": spec.revision, "planner": trace.get("planner"),
                 "model": trace.get("model"), "seed": sess.get("seed"), "scenario": trace.get("scenario"),
                 "success": trace.get("success"), "claimed_success": trace.get("claimed_success"),
                 "replay_matched": bool(matched), "replay_truth": session.truth(),
                 "robot_time_s": round(session.sim_time, 2), "record_hz": 1.0 / session.cfg.sim.policy_dt,
                 "controller": {k: getattr(session.cfg.controller, k) for k in
                                ("kp_lin", "kf_lin", "kp_rot", "kf_rot", "max_lin_vel", "force_deadband",
                                 "wrench_filter_hz", "protective_stop_force")},
                 "record_wall_s": round(time.perf_counter() - t_wall, 1)},
        "calls": calls, "disturbances": sorted(disturb, key=lambda x: x["t"]),
        "phases": phases, "phase_text": {p: PHASES.get(p, p.replace("_", " ")) for p in phases},
        "messages": [[round(float(t), 2), m] for t, m in session.expert.log],
        "pose_scale": POS_SCALE, "quat_scale": QUAT_SCALE,
        "n_frames": int(poses.shape[0]), "n_moving": int(poses.shape[1]),
        "poses_b64": _b64_int16(poses.reshape(-1)),
        **r,
    }
    doc["mode"] = [1 if m == "disturbance" else 0 for m in doc["mode"]]
    session.close()
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run_build output folder with trace.json")
    ap.add_argument("--spec", help="spec file (default: the packaged spec named in the trace)")
    ap.add_argument("--out", help="output file (default: <run_dir>/scene3d.json)")
    ap.add_argument("--opening-status", action="store_true",
                    help="settle 0.4 s before the first call, as a live Nemotron run does. The physics is "
                         "re-simulated, so tiny numerical differences can still change a later outcome: "
                         "the file records whether every outcome matched the trace")
    args = ap.parse_args(argv)
    doc = record(os.path.join(args.run_dir, "trace.json"), args.spec, opening_status=args.opening_status)
    out = args.out or os.path.join(args.run_dir, "scene3d.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"{out}: {doc['n_frames']} frames, replay {'matched' if doc['meta']['replay_matched'] else 'DIFFERED'}"
          f" ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
