"""Run a skill with a learned GR00T policy instead of the force-guided expert (level 2).

The runner plugs into CellSession.skill_runners. It renders the two camera views, builds the
state vector exactly as the demo recorder did, asks the GR00T server for a 16-step action
chunk, executes the first ``execute_horizon`` steps through the same 20 Hz action interface
(the 500 Hz admittance controller underneath is unchanged), and asks again.

The skill counts as done only when the policy has done everything the expert does: grasped
the wire, left it inside the fork's slot, opened the gripper and lifted clear, held for half a
second. Otherwise it runs into the time limit and the skill reports ``timeout``, so the
planner sees an ordinary failed skill and can retry (by default the retry uses the expert).

With ``ensemble_decay`` set, the runner asks for a new chunk every ``execute_horizon`` steps
and executes the weighted average of every chunk that covers the current step (temporal
ensembling, as in ACT), weight exp(-decay x the chunk's age in steps): consecutive chunks are
independent samples of the action distribution, and averaging them steadies the motion and
keeps the policy from hopping between plans.

With ``restarts`` > 0, a try that stalls (the gripper let go of the wire outside the slot, or
no new milestone for 12 s) is cleared away the way a failed skill is (gripper open, tool up)
and the policy starts over from there, as often as allowed within the time limit.

Along the way the runner notes how far the policy got (``self.last["milestones"]``, seconds
after the start): the wire really in the hand, lifted above the prongs, carried over the fork
lined up with its slot, inside the slot, released there. The evaluation turns these into a
funnel.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

from . import groot_features as gf

REFUSED = ("infeasible_spec", "unknown_fork", "previous_fork_not_seated")
MILESTONES = ("in_hand", "lifted", "over_slot", "inside", "released")
TRAJECTORY_COLUMNS = ["t", "tcp_x", "tcp_y", "tcp_z", "tcp_yaw", "gripper", "fx", "fy", "fz", "slot_y", "slot_z",
                      "inside", "a_dx", "a_dy", "a_dz", "a_dyaw", "a_grip"]
IN_HAND_MM = 15.0       # closed gripper with a perceived wire point this close to the TCP
OVER_SLOT_MM = 5.0      # wire in hand, crossing the fork's slot plane this close to the slot (any height)
LET_GO_S = 2.0          # with restarts: the gripper open this long after holding the wire, the wire not in the slot
NO_PROGRESS_S = 12.0    # with restarts: no new milestone for this long (a good try takes ~15 s in all)


def ensemble_action(active: List[Tuple[int, np.ndarray]], step: int, decay: float) -> np.ndarray:
    """Temporal ensembling: every chunk's action for ``step`` (chunks as (step asked at, chunk)),
    averaged with weight exp(-decay x the chunk's age in steps)."""
    ages = np.array([step - s0 for s0, _ in active], dtype=float)
    w = np.exp(-float(decay) * ages)
    return (w[:, None] * np.stack([np.asarray(c)[step - s0] for s0, c in active])).sum(0) / w.sum()


class GrootRunner:
    def __init__(self, client, skills: Iterable[str] = ("route_fork",), forks: Optional[Iterable[str]] = None,
                 attempts: Optional[Iterable[int]] = (0,), execute_horizon: int = 8, max_seconds: float = 40.0,
                 name: str = "GR00T N1.7 (fine-tuned)", ensemble_decay: Optional[float] = None,
                 record_trajectory: bool = False, restarts: int = 0):
        self.client = client
        self.skills = set(skills)
        self.forks = None if forks is None else set(forks)
        self.attempts = None if attempts is None else set(int(a) for a in attempts)
        self.execute_horizon = int(execute_horizon)
        self.max_seconds = float(max_seconds)
        self.name = name
        self.ensemble_decay = None if ensemble_decay is None else float(ensemble_decay)
        self.record_trajectory = bool(record_trajectory)       # per step, in self.last["trajectory"]
        self.restarts = int(restarts)                            # fresh starts after a stalled try
        self._cams: Dict[int, gf.Cameras] = {}
        self.last: Dict[str, Any] = {}
        self._state_keys: Optional[List[str]] = None     # the state keys the served model was trained with

    def wants(self, skill: str, target: str = "", attempt: int = 0) -> bool:
        if skill not in self.skills:
            return False
        if self.forks is not None and target and target not in self.forks:
            return False
        return self.attempts is None or int(attempt) in self.attempts

    def cameras(self, session) -> gf.Cameras:
        key = id(session.env.cell.sim.model)
        if key not in self._cams:
            self._cams[key] = gf.Cameras(session.env.cell.sim.model)
        return self._cams[key]

    def state_keys(self) -> List[str]:
        """Asked once from the server; models trained before the geometry keys get the base ones."""
        if self._state_keys is None:
            keys = self.client.modality_keys().get("state") or [k for k, _ in gf.STATE_LAYOUT]
            known = {k for k, _ in gf.STATE_LAYOUT}
            unknown = [k for k in keys if k not in known]
            if unknown:
                raise ValueError(f"the served model expects state keys this runner does not know: {unknown}")
            self._state_keys = list(keys)
        return self._state_keys

    def _chunk(self, session, obs, i: int, text: str) -> np.ndarray:
        imgs = self.cameras(session).render(session.env.cell.sim.data)
        parts = gf.state_parts(obs, "route_fork", i, session.cfg)
        out = self.client.get_action(gf.observation_for_policy(imgs, parts, text, self.state_keys()))
        chunk = gf.join_action({k: np.asarray(v)[0] for k, v in out.items()})
        return np.clip(chunk, -1.0, 1.0)

    def route_fork(self, session, i: int) -> Generator[np.ndarray, Dict[str, np.ndarray], bool]:
        """The policy routes the wire into fork ``i``. With ``restarts`` > 0 a try that stalls
        (let go of the wire outside the slot, or no new milestone for a while) is cleared away,
        gripper open and tool up, and the policy starts over, all within ``max_seconds``."""
        t0 = float(session.obs["time"][0])
        self.last = {"fork": session.route[i], "calls": 0, "grasped": False, "milestones": {}, "furthest": None,
                     "tries": 1, "stalls": []}
        if self.record_trajectory:
            self.last["trajectory"] = []
            self.last["trajectory_columns"] = TRAJECTORY_COLUMNS
        while True:
            watch = len(self.last["stalls"]) < self.restarts      # the last try runs until the time limit
            stall = yield from self._try(session, i, t0, watch)
            if stall is None:
                return True
            self.last["stalls"].append({"t": round(float(session.obs["time"][0]) - t0, 2), "why": stall})
            yield from session._clear_board()
            self.last["tries"] += 1

    def _try(self, session, i: int, t0: float,
             watch: bool) -> Generator[np.ndarray, Dict[str, np.ndarray], Optional[str]]:
        """One go at the fork: None once the wire is routed, else (only when ``watch``) why it stalled."""
        from harness_core.perception import cable_crossing_in_fork

        cfg = session.cfg
        text = gf.instruction("route_fork", session.route[i])
        obs = session.obs
        bz = float(obs["board_z"][0])
        z_top = bz + cfg.fork.post_height + cfg.fork.prong_height
        z_clear = z_top + 0.03
        wire_d = 2.0 * cfg.wire.radius
        inside_since = open_outside_since = None
        held, seen, progress_t = False, set(), float(obs["time"][0])
        queue: list = []
        active: List[Tuple[int, np.ndarray]] = []          # (step it was asked at, chunk) when ensembling
        step = next_ask = 0
        traj = self.last.get("trajectory")
        while True:
            if self.ensemble_decay is None:
                if not queue:
                    queue = list(self._chunk(session, obs, i, text)[: self.execute_horizon])
                    self.last["calls"] += 1
                action = queue.pop(0)
            else:
                active = [(s0, c) for s0, c in active if step - s0 < len(c)]
                if step >= next_ask or not active:              # never run out, even with a horizon > the chunk
                    active.append((step, self._chunk(session, obs, i, text)))
                    next_ask = step + self.execute_horizon
                    self.last["calls"] += 1
                action = ensemble_action(active, step, self.ensemble_decay)
            sent = np.asarray(action, dtype=float)
            obs = yield action
            step += 1
            t = float(obs["time"][0])
            opening = float(obs["gripper"][0])
            closed = opening < wire_d + 0.003
            if closed:
                self.last["grasped"] = True
            chk = cable_crossing_in_fork(obs["cable"], obs["forks"][i], cfg.fork, bz)
            now = self._milestones(obs, i, closed, z_top, chk, cfg.fork.slot_width, opening > 0.02, t - t0)
            if set(now) - seen:
                seen.update(now)
                progress_t = t
            held = held or "in_hand" in now
            if traj is not None:
                tcp = obs["tcp_pos"]
                found = bool(np.isfinite(chk["y"]))
                traj.append([round(t - t0, 3), *(round(float(v), 4) for v in tcp), round(float(obs["tcp_yaw"][0]), 3),
                             round(opening, 4), *(round(float(v), 2) for v in obs["wrench"][:3]),
                             round(float(chk["y"]), 4) if found else None,
                             round(float(chk["z"]), 4) if found else None, int(bool(chk["inside"])),
                             *(round(float(v), 3) for v in sent)])
            if self.last["grasped"] and chk["inside"] and opening > 0.02 and float(obs["tcp_pos"][2]) > z_clear:
                inside_since = t if inside_since is None else inside_since
                if t - inside_since >= 0.5:
                    self.last["seconds"] = round(t - t0, 2)
                    return None
            else:
                inside_since = None
            if not watch:
                continue
            if held and opening > 0.02 and not chk["inside"]:
                open_outside_since = t if open_outside_since is None else open_outside_since
                if t - open_outside_since >= LET_GO_S:
                    return "let go outside the slot"
            else:
                open_outside_since = None
            if t - progress_t >= NO_PROGRESS_S:
                return f"no progress for {NO_PROGRESS_S:.0f} s"

    def _milestones(self, obs, i: int, closed: bool, z_top: float, chk: Dict[str, Any], slot_width: float,
                    opened: bool, t: float) -> List[str]:
        """Note the milestones reached for the first time (seconds after the start); returns those true now."""
        reached = self.last["milestones"]
        tcp = np.asarray(obs["tcp_pos"], dtype=float)
        cable = np.asarray(obs["cable"], dtype=float)
        in_hand = closed and float(np.min(np.linalg.norm(cable - tcp, axis=1))) < IN_HAND_MM / 1000.0
        fork_mm = 1000.0 * float(np.linalg.norm(np.asarray(obs["forks"][i][:2], dtype=float) - tcp[:2]))
        self.last["fork_distance_mm"] = round(fork_mm, 1)
        lined_up = abs(float(chk["y"])) < slot_width / 2 + OVER_SLOT_MM / 1000.0
        inside = bool(chk["inside"])
        now = [name for name, ok in (("in_hand", in_hand), ("lifted", in_hand and tcp[2] > z_top + 0.01),
                                     ("over_slot", in_hand and lined_up), ("inside", inside),
                                     ("released", inside and opened)) if ok]
        for name in now:
            if name not in reached:
                reached[name] = round(t, 2)
                if self.last["furthest"] is None or MILESTONES.index(name) > MILESTONES.index(self.last["furthest"]):
                    self.last["furthest"] = name
        return now

    def close(self) -> None:
        for c in self._cams.values():
            c.close()
        self._cams.clear()


def connect_runner(address: Optional[str] = None, attempts: str = "0", timeout_ms: int = 120000,
                   restarts: int = 0, max_seconds: float = 40.0) -> GrootRunner:
    """A runner for the GR00T policy server at HOST:PORT (default 127.0.0.1:5556).

    ``attempts`` says which route_fork attempts the policy takes: "0" (the first one; a retry
    after a failure goes to the expert), "0,1", or "all". ``restarts``: fresh starts the
    policy gets after a stalled try, within ``max_seconds``."""
    from .groot_client import DEFAULT_PORT, GrootClient
    host, port = "127.0.0.1", DEFAULT_PORT
    if address:
        h, _, p = address.rpartition(":")
        if p.isdigit():
            host, port = h or host, int(p)
        else:
            host = address
    client = GrootClient(host, port, timeout_ms=timeout_ms)
    if not client.ping():
        client.close()
        raise SystemExit(f"no GR00T policy server at {host}:{port} (start gr00t/eval/run_gr00t_server.py "
                         f"with --port {port})")
    chosen = None if attempts == "all" else [int(a) for a in str(attempts).split(",") if a.strip()]
    return GrootRunner(client, attempts=chosen, restarts=restarts, max_seconds=max_seconds)


def route_stats(tool_calls: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    """route_fork calls that ran (refusals left out), by controller, and how many routed the wire."""
    out = {"groot_routes": 0, "groot_ok": 0, "expert_routes": 0, "expert_ok": 0}
    for c in tool_calls or []:
        r = c.get("result") or {}
        if c.get("name") != "route_fork" or r.get("skill") != "route_fork" or r.get("outcome") in REFUSED:
            continue
        who = "groot" if str(r.get("executed_by", "")).startswith("GR00T") else "expert"
        out[f"{who}_routes"] += 1
        out[f"{who}_ok"] += bool(r.get("ok"))
    return out
