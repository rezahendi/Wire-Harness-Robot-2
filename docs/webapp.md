# Mission Control: the web app

One page to plan, run and watch builds of the simulated cell:

* **Left:** pick a harness (or upload a spec), see its drawing and the spec checks, choose the
  options and start a build. Past builds are listed below and open as replays.
* **Centre:** the cell in 3D, live. The ring under the tool and the tag above it say who is
  moving the robot at that moment: **GR00T N1.7** (green), **force-controlled seating**
  (orange), the **force-guided expert skill** (blue), a scripted **disturbance** (red). The
  purple halo marks the fixture of the planner's current step. Drag to turn, scroll to zoom;
  *Follow the tool* keeps the camera on the gripper.
* **Right:** every decision of the planner with its reasoning (Nemotron), who executed it, how
  long it took and the highest contact force; below, the robot's phase, force, gripper and
  speed, a 12-second force trace and what the simulator says is done.
* **Results tab:** the GR00T evaluation on 40 unseen boards (from `webapp/results.json`).

After a build, the timeline under the 3D view replays it: coloured by who was acting, with one
label per step; space plays or pauses, the arrow keys step a second (shift: five).

## Run it on the GPU VM (GR00T + Nemotron)

```bash
tmux new -s app
bash ~/Wire-Harness-Robot-2/scripts/mission_control.sh
```

The script installs the web server into `~/simenv` the first time, starts GR00T's policy
server with `~/ckpt/route_v6/checkpoint-20000` (about a minute to load; `CKPT=...` for another
checkpoint) and serves the app on port 8000 of the VM. The Nebius key comes from
`NEBIUS_API_KEY` or `~/.config/nebius/api_key`; without it the app offers the scripted planner
only. `Ctrl-b d` leaves it running; `Ctrl-c` stops the app and the policy server.

On the laptop, forward the port and open the page:

```bash
ssh -L 8000:localhost:8000 reza@<the VM's IP>
```

then <http://localhost:8000>. Builds are kept in `~/runs/webapp/<build id>/` (`events.json` for
the replay, `trace.json`, `report.md`, `scene3d.json` for the standalone viewer).

## Run it anywhere else

```bash
pip install fastapi "uvicorn[standard]"
python -m harness_agent.webapp                                   # http://127.0.0.1:8000
python -m harness_agent.webapp --groot 127.0.0.1:5556            # with a GR00T policy server
```

Without a GPU, a replay server can stand in for GR00T on a board whose expert demonstrations
were recorded (it plays the episode recorded for each fork it is asked about):

```bash
python -m harness_agent.groot_data record --out data/board3 --seeds 3 --popped 0
python -m harness_agent.groot_replay_server data/board3 --by-instruction --seed 3 &
python -m harness_agent.webapp --groot 127.0.0.1:5556            # then start a build on board 3
```

## Options of a build

| option | what it does |
|---|---|
| Planner | **Nemotron** on Nebius Token Factory decides every step, or the **scripted** rule policy (no LLM) |
| Routing | **GR00T** routes each fork first (force-controlled seating finishes a snap-in it is stuck on for 1.5 s; a retry goes to the expert, and once a fork has lost its wire the expert does the rest of the routing), or the **expert** routes every fork |
| Disturbance | the wire is pulled out of the second fork after it was routed, and/or the connector slips out of the fingers; the planner is not told |
| Camera | a vision model on Token Factory checks every fixture in rendered images before the planner may finish |
| Board | with *varied* on, the number picks a randomised board (the fixtures moved a little, the wire's stiffness, friction and slack varied); off, the spec's nominal board |
| Pace | real time, or as fast as the simulator runs |

GR00T was fine-tuned on the Door module (`demo_3fork.yaml`); on other harnesses the expert
does more of the work.

## The server

`harness_agent/webapp/server.py` (FastAPI) runs one build at a time in a worker thread
(`builds.py`) and streams it over a websocket. Each build emits `scene` (the cell's geometry,
once), `frame` (20 per robot second: every moving body's pose, the tool, the force, the
gripper, the skill's phase and new log lines), `planner`, `plan` (Nemotron's reasoning),
`call_start` / `call_end`, `disturbance`, `visual_check` and finally `done` or `failed`.

| endpoint | |
|---|---|
| `GET /api/status` | key present, GR00T server reachable, a build running |
| `GET /api/specs`, `POST /api/specs` | the packaged and uploaded specs with their checks |
| `GET /api/specs/{file}/drawing.png` | the assembly drawing |
| `POST /api/builds`, `GET /api/builds` | start a build; this run's and earlier builds |
| `GET /api/builds/{id}/events` | every event of a build (the replay) |
| `GET /api/builds/{id}/files/{path}` | its report, trace and 3D scene |
| `WS /ws/builds/{id}` | the events so far, then live until `done` |
| `GET /api/results` | `results.json` for the Results tab |

Tests: `harness_agent/test/test_webapp.py` (needs `httpx`; the whole-build test runs with
`HARNESS_SLOW_TESTS=1`).
