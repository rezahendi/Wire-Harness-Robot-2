# GR00T N1.7 as the routing skill: runbook

Level 3 (Nemotron) decides the next step, level 2 (the skill) works out every move, level 1
(the 500 Hz admittance controller) makes it soft and force-limited. Here NVIDIA Isaac GR00T
N1.7 learns level 2 for `route_fork` from the force-guided expert's demonstrations, one
language-conditioned policy for every fork ("route the wire into fork F2"). The controller
underneath and the planner above stay the same.

Measured on a preemptible 1x L40S VM (8 vCPU, 32 GiB, about $0.78/h in eu-north1):

| step | time | cost |
|---|---|---|
| 1. VM + setup | 30 min | ~$0.40 |
| 2. smoke test (replay through GR00T's own server) | 10 min | ~$0.15 |
| 3. record 400 routing demos (7 workers) | ~2 h; the simulator's physics sets the pace, see [Recording speed](#recording-speed) | ~$1.60 |
| 4. fine-tune GR00T N1.7, 6,000 steps at batch 32 | 80-85 min (1.2-1.3 steps/s) | ~$1.10 |
| 5. closed-loop evaluation, 30 trials | see `summary.md` (wall time is printed) | <$1 |

**Stop the VM whenever nothing runs.** A stopped VM costs only its disk. A preemptible VM can
be stopped by Nebius at any time (60 s warning); its disk survives, the public IP may change.

## 1. Create the VM

Nebius AI Cloud console → Compute → Virtual machines → Create (project in **eu-north1**, where
L40S and H100 are offered):

* GPU: **1x L40S** (48 GB; fine-tuning at batch 32 fits). An H100 trains faster per hour.
* Preemptible: yes (much cheaper; everything long runs in tmux and checkpoints are saved).
* Preset: 8 vCPU / 32 GiB. Recording runs one build per vCPU and is limited by the CPU, so
  a preset with more vCPUs records proportionally faster (training does not need them).
* Boot disk: Ubuntu 24.04 **with CUDA**, 200 GiB (each fine-tuning checkpoint is ~36 GB).
* Access: a username and the public key from WSL (`ssh-keygen -t ed25519`, then
  `cat ~/.ssh/id_ed25519.pub`); a public IP.

Then, from WSL: `ssh <username>@<public-ip>` and type `yes` the first time.

## 2. Code and setup (on the VM)

```bash
git clone https://github.com/rezahendi/Wire-Harness-Robot-2.git   # private repo: user name + a GitHub token
bash Wire-Harness-Robot-2/scripts/setup_groot_vm.sh                # ~15 min: packages, ~/simenv, ~/Isaac-GR00T
```

GR00T N1.7's vision-language backbone, `nvidia/Cosmos-Reason2-2B`, is gated: open its page on
huggingface.co and accept the licence, then log in with a **read** token:

```bash
cd ~/Isaac-GR00T && uv run huggingface-cli login
```

Use tmux for everything long: `tmux new -s groot` once; `Ctrl-b c` opens a new window,
`Ctrl-b 0`/`1`/`2` switches, `Ctrl-b d` detaches, `tmux a -t groot` comes back (also after
the SSH connection drops: whatever runs in tmux keeps running).

**Port 5556.** GR00T's server defaults to 5555, but on NVIDIA's VM images the DCGM host engine
(`nv-hostengine`) already listens there ("Address already in use"). Every command here uses
5556, which is also this repo's default.

## 3. Smoke test: the whole loop, before any training

Record two builds, serve them with **GR00T's own server** in replay mode, and let the
simulator execute the served actions through the GR00T client. Replaying the expert's own
actions on the same seeds must route every fork; that checks the data format (including
GR00T's video decoder on our files), the protocol, the observation builder, the chunk
execution and the success check.

```bash
source ~/harness_env.sh
python -m harness_agent.groot_data record --out ~/data/smoke --seeds 1000-1001 --popped 0
python -m harness_agent.groot_data check ~/data/smoke --preview ~/data/smoke_preview.png

# window 1: GR00T's server replaying the recorded set
cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py --dataset-path ~/data/smoke \
    --modality-config-path ~/Wire-Harness-Robot-2/groot/harness_config.py \
    --embodiment-tag NEW_EMBODIMENT --execution-horizon 8 --port 5556

# window 2
source ~/harness_env.sh
python -m harness_agent.groot_eval --replay-dataset ~/data/smoke --seeds 1000-1001 --forks F1,F2,F3 --out ~/eval/smoke
```

Expected: 6/6 routed (measured: 6/6, about 1 ms per chunk). Stop the replay server with Ctrl-c.

## 4. Record the training set

```bash
source ~/harness_env.sh
python -m harness_agent.groot_data bench          # renderer and speed of this machine, 1 min
python -m harness_agent.groot_data record --out ~/data/harness_route --builds 330 --workers 7
python -m harness_agent.groot_data check ~/data/harness_route --preview ~/data/route_preview.png
```

330 randomised builds give about 1,000 routing episodes (F1, F2, F3, plus re-routes of F2
in the quarter of builds where the wire is pulled out). Seeds start at 1000, so the
benchmark seeds 0-99 stay unseen. Every finished build prints a progress line with the
episodes so far and the time left.

* **Stop early:** Ctrl-c once. The recorder packages the episodes finished so far into the
  dataset (about a minute) and writes `meta/stats.json`.
* **Resume** after a preempted VM: run the same command again; finished builds are skipped.
* **Add more data later:** record into a new folder, then combine:
  `python -m harness_agent.groot_data merge ~/data/harness_route ~/data/harness_route2 --out ~/data/route_all`
  (episodes recorded twice are kept once).

The first set (`harness_route`, 403 episodes: F1 132, F2 146, F3 125; 12-24 s each, median
15 s) was stopped early this way.

Two options make demos that teach recovery:

* `--noise 1.0`: while recording, the executed motion is pushed (a smooth jitter, and now and
  then a 1.5-3 cm kick while the gripper moves above the wire), but the expert's clean action
  is what gets recorded. The closed-loop expert brings the gripper back, so the demos show how
  to get back on track (DART). Each build gets a random push scale up to the given value; no
  pushes during the touch-down, closing, seating and release.
* `--hold-after 1.0`: one second of standing still is recorded after each successful call,
  so the policy learns to stop when the wire is in.

### Recording speed

`bench` measures one worker: the expert routes F1 without cameras, then the two policy views
are rendered. On the L40S VM (measured while a fine-tuning run shared the CPUs):

| | per 20 Hz step |
|---|---|
| physics and control (MuJoCo at 1 kHz, admittance control at 500 Hz) | 62 ms |
| both camera views, rendered by the GPU through EGL (`NVIDIA L40S/PCIe/SSE2`) | 4.4 ms |

So a worker records at ~0.75x real time, and the constraint solver is most of it (the wire's
contacts with elliptic friction cones; about 80% of MuJoCo's step time). The recorder stops
each build once the forks are routed when only `route_fork` is recorded, which skips the
connector stage (a fifth to a third of a build's simulated time) without changing a single
recorded step. More vCPUs is the remaining lever.

If `bench` names `llvmpipe` instead of the GPU, the cameras render in software, ~290 ms per
step on a 2-vCPU test machine, many times slower. The fix is NVIDIA's EGL library for the
installed driver; check what is there first, and only install a package whose version
matches the running driver exactly (a mismatched library breaks CUDA until the next reboot):

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
dpkg -l | grep -E "libnvidia-(gl|compute)"
```

## 5. Fine-tune GR00T N1.7

```bash
cd ~/Isaac-GR00T
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/data/harness_route \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ~/Wire-Harness-Robot-2/groot/harness_config.py \
    --num-gpus 1 --output-dir ~/ckpt/route_v1 \
    --max-steps 6000 --save-steps 2000 --save-total-limit 2 \
    --global-batch-size 32 --dataloader-num-workers 8 2>&1 | tee ~/ckpt/route_v1_train.log
```

It tunes the projector and the diffusion action head (the vision-language backbone stays
frozen), with GR00T's default augmentation. `--save-total-limit 2` keeps the disk in check:
a checkpoint with optimizer state is ~36 GB. The recorder already wrote `meta/stats.json` in
GR00T's format (same fingerprints), so training reuses it. Measured on the 403-episode set:
the loss falls from ~1.2 to ~0.1-0.2 within the first 1,000 steps, 1.2-1.3 steps/s.

## 6. Serve the policy and evaluate it in closed loop

```bash
# window 1
cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/ckpt/route_v1/checkpoint-6000 --embodiment-tag NEW_EMBODIMENT --port 5556

# window 2
source ~/harness_env.sh
python -m harness_agent.groot_eval --forks F1,F2,F3 --seeds 0-9 --workers 4 --out ~/eval/route_v1 --video
python -m harness_agent.groot_eval --forks F1,F2,F3 --seeds 0-9 --workers 4 --expert --out ~/eval/expert  # baseline
```

`--workers 4` runs four trials at once against the one server: the server answers one request
at a time (~245 ms per action chunk on the L40S), while the simulator and cameras need the
CPU, so parallel trials keep both busy. `summary.md` has the success rate per fork with a 95%
interval, time, contact force, how far the policy got (wire in the hand, lifted, over the
slot, inside, released) and the server's round-trip time; `results.jsonl` has every trial;
`videos/` one clip per trial.

Two options change how the same checkpoint is run, without retraining:

- `--execute-horizon N --ensemble DECAY` (e.g. `4 --ensemble 0.1`): a new chunk every N steps,
  and every step executes the weighted average of all chunks that cover it, weight
  exp(-DECAY x the chunk's age in steps) (temporal ensembling, as in ACT). Consecutive chunks
  are separate samples of the policy; averaging them steadies the motion near the slot.
- `--trajectories`: every trial's step-by-step record in `trajectories/<fork>_seed<NNN>.json`
  (time, tool position and yaw, gripper opening, force, where the wire crosses the slot plane,
  inside or not, and the action sent), for looking at what the policy does where it fails.
- `--seat-assist SECONDS` (e.g. `1.5`), a hybrid: when the policy has held the wire lined up
  over the slot, low, for that long in all without getting it in, or starts to open the
  gripper there, the expert's force-controlled seating takes over (`seat_from_here`: turn the
  gripper so the wire leaves it along the route, lift a wire that lies on a prong clear and
  line it up again, lower it beyond the fork, centre it, ramp the tension, wiggle it past the
  lips, let go, back off). The summary says how many trials were handed over and how many the
  policy routed alone; report the two apart.
- `--record-takeovers` (with `--seat-assist`): every successful takeover is also saved as a
  training episode in `<out>/takeovers` (camera views, states and the expert's actions,
  starting from the policy's own stuck state, ending with a second's stand-still). That is
  DAgger-style data: the demos the policy is missing are the ones from the states it gets
  itself into. Run it on training boards (e.g. `--seeds 8000-8199`), never on 0-99, and add
  the folder to the next set: `groot_data merge ~/data/route_v5 <out>/takeovers ... --out
  ~/data/route_v6`.

### Results so far

Closed loop on 20 boards never used for training (59 trials; one board where the expert
could not set up F3 is skipped), success = wire seated, released and cleared:

| model | demos | state | steps | routed |
|---|---|---|---|---|
| `route_v1` | 403 | base (47) | 6,000 | 11/59 (19%) |
| `route_v2` | 1,165 (762 with recovery pushes, end hold) | base (47) | 12,000 | 10/59 (17%); again: 14/59; re-planning every 0.2 s: 8/59 |
| `route_v3` | 1,250 with pushes and hold | v3 (66) | 12,000 | **19/59 (32%)** |
| `route_v3`, chunk every 4 steps, ensembled | | | | 20/59 (34%), no protective stop |
| `route_v3`, ensembled + up to 2 restarts in 60 s | | | | **23/59 (39%)** |
| `route_v3`, 8 denoising steps | | | | 13/59 |
| `route_v3_sd0`: the same demos, state dropout off | | v3 (66) | 12,000 | 15/59; with restarts 18/59 |

Successful trials take ~15 s, like the expert, at ~190 ms per action chunk with 4 workers.
The same model runs differ by about 4 successes in 59 (`route_v2`: 10 and 14), so read the
table with that in mind.

What changed things was the state, not the amount of data. Three times the demos, recovery
pushes included, changed nothing (`route_v1` to `route_v2`). The gripper-relative geometry of
`route_v3` (route, wire and slot keys) took seating from 38-56% of the wires carried over the
slot to 77%, and halved the contact force. Ensembling overlapping chunks steadies the motion
(more grasps, no protective stops); restarts rarely save a trial (3 of 39). More denoising
steps and switching GR00T's state dropout off both made things worse.

Where `route_v3` still fails, from its step-by-step trajectories and the test boards rebuilt
in simulation:

* Grasp (most failures): the policy closes the gripper at nearly the same spot on every
  board; its grasp points spread half as much as the expert's (8 vs 17.5 mm on F1). Where the
  wire lies there it holds it (5 mm finger gap); where it lies 15-26 mm to the side the fingers
  close on nothing. The policy does not work out where along the wire to grasp from 8 coarse
  wire points.
* Seat (F1 mostly): the tool presses down 4-5 cm past the fork instead of the expert's 7-9 cm,
  lower than the seating height, while the wire pulls little; the wire then rests on the lips
  at the prong tops and never snaps in. The expert keeps moving away from the fork until the
  wire pulls back, so the wire is taut over the slot.

`route_v5` (state v4) gives the policy the expert's plan, made once from the same perception at
the start of the skill: where on the wire to grasp, the wire's pull along the route, the
seating height and the planned holding distance (pick and seat keys below). The funnel's
"wire in hand" now means the fingers stopped on the wire (2-9 mm apart), not just closed next
to it.

On 40 test boards (seeds 0-39, 119 trials):

| model | routed | protective stops | in hand / lifted / over the slot / inside / released |
|---|---|---|---|
| `route_v3` | 41/119 (34%) | 8 | 92 / 68 / 59 / 44 / 41 |
| `route_v5` (1,261 demos, state v4, 12,000 steps) | 53/119 (45%) | 10 | 101 / 100 / 95 / 63 / 54 |
| `route_v5`, chunk every 4 steps, ensembled | **69/119 (58%)**: F1 20/40, F2 26/40, F3 23/39 | 2 | 111 / 108 / 103 / 71 / 70 |

Trial by trial, `route_v5` routed 32 wires `route_v3` did not and missed 20 it did; the
ensembled run beat the plain one 34 to 18. The plan keys fixed the grasp (wire in hand in 111
of 119 trials, from 92). What is left is the seat: in 32 trials the policy brings the wire
over the slot, low, and it ends up resting on the lips at the prong tops (60-63 mm up), 2-10 mm
off the slot centre, the tool pulling 15-30 N; after 2-7 s there the policy opens the gripper.
Of the 50 failed trials, 31 had the wire held, lined up and low at some point; a successful
seat goes in 0.05-1.65 s after that (median 0.55 s; 4 of 69 took longer).

The expert can finish those seats. The simulator is deterministic, so replaying the policy's
recorded actions on the same board reproduces its trial to 0.1-0.5 mm; replayed up to where
the seat assist would take over (1.5 s stuck, or the gripper starting to open) and handed to
the expert, the assist takes over in 27 of the 50 failed trials and the expert seats 26 of
those wires: the hybrid would route about 95 of 119 (80%) where the policy alone routes 69 (to
be confirmed in closed loop: `--seat-assist 1.5`). What the expert needed from a policy's
stuck state, beyond its own seating: turning the gripper back to the route (the policy holds
it ~10 degrees off, the stiff wire leaves the fingers skewed, and centring it at the fork then
takes a 12 mm sideways offset that crosses a fork turned 25 degrees too steeply to pass the
lips), and lifting a wire that lies on a prong clear before lining up again. The one it cannot finish: a wire grasped so far
from the fork that it stays slack at the longest holding distance (it would need a new grasp).

Copy results home from WSL:

```bash
scp -r <username>@<public-ip>:eval/route_v1 ~/harness_eval_route_v1
```

## 7. GR00T inside whole builds

With the policy server still running, the planner's `route_fork` calls can go to GR00T: the
tool call becomes GR00T's instruction, and if GR00T fails, the board is cleared and the
retry goes to the force-guided expert (`--groot-attempts all` lets GR00T take retries too).

```bash
source ~/harness_env.sh
cd ~/Wire-Harness-Robot-2
# one build, scripted decisions, video
python -m harness_agent.run_build --spec harness_agent/specs/demo_3fork.yaml --planner scripted \
    --seed 3 --randomize --groot --video
# Nemotron decides (Token Factory key in this shell: export NEBIUS_API_KEY=...)
python -m harness_agent.run_build --spec harness_agent/specs/demo_3fork.yaml --planner nemotron \
    --seed 3 --randomize --groot --video
# the recovery benchmark with GR00T routing: planner labels get "+groot"
python -m harness_agent.benchmark --planners scripted --groot --seeds 0-4 --out ~/runs/bench
```

`report.md` lists who executed every step; the benchmark table adds "GR00T routed" (routes
GR00T completed / routes it attempted).

## 8. A whole round, unattended

`scripts/groot_round.sh` chains the steps above: record demos (with pushes and the end hold),
package them, fine-tune, check the fit open loop (GR00T's `open_loop_eval` on training demos
and on the expert's demos from 10 test boards), evaluate on 20 test boards (60 trials, with the
milestone funnel), and evaluate the previous model on the same boards. Each step is skipped
when its result exists, so the same command continues after a preempted VM. Log:
`~/rounds/<name>.log`.

```bash
tmux new -s round
bash ~/Wire-Harness-Robot-2/scripts/groot_round.sh route_v3     # ~6 h; Ctrl-b d to leave it
```

Defaults: 400 builds from seed 3000 (~1,250 demos), a fresh set (sets with a different state
layout cannot be combined), 12,000 steps, `route_v2/checkpoint-12000` as the comparison.
Change them with environment variables, e.g. `BUILDS=200 STEPS=8000 bash .../groot_round.sh
route_v3b`; `EXTRA_TRAIN_ARGS="--tune-visual"` also fine-tunes the vision encoder.
`DATASET=~/data/route_v3` trains on a packaged set without recording one (and reuses its
`_val` set), for trying training settings on the same demos, e.g. `DATASET=~/data/route_v3
EXTRA_TRAIN_ARGS="--state-dropout-prob 0.0" BASELINE="" bash .../groot_round.sh route_v3_sd0`.
At the end the round runs `groot_evals.sh` variants of the new checkpoint (`VARIANTS`, `ens4` by
default) and compares with `route_v3/checkpoint-12000` (`BASELINE`).
When it prints "round finished",
stop the VM in the console: shutting it down from inside makes Nebius restart it and keep
charging.

`scripts/groot_evals.sh` then runs the trained checkpoint in other ways on the same boards, no
retraining (~30 min per variant on 20 boards, `~/eval/<name>_<variant>`): `ens4` (a chunk
every 4 steps, ensembled, with trajectories), `restarts` (up to 2 fresh starts after a stalled
try, 60 s), `best` (both, with videos), `steps8` (8 denoising steps instead of 4: a copy of
`config.json` next to links to the weights), `assist` (ens4 with the seat assist after 1.5 s:
the hybrid), `assist_best` (assist with restarts and videos) and `takeovers` (ens4 + assist on
200 training boards, `TAKEOVER_SEEDS=8000-8199`, recording every takeover as a training
episode for the next round).

```bash
tmux new -s evals
bash ~/Wire-Harness-Robot-2/scripts/groot_evals.sh route_v3
# the hybrid and the DAgger-style data for the next round, on 40 test boards:
EVAL_SEEDS=0-39 EVAL_WORKERS=6 VARIANTS="assist assist_best takeovers" \
    bash ~/Wire-Harness-Robot-2/scripts/groot_evals.sh route_v5
```

`scripts/groot_dagger.sh` does all of it in one go, unattended (~11 h): the current model
with the seat assist on the 40 test boards (alone and with restarts), takeovers recorded on
200 training boards, then the next round on the current model's set + the takeovers + 400
new builds (30% seat recoveries), 20,000 steps, evaluated alone and with the assist. Each
step carries on after a preempted VM when the same command is run again (an interrupted
evaluation keeps its finished trials).

```bash
tmux new -s night
bash ~/Wire-Harness-Robot-2/scripts/groot_dagger.sh route_v5 route_v6
```

The next round then trains on the earlier set, the takeovers and new demos together:
`BASE="~/data/route_v5 ~/eval/route_v5_takeovers/takeovers"` (sets with the same state
layout, listed before the new recording). `SEAT_RECOVERIES=0.3` records 30% of the new
routes as seat recoveries: after the carry the tool comes down off the slot (3-12 mm to the
side, closer or lower than the expert would, turned up to 15 degrees; not recorded) and the
episode is the expert seating the wire from there. Pushes while the expert seats would not do
this: its descent snaps the wire in, the seating phase lasts a quarter second, and it undoes
a push within one step.

## Troubleshooting

| message | cause and fix |
|---|---|
| `Address already in use` / `no GR00T policy server at 127.0.0.1:5555` | 5555 is taken by `nv-hostengine`; use `--port 5556` on both sides |
| `uv sync` fails on a torchcodec wheel (a Git LFS pointer, not a wheel) | `git lfs pull --include "scripts/deployment/dgpu/wheels/*"` in ~/Isaac-GR00T (the setup script does it) |
| `GatedRepoError` for `nvidia/Cosmos-Reason2-2B` | accept the licence on its Hugging Face page, log in again |
| `client_loop: send disconnect: Broken pipe` | the SSH connection dropped; reconnect and `tmux a -t groot`, the job kept running |
| `No space left on device` during training | old checkpoints; keep `--save-total-limit 2`, delete smoke-test checkpoints |

## What the policy sees and does

Defined in `harness_agent/harness_agent/groot_features.py`, matched by `groot/harness_config.py`
(`harness_config_v3.py` and `harness_config_base.py` for the older layouts;
`python -m harness_agent.groot_data config-for <set>` names the right one, and the round script
uses it):

* video: `scene` (fixed camera over the board) and `wrist`, 256 x 256
* state (79, layout v4): TCP pose, commanded lead, gripper opening, force/torque, the target
  fork's CAD pose and the point the wire is fixed at before it, 8 perceived wire keypoints (the
  base 47, all that `route_v1`/`route_v2` saw); geometry measured from the gripper: its position
  in the target fork's route frame, the nearest perceived wire point in the gripper frame, and
  where the wire crosses the slot (v3, 66, `route_v3`); and the gripper against the skill's plan
  (`harness_core.expert.plan_route`, made once when the skill starts, as the expert makes it):
  the planned grasp point on the wire in the gripper frame, the wire's pull along the route,
  the tool height against the seating height and the distance past the fork against the planned
  one. Offsets are coarse and fine (tanh of a 1 cm, 5 mm or 5 N scale). The runner asks the
  served model which keys it was trained with, so older checkpoints still run; recordings keep
  the perception behind every state (`raw/`), so a later layout can be computed from them.
* action (5, 20 Hz, chunks of 16, 8 executed per call): TCP step and yaw step for the
  admittance controller, gripper command
* language: "route the wire into fork F1/F2/F3"

The same information the expert uses; no simulator ground truth.
