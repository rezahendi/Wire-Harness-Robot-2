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

### Results so far

Closed loop on 20 boards never used for training (59 trials; one board where the expert
could not set up F3 is skipped), success = wire seated, released and cleared:

| model | demos | state | steps | routed |
|---|---|---|---|---|
| `route_v1` | 403 | base (47) | 6,000 | 11/59 (19%) |
| `route_v2` | 1,165 (762 with recovery pushes, end hold) | base (47) | 12,000 | 10/59 (17%); re-planning every 0.2 s: 8/59 |

Successful trials take ~15 s, like the expert, at ~190 ms per action chunk with 4 workers.
Three times the demos, recovery pushes included, changed nothing, so data volume is not the
limit. Where `route_v1` fails (59 trials): the wire is in the hand in 35, lifted in 27, carried
over the slot in 26, inside the slot in 11. Both losses, grasping and seating, are precision
steps: with the base state the policy has to work out millimetre offsets from table
coordinates and 8 wire points 8 cm apart. `route_v3` adds that geometry measured from the
gripper (route, wire and slot keys below).

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
route_v3b`; `EXTRA_TRAIN_ARGS="--tune-visual"` also fine-tunes the vision encoder. When it prints "round finished",
stop the VM in the console: shutting it down from inside makes Nebius restart it and keep
charging.

## Troubleshooting

| message | cause and fix |
|---|---|
| `Address already in use` / `no GR00T policy server at 127.0.0.1:5555` | 5555 is taken by `nv-hostengine`; use `--port 5556` on both sides |
| `uv sync` fails on a torchcodec wheel (a Git LFS pointer, not a wheel) | `git lfs pull --include "scripts/deployment/dgpu/wheels/*"` in ~/Isaac-GR00T (the setup script does it) |
| `GatedRepoError` for `nvidia/Cosmos-Reason2-2B` | accept the licence on its Hugging Face page, log in again |
| `client_loop: send disconnect: Broken pipe` | the SSH connection dropped; reconnect and `tmux a -t groot`, the job kept running |
| `No space left on device` during training | old checkpoints; keep `--save-total-limit 2`, delete smoke-test checkpoints |

## What the policy sees and does

Defined in `harness_agent/harness_agent/groot_features.py`, matched by `groot/harness_config.py`:

* video: `scene` (fixed camera over the board) and `wrist`, 256 x 256
* state (66): TCP pose, commanded lead, gripper opening, force/torque, the target fork's CAD
  pose and the point the wire is fixed at before it, 8 perceived wire keypoints (the base 47,
  all that `route_v1`/`route_v2` saw), plus geometry measured from the gripper: its position in
  the target fork's route frame, the nearest perceived wire point in the gripper frame, and
  where the wire crosses the slot, each coarse and fine (tanh of a 1 cm or 5 mm scale). The
  runner asks the served model which keys it was trained with, so older checkpoints still run.
* action (5, 20 Hz, chunks of 16, 8 executed per call): TCP step and yaw step for the
  admittance controller, gripper command
* language: "route the wire into fork F1/F2/F3"

The same information the expert uses; no simulator ground truth.
