# GR00T N1.7 as the routing skill: runbook

Level 3 (Nemotron) decides the next step, level 2 (the skill) works out every move, level 1
(the 500 Hz admittance controller) makes it soft and force-limited. Here NVIDIA Isaac GR00T
N1.7 learns level 2 for `route_fork` from the force-guided expert's demonstrations, one
language-conditioned policy for every fork ("route the wire into fork F2"). The controller
underneath and the planner above stay the same.

| step | where | time | cost (on-demand, preemptible is cheaper) |
|---|---|---|---|
| 1. VM + setup | Nebius AI Cloud, 1x L40S | 20 min | ~$0.50 |
| 2. smoke test (replay through GR00T's own server) | same VM | 10 min | ~$0.25 |
| 3. record ~1,000 routing demos | same VM, CPU cores | 45-60 min | ~$1.50 |
| 4. fine-tune GR00T N1.7 | same VM, GPU | 1-2 h | ~$2-3 |
| 5. closed-loop evaluation, 60 trials | same VM | ~30 min | ~$0.75 |

**Stop the VM whenever nothing runs.** A stopped VM costs only its disk.

## 1. Create the VM

Nebius AI Cloud console → Compute → Virtual machines → Create:

* GPU: **1x L40S** (48 GB; fine-tuning peaks around 35 GB). An H100 is faster per hour of training.
* Preemptible: yes if offered (much cheaper; jobs below run in tmux and checkpoints are saved, so an interruption only costs time).
* vCPUs: as many as the L40S preset offers (the demo recorder runs one build per core).
* Boot disk: the Ubuntu 22.04 (or 24.04) image **with NVIDIA drivers / CUDA 12**, 200 GB.
* Access: your username and the public key from WSL (`cat ~/.ssh/id_ed25519.pub`); a public IP.

Then, from WSL: `ssh <username>@<public-ip>`

## 2. Code and setup (on the VM)

The repo is private, so give the VM a read-only deploy key:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && cat ~/.ssh/id_ed25519.pub
# GitHub -> rezahendi/Wire-Harness-Robot-2 -> Settings -> Deploy keys -> Add (read-only), paste the key
git clone git@github.com:rezahendi/Wire-Harness-Robot-2.git
bash Wire-Harness-Robot-2/scripts/setup_groot_vm.sh          # ~15 min: system packages, ~/simenv, ~/Isaac-GR00T
cd ~/Isaac-GR00T && uv run huggingface-cli login             # paste your Hugging Face read token
```

Use `tmux` for everything long (`tmux new -s groot`; detach with Ctrl-b d, back with `tmux a -t groot`).

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

# terminal 2 (tmux window): GR00T's server replaying the recorded set
cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py --dataset-path ~/data/smoke \
    --modality-config-path ~/Wire-Harness-Robot-2/groot/harness_config.py \
    --embodiment-tag NEW_EMBODIMENT --execution-horizon 8 --port 5555

# terminal 1
python -m harness_agent.groot_eval --replay-dataset ~/data/smoke --seeds 1000-1001 --forks F1,F2,F3 --out ~/eval/smoke
```

Expected: 6/6 routed. Stop the replay server (Ctrl-c) afterwards.

## 4. Record the training set

```bash
source ~/harness_env.sh
python -m harness_agent.groot_data record --out ~/data/harness_route --builds 330 --workers $(( $(nproc) - 2 ))
python -m harness_agent.groot_data check ~/data/harness_route --preview ~/data/route_preview.png
```

330 randomised builds give about 1,000 routing episodes (F1, F2, F3, plus re-routes of F2
in the quarter of builds where the wire is pulled out). Seeds start at 1000, so the
benchmark seeds 0-99 stay unseen.

## 5. Fine-tune GR00T N1.7

```bash
cd ~/Isaac-GR00T
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/data/harness_route \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ~/Wire-Harness-Robot-2/groot/harness_config.py \
    --num-gpus 1 --output-dir ~/ckpt/route_v1 \
    --max-steps 6000 --save-steps 2000 --global-batch-size 32 --dataloader-num-workers 8
```

It tunes the projector and the diffusion action head (the vision-language backbone stays
frozen), with GR00T's default augmentation. The first run also writes `meta/stats.json`.

## 6. Serve the policy and evaluate it in closed loop

```bash
# tmux window 2
cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/ckpt/route_v1/checkpoint-6000 --embodiment-tag NEW_EMBODIMENT --port 5555

# tmux window 1
source ~/harness_env.sh
python -m harness_agent.groot_eval --forks F1,F2,F3 --seeds 0-19 --out ~/eval/route_v1 --video
python -m harness_agent.groot_eval --forks F1,F2,F3 --seeds 0-19 --expert --out ~/eval/expert   # baseline
```

`summary.md` has the success rate per fork with a 95% interval, time and contact force;
`videos/` has one clip per trial. Copy results home from WSL:

```bash
scp -r <username>@<public-ip>:eval/route_v1 ~/harness_eval_route_v1
```

## What the policy sees and does

Defined in `harness_agent/harness_agent/groot_features.py`, matched by `groot/harness_config.py`:

* video: `scene` (fixed camera over the board) and `wrist`, 256 x 256
* state (47): TCP pose, commanded lead, gripper opening, force/torque, the target fork's CAD
  pose and the point the wire is fixed at before it, 8 perceived wire keypoints
* action (5, 20 Hz, chunks of 16, 8 executed per call): TCP step and yaw step for the
  admittance controller, gripper command
* language: "route the wire into fork F1/F2/F3"

The same information the expert uses; no simulator ground truth.
