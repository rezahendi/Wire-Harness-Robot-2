# Feedback: Nebius Token Factory, Nebius AI Cloud, NVIDIA Nemotron and Isaac GR00T N1.7

What we learned building this project (September and October 2026), with the numbers behind it.
Everything here comes from our own runs; `runs/*/trace.json` has the raw planner calls and
`docs/groot.md` the GR00T runbook.

# Nebius Token Factory and NVIDIA Nemotron

## What worked very well

**Nemotron 3 Super as a tool-calling supervisor.** Across 83 builds and 885 model turns, every
turn came back with exactly one well-formed tool call: no turn without a call, no turn with
several, no argument we could not parse, no tool that does not exist. We had written a fallback
parser for tool calls embedded in text and a "please call a tool" nudge; neither was ever used.
Mean latency was 1.3 s per decision, which is fast next to the 10-20 s a robot skill takes. It
read refusals correctly: when `route_fork F3` was refused because F2 had lost the wire, it went
back to F2 every time, with no rule written for that case.

**OpenAI compatibility.** A standard-library client (`urllib`, no SDK) handled tool calls, image
input (base64 data URLs), several images per message and usage accounting without special
cases for Nemotron, Kimi K3, MiniCPM-V or Gemma 3.

**Fast, clear refusals.** Sending an image to a text-only model failed in 0.2 s with "This model
does not support image input". That made capability discovery by probing cheap.

## What would have helped

1. **An NVIDIA vision model on Token Factory.** None of the four NVIDIA models on our account
   (Nemotron 3 Nano 30B, Nemotron 3 Super 120B, Nemotron 3 Ultra 550B, Nemotron 3.5 Lightning)
   accepts images. The camera check of an NVIDIA-planned robot therefore runs on Kimi K3 and
   MiniCPM-V. A Nemotron VL or Cosmos Reason model would let a physical-AI loop stay on NVIDIA
   models end to end, which is what this hackathon track asks for.

2. **Capabilities in `GET /v1/models`.** The model list gives ids only. We had to send a test
   image to every model to learn which ones take images (`check_nebius --vision-all`), and we
   could not see context length, tool-calling support or whether a model reasons before
   answering. A `capabilities` field would save every team that probing.

3. **A separate reasoning budget.** Kimi K3's reasoning tokens count against `max_tokens`. At
   700 tokens it answered 28 of 130 inspection images (and got all 28 right); at 4,000 it
   answered 114 of 134, with a 16 s median latency. A per-request thinking budget, or an effort
   setting, would make reasoning VLMs usable inside a control loop without guessing a
   `max_tokens` that is long enough for the thinking and short enough for the latency.

4. **Reasoning field naming.** Reasoning arrives as `reasoning_content` from some models and
   `reasoning` from others. One documented field would simplify clients.

5. **Per-key spend in the API.** Usage per request is returned (thank you), but the remaining
   credit is only in the web console. For long unattended benchmark runs, a balance endpoint
   would let a job stop before it runs dry.

## Numbers for reference

| model | used for | calls in our runs | median latency | notes |
|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b | planner (tool calls) | 885 turns in 83 builds | 1.3 s | 0 malformed calls; 40/40 builds in the final benchmark; 4.0 M prompt + 75 k completion tokens in all |
| moonshotai/Kimi-K3 | camera check, first opinion | ~670 images | 12-27 s | 100% right on the 114/134 it answered (style v3 + examples) |
| openbmb/MiniCPM-V-4_5 | camera check, fallback | ~800 images | 0.8 s | fast; examples in the prompt made it worse |
| google/gemma-3-27b-it | compared | ~800 images | 2-3 s | leaned towards "seated" |

# NVIDIA Isaac GR00T N1.7

We fine-tuned GR00T N1.7 (`nvidia/GR00T-N1.7-3B`) as the cell's wire-routing skill: two cameras,
a 79-value state with the wrist force/torque reading, 5-D actions at 20 Hz and the instruction
"route the wire into fork F2", about 2,700 episodes in the last round.

## What worked very well

**A new embodiment without touching GR00T's code.** A modality config for `NEW_EMBODIMENT` and
LeRobot v2 data were all it took. Before any training, GR00T's own policy server replaying our
recorded set routed 6 of 6 wires through our client, which checked the data, the config and the
client end to end in ten minutes.

**One policy for every clip.** The language instruction carries the target fork, so one model
routes F1, F2 and F3, and the planner's tool call becomes GR00T's instruction unchanged.

**Training cost.** 20,000 steps at batch 32 take about 4.5 hours on one L40S (1.2-1.3 steps/s),
so a full round (record, train, evaluate) fits in a night on one GPU.

## What would have helped

1. **The default port.** The policy server listens on 5555, where DCGM's `nv-hostengine` already
   runs on NVIDIA's GPU images, so the server fails with "Address already in use". A different
   default, or a message that names the cause, would save the first hour.
2. **A clone without Git LFS.** `uv sync` fails on the torchcodec wheel when the repository was
   cloned without LFS (the wheel is an LFS pointer). A check in the setup that says "run git lfs
   pull" would make the error obvious.
3. **The gated backbone, up front.** `nvidia/Cosmos-Reason2-2B` is gated on Hugging Face and the
   setup only finds out with a `GatedRepoError` once it downloads. One line at the top of the
   fine-tuning guide would do.
4. **Checkpoint size.** A checkpoint with optimizer state is about 36 GB, so a 200 GB disk fills
   during a long run unless `--save-total-limit` is set. A weights-only option for intermediate
   checkpoints would help.
5. **Batched serving.** The policy server answers one request at a time (about 245 ms per action
   chunk on an L40S). Closed-loop evaluation runs several simulators in parallel, and their
   requests wait in line; batching requests from several clients would make evaluation several
   times faster.
6. **Temporal ensembling as an option.** Averaging overlapping action chunks (as in ACT) steadied
   the motion near the clip for us; we wrote it in our client. A built-in option would let
   others try it with one flag.
7. **What the knobs trade off.** More denoising steps and switching state dropout off both made
   our results worse. A short note on when to change them would save experiments.

# Nebius AI Cloud

We used one L40S VM (8 vCPU, 32 GiB, CUDA image) for everything GPU-side: recording demos in
MuJoCo with GPU rendering through EGL, fine-tuning GR00T, closed-loop evaluation, and the live
Mission Control runs with GR00T served next to the simulator and Nemotron on Token Factory.

## What worked very well

**One machine for the whole loop.** On the CUDA image, the two policy cameras rendered through
EGL in 4.4 ms per step on the L40S, and GR00T trained and served on the same machine as the
simulator, so a live build needed no second server.

**Preemptible VMs for the long jobs.** Recording and training ran on a preemptible VM at a
fraction of the price; with every step resumable, a preemption cost minutes, not the run.

## What would have helped

1. **Shutting down from inside.** `sudo shutdown` inside the VM did not stop it: it came back
   up and kept billing, and only a stop in the console ended the charges. A warning in the
   console (or treating an OS shutdown as a stop) would save money for everyone who stops a
   machine the way they would stop a laptop.
