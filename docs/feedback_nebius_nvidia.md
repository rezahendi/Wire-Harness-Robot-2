# Feedback: building a physical-AI agent on Nebius Token Factory with NVIDIA Nemotron

What we learned building this project (September 2026), with the numbers behind it. Everything
here comes from our own runs; `runs/*/trace.json` has the raw calls.

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
