"""Check the Nebius Token Factory setup before running a build.

    export NEBIUS_API_KEY=...
    python -m harness_agent.check_nebius

Lists the NVIDIA models the key can use, picks the planner / vision / fast models, and
makes two tiny calls: one tool call (the agent depends on it) and one image question
(the visual checks will). Costs a fraction of a cent.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import zlib
from typing import Optional

from .llm import LLMError, TokenFactoryClient


def _png(width: int = 64, height: int = 48, left=(30, 70, 220), right=(240, 200, 20)) -> bytes:
    """A small two-colour PNG (left half, right half), built without any imaging library."""
    row = bytes(left) * (width // 2) + bytes(right) * (width - width // 2)
    raw = b"".join(b"\x00" + row for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


VISION_HINTS = ("vl", "omni", "vision", "gemma-3", "minicpm-v", "minimax-m", "qwen3.5", "kimi-k3",
                "llava", "pixtral", "lightning")


def _vision_test(client: TokenFactoryClient, model: str):
    """Ask for the two colours of the test image. Returns (verdict, answer, seconds).

    verdicts: OK, wrong (answered, but not blue then yellow: the image was probably
    ignored), thinking only (a reasoning model used up its tokens before answering),
    no answer, no images (the request was refused).
    """
    url = "data:image/png;base64," + base64.b64encode(_png()).decode("ascii")
    t0 = time.perf_counter()
    try:
        msg = client.chat(model, [{"role": "user", "content": [
            {"type": "text", "text": "This image has two colours, one on the left half and one on the "
                                     "right half. Name them, left first, in two words."},
            {"type": "image_url", "image_url": {"url": url}}]}], max_tokens=1536)
    except LLMError as exc:
        return "no images", str(exc).split(":", 2)[-1].strip(), time.perf_counter() - t0
    dt = time.perf_counter() - t0
    answer = (msg.get("content") or "").strip().lower()
    if not answer:
        return ("thinking only" if msg.get("reasoning") else "no answer"), "", dt
    left = answer.find("blue")
    right = min((answer.find(w) for w in ("yellow", "gold") if w in answer), default=-1)
    return ("OK" if 0 <= left < right else "wrong"), answer, dt


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", default=None, help="model id to test for tool calling")
    ap.add_argument("--vision", default=None, help="model id to test with an image")
    ap.add_argument("--vision-all", action="store_true",
                    help="test every model that might take images (and all NVIDIA models)")
    args = ap.parse_args(argv)
    try:
        client = TokenFactoryClient()
        models = client.list_models()
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1
    nv = [m for m in models if "nemotron" in m.lower() or m.lower().startswith("nvidia/")]
    print(f"{len(models)} models available, {len(nv)} from NVIDIA:")
    for m in nv:
        print(f"   {m}")
    ok = True
    try:
        planner = args.planner or client.planner_model()
    except LLMError as exc:
        print(f"no planner model: {exc}")
        return 1
    print(f"\nplanner model: {planner}")
    tools = [{"type": "function", "function": {
        "name": "set_lamp", "description": "Switch the workbench lamp.",
        "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}}]
    t0 = time.perf_counter()
    try:
        msg = client.chat(planner, [{"role": "system", "content": "You operate a workbench. Use the tools."},
                                    {"role": "user", "content": "Turn the lamp on, please."}],
                          tools=tools, max_tokens=1024)
        calls = msg.get("tool_calls") or []
        good = bool(calls) and calls[0]["name"] == "set_lamp" and calls[0]["arguments"].get("on") is True
        print(f"   tool call: {json.dumps(calls)[:200]}  ({time.perf_counter() - t0:.1f} s) "
              f"-> {'OK' if good else 'UNEXPECTED'}")
        ok &= good
    except LLMError as exc:
        print(f"   tool call FAILED: {exc}")
        ok = False
    if args.vision_all:
        cands = [m for m in models if any(k in m.lower() for k in VISION_HINTS)]
        cands += [m for m in nv if m not in cands]
        print(f"\nimage input test on {len(cands)} models (two colours, left and right):")
        working = []
        for m in cands:
            verdict, answer, dt = _vision_test(client, m)
            print(f"   {m:45s} {verdict:13s} {dt:5.1f} s  {answer[:60]!r}", flush=True)
            if verdict == "OK":
                working.append((m, dt))
        print("\nmodels that read the image correctly: "
              + (", ".join(f"{m} ({dt:.1f} s)" for m, dt in working) or "none"))
        if working:
            best = min(working, key=lambda w: (0 if w[0] in nv else 1, w[1]))[0]
            print(f"suggested (NVIDIA first, then fastest):  export HARNESS_VISION_MODEL={best}")
    else:
        try:
            vision = args.vision or client.vision_model()
            print(f"\nvision model: {vision}")
            verdict, answer, dt = _vision_test(client, vision)
            print(f"   image answer: {answer[:80]!r}  ({dt:.1f} s) -> {verdict}")
            ok &= verdict == "OK"
        except LLMError as exc:
            print(f"   vision check skipped: {exc}")
            print("   (try: check_nebius --vision-all)")
    print(f"\nusage: {client.usage.as_dict()}")
    print("\nREADY" if ok else "\nSOMETHING NEEDS ATTENTION (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
