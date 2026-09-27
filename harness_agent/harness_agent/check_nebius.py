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


def _png(width: int = 48, height: int = 48, rgb=(220, 40, 40)) -> bytes:
    """A small solid-colour PNG, built without any imaging library."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", default=None, help="model id to test for tool calling")
    ap.add_argument("--vision", default=None, help="model id to test with an image")
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
    try:
        vision = args.vision or client.vision_model()
        print(f"\nvision model: {vision}")
        url = "data:image/png;base64," + base64.b64encode(_png()).decode("ascii")
        t0 = time.perf_counter()
        msg = client.chat(vision, [{"role": "user", "content": [
            {"type": "text", "text": "What colour is this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": url}}]}], max_tokens=512)
        answer = (msg.get("content") or "").strip()
        good = "red" in answer.lower()
        print(f"   image answer: {answer[:80]!r}  ({time.perf_counter() - t0:.1f} s) -> "
              f"{'OK' if good else 'UNEXPECTED'}")
        ok &= good
    except LLMError as exc:
        print(f"   vision check skipped: {exc}")
    print(f"\nusage: {client.usage.as_dict()}")
    print("\nREADY" if ok else "\nSOMETHING NEEDS ATTENTION (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
