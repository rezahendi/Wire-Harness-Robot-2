"""Minimal client for Nebius Token Factory (OpenAI-compatible chat completions).

Standard library only, so it runs in the ROS workspace, in a plain venv and inside a
Nebius Serverless Job without extra dependencies.

    export NEBIUS_API_KEY=...                   # from tokenfactory.nebius.com
    python -m harness_agent.check_nebius        # lists models, tests a tool call

Model names on Token Factory change over time, so the planner and vision models are
picked from ``GET /v1/models`` by preference lists (override with HARNESS_PLANNER_MODEL /
HARNESS_VISION_MODEL / HARNESS_FAST_MODEL).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# substrings in order of preference (matched case-insensitively against model ids)
PLANNER_PREFERENCE = ("nemotron-3-super", "nemotron-3-ultra", "nemotron-3_5", "nemotron-3-nano-30b",
                      "nemotron")
VISION_PREFERENCE = ("nemotron-3-nano-omni", "nano-omni", "cosmos-reason", "nemotron-nano-2-vl", "omni",
                     "minicpm-v", "gemma-3", "kimi-k3", "-vl")
FAST_PREFERENCE = ("nemotron-3-nano-30b", "nemotron-3-nano", "lightning", "nemotron")


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, model: str, usage: Optional[Dict[str, Any]], seconds: float) -> None:
        with self._lock:                  # vision_eval scores images from several threads
            self._add(model, usage, seconds)

    def _add(self, model: str, usage: Optional[Dict[str, Any]], seconds: float) -> None:
        p = int((usage or {}).get("prompt_tokens", 0) or 0)
        c = int((usage or {}).get("completion_tokens", 0) or 0)
        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c
        self.seconds += seconds
        m = self.per_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                              "seconds": 0.0})
        m["calls"] += 1
        m["prompt_tokens"] += p
        m["completion_tokens"] += c
        m["seconds"] += seconds

    def as_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 1),
                "per_model": self.per_model}


class TokenFactoryClient:
    """Chat completions with tools against Nebius Token Factory."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 120.0, max_retries: int = 4):
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY", "")
        if not self.api_key:
            raise LLMError("NEBIUS_API_KEY is not set (create a key at tokenfactory.nebius.com)")
        self.base_url = (base_url or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/") + "/"
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self._models: Optional[List[str]] = None

    # ------------------------------------------------------------- http
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path.lstrip("/")
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    raise LLMError(f"non-JSON answer from {url}: {raw[:300]}") from None
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise LLMError(f"HTTP {e.code} from {url}: {detail}") from None
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                # URLError: DNS/connect; OSError: read timeouts and resets; HTTPException:
                # truncated responses. All transient, so retry with backoff.
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise LLMError(f"cannot reach {url}: {getattr(e, 'reason', None) or e!r}") from None
        raise LLMError("unreachable")

    # ----------------------------------------------------------- models
    def list_models(self) -> List[str]:
        if self._models is None:
            data = self._request("GET", "models")
            self._models = sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
        return self._models

    def pick_model(self, preference: Sequence[str], env_var: Optional[str] = None) -> str:
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]
        models = self.list_models()
        for pat in preference:
            hits = [m for m in models if pat.lower() in m.lower()]
            if hits:
                return sorted(hits, key=len)[0]
        raise LLMError(f"no model matching {list(preference)} among {len(models)} available models")

    def planner_model(self) -> str:
        return self.pick_model(PLANNER_PREFERENCE, "HARNESS_PLANNER_MODEL")

    def vision_model(self) -> str:
        return self.pick_model(VISION_PREFERENCE, "HARNESS_VISION_MODEL")

    def fast_model(self) -> str:
        return self.pick_model(FAST_PREFERENCE, "HARNESS_FAST_MODEL")

    # ------------------------------------------------------------- chat
    def chat(self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             temperature: float = 0.2, max_tokens: int = 2048, **extra) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(extra)
        t0 = time.perf_counter()
        data = self._request("POST", "chat/completions", body)
        self.usage.add(model, data.get("usage"), time.perf_counter() - t0)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise LLMError(f"unexpected response: {str(data)[:400]}") from None
        return normalise_message(msg)


def normalise_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Assistant message with tool calls as a list of {id, name, arguments(dict)}.

    Some serving stacks return tool calls inside the text instead of the structured
    field; those are recovered here so the agent loop does not care.
    """
    content = msg.get("content") or ""
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""), "arguments": args})
    if not calls and content:
        calls = _tool_calls_from_text(content)
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {"role": "assistant", "content": _strip_think(content), "tool_calls": calls,
            "reasoning": reasoning, "raw": msg}


_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    calls = []
    for i, m in enumerate(_TOOLCALL_RE.finditer(text)):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append({"id": f"text_call_{i}", "name": name, "arguments": args})
    return calls


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def assistant_message_for_history(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Re-encode a normalised assistant message in OpenAI wire format."""
    out: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                             for c in msg["tool_calls"]]
    return out
