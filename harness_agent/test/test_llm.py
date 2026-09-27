"""Token Factory client against a local fake server (no network, no key needed)."""

import http.server
import json
import threading

import pytest

from harness_agent import check_nebius, llm
from harness_agent.llm import LLMError, TokenFactoryClient


class FakeTokenFactory(http.server.BaseHTTPRequestHandler):
    """Plays back queued (status, body, delay) answers; records the requests."""

    script = []
    seen = []

    def log_message(self, *args):
        pass

    def _answer(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        type(self).seen.append((self.command, self.path, body))
        status, payload, delay = type(self).script.pop(0) if type(self).script else (200, {}, 0.0)
        if delay:
            threading.Event().wait(delay)          # not time.sleep: the tests patch that out
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # the client gave up (timeout test)

    do_GET = _answer
    do_POST = _answer


@pytest.fixture
def server(monkeypatch):
    FakeTokenFactory.script = []
    FakeTokenFactory.seen = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeTokenFactory)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)      # no real backoff in tests
    for var in ("no_proxy", "NO_PROXY"):                          # never send 127.0.0.1 to a proxy
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    monkeypatch.delenv("HARNESS_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("HARNESS_VISION_MODEL", raising=False)
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1/"
    httpd.shutdown()


def reply(content="", tool_calls=None, reasoning=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def client(url, **kw):
    return TokenFactoryClient(api_key="test-key", base_url=url, **kw)


def test_models_are_picked_by_preference(server):
    FakeTokenFactory.script = [(200, {"data": [{"id": "google/gemma-3-27b-it"},
                                               {"id": "nvidia/nemotron-3-super-120b-a12b"},
                                               {"id": "nvidia/nemotron-3-nano-30b-a3b"},
                                               {"id": "openbmb/MiniCPM-V-4_5"}]}, 0.0)]
    c = client(server)
    assert c.planner_model() == "nvidia/nemotron-3-super-120b-a12b"
    assert c.vision_model() == "openbmb/MiniCPM-V-4_5"
    assert c.fast_model() == "nvidia/nemotron-3-nano-30b-a3b"
    assert len(FakeTokenFactory.seen) == 1                      # the model list is cached
    assert FakeTokenFactory.seen[0][:2] == ("GET", "/v1/models")


def test_tool_call_round_trip_and_usage(server):
    FakeTokenFactory.script = [(200, reply(tool_calls=[{"id": "c1", "type": "function", "function": {
        "name": "route_fork", "arguments": "{\"fork_id\": \"F2\"}"}}]), 0.0)]
    c = client(server)
    msg = c.chat("m", [{"role": "user", "content": "go"}], tools=[{"type": "function"}])
    assert msg["tool_calls"] == [{"id": "c1", "name": "route_fork", "arguments": {"fork_id": "F2"}}]
    sent = FakeTokenFactory.seen[0][2]
    assert sent["tool_choice"] == "auto" and sent["model"] == "m"
    assert c.usage.calls == 1 and c.usage.prompt_tokens == 10 and c.usage.completion_tokens == 3


def test_transient_errors_are_retried(server):
    FakeTokenFactory.script = [(503, {"error": "busy"}, 0.0), (429, {"error": "slow down"}, 0.0),
                               (200, reply("done"), 0.0)]
    assert client(server).chat("m", [])["content"] == "done"
    assert len(FakeTokenFactory.seen) == 3


def test_read_timeout_is_retried_not_raised(server):
    FakeTokenFactory.script = [(200, reply("late"), 1.0), (200, reply("on time"), 0.0)]
    assert client(server, timeout=0.3).chat("m", [])["content"] == "on time"


def test_client_errors_are_reported_with_the_server_detail(server):
    FakeTokenFactory.script = [(400, {"error": {"message": "image input is not supported"}}, 0.0)]
    with pytest.raises(LLMError, match="image input is not supported"):
        client(server).chat("m", [])
    assert len(FakeTokenFactory.seen) == 1                      # a 400 is not retried


def test_garbage_answers_raise_llm_errors(server):
    FakeTokenFactory.script = [(200, b"<html>gateway</html>", 0.0), (200, {"no": "choices"}, 0.0)]
    c = client(server)
    with pytest.raises(LLMError, match="non-JSON"):
        c.chat("m", [])
    with pytest.raises(LLMError, match="unexpected response"):
        c.chat("m", [])


@pytest.mark.parametrize("answer,verdict", [
    (reply("Blue, yellow."), "OK"),
    (reply("blue and golden"), "OK"),
    (reply("Yellow, blue"), "wrong"),
    (reply("I cannot see images."), "wrong"),
    (reply("", reasoning="The left half looks..."), "thinking only"),
    (reply(""), "no answer"),
])
def test_vision_check_verdicts(server, answer, verdict):
    FakeTokenFactory.script = [(200, answer, 0.0)]
    got, _, _ = check_nebius._vision_test(client(server), "m")
    assert got == verdict
    content = FakeTokenFactory.seen[0][2]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_vision_check_reports_refused_images(server):
    FakeTokenFactory.script = [(400, {"error": "model does not accept image_url"}, 0.0)]
    got, detail, _ = check_nebius._vision_test(client(server), "m")
    assert got == "no images" and "image_url" in detail
