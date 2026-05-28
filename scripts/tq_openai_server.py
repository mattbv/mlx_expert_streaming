#!/usr/bin/env python3
import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MAX_REQUEST_BYTES = 1_000_000


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible TurboQuant-MLX server")
    parser.add_argument("--model", default="manjunathshiva/Qwen3.6-35B-A3B-tq3-g32")
    parser.add_argument("--cache-budget-gb", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_runtime(model_id, cache_budget_gb):
    from turboquant_mlx.stream.loader import load_streaming

    started = time.monotonic()
    model, tokenizer, cache = load_streaming(model_id, cache_budget_gb=cache_budget_gb)
    elapsed = time.monotonic() - started
    print(f"loaded model={model_id} cache_budget_gb={cache_budget_gb} elapsed={elapsed:.1f}s", flush=True)
    return model, tokenizer, cache


def build_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        rendered.append(f"{role}: {content}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


class ServerState:
    args = None
    model = None
    tokenizer = None
    cache = None


class Handler(BaseHTTPRequestHandler):
    server_version = "TurboQuantOpenAI/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": ServerState.args.model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "request body is empty or too large"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self.send_json(400, {"error": f"invalid JSON: {exc}"})
            return

        messages = body.get("messages")
        if not isinstance(messages, list):
            self.send_json(400, {"error": "messages must be a list"})
            return
        if body.get("stream"):
            self.send_json(400, {"error": "streaming responses are not implemented"})
            return

        try:
            max_tokens = int(body.get("max_tokens") or ServerState.args.max_tokens)
            temperature = float(body.get("temperature", ServerState.args.temperature))
        except (TypeError, ValueError):
            self.send_json(400, {"error": "max_tokens and temperature must be numeric"})
            return
        if max_tokens < 1 or max_tokens > ServerState.args.max_tokens:
            self.send_json(400, {"error": f"max_tokens must be between 1 and {ServerState.args.max_tokens}"})
            return
        if temperature < 0:
            self.send_json(400, {"error": "temperature must be non-negative"})
            return
        prompt = build_prompt(ServerState.tokenizer, messages)

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        started = time.monotonic()
        text = generate(
            ServerState.model,
            ServerState.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        elapsed = time.monotonic() - started
        approx_tokens = max(1, len(text.split()))

        self.send_json(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": ServerState.args.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": approx_tokens,
                    "total_tokens": None,
                },
                "benchmark": {
                    "elapsed_seconds": elapsed,
                    "approx_words_per_second": approx_tokens / elapsed if elapsed else None,
                    "cache_stats": json_safe(getattr(ServerState.cache, "stats", lambda: None)()),
                },
            },
        )


def main():
    args = parse_args()
    ServerState.args = args
    ServerState.model, ServerState.tokenizer, ServerState.cache = load_runtime(
        args.model,
        args.cache_budget_gb,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving http://{args.host}:{args.port}/v1 model={args.model}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
