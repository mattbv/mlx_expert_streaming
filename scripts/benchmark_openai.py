#!/usr/bin/env python3
import argparse
import json
import time

import requests


PROMPTS = [
    "Write a Python function that validates JSON and returns a useful error message.",
    "Explain the difference between mmap and pread in one short paragraph.",
    "Given a list of integers, write a concise function that returns the two largest values.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible chat endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="manjunathshiva/Qwen3.6-35B-A3B-tq3-g32")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    results = []

    for index, prompt in enumerate(PROMPTS, start=1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        started = time.monotonic()
        response = requests.post(url, json=payload, timeout=args.timeout)
        elapsed = time.monotonic() - started
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        words = len(content.split())
        results.append({"prompt": index, "elapsed_seconds": elapsed, "words": words})
        print(f"prompt={index} elapsed={elapsed:.2f}s words={words} words_per_second={words / elapsed:.2f}")
        print(content[:500].strip())
        print("---")

    total_elapsed = sum(item["elapsed_seconds"] for item in results)
    total_words = sum(item["words"] for item in results)
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "total_elapsed_seconds": total_elapsed,
        "total_words": total_words,
        "average_words_per_second": total_words / total_elapsed if total_elapsed else None,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
