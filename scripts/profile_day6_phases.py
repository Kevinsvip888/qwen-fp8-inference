#!/usr/bin/env python3
"""Capture short, requested prefill/decode windows on the existing Day 6 server.

Run in .venv-serve. Uses the same frozen synthetic token-ID request for both
models. Preserves all earlier traces and records raw responses, protocol/file
hashes, and phase-annotation evidence. Kernel sums are diagnostic GPU work,
not request latency. No server is started or stopped by this script.
"""

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


WINDOWS = {"prefill": (0, 1), "decode": (8, 8)}
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def freeze_json(path, value):
    if path.exists():
        check(json.loads(path.read_text()) == value, f"Frozen data changed: {path}")
    else:
        save_json(path, value)


def inspect_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents", [])
    kernels = [event for event in events
               if "kernel" in {c.strip().lower() for c in event.get("cat", "").split(",")}
               and event.get("ph") == "X" and event.get("dur", 0) > 0]
    totals = defaultdict(float)
    counts = Counter()
    for event in kernels:
        name = event.get("name", "unknown")
        totals[name] += float(event["dur"]) / 1000
        counts[name] += 1
    annotations = Counter()
    for event in events:
        if event.get("ph") != "X" or "kernel" in event.get("cat", "").lower():
            continue
        name = event.get("name", "")
        if any(word in name.lower() for word in ("prefill", "decode", "context", "generation", "forward")):
            annotations[name] += 1
    return {
        "file": path.name, "sha256": sha256(path),
        "gpu_kernel_events": len(kernels),
        "summed_kernel_ms": sum(totals.values()),
        "top_kernels": [{"name": name, "summed_ms": totals[name], "calls": counts[name]}
                        for name in sorted(totals, key=totals.get, reverse=True)[:15]],
        "phase_annotation_candidates": dict(annotations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("BF16", "Random32"))
    parser.add_argument("--repetitions", type=int, default=1, choices=(1, 3))
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/qwen-inference-lab"))
    args = parser.parse_args()
    root = args.root.resolve()
    alias = "qwen-profile-bf16" if args.method == "BF16" else "qwen-profile-fp8"
    pointer = root / f"logs/profiling/day6/{args.method}_trace_dir.txt"
    directory = Path(pointer.read_text().strip()).resolve()
    base = root / "outputs/profiling/day6"
    check(directory.is_relative_to(base.resolve()) and directory.is_dir(), "Invalid trace directory")
    opener = build_opener(ProxyHandler({}))

    def api(path, payload=None, timeout=120):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request("http://127.0.0.1:8000" + path, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read()
                return json.loads(body) if body.strip() else {}
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code} at {path}: {detail[:2000]}") from error

    models = api("/v1/models")
    check([item["id"] for item in models["data"]] == [alias], f"Expected server alias: {alias}")

    # The request is synthetic profiling input, separate from Day 5 random-text workloads.
    from transformers import AutoTokenizer

    tokenizer_path = root / "models/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prohibited = set(tokenizer.all_special_ids)
    prohibited.update(token_id for token_id, token in tokenizer.added_tokens_decoder.items()
                      if token.special)
    allowed = [token_id for token_id in range(tokenizer.vocab_size) if token_id not in prohibited]
    prompt = random.Random(42).choices(allowed, k=1024)
    request_body = {
        "prompt": prompt, "max_tokens": 128, "min_tokens": 128,
        "temperature": 0, "top_p": 1, "seed": 42, "n": 1,
        "ignore_eos": True, "add_special_tokens": False, "stream": False,
    }
    request_path = base / "phase_request_v1.json"
    freeze_json(request_path, request_body)
    source = json.loads((root / "logs/model_revision.json").read_text())
    check(source["revision"] == REVISION, "Unexpected model revision")
    uuid = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    exported = root / "models/Qwen3-4B-Instruct-2507-FP8-Random32-seed42/calibration_manifest.json"
    identity = json.loads(exported.read_text())["identity"]
    check(identity["method"] == "Random32" and identity["pool_seed"] == 42, "Unexpected FP8 checkpoint")
    protocol = {
        "version": 1, "request_sha256": sha256(request_path),
        "input_tokens": 1024, "output_tokens": 128, "concurrency": 1,
        "warmup_requests": 4, "windows": WINDOWS, "gpu_uuid": uuid,
        "source_revision": source["revision"], "fp8_manifest_sha256": sha256(exported),
        "packages": {name: version(name) for name in ("torch", "vllm", "transformers")},
        "server_script_sha256": sha256(root / "scripts/serve_day6_profile.sh"),
        "client_script_sha256": sha256(Path(__file__)),
        "tokenizer_files_sha256": {
            name: sha256(tokenizer_path / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
            if (tokenizer_path / name).is_file()
        },
        "capture_intent": "One requested prefill iteration; eight requested decode iterations after skipping eight engine iterations.",
        "interpretation": "Phase boundaries must be checked using saved annotation/trace evidence. GPU kernel sums are not end-to-end latency.",
    }
    # JSON round-trip normalizes tuples before comparing with an earlier saved protocol.
    protocol = json.loads(json.dumps(protocol))
    protocol_path = base / "phase_protocol_v1.json"
    freeze_json(protocol_path, protocol)
    print("PASS: fixed request and common profiling protocol verified.", flush=True)

    def complete():
        response = api("/v1/completions", {**request_body, "model": alias}, timeout=300)
        usage = response["usage"]
        check(usage["prompt_tokens"] == 1024 and usage["completion_tokens"] == 128,
              f"Unexpected token usage: {usage}")
        return response

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = directory / "captures" / stamp
    session.mkdir(parents=True)
    print("Warming up: four requests, 1024 input / 128 output tokens.", flush=True)
    for index in range(4):
        save_json(session / f"warmup_{index+1}.json", complete())
    records = []
    for repetition in range(1, args.repetitions + 1):
        for phase, (delay, limit) in WINDOWS.items():
            prefix = f"{args.method.lower()}_{phase}_r{repetition}_{stamp}"
            settings = {"profile_prefix": prefix, "delay_iterations": delay, "max_iterations": limit}
            save_json(session / f"{phase}_r{repetition}_start.json", settings)
            print(f"Capturing {phase}: delay={delay}, maximum iterations={limit}", flush=True)
            api("/start_profile", settings)
            try:
                response = complete()
                save_json(session / f"{phase}_r{repetition}_response.json", response)
            finally:
                print("Stopping profiler and writing trace...", flush=True)
                api("/stop_profile", {}, timeout=600)

            files = sorted(directory.glob(prefix + "*.pt.trace.json*"))
            check(files, "No trace with the requested session prefix. Check whether /start_profile accepted session settings.")
            traces = [inspect_trace(path) for path in files]
            count = sum(item["gpu_kernel_events"] for item in traces)
            check(count > 0, f"No GPU kernels in {phase} capture")
            record = {
                "method": args.method, "requested_phase": phase, "repetition": repetition,
                "profile_settings": settings, "request_sha256": sha256(request_path),
                "protocol_sha256": sha256(protocol_path), "traces": traces,
                "gpu_kernel_events": count,
                "summed_kernel_ms": sum(item["summed_kernel_ms"] for item in traces),
            }
            records.append(record)
            save_json(session / f"{phase}_r{repetition}_summary.json", record)
            print(f"  GPU kernels: {count}; summed kernel time: {record['summed_kernel_ms']:.3f} ms", flush=True)
            annotations = Counter()
            for trace in traces:
                annotations.update(trace["phase_annotation_candidates"])
            if annotations:
                print("  Annotation candidates:", flush=True)
                for name, count in annotations.most_common(5):
                    print(f"    {count} × {name[:220]}", flush=True)
            else:
                print("  No phase labels found by the name filter; inspect trace before attributing phases.", flush=True)
    save_json(session / "capture_summary.json", {
        "method": args.method, "protocol_sha256": sha256(protocol_path),
        "request_sha256": sha256(request_path), "captures": records,
    })
    print("\nPASS: fixed request, exact token lengths, unique session traces, and GPU events verified.")
    print("Phase attribution remains subject to the annotation/trace check.")
    print("Saved:", session)


if __name__ == "__main__":
    main()
