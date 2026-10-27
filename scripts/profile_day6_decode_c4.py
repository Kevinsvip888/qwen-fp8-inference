#!/usr/bin/env python3
"""Capture bounded, steady decode windows at concurrency four.

Requires the existing Day 6 profiling server and profile_day6_phases.py next
to this script. Reuses the frozen C1 prompt and validates trace annotations
for four generation requests and zero context requests. Does not change the
Day 5 benchmark or the C1 protocol. Measured kernel sums are diagnostic work,
not unprofiled request latency.
"""

import argparse
import gzip
import json
import statistics
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from profile_day6_phases import check, freeze_json, inspect_trace, save_json, sha256


DELAY = 16
LIMIT = 8
CONCURRENCY = 4
EXPECTED_LABEL = "execute_context_0(0)_generation_4(4)"


def read_capture(directory, prefix):
    files = sorted(directory.glob(prefix + "*.pt.trace.json*"))
    check(files, "No trace found with the requested session prefix")
    labels = Counter()
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            events = json.load(handle).get("traceEvents", [])
        for event in events:
            name = event.get("name", "")
            if event.get("ph") == "X" and name.startswith("execute_context_"):
                labels[name] += 1
    print("  Phase annotations:", dict(labels), flush=True)
    check(labels and set(labels) == {EXPECTED_LABEL},
          "Capture was not exclusively four-request decode. Traces were kept; inspect labels before comparing.")
    traces = [inspect_trace(path) for path in files]
    kernels = sum(trace["gpu_kernel_events"] for trace in traces)
    check(kernels > 0, "No GPU kernel events captured")
    return {
        "traces": traces, "phase_annotations": dict(labels),
        "gpu_kernel_events": kernels,
        "summed_kernel_ms": sum(trace["summed_kernel_ms"] for trace in traces),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("BF16", "Random32"), required=True)
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=3)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/qwen-inference-lab"))
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "outputs/profiling/day6"
    alias = "qwen-profile-bf16" if args.method == "BF16" else "qwen-profile-fp8"
    pointer = root / f"logs/profiling/day6/{args.method}_trace_dir.txt"
    directory = Path(pointer.read_text().strip()).resolve()
    check(directory.is_relative_to(base.resolve()) and directory.is_dir(), "Invalid trace directory")

    parent_path = base / "phase_protocol_v1.json"
    parent = json.loads(parent_path.read_text())
    request_path = base / "phase_request_v1.json"
    request_body = json.loads(request_path.read_text())
    check(sha256(request_path) == parent["request_sha256"], "Frozen prompt changed")
    check(len(request_body["prompt"]) == 1024 and request_body["max_tokens"] == 128,
          "Unexpected request lengths")
    bindings = (
        ("server_script_sha256", root / "scripts/serve_day6_profile.sh"),
        ("client_script_sha256", root / "scripts/profile_day6_phases.py"),
        ("fp8_manifest_sha256", root / "models" /
         "Qwen3-4B-Instruct-2507-FP8-Random32-seed42/calibration_manifest.json"),
    )
    for key, path in bindings:
        check(sha256(path) == parent[key], f"Changed file: {path}")
    packages = {name: version(name) for name in ("torch", "vllm", "transformers")}
    check(packages == parent["packages"], "Serving package versions changed")
    uuid = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    check(uuid == parent["gpu_uuid"], "GPU UUID changed")

    def api(path, payload=None, timeout=120):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request("http://127.0.0.1:8000" + path, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            # Each calling thread owns its opener; no shared HTTP handler state.
            with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
                body = response.read()
                return json.loads(body) if body.strip() else {}
        except HTTPError as error:
            body = error.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code} at {path}: {body[:2000]}") from error

    models = api("/v1/models")
    check([item["id"] for item in models["data"]] == [alias], f"Expected server alias: {alias}")
    protocol = {
        "version": 1, "phase": "steady_decode_c4",
        "parent_protocol_sha256": sha256(parent_path),
        "request_sha256": sha256(request_path),
        "input_tokens_per_request": 1024, "output_tokens_per_request": 128,
        "concurrency": CONCURRENCY, "requests_per_batch": CONCURRENCY,
        "warmup_batches": 4, "delay_iterations": DELAY, "max_iterations": LIMIT,
        "expected_phase_annotation": EXPECTED_LABEL,
        "gpu_uuid": uuid, "packages": packages,
        "client_script_sha256": sha256(Path(__file__)),
        "interpretation": "Fixed-prompt GPU profiling workload. Kernel sums are not end-to-end latency or the exact Day 5 synthetic prompt set.",
    }
    protocol_path = base / "decode_c4_protocol_v1.json"
    freeze_json(protocol_path, protocol)
    print("PASS: same GPU, environment, server source, and fixed prompt verified.", flush=True)

    def batch():
        barrier = threading.Barrier(CONCURRENCY, timeout=30)

        def generate(index):
            barrier.wait()
            response = api("/v1/completions", {**request_body, "model": alias}, timeout=300)
            usage = response["usage"]
            check(usage["prompt_tokens"] == 1024 and usage["completion_tokens"] == 128,
                  f"Unexpected request {index} token usage: {usage}")
            return response

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [executor.submit(generate, index) for index in range(CONCURRENCY)]
            return [future.result() for future in futures]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = directory / "captures_c4" / stamp
    session.mkdir(parents=True)
    print("Warming up: four batches of four simultaneous requests.", flush=True)
    for index in range(4):
        save_json(session / f"warmup_batch_{index+1}.json", batch())

    records = []
    for repetition in range(1, args.repetitions + 1):
        prefix = f"{args.method.lower()}_decode_c4_r{repetition}_{stamp}"
        settings = {"profile_prefix": prefix, "delay_iterations": DELAY, "max_iterations": LIMIT}
        save_json(session / f"r{repetition}_start.json", settings)
        print(f"Capturing repetition {repetition}: skip {DELAY}, record {LIMIT} engine steps.", flush=True)
        api("/start_profile", settings)
        try:
            responses = batch()
            save_json(session / f"r{repetition}_responses.json", responses)
        finally:
            print("Stopping profiler and writing trace...", flush=True)
            api("/stop_profile", {}, timeout=600)

        record = read_capture(directory, prefix)
        record.update({
            "method": args.method, "requested_phase": "steady_decode_c4",
            "repetition": repetition, "profile_settings": settings,
            "request_sha256": sha256(request_path), "protocol_sha256": sha256(protocol_path),
        })
        save_json(session / f"r{repetition}_summary.json", record)
        records.append(record)
        print(f"  GPU kernels: {record['gpu_kernel_events']}; summed kernel time: {record['summed_kernel_ms']:.3f} ms", flush=True)
    totals = [record["summed_kernel_ms"] for record in records]
    summary = {
        "method": args.method, "protocol_sha256": sha256(protocol_path),
        "request_sha256": sha256(request_path), "captures": records,
        "mean_summed_kernel_ms": statistics.mean(totals),
        "sample_sd_ms": statistics.stdev(totals) if len(totals) > 1 else None,
        "note": "GPU work covers eight engine steps for four active requests. It is not a per-request sum or wall-clock latency.",
    }
    save_json(session / "capture_summary.json", summary)
    print(f"\nMean GPU kernel work across eight steps: {statistics.mean(totals):.3f} ms")
    print("PASS: exact token counts and exclusively four-request decode labels verified.")
    print("Saved:", session)


if __name__ == "__main__":
    main()
